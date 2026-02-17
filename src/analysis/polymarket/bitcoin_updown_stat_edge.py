"""Relative statistical edge analysis for Polymarket Bitcoin Up/Down markets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.figure import Figure

from src.analysis.polymarket.util import (
    REL_PRICE_Z_BINS,
    TTE_BUCKETS,
    assign_rel_notional_bin,
    assign_rel_price_z_bin,
    assign_tte_bucket,
    blocked_market_split,
    compute_signal_candidates,
    is_up_down_outcomes,
    parse_resolution,
    parse_tenor_bucket,
)
from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


class BitcoinUpDownStatEdgeAnalysis(Analysis):
    """Find relative statistical edge in Polymarket Bitcoin Up/Down markets."""

    EDGE_COLUMNS = [
        "signal_id",
        "rule_def",
        "n_trades_train",
        "n_markets_train",
        "ev_train",
        "ev_val",
        "ev_test",
        "win_gap_test",
        "q_value",
        "effect_size",
        "equal_market_sign",
        "equal_notional_sign",
    ]

    OUTPUT_COLUMNS = [
        "market_id",
        "question",
        "tenor_bucket",
        "trade_ts",
        "end_ts",
        "tte_bucket",
        "token_id",
        "outcome_label",
        "p",
        "shares",
        "notional",
        "won",
        "edge",
        "edge_bps",
        "peer_key",
        "rel_price_z",
        "rel_notional_pct",
    ]

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
        blocks_dir: Path | str | None = None,
    ):
        super().__init__(
            name="bitcoin_updown_stat_edge",
            description="Relative statistical edge analysis for Polymarket Bitcoin Up/Down markets",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "polymarket" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "polymarket" / "markets")
        self.blocks_dir = Path(blocks_dir or base_dir / "data" / "polymarket" / "blocks")

        self._last_coverage: dict[str, Any] = {}
        self._last_edge_candidates = pd.DataFrame(columns=self.EDGE_COLUMNS)
        self._last_heatmap_figure: Figure | None = None

    def run(self) -> AnalysisOutput:
        """Execute the analysis and return outputs."""
        con = duckdb.connect()

        markets_df = con.execute(
            f"""
            SELECT id, question, outcomes, outcome_prices, clob_token_ids, end_date
            FROM '{self.markets_dir}/*.parquet'
            WHERE lower(question) LIKE 'bitcoin up or down%'
            """
        ).df()

        token_labels_df, all_tokens, coverage_base = self._build_market_tables(markets_df)
        coverage = self._compute_coverage(con, all_tokens, coverage_base)
        self._last_coverage = coverage

        if token_labels_df.empty:
            self._last_edge_candidates = pd.DataFrame(columns=self.EDGE_COLUMNS)
            self._last_heatmap_figure = self._create_empty_heatmap()
            return AnalysisOutput(
                figure=self._create_empty_main_figure(),
                data=pd.DataFrame(columns=self.OUTPUT_COLUMNS),
                chart=self._empty_chart(),
                metadata={"coverage": coverage, "message": "No resolved Up/Down markets for analysis."},
            )

        con.register("token_labels", token_labels_df)
        trades_df = con.execute(
            f"""
            WITH ctf AS (
                SELECT
                    block_number,
                    CASE WHEN maker_asset_id = '0' THEN taker_asset_id ELSE maker_asset_id END AS token_id,
                    CASE
                        WHEN maker_asset_id = '0' THEN maker_amount::DOUBLE / NULLIF(taker_amount::DOUBLE, 0)
                        ELSE taker_amount::DOUBLE / NULLIF(maker_amount::DOUBLE, 0)
                    END AS p,
                    CASE
                        WHEN maker_asset_id = '0' THEN taker_amount::DOUBLE
                        ELSE maker_amount::DOUBLE
                    END / 1e6 AS shares
                FROM '{self.trades_dir}/*.parquet'
                WHERE (maker_asset_id = '0' OR taker_asset_id = '0')
                  AND maker_amount > 0
                  AND taker_amount > 0
            ),
            joined AS (
                SELECT
                    tl.market_id,
                    tl.question,
                    tl.tenor_bucket,
                    CAST(b.timestamp AS TIMESTAMP) AS trade_ts,
                    tl.end_ts,
                    tl.token_id,
                    tl.outcome_label,
                    tl.won,
                    ctf.p,
                    ctf.shares,
                    ctf.p * ctf.shares AS notional,
                    DATEDIFF('second', CAST(b.timestamp AS TIMESTAMP), tl.end_ts) AS tte_seconds
                FROM ctf
                INNER JOIN token_labels tl ON ctf.token_id = tl.token_id
                LEFT JOIN '{self.blocks_dir}/*.parquet' b ON ctf.block_number = b.block_number
                WHERE b.timestamp IS NOT NULL
            )
            SELECT
                market_id,
                question,
                tenor_bucket,
                trade_ts,
                end_ts,
                token_id,
                outcome_label,
                won,
                p,
                shares,
                notional,
                tte_seconds
            FROM joined
            WHERE tte_seconds >= 0
              AND p >= 0
              AND p <= 1
            """
        ).df()

        if trades_df.empty:
            self._last_edge_candidates = pd.DataFrame(columns=self.EDGE_COLUMNS)
            self._last_heatmap_figure = self._create_empty_heatmap()
            return AnalysisOutput(
                figure=self._create_empty_main_figure(),
                data=pd.DataFrame(columns=self.OUTPUT_COLUMNS),
                chart=self._empty_chart(),
                metadata={"coverage": coverage, "message": "No mapped trade rows after filtering."},
            )

        trades_df["trade_ts"] = pd.to_datetime(trades_df["trade_ts"], utc=True)
        trades_df["end_ts"] = pd.to_datetime(trades_df["end_ts"], utc=True)
        trades_df["tte_bucket"] = trades_df["tte_seconds"].map(assign_tte_bucket)
        trades_df["peer_key"] = trades_df["tenor_bucket"] + "|" + trades_df["tte_bucket"]

        peer_median = trades_df.groupby("peer_key", observed=False)["p"].transform("median")
        peer_mad = trades_df.groupby("peer_key", observed=False)["p"].transform(
            lambda s: np.median(np.abs(s - np.median(s)))
        )
        trades_df["peer_size"] = trades_df.groupby("peer_key", observed=False)["p"].transform("size")
        trades_df["rel_price_z"] = np.where(peer_mad > 0, (trades_df["p"] - peer_median) / peer_mad, 0.0)
        trades_df["rel_notional_pct"] = (
            trades_df.groupby("peer_key", observed=False)["notional"].rank(method="average", pct=True) * 100.0
        )
        trades_df["rel_price_z_bin"] = trades_df["rel_price_z"].map(assign_rel_price_z_bin)
        trades_df["rel_notional_bin"] = trades_df["rel_notional_pct"].map(assign_rel_notional_bin)

        trades_df["edge"] = trades_df["won"].astype(float) - trades_df["p"]
        trades_df["edge_bps"] = 10000.0 * trades_df["edge"]

        market_split = blocked_market_split(trades_df[["market_id", "end_ts"]].drop_duplicates())
        trades_df = trades_df.merge(market_split, on="market_id", how="left")

        candidate_input = trades_df[trades_df["peer_size"] >= 100].copy()
        candidates = compute_signal_candidates(candidate_input)
        self._last_edge_candidates = candidates

        main_figure, chart = self._create_main_figure_and_chart(trades_df)
        self._last_heatmap_figure = self._create_heatmap(candidates, trades_df)

        output_df = trades_df[self.OUTPUT_COLUMNS].copy()

        metadata = {
            "coverage": coverage,
            "candidate_counts": {
                "final": int((candidates["candidate_status"] == "final").sum()) if not candidates.empty else 0,
                "exploratory": int((candidates["candidate_status"] == "exploratory").sum())
                if not candidates.empty
                else 0,
            },
            "rows": int(len(output_df)),
        }

        return AnalysisOutput(figure=main_figure, data=output_df, chart=chart, metadata=metadata)

    def save(
        self,
        output_dir: Path | str,
        formats: list[str] | None = None,
        dpi: int = 300,
    ) -> dict[str, Path]:
        """Save standard outputs plus edge-candidate and coverage sidecar files."""
        saved = super().save(output_dir=output_dir, formats=formats, dpi=dpi)
        output_dir = Path(output_dir)
        selected_formats = set(formats or ["png", "pdf", "csv"])

        if "csv" in selected_formats:
            candidate_path = output_dir / "bitcoin_updown_edge_candidates.csv"
            candidate_df = self._last_edge_candidates.copy()
            candidate_df = candidate_df[candidate_df["candidate_status"] != "rejected"]
            if candidate_df.empty:
                candidate_df = pd.DataFrame(columns=self.EDGE_COLUMNS)
            candidate_df[self.EDGE_COLUMNS].to_csv(candidate_path, index=False)
            saved["csv_candidates"] = candidate_path

        if "json" in selected_formats:
            coverage_path = output_dir / "bitcoin_updown_coverage.json"
            coverage_path.write_text(json.dumps(self._last_coverage, indent=2))
            saved["json_coverage"] = coverage_path

        if "png" in selected_formats and self._last_heatmap_figure is not None:
            heatmap_path = output_dir / "bitcoin_updown_relative_edge_heatmap.png"
            self._last_heatmap_figure.savefig(heatmap_path, dpi=dpi, bbox_inches="tight")
            saved["png_heatmap"] = heatmap_path
            plt.close(self._last_heatmap_figure)
            self._last_heatmap_figure = None

        return saved

    def _build_market_tables(self, markets_df: pd.DataFrame) -> tuple[pd.DataFrame, set[str], dict[str, int]]:
        token_rows: list[dict[str, Any]] = []
        all_tokens: set[str] = set()

        market_total = 0
        resolved_market_total = 0

        for row in markets_df.to_dict(orient="records"):
            outcomes_json = row.get("outcomes")
            if not is_up_down_outcomes(outcomes_json):
                continue

            try:
                token_ids_raw = json.loads(row.get("clob_token_ids", "[]"))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(token_ids_raw, list) or len(token_ids_raw) != 2:
                continue

            token_ids = [str(token_ids_raw[0]), str(token_ids_raw[1])]
            if not token_ids[0] or not token_ids[1]:
                continue

            market_total += 1
            all_tokens.update(token_ids)

            resolution = parse_resolution(row.get("outcome_prices"), outcomes_json)
            if not resolution.resolved or resolution.winner_index is None:
                continue

            resolved_market_total += 1
            tenor_bucket = parse_tenor_bucket(row.get("question"))
            if tenor_bucket == "other":
                continue

            end_ts = pd.to_datetime(row.get("end_date"), utc=True, errors="coerce")
            if pd.isna(end_ts):
                continue
            end_ts = end_ts.tz_convert("UTC").tz_localize(None)
            outcomes = resolution.outcomes or ("Up", "Down")
            winner_index = int(resolution.winner_index)

            for idx, token_id in enumerate(token_ids):
                token_rows.append(
                    {
                        "token_id": token_id,
                        "market_id": str(row["id"]),
                        "question": str(row["question"]),
                        "tenor_bucket": tenor_bucket,
                        "end_ts": end_ts,
                        "outcome_label": outcomes[idx],
                        "won": idx == winner_index,
                    }
                )

        token_labels_df = pd.DataFrame(token_rows)
        coverage_base = {
            "market_total": market_total,
            "resolved_market_total": resolved_market_total,
            "token_total": len(all_tokens),
        }
        return token_labels_df, all_tokens, coverage_base

    def _compute_coverage(
        self,
        con: duckdb.DuckDBPyConnection,
        all_tokens: set[str],
        coverage_base: dict[str, int],
    ) -> dict[str, Any]:
        if not all_tokens:
            return {
                **coverage_base,
                "token_seen_in_trades": 0,
                "token_coverage_rate": 0.0,
                "trade_rows_mapped": 0,
                "unmapped_trade_rows": 0,
            }

        coverage_tokens = pd.DataFrame({"token_id": sorted(all_tokens)})
        con.register("coverage_tokens", coverage_tokens)

        total_ctf_rows = con.execute(
            f"""
            SELECT COUNT(*) AS n
            FROM '{self.trades_dir}/*.parquet'
            WHERE maker_asset_id = '0' OR taker_asset_id = '0'
            """
        ).fetchone()[0]

        mapped_stats = con.execute(
            f"""
            WITH ctf AS (
                SELECT CASE WHEN maker_asset_id = '0' THEN taker_asset_id ELSE maker_asset_id END AS token_id
                FROM '{self.trades_dir}/*.parquet'
                WHERE maker_asset_id = '0' OR taker_asset_id = '0'
            )
            SELECT
                COUNT(*) AS mapped_rows,
                COUNT(DISTINCT ctf.token_id) AS token_seen
            FROM ctf
            INNER JOIN coverage_tokens ct ON ctf.token_id = ct.token_id
            """
        ).fetchone()

        mapped_rows = int(mapped_stats[0] or 0)
        token_seen = int(mapped_stats[1] or 0)
        token_total = max(int(coverage_base["token_total"]), 1)

        return {
            **coverage_base,
            "token_seen_in_trades": token_seen,
            "token_coverage_rate": round(token_seen / token_total, 6),
            "trade_rows_mapped": mapped_rows,
            "unmapped_trade_rows": int(total_ctf_rows - mapped_rows),
        }

    def _create_main_figure_and_chart(self, trades_df: pd.DataFrame) -> tuple[Figure, ChartConfig]:
        focus_df = trades_df[trades_df["split"] == "test"].copy()
        if focus_df.empty:
            focus_df = trades_df.copy()

        def _equal_notional(group: pd.DataFrame) -> float:
            total_notional = float(group["notional"].sum())
            if total_notional <= 0:
                return float("nan")
            return float((group["edge"] * group["notional"]).sum() / total_notional)

        def _equal_market(group: pd.DataFrame) -> float:
            market_edges = group.groupby("market_id", observed=False)["edge"].mean()
            if market_edges.empty:
                return float("nan")
            return float(market_edges.mean())

        grouped = []
        for rel_bin in REL_PRICE_Z_BINS:
            subset = focus_df[focus_df["rel_price_z_bin"] == rel_bin]
            grouped.append(
                {
                    "rel_price_z_bin": rel_bin,
                    "equal_market_edge_bps": _equal_market(subset) * 10000.0 if not subset.empty else np.nan,
                    "equal_notional_edge_bps": _equal_notional(subset) * 10000.0 if not subset.empty else np.nan,
                }
            )
        plot_df = pd.DataFrame(grouped)

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(plot_df))
        width = 0.38

        ax.bar(
            x - width / 2,
            plot_df["equal_market_edge_bps"],
            width=width,
            label="Equal-Market",
            color="#4C72B0",
        )
        ax.bar(
            x + width / 2,
            plot_df["equal_notional_edge_bps"],
            width=width,
            label="Equal-Notional",
            color="#55A868",
        )

        ax.axhline(0, color="#444444", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["rel_price_z_bin"], rotation=15)
        ax.set_ylabel("Edge (bps)")
        ax.set_xlabel("Relative Price Z-Score Bin")
        ax.set_title("Bitcoin Up/Down Relative Edge by Price-Deviation Bin")
        ax.legend(loc="best")
        plt.tight_layout()

        chart = ChartConfig(
            type=ChartType.BAR,
            data=plot_df.to_dict(orient="records"),
            xKey="rel_price_z_bin",
            yKeys=["equal_market_edge_bps", "equal_notional_edge_bps"],
            title="Bitcoin Up/Down Relative Edge by Price-Deviation Bin",
            xLabel="Relative Price Z-Score Bin",
            yLabel="Edge (bps)",
            yUnit=UnitType.NUMBER,
        )
        return fig, chart

    def _create_heatmap(self, candidates: pd.DataFrame, trades_df: pd.DataFrame) -> Figure:
        source = candidates[candidates["candidate_status"] != "rejected"].copy()
        if not source.empty:
            heat_df = source.groupby(["tte_bucket", "rel_price_z_bin"], observed=False)["ev_test"].mean().reset_index()
            heat_df["value"] = heat_df["ev_test"] * 10000.0
        else:
            focus_df = trades_df[trades_df["split"] == "test"].copy()
            if focus_df.empty:
                focus_df = trades_df.copy()
            heat_df = (
                focus_df.groupby(["tte_bucket", "rel_price_z_bin"], observed=False)["edge"]
                .mean()
                .reset_index()
                .rename(columns={"edge": "value"})
            )
            heat_df["value"] = heat_df["value"] * 10000.0

        pivot = (
            heat_df.pivot(index="tte_bucket", columns="rel_price_z_bin", values="value")
            .reindex(index=TTE_BUCKETS, columns=REL_PRICE_Z_BINS)
            .astype(float)
        )

        vals = pivot.to_numpy()
        vmin = np.nanmin(vals) if not np.isnan(vals).all() else -1.0
        vmax = np.nanmax(vals) if not np.isnan(vals).all() else 1.0
        if vmin == vmax:
            vmax = vmin + 1e-6

        fig, ax = plt.subplots(figsize=(12, 6))
        if vmin < 0 < vmax:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
        im = ax.imshow(vals, aspect="auto", cmap="RdBu_r", norm=norm)
        fig.colorbar(im, ax=ax, label="EV Test (bps)")

        ax.set_xticks(range(len(REL_PRICE_Z_BINS)))
        ax.set_xticklabels(REL_PRICE_Z_BINS, rotation=15)
        ax.set_yticks(range(len(TTE_BUCKETS)))
        ax.set_yticklabels(TTE_BUCKETS)
        ax.set_xlabel("Relative Price Z-Score Bin")
        ax.set_ylabel("Time-To-Expiry Bucket")
        ax.set_title("Bitcoin Up/Down Relative Edge Heatmap (EV Test, bps)")

        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                if np.isnan(vals[i, j]):
                    continue
                ax.text(j, i, f"{vals[i, j]:.1f}", ha="center", va="center", color="black", fontsize=8)

        plt.tight_layout()
        return fig

    def _create_empty_main_figure(self) -> Figure:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No eligible Up/Down data for edge analysis", ha="center", va="center")
        ax.axis("off")
        plt.tight_layout()
        return fig

    def _create_empty_heatmap(self) -> Figure:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No candidate signals for heatmap", ha="center", va="center")
        ax.axis("off")
        plt.tight_layout()
        return fig

    def _empty_chart(self) -> ChartConfig:
        return ChartConfig(
            type=ChartType.BAR,
            data=[],
            xKey="rel_price_z_bin",
            yKeys=["equal_market_edge_bps", "equal_notional_edge_bps"],
            title="Bitcoin Up/Down Relative Edge by Price-Deviation Bin",
            yUnit=UnitType.NUMBER,
        )
