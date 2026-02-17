"""Integration tests for Bitcoin Up/Down stat-edge analysis."""

from __future__ import annotations

import json
from pathlib import Path

from src.analysis.polymarket.bitcoin_updown_stat_edge import BitcoinUpDownStatEdgeAnalysis
from src.common.analysis import AnalysisOutput


def test_run_outputs_expected_schema(
    polymarket_trades_dir: Path,
    polymarket_markets_dir: Path,
    polymarket_blocks_dir: Path,
):
    analysis = BitcoinUpDownStatEdgeAnalysis(
        trades_dir=polymarket_trades_dir,
        markets_dir=polymarket_markets_dir,
        blocks_dir=polymarket_blocks_dir,
    )
    output = analysis.run()

    assert isinstance(output, AnalysisOutput)
    assert output.data is not None
    assert not output.data.empty

    expected_cols = {
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
    }
    assert expected_cols.issubset(set(output.data.columns))

    # Token-to-resolution mapping check from fixture setup:
    # token_yes_a resolves as winner; token_yes_b resolves as loser.
    yes_a = output.data[output.data["token_id"] == "token_yes_a"]
    yes_b = output.data[output.data["token_id"] == "token_yes_b"]
    assert not yes_a.empty
    assert not yes_b.empty
    assert yes_a["won"].all()
    assert (~yes_b["won"]).all()

    coverage = output.metadata["coverage"]
    for key in [
        "market_total",
        "resolved_market_total",
        "token_total",
        "token_seen_in_trades",
        "token_coverage_rate",
        "trade_rows_mapped",
        "unmapped_trade_rows",
    ]:
        assert key in coverage


def test_save_writes_sidecar_files(
    tmp_path: Path,
    polymarket_trades_dir: Path,
    polymarket_markets_dir: Path,
    polymarket_blocks_dir: Path,
):
    analysis = BitcoinUpDownStatEdgeAnalysis(
        trades_dir=polymarket_trades_dir,
        markets_dir=polymarket_markets_dir,
        blocks_dir=polymarket_blocks_dir,
    )
    saved = analysis.save(tmp_path, formats=["png", "csv", "json"])

    assert "csv" in saved
    assert "csv_candidates" in saved
    assert "json_coverage" in saved
    assert "png_heatmap" in saved
    assert saved["csv"].exists()
    assert saved["csv_candidates"].exists()
    assert saved["json_coverage"].exists()
    assert saved["png_heatmap"].exists()

    coverage = json.loads(saved["json_coverage"].read_text())
    assert "token_coverage_rate" in coverage
