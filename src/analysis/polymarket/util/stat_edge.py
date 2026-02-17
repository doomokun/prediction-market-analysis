"""Statistical helpers for relative-edge signal extraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

TTE_BUCKETS = (
    "[0,5m)",
    "[5,15m)",
    "[15,60m)",
    "[1,4h)",
    "[4,12h)",
    "[12,24h)",
    "[24,72h)",
    "[72h+]",
)

REL_PRICE_Z_BINS = (
    "(-inf,-2]",
    "(-2,-1]",
    "(-1,1]",
    "(1,2]",
    "(2,inf)",
)

REL_NOTIONAL_BINS = (
    "[0,50)",
    "[50,80)",
    "[80,95)",
    "[95,100]",
)

_REL_NOTIONAL_EDGES = (0.0, 50.0, 80.0, 95.0, 100.0000001)


@dataclass(frozen=True)
class SignalThresholds:
    """Thresholds for signal screening."""

    min_trades_train: int = 300
    min_markets_train: int = 30
    min_abs_ev_train: float = 0.0075
    max_q_value: float = 0.10


def assign_tte_bucket(tte_seconds: float | int) -> str:
    """Map time-to-expiry in seconds to a canonical bucket."""
    tte = float(tte_seconds)
    if tte < 5 * 60:
        return "[0,5m)"
    if tte < 15 * 60:
        return "[5,15m)"
    if tte < 60 * 60:
        return "[15,60m)"
    if tte < 4 * 60 * 60:
        return "[1,4h)"
    if tte < 12 * 60 * 60:
        return "[4,12h)"
    if tte < 24 * 60 * 60:
        return "[12,24h)"
    if tte < 72 * 60 * 60:
        return "[24,72h)"
    return "[72h+]"


def assign_rel_price_z_bin(z_value: float | int) -> str:
    """Map relative price z-score to canonical bins."""
    z = float(z_value)
    if z <= -2.0:
        return "(-inf,-2]"
    if z <= -1.0:
        return "(-2,-1]"
    if z <= 1.0:
        return "(-1,1]"
    if z <= 2.0:
        return "(1,2]"
    return "(2,inf)"


def assign_rel_notional_bin(percentile: float | int) -> str:
    """Map notional percentile rank to canonical bins."""
    p = float(percentile)
    if p < 0:
        p = 0
    if p > 100:
        p = 100
    idx = np.searchsorted(_REL_NOTIONAL_EDGES, p, side="right") - 1
    idx = max(0, min(idx, len(REL_NOTIONAL_BINS) - 1))
    return REL_NOTIONAL_BINS[idx]


def blocked_market_split(
    market_end: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> pd.DataFrame:
    """Assign blocked train/val/test splits by market end time."""
    required = {"market_id", "end_ts"}
    missing = required - set(market_end.columns)
    if missing:
        raise ValueError(f"market_end missing columns: {sorted(missing)}")

    df = market_end.copy()
    df["end_ts"] = pd.to_datetime(df["end_ts"], utc=True)
    df = df.sort_values(["end_ts", "market_id"], kind="mergesort").reset_index(drop=True)

    n = len(df)
    train_cut = int(np.floor(n * train_ratio))
    val_cut = int(np.floor(n * (train_ratio + val_ratio)))

    split = np.full(n, "test", dtype=object)
    split[:train_cut] = "train"
    split[train_cut:val_cut] = "val"
    df["split"] = split
    return df[["market_id", "split"]]


def bh_fdr(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg FDR correction."""
    p = pd.to_numeric(p_values, errors="coerce")
    valid_mask = p.notna()
    q = pd.Series(np.nan, index=p.index, dtype=float)
    if valid_mask.sum() == 0:
        return q

    p_valid = p[valid_mask].to_numpy()
    order = np.argsort(p_valid)
    ranked = p_valid[order]
    m = float(len(ranked))

    adjusted = ranked * m / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    back = np.empty_like(adjusted)
    back[order] = adjusted
    q.loc[valid_mask] = back
    return q


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    denom = weights.sum()
    if denom <= 0:
        return float("nan")
    return float((values * weights).sum() / denom)


def _equal_market_mean(df: pd.DataFrame) -> float:
    market_mean = df.groupby("market_id", observed=False)["edge"].mean()
    if market_mean.empty:
        return float("nan")
    return float(market_mean.mean())


def _sign(x: float) -> int:
    if pd.isna(x) or x == 0:
        return 0
    return 1 if x > 0 else -1


def _safe_mean(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(series.mean())


def _ttest_p_value(edges: pd.Series) -> float:
    if len(edges) < 2:
        return float("nan")
    result = stats.ttest_1samp(edges.to_numpy(), popmean=0.0, nan_policy="omit")
    return float(result.pvalue) if result.pvalue is not None else float("nan")


def _effect_size(edges: pd.Series) -> float:
    std = float(edges.std(ddof=1))
    if len(edges) < 2 or std == 0:
        return 0.0
    return float(edges.mean() / std)


def compute_signal_candidates(
    trades: pd.DataFrame,
    thresholds: SignalThresholds | None = None,
) -> pd.DataFrame:
    """Compute candidate signals with train/val/test metrics."""
    if thresholds is None:
        thresholds = SignalThresholds()

    required = {
        "market_id",
        "split",
        "edge",
        "won",
        "p",
        "notional",
        "tenor_bucket",
        "tte_bucket",
        "rel_price_z_bin",
        "rel_notional_bin",
    }
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(f"trades missing columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    group_cols = ["tenor_bucket", "tte_bucket", "rel_price_z_bin", "rel_notional_bin"]

    for key, group in trades.groupby(group_cols, observed=False):
        tenor_bucket, tte_bucket, rel_price_z_bin, rel_notional_bin = key
        train = group[group["split"] == "train"]
        val = group[group["split"] == "val"]
        test = group[group["split"] == "test"]

        n_trades_train = int(len(train))
        n_markets_train = int(train["market_id"].nunique())

        ev_train = _safe_mean(train["edge"])
        ev_val = _safe_mean(val["edge"])
        ev_test = _safe_mean(test["edge"])
        win_gap_test = _safe_mean(test["won"]) - _safe_mean(test["p"])

        eq_market = _equal_market_mean(train)
        eq_notional = _weighted_mean(train["edge"], train["notional"])
        eq_market_sign = _sign(eq_market)
        eq_notional_sign = _sign(eq_notional)
        aligned_sign = eq_market_sign != 0 and eq_market_sign == eq_notional_sign

        p_value = _ttest_p_value(train["edge"])
        effect = _effect_size(train["edge"])

        rule_def = (
            f"tenor={tenor_bucket}, tte={tte_bucket}, "
            f"rel_price_z={rel_price_z_bin}, rel_notional_pct={rel_notional_bin}"
        )
        signal_id = f"{tenor_bucket}|{tte_bucket}|{rel_price_z_bin}|{rel_notional_bin}"

        rows.append(
            {
                "signal_id": signal_id,
                "rule_def": rule_def,
                "tenor_bucket": tenor_bucket,
                "tte_bucket": tte_bucket,
                "rel_price_z_bin": rel_price_z_bin,
                "rel_notional_bin": rel_notional_bin,
                "n_trades_train": n_trades_train,
                "n_markets_train": n_markets_train,
                "ev_train": ev_train,
                "ev_val": ev_val,
                "ev_test": ev_test,
                "win_gap_test": win_gap_test,
                "p_value": p_value,
                "effect_size": effect,
                "equal_market_sign": eq_market_sign,
                "equal_notional_sign": eq_notional_sign,
                "aligned_sign": aligned_sign,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
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
                "candidate_status",
            ]
        )

    result = pd.DataFrame(rows)
    result["q_value"] = bh_fdr(result["p_value"])

    train_pass = (
        (result["n_trades_train"] >= thresholds.min_trades_train)
        & (result["n_markets_train"] >= thresholds.min_markets_train)
        & (result["ev_train"].abs() >= thresholds.min_abs_ev_train)
        & (result["q_value"] < thresholds.max_q_value)
        & result["aligned_sign"]
    )

    train_sign = np.sign(result["ev_train"])
    val_sign = np.sign(result["ev_val"])
    test_sign = np.sign(result["ev_test"])
    val_pass = train_sign.eq(val_sign) & train_sign.ne(0)
    test_pass = train_sign.eq(test_sign) & train_sign.ne(0)

    status = np.where(
        train_pass & val_pass & test_pass,
        "final",
        np.where(train_pass, "exploratory", "rejected"),
    )
    result["candidate_status"] = status

    result = result.sort_values(
        ["candidate_status", "q_value", "n_trades_train", "signal_id"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return result
