"""Tests for Polymarket statistical-edge utilities."""

from __future__ import annotations

import pandas as pd

from src.analysis.polymarket.util.stat_edge import (
    SignalThresholds,
    assign_rel_notional_bin,
    assign_rel_price_z_bin,
    assign_tte_bucket,
    bh_fdr,
    compute_signal_candidates,
)


def test_bucket_assignment_boundaries():
    assert assign_tte_bucket(0) == "[0,5m)"
    assert assign_tte_bucket(299) == "[0,5m)"
    assert assign_tte_bucket(300) == "[5,15m)"
    assert assign_tte_bucket(900) == "[15,60m)"
    assert assign_tte_bucket(3600) == "[1,4h)"
    assert assign_tte_bucket(259200) == "[72h+]"

    assert assign_rel_price_z_bin(-3.0) == "(-inf,-2]"
    assert assign_rel_price_z_bin(-2.0) == "(-inf,-2]"
    assert assign_rel_price_z_bin(-1.5) == "(-2,-1]"
    assert assign_rel_price_z_bin(0.0) == "(-1,1]"
    assert assign_rel_price_z_bin(1.8) == "(1,2]"
    assert assign_rel_price_z_bin(3.0) == "(2,inf)"

    assert assign_rel_notional_bin(0) == "[0,50)"
    assert assign_rel_notional_bin(49.9) == "[0,50)"
    assert assign_rel_notional_bin(50) == "[50,80)"
    assert assign_rel_notional_bin(80) == "[80,95)"
    assert assign_rel_notional_bin(95) == "[95,100]"
    assert assign_rel_notional_bin(100) == "[95,100]"


def test_bh_fdr_known_case():
    q = bh_fdr(pd.Series([0.01, 0.02, 0.20]))
    assert q.round(4).tolist() == [0.03, 0.03, 0.2]


def _build_signal_rows(
    signal_suffix: str,
    rel_notional_bin: str,
    val_positive: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    # 40 markets -> satisfies min market threshold.
    markets = [f"m_{signal_suffix}_{i}" for i in range(40)]

    def add_rows(split: str, count: int, success_mod: int):
        for i in range(count):
            market_id = markets[i % len(markets)]
            won = 1.0 if i % 10 < success_mod else 0.0
            p = 0.40
            rows.append(
                {
                    "market_id": market_id,
                    "split": split,
                    "edge": won - p,
                    "won": won,
                    "p": p,
                    "notional": 10.0 + (i % 7),
                    "tenor_bucket": "1h",
                    "tte_bucket": "[15,60m)",
                    "rel_price_z_bin": "(-1,1]",
                    "rel_notional_bin": rel_notional_bin,
                }
            )

    add_rows(split="train", count=400, success_mod=7)  # +0.30 mean edge
    add_rows(split="val", count=200, success_mod=7 if val_positive else 2)
    add_rows(split="test", count=200, success_mod=7)
    return rows


def test_compute_signal_candidates_final_and_exploratory():
    rows = []
    rows.extend(_build_signal_rows(signal_suffix="final", rel_notional_bin="[80,95)", val_positive=True))
    rows.extend(_build_signal_rows(signal_suffix="explore", rel_notional_bin="[95,100]", val_positive=False))
    df = pd.DataFrame(rows)

    thresholds = SignalThresholds(
        min_trades_train=300,
        min_markets_train=30,
        min_abs_ev_train=0.0075,
        max_q_value=0.10,
    )
    candidates = compute_signal_candidates(df, thresholds=thresholds)

    # Both signals pass train gate.
    kept = candidates[candidates["candidate_status"] != "rejected"]
    assert len(kept) == 2

    status_map = dict(zip(kept["rel_notional_bin"], kept["candidate_status"]))
    assert status_map["[80,95)"] == "final"
    assert status_map["[95,100]"] == "exploratory"

    # Required output fields are present.
    required = {
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
    }
    assert required.issubset(set(candidates.columns))
