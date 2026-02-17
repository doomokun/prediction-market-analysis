"""Tests for Bitcoin Up/Down parser helpers."""

from __future__ import annotations

import json

from src.analysis.polymarket.util.bitcoin_updown_parser import (
    is_up_down_outcomes,
    parse_resolution,
    parse_tenor_bucket,
)


def test_parse_tenor_bucket_variants():
    assert parse_tenor_bucket("Bitcoin Up or Down - February 3, 5:00PM-5:05PM ET") == "5m"
    assert parse_tenor_bucket("Bitcoin Up or Down - February 3, 5:00PM-5:15PM ET") == "15m"
    assert parse_tenor_bucket("Bitcoin Up or Down - February 3, 5PM ET") == "1h"
    assert parse_tenor_bucket("Bitcoin Up or Down on February 3?") == "1d"
    assert parse_tenor_bucket("Bitcoin Up or Down - February 3, 5:00PM-5:30PM ET") == "other"


def test_is_up_down_outcomes_normalization():
    assert is_up_down_outcomes(json.dumps(["Up", "Down"]))
    assert is_up_down_outcomes(json.dumps(["Up", "Down "]))
    assert not is_up_down_outcomes(json.dumps(["Yes", "No"]))


def test_parse_resolution_thresholds():
    resolved = parse_resolution(json.dumps([1.0, 0.0]), json.dumps(["Up", "Down"]))
    assert resolved.resolved is True
    assert resolved.winner_index == 0
    assert resolved.winner_label == "Up"

    borderline = parse_resolution(json.dumps([0.999, 0.001]), json.dumps(["Up", "Down"]))
    assert borderline.resolved is True
    assert borderline.winner_index == 0

    unresolved = parse_resolution(json.dumps([0.998, 0.002]), json.dumps(["Up", "Down"]))
    assert unresolved.resolved is False
    assert unresolved.winner_index is None
