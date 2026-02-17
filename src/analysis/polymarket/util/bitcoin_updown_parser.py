"""Parsing helpers for Polymarket Bitcoin Up/Down market metadata."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

RESOLUTION_HIGH_THRESHOLD = 0.999
RESOLUTION_LOW_THRESHOLD = 0.001

_QUESTION_PREFIX = "bitcoin up or down"
_HOURLY_RE = re.compile(
    r"^bitcoin up or down\s*-\s*[a-z]+\s+\d{1,2},\s+\d{1,2}(?::\d{2})?\s*[ap]m\s*et$",
    re.IGNORECASE,
)
_DAILY_RE = re.compile(
    r"^bitcoin up or down(?:\s+on)?\s+[a-z]+\s+\d{1,2}(?:,?\s+\d{4})?\??$",
    re.IGNORECASE,
)
_RANGE_RE = re.compile(
    r"^bitcoin up or down\s*-\s*[a-z]+\s+\d{1,2},\s*"
    r"(?P<start>\d{1,2}(?::\d{2})?\s*[ap]m)\s*-\s*"
    r"(?P<end>\d{1,2}(?::\d{2})?\s*[ap]m)\s*et$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ResolutionResult:
    """Resolution parsing result for a binary market."""

    resolved: bool
    winner_index: int | None
    winner_label: str | None
    prices: tuple[float, float] | None
    outcomes: tuple[str, str] | None


def _parse_time_minutes(clock: str) -> int:
    """Convert a 12-hour time string (e.g. 5:15PM) into minutes since midnight."""
    text = clock.strip().upper().replace(" ", "")
    if ":" in text:
        hour_text, minute_ampm = text.split(":", 1)
        minute_text = minute_ampm[:-2]
        ampm = minute_ampm[-2:]
    else:
        hour_text = text[:-2]
        minute_text = "00"
        ampm = text[-2:]

    hour = int(hour_text) % 12
    minute = int(minute_text)
    if ampm == "PM":
        hour += 12
    return hour * 60 + minute


def _load_json_list(raw: str | None) -> list[Any] | None:
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(data, list):
        return None
    return data


def is_up_down_outcomes(outcomes_json: str | None) -> bool:
    """Return True if outcomes are semantically Up/Down."""
    outcomes = _load_json_list(outcomes_json)
    if not outcomes or len(outcomes) != 2:
        return False

    normalized = [str(item).strip().lower() for item in outcomes]
    return normalized == ["up", "down"]


def parse_tenor_bucket(question: str | None) -> str:
    """Classify market tenor from the question text."""
    if not question:
        return "other"

    text = question.strip().lower()
    if not text.startswith(_QUESTION_PREFIX):
        return "other"

    range_match = _RANGE_RE.match(text)
    if range_match:
        start_minutes = _parse_time_minutes(range_match.group("start"))
        end_minutes = _parse_time_minutes(range_match.group("end"))
        # Handle wrap-around ranges (e.g., 11:50PM-12:05AM ET).
        if end_minutes < start_minutes:
            end_minutes += 24 * 60
        delta = end_minutes - start_minutes
        if delta == 5:
            return "5m"
        if delta == 15:
            return "15m"
        if delta == 60:
            return "1h"
        return "other"

    if _HOURLY_RE.match(text):
        return "1h"

    if _DAILY_RE.match(text):
        return "1d"

    return "other"


def parse_resolution(
    outcome_prices_json: str | None,
    outcomes_json: str | None,
    high_threshold: float = RESOLUTION_HIGH_THRESHOLD,
    low_threshold: float = RESOLUTION_LOW_THRESHOLD,
) -> ResolutionResult:
    """Parse final resolution status from outcome prices."""
    prices_raw = _load_json_list(outcome_prices_json)
    outcomes_raw = _load_json_list(outcomes_json)
    if not prices_raw or not outcomes_raw or len(prices_raw) != 2 or len(outcomes_raw) != 2:
        return ResolutionResult(False, None, None, None, None)

    try:
        prices = (float(prices_raw[0]), float(prices_raw[1]))
        outcomes = (str(outcomes_raw[0]).strip(), str(outcomes_raw[1]).strip())
    except (TypeError, ValueError):
        return ResolutionResult(False, None, None, None, None)

    max_price = max(prices)
    min_price = min(prices)
    if max_price < high_threshold or min_price > low_threshold:
        return ResolutionResult(False, None, None, prices, outcomes)

    winner_index = 0 if prices[0] >= prices[1] else 1
    winner_label = outcomes[winner_index]
    return ResolutionResult(True, winner_index, winner_label, prices, outcomes)
