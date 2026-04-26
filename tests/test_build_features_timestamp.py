"""Tests for parse_ts_to_unix in tjtb.features.build_features."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tjtb.features.build_features import parse_ts_to_unix


def _expected_utc_unix(s: str) -> float:
    """Reference: normalize same rules as production (micro pad/truncate, UTC)."""
    # Only used to cross-check a few fixtures; mirrors implementation intent.
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return float(dt.astimezone(timezone.utc).timestamp())


def test_naive_five_digit_fractional_seconds() -> None:
    s = "2026-04-18T15:38:42.34867"
    u = parse_ts_to_unix(s)
    ref = _expected_utc_unix("2026-04-18T15:38:42.348670+00:00")
    assert u == ref


def test_trailing_z_with_fractional() -> None:
    s = "2026-04-18T15:38:42.34867Z"
    u = parse_ts_to_unix(s)
    ref = _expected_utc_unix("2026-04-18T15:38:42.348670+00:00")
    assert u == ref


def test_explicit_offset_six_digit_fraction() -> None:
    s = "2026-04-18T15:38:42.348670+00:00"
    u = parse_ts_to_unix(s)
    ref = _expected_utc_unix(s)
    assert u == ref


def test_naive_no_fractional() -> None:
    s = "2026-04-18T15:38:42"
    u = parse_ts_to_unix(s)
    ref = _expected_utc_unix("2026-04-18T15:38:42+00:00")
    assert u == ref


def test_deterministic_repeatable() -> None:
    s = "2026-04-18T15:38:42.34867Z"
    assert parse_ts_to_unix(s) == parse_ts_to_unix(s)


def test_whitespace_stripped() -> None:
    assert parse_ts_to_unix("  2026-04-18T15:38:42Z  ") == parse_ts_to_unix("2026-04-18T15:38:42Z")


def test_long_fraction_truncated_deterministically() -> None:
    s = "2026-04-18T15:38:42.123456789+00:00"
    u = parse_ts_to_unix(s)
    ref = _expected_utc_unix("2026-04-18T15:38:42.123456+00:00")
    assert u == ref


def test_invalid_raises() -> None:
    with pytest.raises(ValueError):
        parse_ts_to_unix("not-a-timestamp")


def test_empty_raises() -> None:
    with pytest.raises(ValueError):
        parse_ts_to_unix("   ")
