"""Tests for true_failed_absorption_study path math and OCO helper."""

from __future__ import annotations

import math

from tjtb.research.true_failed_absorption_study import (
    _excursion_from_path,
    _oco_first_exit,
    _trim_mids_from_entry,
)


def test_oco_short_stop_before_tp():
    entry_ts = 1000.0
    entry = 100.0
    stop_dist = 1.0
    mids = [
        (entry_ts, entry),
        (entry_ts + 10.0, entry + 0.5),
        (entry_ts + 20.0, entry + 1.01),
    ]
    r = _oco_first_exit(
        side="short",
        mids=mids,
        entry_ts=entry_ts,
        entry=entry,
        stop_dist=stop_dist,
        inv_level=entry + stop_dist,
        horizon=500.0,
        tp_mult=1.0,
        fee_rate=0.0,
    )
    assert r["exit_kind"] == "stop"
    assert r["net_r"] < -0.5


def test_excursion_short_reaches_1r():
    entry_ts = 0.0
    entry = 100.0
    stop = 1.0
    mids = [(0.0, entry), (10.0, 98.5), (20.0, 97.0)]
    ex = _excursion_from_path(side="short", mids=mids, entry_ts=entry_ts, entry=entry, stop_dist=stop, horizon=100.0)
    assert ex["mfe_r"] >= 1.0 - 1e-6
    assert ex.get("time_to_first_1R") is not None


def test_trim_mids_stable():
    p = _trim_mids_from_entry([(0.0, 10.0), (5.0, 11.0)], 2.0, 10.5)
    assert len(p) >= 2
    assert p[0][0] == 2.0
    assert math.isfinite(p[0][1])
