"""Tests for reports/failed_absorption_continuation_study.py (importlib load)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_study_mod():
    root = Path(__file__).resolve().parent.parent
    path = root / "reports" / "failed_absorption_continuation_study.py"
    spec = importlib.util.spec_from_file_location("failed_absorption_continuation_study", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_trim_mids_and_oco_deterministic():
    mod = _load_study_mod()
    mids = [(0.0, 100.0), (10.0, 100.0), (20.0, 98.0), (900.0, 97.0)]
    path = mod._trim_mids_from_entry(mids, entry_ts=5.0, entry_price=100.0)
    assert path[0][0] == 5.0
    assert abs(path[0][1] - 100.0) < 1e-6
    net = mod._oco_net_r_first_exit(100.0, 2.0, 100.5, mids, 5.0, 5.0 + 900.0, tp_mult=1.0)
    assert isinstance(net, float)


def test_run_study_synthetic_ndjson(tmp_path):
    mod = _load_study_mod()
    raw = tmp_path / "bybit_test.ndjson"
    rows = [
        {
            "payload": {
                "topic": "orderbook.50.ETHUSDT",
                "type": "snapshot",
                "ts": 1700000000000,
                "data": {"s": "ETHUSDT", "b": [["3000", "2.0"]], "a": [["3000.5", "1.0"]]},
            }
        },
        {
            "payload": {
                "topic": "orderbook.50.ETHUSDT",
                "type": "delta",
                "ts": 1700000005000,
                "data": {"s": "ETHUSDT", "b": [["3000", "1.5"]], "a": [["3000.5", "1.2"]]},
            }
        },
    ]
    raw.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = tmp_path / "out.json"
    doc = mod.run_study(data_source="bybit", raw_dir=tmp_path, symbol="ETHUSDT", json_output=out)
    assert out.is_file()
    assert "signals" in doc
    assert "aggregates" in doc
    assert "decision" in doc
    assert doc["parameters"]["fee_adjusted"] is True


def test_simulate_short_monotonic_down():
    mod = _load_study_mod()
    entry_ts = 0.0
    mids = [(0.0, 100.0), (60.0, 94.0)]
    r = mod._simulate_short_path(
        mids,
        entry_ts=entry_ts,
        entry_price=100.0,
        stop=2.0,
        ref_level=101.0,
        horizon=900.0,
    )
    assert r.hit_1r is True
    assert r.continuation_60s_r == pytest.approx(3.0)  # (100-94)/2


def test_empty_raw_dir(tmp_path):
    mod = _load_study_mod()
    doc = mod.run_study(data_source="bybit", raw_dir=tmp_path, symbol="ETHUSDT", json_output=tmp_path / "o.json")
    assert doc["signals"] == []
    assert doc["decision"]["verdict"] == "NO_CONTINUATION_EDGE_REDESIGN_REQUIRED"
