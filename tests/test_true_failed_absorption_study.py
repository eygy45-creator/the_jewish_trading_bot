"""Tests for true_failed_absorption_study path math and file-by-file aggregation."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tjtb.research.true_failed_absorption_study import (
    _accumulate_row_into_bucket,
    _empty_trade_bucket,
    _excursion_from_path,
    _finalize_trade_bucket,
    _merge_aggregatable_stats,
    _merge_trade_buckets,
    _oco_first_exit,
    _process_raw_file,
    _trim_mids_from_entry,
    run_true_failed_absorption_study,
)


def _write_ndjson(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _sample_eth_rows(ts_base: int = 1700000000000) -> list[dict]:
    return [
        {
            "payload": {
                "topic": "orderbook.50.ETHUSDT",
                "type": "snapshot",
                "ts": ts_base,
                "data": {"s": "ETHUSDT", "b": [["3000", "2.0"]], "a": [["3000.5", "1.0"]]},
            }
        },
        {
            "payload": {
                "topic": "publicTrade.ETHUSDT",
                "ts": ts_base + 100,
                "data": [{"T": ts_base + 100, "S": "Sell", "s": "ETHUSDT", "v": "1.0", "p": "3000.2"}],
            }
        },
    ]


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


def test_cli_exposes_partials_dir_force_max_files():
    env = {**os.environ, "PYTHONPATH": "src"}
    proc = subprocess.run(
        [sys.executable, "-m", "tjtb.research.true_failed_absorption_study", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--partials-dir" in proc.stdout
    assert "--force" in proc.stdout
    assert "--max-files" in proc.stdout


def test_trim_mids_stable():
    p = _trim_mids_from_entry([(0.0, 10.0), (5.0, 11.0)], 2.0, 10.5)
    assert len(p) >= 2
    assert p[0][0] == 2.0
    assert math.isfinite(p[0][1])


def test_merge_trade_buckets_counts():
    b1 = _empty_trade_bucket()
    b2 = _empty_trade_bucket()
    row_win = {"oco_market_net_r_1R": 0.5, "mfe_r": 1.2}
    row_loss = {"oco_market_net_r_1R": -0.3, "mfe_r": 0.4}
    _accumulate_row_into_bucket(b1, row_win, "oco_market_net_r_1R")
    _accumulate_row_into_bucket(b2, row_loss, "oco_market_net_r_1R")
    merged = _merge_trade_buckets(b1, b2)
    summary = _finalize_trade_bucket(merged)
    assert summary["n"] == 2
    assert summary["win_rate"] == pytest.approx(0.5)
    assert summary["net_expectancy_after_fees"] == pytest.approx(0.1)


def test_merge_aggregatable_stats_entry_models():
    s1 = {
        "overall": _empty_trade_bucket(),
        "by_side": {"short": _empty_trade_bucket(), "long": _empty_trade_bucket()},
        "by_session": {},
        "by_regime": {},
        "entry_models": {
            "A": {"n": 2, "net_r_1r_sum": 1.0},
            "B_filled": {"n": 1, "net_r_1r_sum": -0.5, "loser_count": 1},
            "C_filled": {"n": 0, "net_r_1r_sum": 0.0},
        },
        "structural_vs_fixed": {"n": 0, "struct_net_sum": 0.0, "fixed_net_sum": 0.0},
        "structural_stop_dist_values": [],
        "retest_score_values": [],
        "adverse": {
            "market_n": 0,
            "market_loser_count": 0,
            "maker_b_filled_n": 0,
            "maker_b_filled_loser_count": 0,
        },
    }
    s2 = {
        **s1,
        "entry_models": {
            "A": {"n": 1, "net_r_1r_sum": 0.5},
            "B_filled": {"n": 0, "net_r_1r_sum": 0.0, "loser_count": 0},
            "C_filled": {"n": 0, "net_r_1r_sum": 0.0},
        },
    }
    merged = _merge_aggregatable_stats(s1, s2)
    assert merged["entry_models"]["A"]["n"] == 3
    assert merged["entry_models"]["A"]["net_r_1r_sum"] == pytest.approx(1.5)


def test_file_by_file_writes_partials_and_aggregate(tmp_path):
    raw_dir = tmp_path / "raw"
    partials_dir = tmp_path / "partials"
    out_json = tmp_path / "study.json"
    raw_dir.mkdir()
    _write_ndjson(raw_dir / "bybit_a.ndjson", _sample_eth_rows(1700000000000))
    _write_ndjson(raw_dir / "bybit_b.ndjson", _sample_eth_rows(1700003600000))

    doc = run_true_failed_absorption_study(
        data_source="bybit",
        raw_dir=raw_dir,
        symbol="ETHUSDT",
        json_output=out_json,
        partials_dir=partials_dir,
    )

    assert (partials_dir / "bybit_a.ndjson.json").is_file()
    assert (partials_dir / "bybit_b.ndjson.json").is_file()
    assert out_json.is_file()
    assert doc["files_in_aggregate"] == 2
    assert doc["raw_event_count_estimate"] == 4
    assert "verdict" in doc
    assert "best_candidates" in doc
    assert "session_attribution" in doc
    assert "overall_market_structure_stop" in doc


def test_resume_skips_existing_partial(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    partials_dir = tmp_path / "partials"
    out_json = tmp_path / "study.json"
    raw_dir.mkdir()
    _write_ndjson(raw_dir / "bybit_a.ndjson", _sample_eth_rows())

    run_true_failed_absorption_study(
        data_source="bybit",
        raw_dir=raw_dir,
        symbol="ETHUSDT",
        json_output=out_json,
        partials_dir=partials_dir,
    )
    partial_path = partials_dir / "bybit_a.ndjson.json"
    assert partial_path.is_file()
    original = partial_path.read_text(encoding="utf-8")
    marker = json.loads(original)
    marker["resume_marker"] = True
    partial_path.write_text(json.dumps(marker), encoding="utf-8")

    calls: list[str] = []
    real_process = _process_raw_file

    def tracked_process(path: Path, *, data_source: str, symbol: str):
        calls.append(path.name)
        return real_process(path, data_source=data_source, symbol=symbol)

    monkeypatch.setattr(
        "tjtb.research.true_failed_absorption_study._process_raw_file",
        tracked_process,
    )

    doc = run_true_failed_absorption_study(
        data_source="bybit",
        raw_dir=raw_dir,
        symbol="ETHUSDT",
        json_output=out_json,
        partials_dir=partials_dir,
    )

    assert calls == []
    assert doc["files_skipped_resume"] == ["bybit_a.ndjson"]
    loaded = json.loads(partial_path.read_text(encoding="utf-8"))
    assert loaded.get("resume_marker") is True


def test_force_reprocesses_partial(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    partials_dir = tmp_path / "partials"
    out_json = tmp_path / "study.json"
    raw_dir.mkdir()
    _write_ndjson(raw_dir / "bybit_a.ndjson", _sample_eth_rows())

    run_true_failed_absorption_study(
        data_source="bybit",
        raw_dir=raw_dir,
        symbol="ETHUSDT",
        json_output=out_json,
        partials_dir=partials_dir,
    )

    calls: list[str] = []

    def tracked_process(path: Path, *, data_source: str, symbol: str):
        calls.append(path.name)
        return _process_raw_file(path, data_source=data_source, symbol=symbol)

    monkeypatch.setattr(
        "tjtb.research.true_failed_absorption_study._process_raw_file",
        tracked_process,
    )

    run_true_failed_absorption_study(
        data_source="bybit",
        raw_dir=raw_dir,
        symbol="ETHUSDT",
        json_output=out_json,
        partials_dir=partials_dir,
        force=True,
    )
    assert calls == ["bybit_a.ndjson"]


def test_max_files_limits_processing(tmp_path):
    raw_dir = tmp_path / "raw"
    partials_dir = tmp_path / "partials"
    out_json = tmp_path / "study.json"
    raw_dir.mkdir()
    _write_ndjson(raw_dir / "bybit_a.ndjson", _sample_eth_rows(1700000000000))
    _write_ndjson(raw_dir / "bybit_b.ndjson", _sample_eth_rows(1700003600000))

    doc = run_true_failed_absorption_study(
        data_source="bybit",
        raw_dir=raw_dir,
        symbol="ETHUSDT",
        json_output=out_json,
        partials_dir=partials_dir,
        max_files=1,
    )

    assert doc["files_in_aggregate"] == 1
    assert (partials_dir / "bybit_a.ndjson.json").is_file()
    assert not (partials_dir / "bybit_b.ndjson.json").exists()
