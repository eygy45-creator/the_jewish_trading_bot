from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

import tjtb.live.live_paper_crypto as live
from tjtb.live.live_paper_crypto import LivePaperEngine
from tjtb.research.eth_geometry_runner import (
    DEFAULT_TIMEOUTS_SEC,
    EthGeometryResult,
    ROUND_TRIP_TAKER_FEE,
    _average_fee_cost_r,
    _per_trade_net_r,
    run_eth_geometry_grid,
    write_eth_geometry_reports,
)
from tjtb.research.stop_grid_runner import _avg_notional_and_lev


def _write_ndjson(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _sample_eth_rows() -> list[dict]:
    return [
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
                "topic": "publicTrade.ETHUSDT",
                "ts": 1700000000100,
                "data": [{"T": 1700000000100, "S": "Sell", "s": "ETHUSDT", "v": "1.0", "p": "3000.2"}],
            }
        },
    ]


def test_eth_geometry_matrix_executes(tmp_path):
    raw = tmp_path / "bybit_20260501.ndjson"
    _write_ndjson(raw, _sample_eth_rows())
    stops = [1.0, 2.0]
    timeouts = [2.0, 5.0]
    res = run_eth_geometry_grid(
        data_source="bybit",
        raw_dir=tmp_path,
        stops=stops,
        timeouts_sec=timeouts,
        symbol="ETHUSDT",
    )
    assert len(res) == len(stops) * len(timeouts)
    for r in res:
        assert isinstance(r, EthGeometryResult)
        assert r.round_trip_fee_rate == ROUND_TRIP_TAKER_FEE


def test_eth_geometry_output_files(tmp_path):
    raw = tmp_path / "bybit_20260501.ndjson"
    _write_ndjson(raw, _sample_eth_rows())
    res = run_eth_geometry_grid(data_source="bybit", raw_dir=tmp_path, stops=[1.0], timeouts_sec=[2.0])
    csv_out = tmp_path / "eth_geometry_results.csv"
    json_out = tmp_path / "eth_geometry_summary.json"
    summary = write_eth_geometry_reports(res, csv_out, json_out)
    assert csv_out.is_file()
    assert json_out.is_file()
    assert "best_by_raw_expectancy" in summary
    assert "best_by_survivability" in summary
    assert "best_realistic_execution_candidate" in summary


def test_default_timeout_grid_includes_long_windows():
    assert DEFAULT_TIMEOUTS_SEC == [120.0, 180.0, 300.0, 480.0, 600.0, 900.0]


def test_default_timeout_grid_is_used_by_runner(tmp_path):
    raw = tmp_path / "bybit_20260501.ndjson"
    _write_ndjson(raw, _sample_eth_rows())
    res = run_eth_geometry_grid(data_source="bybit", raw_dir=tmp_path, stops=[1.0], symbol="ETHUSDT")
    assert len(res) == len(DEFAULT_TIMEOUTS_SEC)
    assert sorted({r.timeout_sec for r in res}) == DEFAULT_TIMEOUTS_SEC


def test_default_matrix_size_matches_geometry_grid(tmp_path):
    raw = tmp_path / "bybit_20260501.ndjson"
    _write_ndjson(raw, _sample_eth_rows())
    default_stops = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
    res = run_eth_geometry_grid(data_source="bybit", raw_dir=tmp_path, symbol="ETHUSDT")
    assert len(res) == len(default_stops) * len(DEFAULT_TIMEOUTS_SEC)


def test_leverage_math_sanity_eth():
    avg_notional, lev = _avg_notional_and_lev([3000.0, 3010.0], stop_size=2.0, account=10_000.0)
    assert avg_notional > 0
    assert lev > 0


def test_fee_math_sanity():
    entry = 3000.0
    gross = 10.0
    net = _per_trade_net_r(entry, gross, ROUND_TRIP_TAKER_FEE)
    assert net == pytest.approx(gross - entry * ROUND_TRIP_TAKER_FEE)
    avg_fr = _average_fee_cost_r([entry], stop_size=10.0)
    assert avg_fr == pytest.approx((ROUND_TRIP_TAKER_FEE * entry) / 10.0)


def test_invalid_stop_rejected(tmp_path):
    raw = tmp_path / "bybit_20260501.ndjson"
    _write_ndjson(raw, _sample_eth_rows())
    with pytest.raises(ValueError):
        run_eth_geometry_grid(data_source="bybit", raw_dir=tmp_path, stops=[1.0, -1.0], timeouts_sec=[2.0])


def test_invalid_timeout_rejected(tmp_path):
    raw = tmp_path / "bybit_20260501.ndjson"
    _write_ndjson(raw, _sample_eth_rows())
    with pytest.raises(ValueError):
        run_eth_geometry_grid(data_source="bybit", raw_dir=tmp_path, stops=[1.0], timeouts_sec=[2.0, 0.0])


def test_production_defaults_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "PAPER_TRADES_PATH", tmp_path / "paper_trades.csv")
    monkeypatch.setattr(live, "OPPORTUNITIES_PATH", tmp_path / "opportunities.csv")
    assert live.TIMEOUT_SEC == 2.0
    eng = LivePaperEngine(logging.getLogger("tjtb.test.eth_geom"))
    top = live.TopState(
        ts=1700000000.0,
        ts_text="2023-11-14T22:13:20+00:00",
        best_bid=100.0,
        best_ask=100.5,
        best_bid_sz=1.0,
        best_ask_sz=1.0,
        spread=0.5,
        mid=100.25,
        micro_dev=0.0,
        tob_imb=0.0,
    )
    eng._take_trade(top, "normal", 2.0, 1.0)
    assert eng.open_trade is not None
    assert eng.open_trade["sl_price"] == top.mid + 1.0
