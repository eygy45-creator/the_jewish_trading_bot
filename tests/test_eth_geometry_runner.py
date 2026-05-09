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
    EthGeometryEngine,
    ExcursionState,
    ROUND_TRIP_TAKER_FEE,
    _average_fee_cost_r,
    _excursion_metrics,
    _per_trade_net_r,
    _percentile,
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


def _top(ts: float, mid: float) -> live.TopState:
    return live.TopState(
        ts=ts,
        ts_text=f"2023-11-14T22:{int(ts):02d}:00+00:00",
        best_bid=mid - 0.25,
        best_ask=mid + 0.25,
        best_bid_sz=1.0,
        best_ask_sz=1.0,
        spread=0.5,
        mid=mid,
        micro_dev=0.0,
        tob_imb=0.0,
    )


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
    assert "excursion_analysis" in summary


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


def test_short_excursion_mfe_and_mae_are_direction_aware():
    state = ExcursionState(trade_id=1, side="short", entry_ts=0.0, entry_price=2000.0, stop_distance=10.0, timeout_sec=900.0)
    for ts, price in [(1.0, 1996.0), (2.0, 1994.0), (3.0, 1998.0), (4.0, 2007.0)]:
        state.update(ts, price)
    assert state.mfe_r == pytest.approx(0.6)
    assert state.mae_r == pytest.approx(-0.7)
    assert state.seconds_to_mfe == pytest.approx(2.0)
    assert state.seconds_to_mae == pytest.approx(4.0)
    assert state.min_price_reached == pytest.approx(1994.0)
    assert state.max_price_reached == pytest.approx(2007.0)


def test_long_excursion_mfe_and_mae_are_direction_aware():
    state = ExcursionState(trade_id=1, side="long", entry_ts=0.0, entry_price=2000.0, stop_distance=10.0, timeout_sec=900.0)
    for ts, price in [(1.0, 2004.0), (2.0, 2006.0), (3.0, 2001.0), (4.0, 1993.0)]:
        state.update(ts, price)
    assert state.mfe_r == pytest.approx(0.6)
    assert state.mae_r == pytest.approx(-0.7)
    assert state.seconds_to_mfe == pytest.approx(2.0)
    assert state.seconds_to_mae == pytest.approx(4.0)
    assert state.min_price_reached == pytest.approx(1993.0)
    assert state.max_price_reached == pytest.approx(2006.0)


def test_excursion_tracking_continues_after_breakeven_exit():
    eng = EthGeometryEngine(logging.getLogger("tjtb.test.excursion"), data_source="bybit", stop_size=1.0, timeout_sec=900.0)
    eng._take_trade(_top(0.0, 100.0), "normal", tp_r=2.0, be_trigger=1.0)
    eng._maybe_manage_open_trade(_top(60.0, 99.0))
    assert eng.open_trade is not None
    assert eng.open_trade["sl_price"] == pytest.approx(100.0)
    eng._maybe_manage_open_trade(_top(120.0, 100.0))
    assert eng.open_trade is None
    assert eng.closed_trades[0]["outcome"] == "sl_or_be"
    assert eng.closed_trades[0]["mfe_r"] == pytest.approx(1.0)
    eng._maybe_manage_open_trade(_top(300.0, 98.8))
    eng._maybe_manage_open_trade(_top(900.0, 100.0))
    assert eng.closed_trades[0]["mfe_r"] == pytest.approx(1.2)
    assert eng.closed_trades[0]["time_to_first_1R"] == pytest.approx(60.0)


def test_excursion_percentiles_are_interpolated():
    assert _percentile([0.0, 1.0, 2.0, 3.0], 50) == pytest.approx(1.5)
    assert _percentile([0.0, 1.0, 2.0, 3.0], 75) == pytest.approx(2.25)
    assert _percentile([-3.0, -2.0, -1.0, 0.0], 90) == pytest.approx(-0.3)


def test_excursion_reachability_and_time_metrics():
    trades = [
        {"mfe_r": 0.2, "mae_r": 0.0, "seconds_to_mfe": 10.0, "time_to_first_0_25R": None, "time_to_first_0_5R": None, "time_to_first_1R": None},
        {"mfe_r": 0.6, "mae_r": -0.1, "seconds_to_mfe": 20.0, "time_to_first_0_25R": 5.0, "time_to_first_0_5R": 12.0, "time_to_first_1R": None},
        {"mfe_r": 1.2, "mae_r": -0.3, "seconds_to_mfe": 40.0, "time_to_first_0_25R": 4.0, "time_to_first_0_5R": 9.0, "time_to_first_1R": 35.0},
    ]
    metrics = _excursion_metrics(trades)
    assert metrics["reachability"]["percent_reaching_0_25R"] == pytest.approx(2 / 3)
    assert metrics["reachability"]["percent_reaching_0_5R"] == pytest.approx(2 / 3)
    assert metrics["reachability"]["percent_reaching_1R"] == pytest.approx(1 / 3)
    assert metrics["average_time_to_0_25R"] == pytest.approx(4.5)
    assert metrics["average_time_to_0_5R"] == pytest.approx(10.5)
    assert metrics["average_time_to_1R"] == pytest.approx(35.0)
    assert metrics["average_time_to_peak_mfe"] == pytest.approx(70.0 / 3)


def test_excursion_null_values_when_targets_never_reached():
    state = ExcursionState(trade_id=1, side="short", entry_ts=0.0, entry_price=100.0, stop_distance=10.0, timeout_sec=900.0)
    state.update(5.0, 98.0)
    out = state.to_output()
    assert out["mfe_r"] == pytest.approx(0.2)
    assert out["time_to_first_0_25R"] is None
    assert out["time_to_first_0_5R"] is None
    assert out["time_to_first_2R"] is None


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
