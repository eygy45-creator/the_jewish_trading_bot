from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from tjtb.live.live_paper_crypto import LivePaperEngine
from tjtb.research.stop_grid_runner import _avg_notional_and_lev, _write_reports, run_stop_grid


def _write_ndjson(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _sample_bybit_rows() -> list[dict]:
    return [
        {
            "payload": {
                "topic": "orderbook.50.BTCUSDT",
                "type": "snapshot",
                "ts": 1700000000000,
                "data": {"s": "BTCUSDT", "b": [["100.0", "2.0"]], "a": [["100.5", "1.0"]]},
            }
        },
        {
            "payload": {
                "topic": "publicTrade.BTCUSDT",
                "ts": 1700000000100,
                "data": [{"T": 1700000000100, "S": "Sell", "v": "1.0", "p": "100.2"}],
            }
        },
    ]


def test_stop_grid_runner_executes_multiple_stop_configs(tmp_path):
    raw = tmp_path / "bybit_20260501.ndjson"
    _write_ndjson(raw, _sample_bybit_rows())
    res = run_stop_grid(data_source="bybit", raw_dir=tmp_path, stops=[1.0, 2.0, 3.0])
    assert len(res) == 3
    assert [r.stop_size for r in res] == [1.0, 2.0, 3.0]


def test_output_files_created(tmp_path):
    raw = tmp_path / "bybit_20260501.ndjson"
    _write_ndjson(raw, _sample_bybit_rows())
    res = run_stop_grid(data_source="bybit", raw_dir=tmp_path, stops=[1.0, 2.0])
    csv_out = tmp_path / "stop_grid_results.csv"
    json_out = tmp_path / "stop_grid_summary.json"
    _write_reports(res, csv_out, json_out)
    assert csv_out.is_file()
    assert json_out.is_file()


def test_default_production_strategy_unchanged(tmp_path, monkeypatch):
    import tjtb.live.live_paper_crypto as live

    monkeypatch.setattr(live, "PAPER_TRADES_PATH", tmp_path / "paper_trades.csv")
    monkeypatch.setattr(live, "OPPORTUNITIES_PATH", tmp_path / "opportunities.csv")
    eng = LivePaperEngine(logging.getLogger("tjtb.test.default"))
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


def test_leverage_estimation_math_sanity():
    avg_notional, lev = _avg_notional_and_lev([100.0, 110.0], stop_size=1.0, account=10_000.0)
    assert avg_notional > 0
    assert lev > 0


def test_invalid_stop_values_rejected(tmp_path):
    raw = tmp_path / "bybit_20260501.ndjson"
    _write_ndjson(raw, _sample_bybit_rows())
    with pytest.raises(ValueError):
        run_stop_grid(data_source="bybit", raw_dir=tmp_path, stops=[1.0, 0.0, 2.0])


def test_stop_grid_supports_symbol_argument(tmp_path):
    raw = tmp_path / "bybit_20260501.ndjson"
    _write_ndjson(
        raw,
        [
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
        ],
    )
    res = run_stop_grid(data_source="bybit", raw_dir=tmp_path, stops=[1.0], symbol="ETHUSDT")
    assert len(res) == 1

