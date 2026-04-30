from __future__ import annotations

import logging

import tjtb.live.live_paper_crypto as live


def _test_logger() -> logging.Logger:
    lg = logging.getLogger("tjtb.test.live.bybit")
    lg.handlers.clear()
    lg.addHandler(logging.NullHandler())
    return lg


def test_coinbase_default_source_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "PAPER_TRADES_PATH", tmp_path / "paper_trades.csv")
    monkeypatch.setattr(live, "OPPORTUNITIES_PATH", tmp_path / "opportunities.csv")
    eng = live.LivePaperEngine(_test_logger())
    assert eng.data_source == "coinbase"
    assert eng.raw_glob == live.RAW_GLOB


def test_bybit_book_envelope_updates_top_of_book(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "PAPER_TRADES_PATH", tmp_path / "paper_trades.csv")
    monkeypatch.setattr(live, "OPPORTUNITIES_PATH", tmp_path / "opportunities.csv")
    eng = live.LivePaperEngine(_test_logger(), data_source="bybit")
    msg = {
        "payload": {
            "topic": "orderbook.50.BTCUSDT",
            "type": "snapshot",
            "ts": 1700000000000,
            "data": {
                "s": "BTCUSDT",
                "b": [["100.0", "2.0"]],
                "a": [["100.5", "1.0"]],
            },
        }
    }
    top, pressure = eng._process_l2_msg(msg)
    assert top is not None
    assert top.best_bid == 100.0
    assert top.best_ask == 100.5
    assert pressure > 0


def test_bybit_public_trade_increments_trade_times(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "PAPER_TRADES_PATH", tmp_path / "paper_trades.csv")
    monkeypatch.setattr(live, "OPPORTUNITIES_PATH", tmp_path / "opportunities.csv")
    eng = live.LivePaperEngine(_test_logger(), data_source="bybit")
    msg = {
        "payload": {
            "topic": "publicTrade.BTCUSDT",
            "data": [
                {"T": 1700000000100, "S": "Buy", "v": "0.01", "p": "100.2"},
                {"T": 1700000000200, "S": "Sell", "v": "0.02", "p": "100.1"},
            ],
        }
    }
    eng._process_trade_msg(msg)
    assert len(eng.trade_times) == 2


def test_bybit_source_requires_no_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "PAPER_TRADES_PATH", tmp_path / "paper_trades.csv")
    monkeypatch.setattr(live, "OPPORTUNITIES_PATH", tmp_path / "opportunities.csv")
    eng = live.LivePaperEngine(_test_logger(), data_source="bybit")
    assert eng.data_source == "bybit"


def test_malformed_bybit_messages_are_skipped_safely(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "PAPER_TRADES_PATH", tmp_path / "paper_trades.csv")
    monkeypatch.setattr(live, "OPPORTUNITIES_PATH", tmp_path / "opportunities.csv")
    eng = live.LivePaperEngine(_test_logger(), data_source="bybit")
    top, pressure = eng._process_l2_msg({"payload": {"topic": "orderbook.50.BTCUSDT", "type": "delta", "data": "bad"}})
    eng._process_trade_msg({"payload": {"topic": "publicTrade.BTCUSDT", "data": "bad"}})
    assert top is None
    assert pressure == 0.0
    assert len(eng.trade_times) == 0


def _mk_top() -> live.TopState:
    return live.TopState(
        ts=1700000000.0,
        ts_text="2023-11-14T22:13:20+00:00",
        best_bid=100.0,
        best_ask=100.5,
        best_bid_sz=2.0,
        best_ask_sz=1.0,
        spread=0.5,
        mid=100.25,
        micro_dev=0.1,
        tob_imb=0.333333,
    )


def test_dry_run_computes_qty_from_risk_fraction(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "PAPER_TRADES_PATH", tmp_path / "paper_trades.csv")
    monkeypatch.setattr(live, "OPPORTUNITIES_PATH", tmp_path / "opportunities.csv")
    monkeypatch.setenv("DATA_SOURCE", "bybit")
    monkeypatch.setenv("EXECUTION_MODE", "bybit_demo_dry_run")
    monkeypatch.setenv("BYBIT_SYMBOL", "BTCUSDT")
    monkeypatch.setenv("RISK_PER_TRADE", "0.0025")
    monkeypatch.setenv("KILL_SWITCH", "false")
    monkeypatch.setenv("BYBIT_DRY_RUN_ALLOW_KILL_SWITCH", "true")
    monkeypatch.setenv("BYBIT_BALANCE_OVERRIDE", "10000")
    eng = live.LivePaperEngine(_test_logger(), data_source="bybit")
    ok, reason = eng._plan_bybit_dry_run_entry(_mk_top(), tp_r=2.0)
    assert ok is True
    assert reason == ""
    # risk=25, stop distance=1.0 => qty=25
    assert eng.last_execution_plan is not None
    assert abs(float(eng.last_execution_plan["qty"]) - 25.0) < 1e-9


def test_dry_run_rejects_invalid_sizing(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "PAPER_TRADES_PATH", tmp_path / "paper_trades.csv")
    monkeypatch.setattr(live, "OPPORTUNITIES_PATH", tmp_path / "opportunities.csv")
    monkeypatch.setenv("EXECUTION_MODE", "bybit_demo_dry_run")
    monkeypatch.setenv("KILL_SWITCH", "false")
    monkeypatch.setenv("BYBIT_DRY_RUN_ALLOW_KILL_SWITCH", "true")
    monkeypatch.setenv("BYBIT_BALANCE_OVERRIDE", "0")
    eng = live.LivePaperEngine(_test_logger(), data_source="bybit")
    ok, reason = eng._plan_bybit_dry_run_entry(_mk_top(), tp_r=2.0)
    assert ok is False
    assert reason == "invalid_account_balance"


def test_dry_run_requires_no_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "PAPER_TRADES_PATH", tmp_path / "paper_trades.csv")
    monkeypatch.setattr(live, "OPPORTUNITIES_PATH", tmp_path / "opportunities.csv")
    monkeypatch.delenv("BYBIT_API_KEY", raising=False)
    monkeypatch.delenv("BYBIT_API_SECRET", raising=False)
    monkeypatch.setenv("EXECUTION_MODE", "bybit_demo_dry_run")
    monkeypatch.setenv("KILL_SWITCH", "false")
    monkeypatch.setenv("BYBIT_DRY_RUN_ALLOW_KILL_SWITCH", "true")
    monkeypatch.setenv("BYBIT_BALANCE_OVERRIDE", "10000")
    eng = live.LivePaperEngine(_test_logger(), data_source="bybit")
    ok, _ = eng._plan_bybit_dry_run_entry(_mk_top(), tp_r=2.0)
    assert ok is True


def test_dry_run_writes_additive_execution_log(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "PAPER_TRADES_PATH", tmp_path / "paper_trades.csv")
    monkeypatch.setattr(live, "OPPORTUNITIES_PATH", tmp_path / "opportunities.csv")
    monkeypatch.setenv("EXECUTION_MODE", "bybit_demo_dry_run")
    monkeypatch.setenv("KILL_SWITCH", "false")
    monkeypatch.setenv("BYBIT_DRY_RUN_ALLOW_KILL_SWITCH", "true")
    monkeypatch.setenv("BYBIT_BALANCE_OVERRIDE", "10000")
    eng = live.LivePaperEngine(_test_logger(), data_source="bybit")
    ok, _ = eng._plan_bybit_dry_run_entry(_mk_top(), tp_r=2.0)
    assert ok is True
    assert eng.execution_dry_run_path.is_file()
    lines = eng.execution_dry_run_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2


def test_dry_run_respects_kill_switch_without_override(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "PAPER_TRADES_PATH", tmp_path / "paper_trades.csv")
    monkeypatch.setattr(live, "OPPORTUNITIES_PATH", tmp_path / "opportunities.csv")
    monkeypatch.setenv("EXECUTION_MODE", "bybit_demo_dry_run")
    monkeypatch.setenv("KILL_SWITCH", "true")
    monkeypatch.delenv("BYBIT_DRY_RUN_ALLOW_KILL_SWITCH", raising=False)
    monkeypatch.setenv("BYBIT_BALANCE_OVERRIDE", "10000")
    eng = live.LivePaperEngine(_test_logger(), data_source="bybit")
    ok, reason = eng._plan_bybit_dry_run_entry(_mk_top(), tp_r=2.0)
    assert ok is False
    assert reason == "kill_switch_active"


def test_existing_csv_required_headers_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "PAPER_TRADES_PATH", tmp_path / "paper_trades.csv")
    monkeypatch.setattr(live, "OPPORTUNITIES_PATH", tmp_path / "opportunities.csv")
    live.LivePaperEngine(_test_logger(), data_source="coinbase")
    trade_header = (tmp_path / "paper_trades.csv").read_text(encoding="utf-8").splitlines()[0]
    opp_header = (tmp_path / "opportunities.csv").read_text(encoding="utf-8").splitlines()[0]
    assert trade_header == "entry_ts,exit_ts,side,entry_price,exit_price,outcome,r_value,regime"
    assert opp_header == "ts,anomaly_percentile,anomaly_score,direction,regime,action,reason"

