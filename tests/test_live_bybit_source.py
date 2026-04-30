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

