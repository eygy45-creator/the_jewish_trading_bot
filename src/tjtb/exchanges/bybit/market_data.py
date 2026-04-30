"""Bybit public market-data message helpers (linear category)."""

from __future__ import annotations

import os
from typing import Any

DEFAULT_SYMBOL = "BTCUSDT"


def get_bybit_symbol(default: str = DEFAULT_SYMBOL) -> str:
    return str(os.environ.get("BYBIT_SYMBOL", default)).strip().upper()


def orderbook_topic(symbol: str) -> str:
    return f"orderbook.50.{symbol}"


def trade_topic(symbol: str) -> str:
    return f"publicTrade.{symbol}"


ORDERBOOK_TOPIC = orderbook_topic(DEFAULT_SYMBOL)
TRADE_TOPIC = trade_topic(DEFAULT_SYMBOL)
SUBSCRIBE_ARGS = [ORDERBOOK_TOPIC, TRADE_TOPIC]


def build_subscribe_args(symbol: str | None = None) -> list[str]:
    sym = (symbol or get_bybit_symbol()).upper()
    return [orderbook_topic(sym), trade_topic(sym)]


def is_public_data_topic(topic: str, symbol: str | None = None) -> bool:
    sym = (symbol or get_bybit_symbol()).upper()
    return topic in {orderbook_topic(sym), trade_topic(sym)}


def parse_orderbook_message(msg: dict[str, Any], symbol: str | None = None) -> dict[str, Any] | None:
    """Return a minimal parsed orderbook payload or None if invalid."""
    sym = (symbol or get_bybit_symbol()).upper()
    if str(msg.get("topic", "")) != orderbook_topic(sym):
        return None
    msg_type = str(msg.get("type", "")).lower()
    if msg_type not in {"snapshot", "delta"}:
        return None
    data = msg.get("data")
    if not isinstance(data, dict):
        return None
    symbol_from_data = str(data.get("s", ""))
    bids = data.get("b")
    asks = data.get("a")
    if symbol_from_data != sym or not isinstance(bids, list) or not isinstance(asks, list):
        return None
    return {
        "kind": "orderbook",
        "type": msg_type,
        "symbol": symbol_from_data,
        "bids": bids,
        "asks": asks,
        "u": data.get("u"),
        "seq": data.get("seq", msg.get("seq")),
        "cts": msg.get("cts"),
        "ts": msg.get("ts"),
    }


def parse_trade_message(msg: dict[str, Any], symbol: str | None = None) -> dict[str, Any] | None:
    """Return a minimal parsed trade payload or None if invalid."""
    sym = (symbol or get_bybit_symbol()).upper()
    if str(msg.get("topic", "")) != trade_topic(sym):
        return None
    data = msg.get("data")
    if not isinstance(data, list):
        return None
    if not data:
        return {
            "kind": "trades",
            "symbol": sym,
            "count": 0,
            "rows": [],
            "ts": msg.get("ts"),
        }
    rows: list[dict[str, Any]] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        symbol_from_row = str(row.get("s", ""))
        if symbol_from_row != sym:
            continue
        side = str(row.get("S", ""))
        size = row.get("v")
        price = row.get("p")
        if side not in {"Buy", "Sell"}:
            continue
        rows.append(
            {
                "symbol": symbol_from_row,
                "side": side,
                "size": size,
                "price": price,
                "trade_ts": row.get("T"),
                "seq": row.get("seq"),
            }
        )
    return {
        "kind": "trades",
        "symbol": sym,
        "count": len(rows),
        "rows": rows,
        "ts": msg.get("ts"),
    }


def normalize_public_message(msg: dict[str, Any], local_ts_iso: str, symbol: str | None = None) -> dict[str, Any] | None:
    """
    Wrap a Bybit WS payload for NDJSON persistence.

    Raw payload is preserved under ``payload`` for fidelity.
    """
    topic = str(msg.get("topic", ""))
    sym = (symbol or get_bybit_symbol()).upper()
    if not is_public_data_topic(topic, sym):
        return None
    parsed = parse_orderbook_message(msg, sym)
    if parsed is None:
        parsed = parse_trade_message(msg, sym)
    if parsed is None:
        return None
    return {
        "source": "bybit",
        "local_ts": local_ts_iso,
        "exchange_ts": msg.get("ts"),
        "topic": topic,
        "kind": parsed.get("kind"),
        "payload": msg,
    }
