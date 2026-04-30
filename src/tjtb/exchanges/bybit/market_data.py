"""Bybit public market-data message helpers (linear category)."""

from __future__ import annotations

from typing import Any

ORDERBOOK_TOPIC = "orderbook.50.BTCUSDT"
TRADE_TOPIC = "publicTrade.BTCUSDT"
SUBSCRIBE_ARGS = [ORDERBOOK_TOPIC, TRADE_TOPIC]


def is_public_data_topic(topic: str) -> bool:
    return topic in {ORDERBOOK_TOPIC, TRADE_TOPIC}


def parse_orderbook_message(msg: dict[str, Any]) -> dict[str, Any] | None:
    """Return a minimal parsed orderbook payload or None if invalid."""
    if str(msg.get("topic", "")) != ORDERBOOK_TOPIC:
        return None
    msg_type = str(msg.get("type", "")).lower()
    if msg_type not in {"snapshot", "delta"}:
        return None
    data = msg.get("data")
    if not isinstance(data, dict):
        return None
    symbol = str(data.get("s", ""))
    bids = data.get("b")
    asks = data.get("a")
    if symbol != "BTCUSDT" or not isinstance(bids, list) or not isinstance(asks, list):
        return None
    return {
        "kind": "orderbook",
        "type": msg_type,
        "symbol": symbol,
        "bids": bids,
        "asks": asks,
        "u": data.get("u"),
        "seq": data.get("seq", msg.get("seq")),
        "cts": msg.get("cts"),
        "ts": msg.get("ts"),
    }


def parse_trade_message(msg: dict[str, Any]) -> dict[str, Any] | None:
    """Return a minimal parsed trade payload or None if invalid."""
    if str(msg.get("topic", "")) != TRADE_TOPIC:
        return None
    data = msg.get("data")
    if not isinstance(data, list):
        return None
    if not data:
        return {
            "kind": "trades",
            "symbol": "BTCUSDT",
            "count": 0,
            "rows": [],
            "ts": msg.get("ts"),
        }
    rows: list[dict[str, Any]] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("s", ""))
        if symbol != "BTCUSDT":
            continue
        side = str(row.get("S", ""))
        size = row.get("v")
        price = row.get("p")
        if side not in {"Buy", "Sell"}:
            continue
        rows.append(
            {
                "symbol": symbol,
                "side": side,
                "size": size,
                "price": price,
                "trade_ts": row.get("T"),
                "seq": row.get("seq"),
            }
        )
    return {
        "kind": "trades",
        "symbol": "BTCUSDT",
        "count": len(rows),
        "rows": rows,
        "ts": msg.get("ts"),
    }


def normalize_public_message(msg: dict[str, Any], local_ts_iso: str) -> dict[str, Any] | None:
    """
    Wrap a Bybit WS payload for NDJSON persistence.

    Raw payload is preserved under ``payload`` for fidelity.
    """
    topic = str(msg.get("topic", ""))
    if not is_public_data_topic(topic):
        return None
    parsed = parse_orderbook_message(msg)
    if parsed is None:
        parsed = parse_trade_message(msg)
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
