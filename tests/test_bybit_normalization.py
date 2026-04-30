from __future__ import annotations

import json
from datetime import datetime, timezone

from tjtb.data.bybit_recorder import output_path_for_day, write_ndjson_line
from tjtb.exchanges.bybit.market_data import (
    ORDERBOOK_TOPIC,
    TRADE_TOPIC,
    build_subscribe_args,
    normalize_public_message,
    parse_orderbook_message,
    parse_trade_message,
)


def test_malformed_message_handling_returns_none() -> None:
    bad = {"topic": ORDERBOOK_TOPIC, "type": "delta", "data": "not-a-map"}
    assert parse_orderbook_message(bad) is None
    assert normalize_public_message(bad, "2026-01-01T00:00:00+00:00") is None


def test_snapshot_vs_delta_parsing_sanity() -> None:
    snapshot = {
        "topic": ORDERBOOK_TOPIC,
        "type": "snapshot",
        "ts": 1700000000000,
        "cts": 1700000000001,
        "data": {"s": "BTCUSDT", "b": [["65000.1", "2.1"]], "a": [["65000.2", "1.3"]], "u": 10, "seq": 20},
    }
    delta = {
        "topic": ORDERBOOK_TOPIC,
        "type": "delta",
        "ts": 1700000000100,
        "cts": 1700000000101,
        "data": {"s": "BTCUSDT", "b": [["65000.1", "1.0"]], "a": [["65000.2", "0"]], "u": 11, "seq": 21},
    }
    s = parse_orderbook_message(snapshot)
    d = parse_orderbook_message(delta)
    assert s is not None and s["type"] == "snapshot"
    assert d is not None and d["type"] == "delta"
    assert s["bids"][0][0] == "65000.1"
    assert d["asks"][0][1] == "0"


def test_trade_parsing_includes_taker_side() -> None:
    msg = {
        "topic": TRADE_TOPIC,
        "type": "snapshot",
        "ts": 1700000000200,
        "data": [{"T": 1700000000190, "s": "BTCUSDT", "S": "Buy", "v": "0.010", "p": "65000.5", "seq": 31}],
    }
    parsed = parse_trade_message(msg)
    assert parsed is not None
    assert parsed["count"] == 1
    assert parsed["rows"][0]["side"] == "Buy"


def test_recorder_import_does_not_require_credentials() -> None:
    from tjtb.data import bybit_recorder as mod

    assert isinstance(mod.DEFAULT_WS_URL, str)
    assert "bybit" in mod.DEFAULT_WS_URL


def test_file_writing_smoke(tmp_path) -> None:
    path = output_path_for_day(tmp_path, datetime(2026, 4, 30, tzinfo=timezone.utc))
    rec = {
        "source": "bybit",
        "local_ts": "2026-04-30T00:00:00+00:00",
        "exchange_ts": 1700000000000,
        "topic": ORDERBOOK_TOPIC,
        "kind": "orderbook",
        "payload": {"topic": ORDERBOOK_TOPIC, "type": "snapshot"},
    }
    write_ndjson_line(path, rec)
    content = path.read_text(encoding="utf-8").strip()
    loaded = json.loads(content)
    assert loaded["source"] == "bybit"
    assert loaded["topic"] == ORDERBOOK_TOPIC


def test_recorder_topic_construction_for_ethusdt() -> None:
    args = build_subscribe_args("ETHUSDT")
    assert "orderbook.50.ETHUSDT" in args
    assert "publicTrade.ETHUSDT" in args

