from __future__ import annotations

import json
from pathlib import Path

import tjtb.data.parse_bybit as pb


def _write_lines(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_snapshot_delta_and_level_removal_best_bid_ask(tmp_path, monkeypatch):
    raw = tmp_path / "bybit_20260430.ndjson"
    out_tr = tmp_path / "trades.csv"
    out_up = tmp_path / "updates.csv"
    out_bs = tmp_path / "book_state.csv"
    monkeypatch.setattr(pb, "OUTPUT_TRADES", out_tr)
    monkeypatch.setattr(pb, "OUTPUT_BOOK_UPDATES", out_up)
    monkeypatch.setattr(pb, "OUTPUT_BOOK_STATE", out_bs)
    pb._init_csv_headers()

    rows = [
        {
            "source": "bybit",
            "payload": {
                "topic": "orderbook.50.BTCUSDT",
                "type": "snapshot",
                "ts": 1700000000000,
                "data": {
                    "s": "BTCUSDT",
                    "b": [["100.0", "2.0"], ["99.5", "1.0"]],
                    "a": [["100.5", "3.0"], ["101.0", "1.0"]],
                    "seq": 10,
                },
            },
        },
        {
            "source": "bybit",
            "payload": {
                "topic": "orderbook.50.BTCUSDT",
                "type": "delta",
                "ts": 1700000000100,
                "data": {"s": "BTCUSDT", "b": [["100.0", "0"]], "a": [["100.5", "2.0"]], "seq": 11},
            },
        },
    ]
    _write_lines(raw, rows)
    st = pb.parse_file(str(raw))
    assert st.snapshots_seen == 1
    assert st.book_updates_written == 6
    lines = out_bs.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 3
    # After removal of 100.0 bid, best bid should be 99.5
    assert "99.5" in lines[-1]


def test_signed_pressure_reconstruction(tmp_path, monkeypatch):
    raw = tmp_path / "bybit_20260430.ndjson"
    out_tr = tmp_path / "trades.csv"
    out_up = tmp_path / "updates.csv"
    out_bs = tmp_path / "book_state.csv"
    monkeypatch.setattr(pb, "OUTPUT_TRADES", out_tr)
    monkeypatch.setattr(pb, "OUTPUT_BOOK_UPDATES", out_up)
    monkeypatch.setattr(pb, "OUTPUT_BOOK_STATE", out_bs)
    pb._init_csv_headers()

    rows = [
        {
            "source": "bybit",
            "payload": {
                "topic": "orderbook.50.BTCUSDT",
                "type": "snapshot",
                "ts": 1700000000000,
                "data": {"s": "BTCUSDT", "b": [["100.0", "1.0"]], "a": [["100.5", "1.0"]], "seq": 1},
            },
        },
        {
            "source": "bybit",
            "payload": {
                "topic": "orderbook.50.BTCUSDT",
                "type": "delta",
                "ts": 1700000000100,
                "data": {"s": "BTCUSDT", "b": [["100.0", "1.5"]], "a": [["100.5", "0.5"]], "seq": 2},
            },
        },
    ]
    _write_lines(raw, rows)
    pb.parse_file(str(raw))
    last = out_bs.read_text(encoding="utf-8").strip().splitlines()[-1]
    # pressure = (+0.5 bid delta) - (-0.5 ask delta) = +1.0
    assert ",1.0," in f",{last},"


def test_malformed_message_handling(tmp_path, monkeypatch):
    raw = tmp_path / "bybit_bad.ndjson"
    out_tr = tmp_path / "trades.csv"
    out_up = tmp_path / "updates.csv"
    out_bs = tmp_path / "book_state.csv"
    monkeypatch.setattr(pb, "OUTPUT_TRADES", out_tr)
    monkeypatch.setattr(pb, "OUTPUT_BOOK_UPDATES", out_up)
    monkeypatch.setattr(pb, "OUTPUT_BOOK_STATE", out_bs)
    pb._init_csv_headers()
    with raw.open("w", encoding="utf-8") as f:
        f.write("{not-json}\n")
        f.write(json.dumps({"source": "bybit", "payload": {"topic": "unknown"}}) + "\n")
    st = pb.parse_file(str(raw))
    assert st.malformed_lines >= 1
    assert out_tr.is_file()

