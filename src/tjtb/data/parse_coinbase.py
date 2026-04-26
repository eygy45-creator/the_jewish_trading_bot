"""
Stream Coinbase Advanced Trade NDJSON recordings into normalized CSVs.

Uses only the standard library. Does not load entire files into memory.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_RAW_GLOB = "data/raw/coinbase_*.ndjson"
OUTPUT_TRADES = Path("data/parsed/trades.csv")
OUTPUT_BOOK = Path("data/parsed/book_updates.csv")
READ_BUFFER = 1024 * 1024
WRITE_BUFFER = 1024 * 1024


@dataclass
class ParseStats:
    trades_written: int = 0
    book_updates_written: int = 0
    snapshot_rows_written: int = 0
    lines_skipped: int = 0


def parse_trade_events(obj: dict[str, Any]) -> list[tuple[str, float, float, str]]:
    """
    Extract normalized trade rows from one top-level WebSocket message dict.

    Expects channel ``market_trades``. Returns rows: (ts, price, size, side)
    with side in ``buy`` | ``sell``.
    """
    if obj.get("channel") != "market_trades":
        return []
    rows: list[tuple[str, float, float, str]] = []
    events = obj.get("events")
    if not isinstance(events, list):
        return rows
    for ev in events:
        if not isinstance(ev, dict):
            continue
        trades = ev.get("trades")
        if not isinstance(trades, list):
            continue
        for t in trades:
            if not isinstance(t, dict):
                continue
            try:
                ts = str(t["time"])
                price = float(t["price"])
                size = float(t["size"])
                raw = str(t["side"]).upper()
                if raw == "BUY":
                    side = "buy"
                elif raw == "SELL":
                    side = "sell"
                else:
                    continue
                rows.append((ts, price, size, side))
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def parse_l2_events(obj: dict[str, Any]) -> list[tuple[str, str, int, str, float, float]]:
    """
    Extract normalized L2 book update rows from one top-level message dict.

    Expects channel ``l2_data``. Maps ``offer`` -> ``ask``.
    Returns rows: (ts, event_type, sequence_num, side, price, size) with side in ``bid`` | ``ask``.
    """
    if obj.get("channel") != "l2_data":
        return []
    rows: list[tuple[str, str, int, str, float, float]] = []
    events = obj.get("events")
    sequence_num = obj.get("sequence_num")
    if not isinstance(sequence_num, int):
        return rows
    if not isinstance(events, list):
        return rows
    for ev in events:
        if not isinstance(ev, dict):
            continue
        ev_type = str(ev.get("type", "")).lower()
        if ev_type not in ("snapshot", "update"):
            continue
        updates = ev.get("updates")
        if not isinstance(updates, list):
            continue
        for u in updates:
            if not isinstance(u, dict):
                continue
            try:
                ts = str(u["event_time"])
                raw_side = str(u["side"]).lower()
                if raw_side == "offer":
                    side = "ask"
                elif raw_side == "bid":
                    side = "bid"
                else:
                    continue
                price = float(u["price_level"])
                size = float(u["new_quantity"])
                rows.append((ts, ev_type, sequence_num, side, price, size))
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def _init_csv_headers() -> None:
    OUTPUT_TRADES.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_BOOK.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_TRADES.open("w", newline="", encoding="utf-8", buffering=WRITE_BUFFER) as f:
        csv.writer(f).writerow(["ts", "price", "size", "side"])
    with OUTPUT_BOOK.open("w", newline="", encoding="utf-8", buffering=WRITE_BUFFER) as f:
        csv.writer(f).writerow(["ts", "event_type", "sequence_num", "side", "price", "size"])


def parse_file(input_path: str) -> ParseStats:
    """
    Stream-parse one NDJSON file; append rows to ``data/parsed/trades.csv`` and
    ``data/parsed/book_updates.csv``.

    Call :func:`_init_csv_headers` once before the first :func:`parse_file` when
    processing multiple sources so outputs are truncated deterministically.
    """
    stats = ParseStats()
    with (
        Path(input_path).open("r", encoding="utf-8", errors="replace", buffering=READ_BUFFER) as inp,
        OUTPUT_TRADES.open("a", newline="", encoding="utf-8", buffering=WRITE_BUFFER) as ft,
        OUTPUT_BOOK.open("a", newline="", encoding="utf-8", buffering=WRITE_BUFFER) as fb,
    ):
        trades_writer = csv.writer(ft)
        book_writer = csv.writer(fb)
        for line in inp:
            stripped = line.strip()
            if not stripped:
                stats.lines_skipped += 1
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                stats.lines_skipped += 1
                continue
            if not isinstance(obj, dict):
                stats.lines_skipped += 1
                continue

            try:
                ch = obj.get("channel")
                if ch == "heartbeats":
                    continue
                if ch == "market_trades":
                    for row in parse_trade_events(obj):
                        trades_writer.writerow(row)
                        stats.trades_written += 1
                elif ch == "l2_data":
                    for row in parse_l2_events(obj):
                        book_writer.writerow(row)
                        stats.book_updates_written += 1
                        if row[1] == "snapshot":
                            stats.snapshot_rows_written += 1
            except Exception:
                stats.lines_skipped += 1
                continue
    return stats


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )
    paths = sorted(Path().glob(DEFAULT_RAW_GLOB))
    if not paths:
        logger.error("No input files matching %s", DEFAULT_RAW_GLOB)
        return

    _init_csv_headers()
    total = ParseStats()
    for p in paths:
        if not p.is_file():
            logger.warning("Not a file, skipping: %s", p)
            total.lines_skipped += 1
            continue
        st = parse_file(str(p.resolve()))
        total.trades_written += st.trades_written
        total.book_updates_written += st.book_updates_written
        total.snapshot_rows_written += st.snapshot_rows_written
        total.lines_skipped += st.lines_skipped
        logger.info(
            "Parsed %s trades=%s book_updates=%s snapshot_rows=%s lines_skipped=%s",
            p.name,
            st.trades_written,
            st.book_updates_written,
            st.snapshot_rows_written,
            st.lines_skipped,
        )

    logger.info(
        "Total trades=%s book_updates=%s snapshot_rows=%s lines_skipped=%s -> %s , %s",
        total.trades_written,
        total.book_updates_written,
        total.snapshot_rows_written,
        total.lines_skipped,
        OUTPUT_TRADES,
        OUTPUT_BOOK,
    )


if __name__ == "__main__":
    main()
