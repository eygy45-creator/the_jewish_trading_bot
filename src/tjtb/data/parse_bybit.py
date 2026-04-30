"""
Parse recorded Bybit public NDJSON into normalized CSV artifacts.

Input lines are produced by ``tjtb.data.bybit_recorder`` and expected to contain
an envelope with ``payload`` preserving raw WS message JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_RAW_GLOB = "data/raw/bybit_*.ndjson"
OUTPUT_TRADES = Path("data/parsed/bybit_trades.csv")
OUTPUT_BOOK_UPDATES = Path("data/parsed/bybit_book_updates.csv")
OUTPUT_BOOK_STATE = Path("data/parsed/bybit_book_state.csv")
READ_BUFFER = 1024 * 1024
WRITE_BUFFER = 1024 * 1024
SIZE_EPS = 1e-12
DEFAULT_SYMBOL = "BTCUSDT"


@dataclass
class ParseStats:
    trades_written: int = 0
    book_updates_written: int = 0
    snapshots_seen: int = 0
    book_state_written: int = 0
    malformed_lines: int = 0


def _iso_from_ms(ms: Any) -> str | None:
    try:
        v = int(ms)
    except (TypeError, ValueError):
        return None
    dt = datetime.fromtimestamp(v / 1000.0, tz=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def _coerce_float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f


def _extract_payload(obj: dict[str, Any]) -> dict[str, Any] | None:
    payload = obj.get("payload")
    return payload if isinstance(payload, dict) else None


def parse_trade_rows(payload: dict[str, Any], symbol: str = DEFAULT_SYMBOL) -> list[tuple[str, float, float, str, int | None]]:
    if str(payload.get("topic", "")) != f"publicTrade.{symbol}":
        return []
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    out: list[tuple[str, float, float, str, int | None]] = []
    for tr in data:
        if not isinstance(tr, dict):
            continue
        ts = _iso_from_ms(tr.get("T"))
        px = _coerce_float(tr.get("p"))
        sz = _coerce_float(tr.get("v"))
        side_raw = str(tr.get("S", "")).lower()
        side = "buy" if side_raw == "buy" else ("sell" if side_raw == "sell" else "")
        if ts is None or px is None or sz is None or not side:
            continue
        seq = tr.get("seq")
        try:
            seq_num = int(seq) if seq is not None else None
        except (TypeError, ValueError):
            seq_num = None
        out.append((ts, px, sz, side, seq_num))
    return out


def parse_orderbook_levels(
    payload: dict[str, Any],
    symbol: str = DEFAULT_SYMBOL,
) -> tuple[str, str, int, list[tuple[str, float, float]]]:
    """
    Return (ts_iso, event_type, sequence_num, levels) where each level is (side, price, size).
    """
    if str(payload.get("topic", "")) != f"orderbook.50.{symbol}":
        return "", "", 0, []
    msg_type = str(payload.get("type", "")).lower()
    if msg_type not in {"snapshot", "delta"}:
        return "", "", 0, []
    data = payload.get("data")
    if not isinstance(data, dict):
        return "", "", 0, []
    if str(data.get("s", "")) != symbol:
        return "", "", 0, []
    ts_iso = _iso_from_ms(payload.get("ts"))
    if ts_iso is None:
        return "", "", 0, []
    seq_raw = data.get("seq", payload.get("seq", 0))
    try:
        seq = int(seq_raw)
    except (TypeError, ValueError):
        seq = 0

    levels: list[tuple[str, float, float]] = []
    bids = data.get("b")
    asks = data.get("a")
    if isinstance(bids, list):
        for lv in bids:
            if not isinstance(lv, list) or len(lv) < 2:
                continue
            px = _coerce_float(lv[0])
            sz = _coerce_float(lv[1])
            if px is None or sz is None:
                continue
            levels.append(("bid", px, sz))
    if isinstance(asks, list):
        for lv in asks:
            if not isinstance(lv, list) or len(lv) < 2:
                continue
            px = _coerce_float(lv[0])
            sz = _coerce_float(lv[1])
            if px is None or sz is None:
                continue
            levels.append(("ask", px, sz))
    return ts_iso, msg_type, seq, levels


def _init_csv_headers() -> None:
    OUTPUT_TRADES.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_TRADES.open("w", newline="", encoding="utf-8", buffering=WRITE_BUFFER) as f:
        csv.writer(f).writerow(["ts", "price", "size", "side", "sequence_num"])
    with OUTPUT_BOOK_UPDATES.open("w", newline="", encoding="utf-8", buffering=WRITE_BUFFER) as f:
        csv.writer(f).writerow(["ts", "event_type", "sequence_num", "side", "price", "size"])
    with OUTPUT_BOOK_STATE.open("w", newline="", encoding="utf-8", buffering=WRITE_BUFFER) as f:
        csv.writer(f).writerow(
            [
                "ts",
                "best_bid",
                "best_ask",
                "best_bid_size",
                "best_ask_size",
                "spread",
                "mid",
                "microprice",
                "tob_imbalance",
                "signed_pressure",
                "l2_event_type",
                "sequence_num",
            ]
        )


def parse_file(input_path: str, symbol: str = DEFAULT_SYMBOL) -> ParseStats:
    stats = ParseStats()
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    with (
        Path(input_path).open("r", encoding="utf-8", errors="replace", buffering=READ_BUFFER) as inp,
        OUTPUT_TRADES.open("a", newline="", encoding="utf-8", buffering=WRITE_BUFFER) as ft,
        OUTPUT_BOOK_UPDATES.open("a", newline="", encoding="utf-8", buffering=WRITE_BUFFER) as fb,
        OUTPUT_BOOK_STATE.open("a", newline="", encoding="utf-8", buffering=WRITE_BUFFER) as fs,
    ):
        tw = csv.writer(ft)
        bw = csv.writer(fb)
        sw = csv.writer(fs)
        for line in inp:
            s = line.strip()
            if not s:
                stats.malformed_lines += 1
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                stats.malformed_lines += 1
                continue
            if not isinstance(obj, dict):
                stats.malformed_lines += 1
                continue
            payload = _extract_payload(obj)
            if payload is None:
                continue

            trade_rows = parse_trade_rows(payload, symbol=symbol)
            for row in trade_rows:
                tw.writerow(row)
                stats.trades_written += 1

            ts_iso, ev_type, seq, levels = parse_orderbook_levels(payload, symbol=symbol)
            if not levels:
                continue
            if ev_type == "snapshot":
                bids.clear()
                asks.clear()
                stats.snapshots_seen += 1

            signed_pressure = 0.0
            for side, px, nq in levels:
                book = bids if side == "bid" else asks
                old = float(book.get(px, 0.0))
                delta = nq - old
                signed_pressure += delta if side == "bid" else -delta
                if nq <= SIZE_EPS:
                    book.pop(px, None)
                else:
                    book[px] = nq
                bw.writerow([ts_iso, ev_type, seq, side, px, nq])
                stats.book_updates_written += 1

            if not bids or not asks:
                continue
            bb = max(bids)
            ba = min(asks)
            if bb >= ba:
                continue
            bsz = float(bids.get(bb, 0.0))
            asz = float(asks.get(ba, 0.0))
            if bsz <= SIZE_EPS or asz <= SIZE_EPS:
                continue
            spread = ba - bb
            mid = (bb + ba) / 2.0
            denom = bsz + asz
            micro = (ba * bsz + bb * asz) / denom
            tob_imb = (bsz - asz) / denom
            sw.writerow([ts_iso, bb, ba, bsz, asz, spread, mid, micro, tob_imb, signed_pressure, ev_type, seq])
            stats.book_state_written += 1
    return stats


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    p = argparse.ArgumentParser(description="Parse Bybit raw NDJSON into normalized CSVs")
    p.add_argument("--glob", default=DEFAULT_RAW_GLOB)
    p.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Bybit symbol, e.g. BTCUSDT or ETHUSDT")
    args = p.parse_args(argv)
    paths = sorted(Path().glob(args.glob))
    if not paths:
        logger.error("No input files matching %s", args.glob)
        return
    symbol = str(args.symbol).strip().upper()
    _init_csv_headers()
    total = ParseStats()
    for path in paths:
        st = parse_file(str(path.resolve()), symbol=symbol)
        total.trades_written += st.trades_written
        total.book_updates_written += st.book_updates_written
        total.snapshots_seen += st.snapshots_seen
        total.book_state_written += st.book_state_written
        total.malformed_lines += st.malformed_lines
        logger.info(
            "Parsed %s trades=%s book_updates=%s snapshots=%s book_state=%s malformed=%s",
            path.name,
            st.trades_written,
            st.book_updates_written,
            st.snapshots_seen,
            st.book_state_written,
            st.malformed_lines,
        )
    logger.info(
        "Total trades=%s book_updates=%s snapshots=%s book_state=%s malformed=%s -> %s, %s, %s",
        total.trades_written,
        total.book_updates_written,
        total.snapshots_seen,
        total.book_state_written,
        total.malformed_lines,
        OUTPUT_TRADES,
        OUTPUT_BOOK_UPDATES,
        OUTPUT_BOOK_STATE,
    )


if __name__ == "__main__":
    main()

