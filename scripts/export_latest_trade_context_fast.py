from __future__ import annotations

import json
import math
import re
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_NDJSON_GLOB = "coinbase_*.ndjson"
MAX_L2_ROWS = 5000
BEFORE_SECONDS = 120
AFTER_SECONDS = 120
MAX_RUNTIME_SECONDS = 9.5
_CHANNEL_RE = re.compile(r'"channel"\s*:\s*"([^"]+)"')
_TS_RE = re.compile(r'"(?:timestamp|local_ts)"\s*:\s*"([^"]+)"')


def _latest_raw_ndjson_path() -> Path | None:
    raw_dir = ROOT / "data" / "raw"
    files = sorted(
        [p for p in raw_dir.glob(RAW_NDJSON_GLOB) if p.is_file()],
        key=lambda p: p.stat().st_mtime,
    )
    return files[-1] if files else None


def _parse_any_ts(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def _trade_window(latest_trade: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    entry_ts = _parse_any_ts(latest_trade.get("entry_ts"))
    exit_ts = _parse_any_ts(latest_trade.get("exit_ts"))
    if entry_ts is None:
        raise ValueError("Latest trade has invalid entry_ts")
    if exit_ts is None or exit_ts < entry_ts:
        exit_ts = entry_ts
    before_ts = entry_ts - timedelta(seconds=BEFORE_SECONDS)
    after_ts = exit_ts + timedelta(seconds=AFTER_SECONDS)
    return before_ts, entry_ts, after_ts


def _best_book(bids: dict[float, float], asks: dict[float, float]) -> tuple[float | None, float | None, float | None, float | None]:
    bid = max(bids.keys()) if bids else None
    ask = min(asks.keys()) if asks else None
    bid_size = bids.get(bid) if bid is not None else None
    ask_size = asks.get(ask) if ask is not None else None
    return bid, ask, bid_size, ask_size


def main() -> None:
    started = time.perf_counter()
    paper_trades_path = ROOT / "data" / "live" / "paper_trades.csv"
    output_path = ROOT / "data" / "live" / "trade_context_latest_fast.csv"

    if not paper_trades_path.is_file():
        raise FileNotFoundError(f"Missing file: {paper_trades_path}")

    trades = pd.read_csv(paper_trades_path, encoding="utf-8")
    if trades.empty:
        raise ValueError("paper_trades.csv is empty")
    if "entry_ts" not in trades.columns:
        raise ValueError("paper_trades.csv missing required column: entry_ts")

    if "exit_ts" in trades.columns:
        trades = trades.copy()
        trades["_exit_ts"] = pd.to_datetime(trades["exit_ts"], utc=True, errors="coerce")
        latest_trade = trades.sort_values("_exit_ts", ascending=False, na_position="last").iloc[0]
    else:
        latest_trade = trades.iloc[-1]

    before_ts, _entry_ts, after_ts = _trade_window(latest_trade)
    raw_path = _latest_raw_ndjson_path()
    if raw_path is None:
        raise FileNotFoundError(f"No files matching {RAW_NDJSON_GLOB} under {ROOT / 'data' / 'raw'}")

    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    seen_snapshot = False
    rows: list[dict[str, Any]] = []

    with raw_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if time.perf_counter() - started > MAX_RUNTIME_SECONDS:
                break
            if len(rows) >= MAX_L2_ROWS:
                break

            ch = _CHANNEL_RE.search(line)
            if not ch or ch.group(1) != "l2_data":
                continue

            ts_hint = _TS_RE.search(line)
            ts_hint_dt = _parse_any_ts(ts_hint.group(1)) if ts_hint else None
            if ts_hint_dt is not None and ts_hint_dt > after_ts:
                break

            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            if payload.get("channel") != "l2_data":
                continue

            line_ts = (
                _parse_any_ts(payload.get("timestamp"))
                or _parse_any_ts(payload.get("local_ts"))
                or ts_hint_dt
            )
            if line_ts is not None and line_ts > after_ts:
                break

            events = payload.get("events", [])
            if not isinstance(events, list):
                continue

            for event in events:
                event_type = str(event.get("type", "")).lower()
                if event_type == "snapshot":
                    bids.clear()
                    asks.clear()
                    seen_snapshot = True

                updates = event.get("updates", [])
                if not isinstance(updates, list):
                    continue

                for upd in updates:
                    side = str(upd.get("side", "")).strip().lower()
                    if side == "offer":
                        side = "ask"
                    if side not in {"bid", "ask"}:
                        continue

                    try:
                        price_f = float(upd.get("price_level"))
                        qty_f = float(upd.get("new_quantity"))
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(price_f) or not math.isfinite(qty_f):
                        continue

                    book = bids if side == "bid" else asks
                    if qty_f <= 0.0:
                        book.pop(price_f, None)
                    else:
                        book[price_f] = qty_f

                if not seen_snapshot:
                    continue

                event_ts = (
                    _parse_any_ts(event.get("event_time"))
                    or _parse_any_ts(event.get("time"))
                    or line_ts
                )
                if event_ts is None:
                    continue
                if event_ts < before_ts:
                    continue
                if event_ts > after_ts:
                    break

                bid, ask, bid_size, ask_size = _best_book(bids, asks)
                spread = (ask - bid) if (ask is not None and bid is not None) else None
                rows.append(
                    {
                        "ts": event_ts.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                        "bid": bid,
                        "ask": ask,
                        "bid_size": bid_size,
                        "ask_size": ask_size,
                        "spread": spread,
                        "raw_json": line.rstrip("\n"),
                    }
                )
                if len(rows) >= MAX_L2_ROWS:
                    break

    out_df = pd.DataFrame(rows, columns=["ts", "bid", "ask", "bid_size", "ask_size", "spread", "raw_json"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False, encoding="utf-8")

    bid_non_null = int(out_df["bid"].notna().sum()) if "bid" in out_df.columns else 0
    ask_non_null = int(out_df["ask"].notna().sum()) if "ask" in out_df.columns else 0
    first_ts = str(out_df["ts"].iloc[0]) if not out_df.empty else ""
    last_ts = str(out_df["ts"].iloc[-1]) if not out_df.empty else ""

    print(f"rows: {len(out_df)}")
    print(f"bid_non_null: {bid_non_null}")
    print(f"ask_non_null: {ask_non_null}")
    print(f"first_ts: {first_ts}")
    print(f"last_ts: {last_ts}")
    print(f"output_path: {output_path}")


if __name__ == "__main__":
    main()
