from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from tjtb.runtime_paths import LIVE_DATA_DIR, OPPORTUNITIES_PATH, PAPER_TRADES_PATH, RAW_DATA_DIR

RAW_NDJSON_GLOB = "coinbase_*.ndjson"


@dataclass(frozen=True)
class TradeWindow:
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp
    trade_ref: str


def _safe_float(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _safe_ts(x: Any) -> pd.Timestamp | None:
    if x is None:
        return None
    ts = pd.to_datetime(x, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def _pick_ts(obj: dict[str, Any]) -> pd.Timestamp | None:
    for k in ("event_time", "time", "ts", "timestamp"):
        if k in obj:
            ts = _safe_ts(obj.get(k))
            if ts is not None:
                return ts
    return None


def sample_raw_event_keys(max_files: int = 1, max_objects_per_file: int = 200) -> list[str]:
    files = sorted([p for p in RAW_DATA_DIR.glob(RAW_NDJSON_GLOB) if p.is_file()], key=lambda x: x.stat().st_mtime)
    if not files:
        return []
    keys: set[str] = set()
    for path in files[-max_files:]:
        n = 0
        for obj in _iter_json_objects(path):
            keys.update(obj.keys())
            n += 1
            if n >= max_objects_per_file:
                break
    return sorted(keys)


def _first_present(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _sanitize_ts_for_filename(ts: str) -> str:
    return re.sub(r"[^0-9T]", "_", ts.strip())[:48]


def _parse_trade_window(entry_ts: str, exit_ts: str | None = None) -> TradeWindow:
    entry = _safe_ts(entry_ts)
    if entry is None:
        raise ValueError(f"Invalid entry timestamp: {entry_ts}")
    if exit_ts:
        exit_parsed = _safe_ts(exit_ts)
    else:
        exit_parsed = None
    exit_final = exit_parsed if exit_parsed is not None else entry
    return TradeWindow(
        entry_ts=entry,
        exit_ts=exit_final,
        start_ts=entry - timedelta(seconds=30),
        end_ts=exit_final + timedelta(seconds=60),
        trade_ref=entry_ts,
    )


def _load_opportunities_for_window(window: TradeWindow) -> pd.DataFrame:
    if not OPPORTUNITIES_PATH.is_file():
        return pd.DataFrame()
    try:
        opp = pd.read_csv(OPPORTUNITIES_PATH, encoding="utf-8")
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()
    if opp.empty:
        return pd.DataFrame()
    ts_col = _first_present(opp, ("ts", "timestamp", "time", "event_time"))
    if ts_col is None:
        return pd.DataFrame()
    opp = opp.copy()
    opp["timestamp"] = pd.to_datetime(opp[ts_col], utc=True, errors="coerce")
    opp = opp.dropna(subset=["timestamp"])
    opp = opp[(opp["timestamp"] >= window.start_ts) & (opp["timestamp"] <= window.end_ts)]
    if opp.empty:
        return pd.DataFrame()
    keep_cols = [
        c
        for c in (
            "timestamp",
            "anomaly_percentile",
            "anomaly_score",
            "direction",
            "regime",
            "action",
            "imbalance",
            "top_of_book_imbalance",
        )
        if c in opp.columns or c == "timestamp"
    ]
    return opp[keep_cols].copy()


def _candidate_raw_files(window: TradeWindow) -> list[Path]:
    files = [p for p in RAW_DATA_DIR.glob(RAW_NDJSON_GLOB) if p.is_file()]
    if not files:
        return []
    wanted_days = {
        window.start_ts.strftime("%Y-%m-%d"),
        window.entry_ts.strftime("%Y-%m-%d"),
        window.end_ts.strftime("%Y-%m-%d"),
    }
    selected: list[Path] = []
    for p in files:
        if any(day in p.name for day in wanted_days):
            selected.append(p)
    if selected:
        return sorted(selected, key=lambda x: x.stat().st_mtime)
    return sorted(files, key=lambda x: x.stat().st_mtime)


def _iter_json_objects(path: Path):
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    yield obj
    except OSError:
        return


def _build_raw_context(window: TradeWindow) -> tuple[pd.DataFrame, set[str]]:
    files = _candidate_raw_files(window)
    if not files:
        return pd.DataFrame(), set()

    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    buy_volume = 0.0
    sell_volume = 0.0
    buy_count = 0
    sell_count = 0
    rows: list[dict[str, Any]] = []
    raw_keys: set[str] = set()

    for path in files:
        for obj in _iter_json_objects(path):
            raw_keys.update(obj.keys())
            channel = str(obj.get("channel", "")).lower()
            events = obj.get("events")
            if not isinstance(events, list):
                continue

            if channel == "market_trades":
                for ev in events:
                    if not isinstance(ev, dict):
                        continue
                    raw_keys.update(ev.keys())
                    trades = ev.get("trades")
                    if not isinstance(trades, list):
                        continue
                    for tr in trades:
                        if not isinstance(tr, dict):
                            continue
                        raw_keys.update(tr.keys())
                        ts = _pick_ts(tr)
                        if ts is None:
                            ts = _pick_ts(ev)
                        if ts is None or ts < window.start_ts or ts > window.end_ts:
                            continue
                        side = str(tr.get("side", "")).lower()
                        size = _safe_float(tr.get("size"))
                        if size is None:
                            size = _safe_float(tr.get("qty"))
                        price = _safe_float(tr.get("price"))
                        if size is None:
                            continue
                        if side in {"buy", "bid"}:
                            buy_volume += size
                            buy_count += 1
                        elif side in {"sell", "ask", "offer"}:
                            sell_volume += size
                            sell_count += 1
                        best_bid = max(bids.keys()) if bids else None
                        best_ask = min(asks.keys()) if asks else None
                        bid_sz = bids.get(best_bid, 0.0) if best_bid is not None else None
                        ask_sz = asks.get(best_ask, 0.0) if best_ask is not None else None
                        spread = (best_ask - best_bid) if (best_bid is not None and best_ask is not None and best_ask > best_bid) else None
                        denom = (bid_sz or 0.0) + (ask_sz or 0.0)
                        imbalance = (((bid_sz or 0.0) - (ask_sz or 0.0)) / denom) if denom > 0 else None
                        microprice = (
                            ((best_ask * (bid_sz or 0.0)) + (best_bid * (ask_sz or 0.0))) / denom
                            if denom > 0 and best_bid is not None and best_ask is not None
                            else None
                        )
                        rows.append(
                            {
                                "ts": ts,
                                "bid": best_bid,
                                "ask": best_ask,
                                "bid_size": bid_sz,
                                "ask_size": ask_sz,
                                "price": price,
                                "size": size,
                                "side": side,
                                "buy_volume": buy_volume,
                                "sell_volume": sell_volume,
                                "aggressive_buyers": buy_count,
                                "aggressive_sellers": sell_count,
                                "imbalance": imbalance,
                                "queue_imbalance": imbalance,
                                "microprice": microprice,
                                "spread": spread,
                                "raw_json": json.dumps(tr, ensure_ascii=True),
                            }
                        )
                continue

            if channel != "l2_data":
                continue

            for ev in events:
                if not isinstance(ev, dict):
                    continue
                ev_type = str(ev.get("type", "")).lower()
                if ev_type == "snapshot":
                    bids.clear()
                    asks.clear()
                updates = ev.get("updates")
                raw_keys.update(ev.keys())
                if not isinstance(updates, list):
                    continue
                for up in updates:
                    if not isinstance(up, dict):
                        continue
                    raw_keys.update(up.keys())
                    ts = _pick_ts(up)
                    if ts is None:
                        continue
                    side = str(up.get("side", "")).lower()
                    if side == "offer":
                        side = "ask"
                    px = _safe_float(up.get("price_level"))
                    if px is None:
                        px = _safe_float(up.get("price"))
                    qty = _safe_float(up.get("new_quantity"))
                    if qty is None:
                        qty = _safe_float(up.get("size"))
                    if qty is None:
                        qty = _safe_float(up.get("qty"))
                    if side not in {"bid", "ask"} or px is None or qty is None:
                        continue

                    book = bids if side == "bid" else asks
                    if qty <= 0:
                        book.pop(px, None)
                    else:
                        book[px] = qty

                    if ts < window.start_ts or ts > window.end_ts:
                        continue
                    best_bid = max(bids.keys()) if bids else None
                    best_ask = min(asks.keys()) if asks else None
                    if best_bid is not None and best_ask is not None and best_bid >= best_ask:
                        continue
                    bid_sz = bids.get(best_bid, 0.0) if best_bid is not None else None
                    ask_sz = asks.get(best_ask, 0.0) if best_ask is not None else None
                    bid_sz_f = float(bid_sz) if bid_sz is not None else 0.0
                    ask_sz_f = float(ask_sz) if ask_sz is not None else 0.0
                    denom = bid_sz_f + ask_sz_f
                    imbalance = ((bid_sz_f - ask_sz_f) / denom) if denom > 0 else None
                    microprice = ((best_ask * bid_sz_f) + (best_bid * ask_sz_f)) / denom if denom > 0 and best_bid is not None and best_ask is not None else None
                    spread = (best_ask - best_bid) if (best_bid is not None and best_ask is not None) else None
                    rows.append(
                        {
                            "ts": ts,
                            "bid": best_bid,
                            "ask": best_ask,
                            "bid_size": bid_sz,
                            "ask_size": ask_sz,
                            "microprice": microprice,
                            "spread": spread,
                            "price": px,
                            "size": qty,
                            "side": side,
                            "buy_volume": buy_volume,
                            "sell_volume": sell_volume,
                            "aggressive_buyers": buy_count,
                            "aggressive_sellers": sell_count,
                            "imbalance": imbalance,
                            "queue_imbalance": imbalance,
                            "raw_json": json.dumps(up, ensure_ascii=True),
                        }
                    )

    if not rows:
        return pd.DataFrame(), raw_keys
    out = pd.DataFrame(rows)
    out = out.sort_values("ts", ascending=True).drop_duplicates(subset=["ts", "side", "price", "size"], keep="last")
    return out.reset_index(drop=True), raw_keys


def _row_phase(ts: pd.Timestamp, window: TradeWindow) -> str:
    if ts == window.entry_ts:
        return "entry"
    if ts == window.exit_ts:
        return "exit"
    if ts < window.entry_ts:
        return "before_entry"
    return "after_entry"


def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    wanted = [
        "ts",
        "bid",
        "ask",
        "bid_size",
        "ask_size",
        "price",
        "size",
        "side",
        "microprice",
        "spread",
        "buy_volume",
        "aggressive_buyers",
        "sell_volume",
        "aggressive_sellers",
        "imbalance",
        "queue_imbalance",
        "volatility",
        "anomaly_score",
        "anomaly_percentile",
        "direction",
        "regime",
        "action",
        "trade_ref",
        "row_phase",
        "raw_json",
    ]
    out = df.copy()
    for c in wanted:
        if c not in out.columns:
            out[c] = pd.NA
    return out[wanted].copy()


def export_trade_context(
    entry_ts: str,
    exit_ts: str | None = None,
    output_dir: Path | None = None,
    paper_trades_path: Path = PAPER_TRADES_PATH,
) -> tuple[pd.DataFrame, Path, str]:
    """
    Export per-trade context window CSV for dashboard download.
    """
    _ = paper_trades_path  # explicit dependency for compatibility with call-sites
    window = _parse_trade_window(entry_ts=entry_ts, exit_ts=exit_ts)
    raw_df, raw_keys = _build_raw_context(window)
    opp_df = _load_opportunities_for_window(window)

    status_parts: list[str] = []
    if raw_df.empty:
        status_parts.append("raw microstructure context unavailable")
    if opp_df.empty:
        status_parts.append("opportunities context unavailable")

    if raw_df.empty and not opp_df.empty:
        merged = opp_df.rename(columns={"timestamp": "ts"}).copy()
    elif raw_df.empty and opp_df.empty:
        merged = pd.DataFrame({"ts": pd.Series(dtype="datetime64[ns, UTC]")})
    elif opp_df.empty:
        merged = raw_df.copy()
    else:
        left = raw_df.copy()
        left["ts"] = pd.to_datetime(left["ts"], utc=True, errors="coerce")
        right = opp_df.rename(columns={"timestamp": "ts"}).copy()
        right["ts"] = pd.to_datetime(right["ts"], utc=True, errors="coerce")
        left = left.sort_values("ts")
        right = right.sort_values("ts")
        merged = pd.merge_asof(left, right, on="ts", direction="nearest", tolerance=pd.Timedelta(seconds=1))

    merged["ts"] = pd.to_datetime(merged.get("ts"), utc=True, errors="coerce")
    merged = merged.dropna(subset=["ts"]).sort_values("ts")
    merged["trade_ref"] = window.trade_ref
    merged["row_phase"] = merged["ts"].apply(lambda ts: _row_phase(ts, window))
    price_for_vol = pd.to_numeric(merged.get("microprice"), errors="coerce")
    ret = price_for_vol.pct_change()
    merged["volatility"] = ret.rolling(window=20, min_periods=5).std()
    if not merged.empty:
        nearest_entry_idx = (merged["ts"] - window.entry_ts).abs().idxmin()
        merged.loc[nearest_entry_idx, "row_phase"] = "entry"
        nearest_exit_idx = (merged["ts"] - window.exit_ts).abs().idxmin()
        merged.loc[nearest_exit_idx, "row_phase"] = "exit"

    merged = _ensure_required_columns(merged)
    merged["ts"] = pd.to_datetime(merged["ts"], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    merged = merged.replace({pd.NA: None})

    out_dir = output_dir or LIVE_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"trade_context_{_sanitize_ts_for_filename(window.trade_ref)}.csv"
    merged.to_csv(output_path, index=False, encoding="utf-8")
    required_micro_cols = [
        "ts",
        "bid",
        "ask",
        "bid_size",
        "ask_size",
        "price",
        "size",
        "side",
        "buy_volume",
        "sell_volume",
        "aggressive_buyers",
        "aggressive_sellers",
        "imbalance",
        "queue_imbalance",
        "microprice",
        "spread",
        "volatility",
        "anomaly_score",
        "anomaly_percentile",
        "regime",
        "action",
        "raw_json",
    ]
    missing_micro = [c for c in required_micro_cols if c not in merged.columns or merged[c].isna().all()]
    if missing_micro:
        status_parts.append(f"missing_or_empty_fields={missing_micro}")
    if raw_keys:
        status_parts.append(f"raw_event_keys={sorted(raw_keys)}")
    status_message = "ok" if not status_parts else "; ".join(status_parts)
    return merged, output_path, status_message
