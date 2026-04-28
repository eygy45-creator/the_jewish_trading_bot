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
_TS_RE = re.compile(r'"(?:timestamp|local_ts)"\s*:\s*"([^"]+)"')
_CHANNEL_RE = re.compile(r'"channel"\s*:\s*"([^"]+)"')
REQUIRED_EXPORT_COLUMNS = [
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
    "imbalance",
    "microprice",
    "spread",
    "anomaly_score",
    "anomaly_percentile",
    "regime",
    "action",
    "raw_json",
]


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
    sampled = sorted(keys)
    # Include nested keys from latest file so dashboard debug shows actionable schema details.
    nested = _sample_latest_raw_keysets(max_objects=max_objects_per_file)
    for group_name in ("event_keys", "trade_keys", "update_keys"):
        for key in nested.get(group_name, []):
            sampled.append(f"{group_name}.{key}")
    return sorted(set(sampled))


def _sample_latest_raw_keysets(max_objects: int = 500) -> dict[str, list[str]]:
    files = sorted([p for p in RAW_DATA_DIR.glob(RAW_NDJSON_GLOB) if p.is_file()], key=lambda x: x.stat().st_mtime)
    if not files:
        return {"top_keys": [], "event_keys": [], "trade_keys": [], "update_keys": []}
    latest = files[-1]
    top_keys: set[str] = set()
    event_keys: set[str] = set()
    trade_keys: set[str] = set()
    update_keys: set[str] = set()
    n = 0
    for obj in _iter_json_objects(latest):
        top_keys.update(obj.keys())
        events = obj.get("events")
        if isinstance(events, list):
            for ev in events:
                if not isinstance(ev, dict):
                    continue
                event_keys.update(ev.keys())
                trades = ev.get("trades")
                if isinstance(trades, list):
                    for tr in trades:
                        if isinstance(tr, dict):
                            trade_keys.update(tr.keys())
                updates = ev.get("updates")
                if isinstance(updates, list):
                    for up in updates:
                        if isinstance(up, dict):
                            update_keys.update(up.keys())
        n += 1
        if n >= max_objects:
            break
    return {
        "top_keys": sorted(top_keys),
        "event_keys": sorted(event_keys),
        "trade_keys": sorted(trade_keys),
        "update_keys": sorted(update_keys),
    }


def _missing_l2_update_keys(raw_keysets: dict[str, list[str]]) -> list[str]:
    update_keys = set(raw_keysets.get("update_keys", []))
    required = {"price_level", "new_quantity", "side"}
    return sorted(required - update_keys)


def _first_present(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _sanitize_ts_for_filename(ts: str) -> str:
    return re.sub(r"[^0-9T]", "_", ts.strip())[:48]


def _parse_trade_window(entry_ts: str, exit_ts: str | None = None, before_seconds: int = 30, after_seconds: int = 30) -> TradeWindow:
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
        start_ts=entry - timedelta(seconds=max(0, int(before_seconds))),
        end_ts=exit_final + timedelta(seconds=max(0, int(after_seconds))),
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


def _extract_line_ts_and_channel(line: str) -> tuple[pd.Timestamp | None, str | None]:
    ts: pd.Timestamp | None = None
    ts_match = _TS_RE.search(line)
    if ts_match:
        ts = _safe_ts(ts_match.group(1))
    ch_match = _CHANNEL_RE.search(line)
    channel = ch_match.group(1).lower() if ch_match else None
    return ts, channel


def _build_raw_context(
    window: TradeWindow,
    max_lines_scanned: int = 300_000,
    max_events_output: int = 5_000,
) -> tuple[pd.DataFrame, set[str], list[str], int, int, str]:
    files = _candidate_raw_files(window)
    if not files:
        return pd.DataFrame(), set(), [], 0, 0, "no_candidate_files"

    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    buy_volume = 0.0
    sell_volume = 0.0
    buy_count = 0
    sell_count = 0
    rows: list[dict[str, Any]] = []
    raw_keys: set[str] = set()
    used_files: list[str] = [p.name for p in files]
    lines_scanned = 0
    stopped_reason = "end_of_window"

    for path in files:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    lines_scanned += 1
                    if lines_scanned > max_lines_scanned:
                        stopped_reason = "max_lines_scanned"
                        break

                    ts_hint, channel_hint = _extract_line_ts_and_channel(line)
                    if ts_hint is not None and ts_hint < window.start_ts:
                        continue
                    if ts_hint is not None and ts_hint > window.end_ts:
                        stopped_reason = "after_ts_reached"
                        break
                    if channel_hint is not None and channel_hint in {"heartbeats", "subscriptions"}:
                        continue
                    if channel_hint is not None and channel_hint != "l2_data":
                        continue

                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(obj, dict):
                        continue

                    raw_keys.update(obj.keys())
                    channel = str(obj.get("channel", "")).lower()
                    if channel in {"heartbeats", "subscriptions"}:
                        continue
                    if channel != "l2_data":
                        continue
                    events = obj.get("events")
                    if not isinstance(events, list):
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
                            if len(rows) >= max_events_output:
                                stopped_reason = "max_events_output"
                                break
                        if len(rows) >= max_events_output:
                            break
                    if len(rows) >= max_events_output:
                        break
            if stopped_reason in {"max_lines_scanned", "after_ts_reached", "max_events_output"}:
                break
        except OSError:
            continue

    if not rows:
        return pd.DataFrame(), raw_keys, used_files, 0, lines_scanned, stopped_reason
    out = pd.DataFrame(rows)
    out = out.sort_values("ts", ascending=True).drop_duplicates(subset=["ts", "side", "price", "size"], keep="last")
    out = out.reset_index(drop=True)
    return out, raw_keys, used_files, int(len(out)), lines_scanned, stopped_reason


def _row_phase(ts: pd.Timestamp, window: TradeWindow) -> str:
    if ts < window.entry_ts:
        return "before"
    if ts > window.exit_ts:
        return "after"
    return "during"


def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    wanted = REQUIRED_EXPORT_COLUMNS + [
        "timestamp",
        "phase",
        "volume",
        "delta",
        "aggressive_buyers",
        "aggressive_sellers",
        "queue_imbalance",
        "volatility",
        "direction",
        "trade_ref",
        "row_phase",
    ]
    out = df.copy()
    for c in wanted:
        if c not in out.columns:
            out[c] = pd.NA
    return out[wanted].copy()


def export_trade_context(
    entry_ts: str,
    exit_ts: str | None = None,
    before_seconds: int = 30,
    after_seconds: int = 30,
    max_lines_scanned: int = 300_000,
    max_events_output: int = 5_000,
    output_dir: Path | None = None,
    paper_trades_path: Path = PAPER_TRADES_PATH,
    **_ignored: Any,
) -> tuple[pd.DataFrame, Path, str]:
    """
    Export per-trade context window CSV for dashboard download.
    """
    _ = paper_trades_path  # explicit dependency for compatibility with call-sites
    window = _parse_trade_window(
        entry_ts=entry_ts,
        exit_ts=exit_ts,
        before_seconds=before_seconds,
        after_seconds=after_seconds,
    )
    raw_df, raw_keys, used_files, raw_event_count, lines_scanned, stopped_reason = _build_raw_context(
        window=window,
        max_lines_scanned=max_lines_scanned,
        max_events_output=max_events_output,
    )
    raw_keysets = _sample_latest_raw_keysets()
    opp_df = _load_opportunities_for_window(window)

    status_parts: list[str] = []
    if raw_df.empty:
        status_parts.append("raw microstructure context unavailable")
    if opp_df.empty:
        status_parts.append("opportunities context unavailable")
    status_parts.append(f"raw_file_used={used_files[-1] if used_files else 'none'}")
    status_parts.append(f"event_count_found={raw_event_count}")
    status_parts.append(f"events_used={raw_event_count}")
    status_parts.append(f"events_output={raw_event_count}")
    status_parts.append(f"lines_scanned={lines_scanned}")
    status_parts.append(f"stopped_reason={stopped_reason}")

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
    microprice_col = merged["microprice"] if "microprice" in merged.columns else pd.Series(index=merged.index, dtype=float)
    price_for_vol = pd.to_numeric(microprice_col, errors="coerce")
    ret = price_for_vol.pct_change(fill_method=None)
    merged["volatility"] = ret.rolling(window=20, min_periods=5).std()
    if not merged.empty:
        nearest_entry_idx = (merged["ts"] - window.entry_ts).abs().idxmin()
        merged.loc[nearest_entry_idx, "row_phase"] = "entry"
        nearest_exit_idx = (merged["ts"] - window.exit_ts).abs().idxmin()
        merged.loc[nearest_exit_idx, "row_phase"] = "exit"
    merged["phase"] = merged["row_phase"]
    merged["volume"] = pd.to_numeric(merged.get("size"), errors="coerce")
    merged["delta"] = pd.to_numeric(merged.get("buy_volume"), errors="coerce") - pd.to_numeric(merged.get("sell_volume"), errors="coerce")

    merged = _ensure_required_columns(merged)
    merged["timestamp"] = pd.to_datetime(merged["ts"], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    merged["ts"] = merged["timestamp"]
    merged = merged.replace({pd.NA: None})

    out_dir = output_dir or LIVE_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"trade_context_{_sanitize_ts_for_filename(window.trade_ref)}.csv"
    merged.to_csv(output_path, index=False, encoding="utf-8")
    required_micro_cols = REQUIRED_EXPORT_COLUMNS
    missing_micro = [c for c in required_micro_cols if c not in merged.columns or merged[c].isna().all()]
    if missing_micro:
        status_parts.append(f"missing_or_empty_fields={missing_micro}")
    if raw_keys:
        status_parts.append(f"raw_event_keys={sorted(raw_keys)}")
    missing_l2_keys = _missing_l2_update_keys(raw_keysets)
    if missing_l2_keys:
        status_parts.append(
            "raw_keys_warning=coinbase_l2_update_keys_missing;"
            f" missing={missing_l2_keys}; latest_keysets={raw_keysets}"
        )
    if merged.empty:
        status_parts.append("reason_empty=no_events_in_window_or_unreconstructable_l2")
    status_message = "ok" if not status_parts else "; ".join(status_parts)
    return merged, output_path, status_message
