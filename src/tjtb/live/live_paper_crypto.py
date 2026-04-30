"""
Live paper-trading loop for Coinbase BTC-USD recorder files.

Paper only: no real orders or API credentials.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import fcntl
import json
import logging
import math
import os
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from tjtb.exchanges.bybit.execution import BybitDemoExecution, ExecutionConfig
from tjtb.features.build_features import parse_ts_to_unix
from tjtb.runtime_paths import (
    HEARTBEAT_PATH,
    LIVE_BOT_LOG_PATH,
    LOGS_DIR,
    OPPORTUNITIES_PATH,
    PAPER_TRADES_PATH,
    RAW_DATA_DIR,
    REPORTS_DIR,
    ensure_runtime_dirs,
)

RAW_GLOB = "coinbase_*.ndjson"
BYBIT_RAW_GLOB = "bybit_*.ndjson"
REPORT_PATH = REPORTS_DIR / "live_status.json"
LOG_PATH = LIVE_BOT_LOG_PATH
LOCK_PATH = LOGS_DIR / "tjtb-live.lock"
OPPORTUNITIES_HEADER = [
    "ts",
    "anomaly_percentile",
    "anomaly_score",
    "direction",
    "regime",
    "action",
    "reason",
]
EXECUTION_DRY_RUN_HEADER = [
    "ts",
    "data_source",
    "execution_mode",
    "symbol",
    "entry_price",
    "stop_price",
    "tp_price",
    "qty",
    "risk_usd",
    "notional",
    "leverage",
    "ok",
    "reason",
]

LOOKBACK_SEC = 15.0
CALIBRATION_SEC = 300.0
COOLDOWN_SEC = 60.0
CLUSTER_WINDOW_SEC = 30.0
MAX_TRADES_PER_DAY = 20
MAX_TRADES_PER_SESSION = 5
TIMEOUT_SEC = 2.0
EPS = 1e-9


def _ensure_runtime_dirs() -> None:
    ensure_runtime_dirs()


def _try_acquire_singleton_lock(logger: logging.Logger) -> object | None:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOCK_PATH, "a+", encoding="utf-8")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        logger.warning("another instance holds %s; exiting", LOCK_PATH)
        fh.close()
        return None
    fh.seek(0)
    fh.truncate()
    fh.write(str(os.getpid()))
    fh.flush()
    return fh


def _start_heartbeat(interval_sec: float, stop: threading.Event) -> threading.Thread:
    def _run() -> None:
        HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
        while not stop.wait(timeout=max(interval_sec, 1.0)):
            HEARTBEAT_PATH.write_text(_utc_now() + "\n", encoding="utf-8")

    t = threading.Thread(target=_run, name="tjtb-heartbeat", daemon=True)
    t.start()
    return t


def _setup_logger() -> logging.Logger:
    _ensure_runtime_dirs()
    logger = logging.getLogger("tjtb.live.paper")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _day_key(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def _session_label(ts: float) -> str:
    h = datetime.fromtimestamp(ts, tz=timezone.utc).hour
    if 0 <= h < 6:
        return "asia"
    if 6 <= h < 12:
        return "london"
    if 12 <= h < 20:
        return "ny"
    return "off_hours"


def _safe_float(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _write_csv_header(path: Path, header: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 0:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(header)


def _append_csv(path: Path, row: list[Any]) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(row)


def _ensure_opportunities_csv_schema(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file() or path.stat().st_size <= 0:
        with path.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(OPPORTUNITIES_HEADER)
        return
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
    except OSError:
        header = []
    if header == OPPORTUNITIES_HEADER:
        return
    bad_path = path.parent / "opportunities.bad.csv"
    if bad_path.is_file():
        bad_path.unlink()
    path.replace(bad_path)
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(OPPORTUNITIES_HEADER)


def _append_opportunity_row(
    ts: str,
    anomaly_percentile: float,
    anomaly_score: float,
    direction: str,
    regime: str,
    action: str,
    reason: str,
) -> None:
    _ensure_opportunities_csv_schema(OPPORTUNITIES_PATH)
    _append_csv(
        OPPORTUNITIES_PATH,
        [ts, anomaly_percentile, anomaly_score, direction, regime, action, reason],
    )


@dataclass
class TopState:
    ts: float
    ts_text: str
    best_bid: float
    best_ask: float
    best_bid_sz: float
    best_ask_sz: float
    spread: float
    mid: float
    micro_dev: float
    tob_imb: float


class RollingStat:
    def __init__(self, window_sec: float) -> None:
        self.window_sec = window_sec
        self.q: deque[tuple[float, float]] = deque()
        self.n = 0
        self.s = 0.0
        self.s2 = 0.0

    def _expire(self, ts: float) -> None:
        lo = ts - self.window_sec
        while self.q and self.q[0][0] < lo:
            _, v = self.q.popleft()
            self.n -= 1
            self.s -= v
            self.s2 -= v * v

    def zscore_before(self, ts: float, y: float) -> float | None:
        self._expire(ts)
        if self.n < 2:
            return None
        mu = self.s / self.n
        var = (self.s2 - (self.s * self.s / self.n)) / (self.n - 1)
        if var <= 1e-12:
            return None
        return (y - mu) / math.sqrt(var)

    def add(self, ts: float, y: float) -> None:
        self.q.append((ts, y))
        self.n += 1
        self.s += y
        self.s2 += y * y


class PercentileRank:
    def __init__(self, window_sec: float) -> None:
        self.window_sec = window_sec
        self.q: deque[tuple[float, float]] = deque()
        self.sorted_vals: list[float] = []

    def _expire(self, ts: float) -> None:
        lo = ts - self.window_sec
        while self.q and self.q[0][0] < lo:
            _, v = self.q.popleft()
            i = bisect.bisect_left(self.sorted_vals, v)
            if i < len(self.sorted_vals) and self.sorted_vals[i] == v:
                self.sorted_vals.pop(i)

    def rank_before(self, ts: float, v: float) -> float | None:
        self._expire(ts)
        n = len(self.sorted_vals)
        if n == 0:
            return None
        r = bisect.bisect_right(self.sorted_vals, v)
        return r / float(n)

    def add(self, ts: float, v: float) -> None:
        self.q.append((ts, v))
        bisect.insort(self.sorted_vals, v)


class IncrementalNDJSON:
    def __init__(self, raw_glob: str) -> None:
        self.raw_glob = raw_glob
        self.file_path: Path | None = None
        self.offset = 0
        self.partial = ""

    def _latest_file(self) -> Path | None:
        files = [p for p in RAW_DATA_DIR.glob(self.raw_glob) if p.is_file()]
        if not files:
            return None
        files.sort(key=lambda p: p.stat().st_mtime)
        return files[-1]

    def read_new_objects(self) -> list[dict[str, Any]]:
        latest = self._latest_file()
        if latest is None:
            return []
        if self.file_path is None or latest != self.file_path:
            self.file_path = latest
            self.offset = 0
            self.partial = ""
        with latest.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(self.offset)
            chunk = f.read()
            self.offset = f.tell()
        if not chunk:
            return []
        buf = self.partial + chunk
        lines = buf.splitlines(keepends=False)
        if chunk and not chunk.endswith("\n"):
            self.partial = lines[-1] if lines else buf
            lines = lines[:-1]
        else:
            self.partial = ""
        out: list[dict[str, Any]] = []
        for ln in lines:
            s = ln.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
        return out


class LivePaperEngine:
    def __init__(self, logger: logging.Logger, data_source: str = "coinbase") -> None:
        self.log = logger
        ds = str(data_source).strip().lower()
        self.data_source = "bybit" if ds == "bybit" else "coinbase"
        self.exchange = "bybit" if self.data_source == "bybit" else "coinbase"
        self.raw_glob = BYBIT_RAW_GLOB if self.data_source == "bybit" else RAW_GLOB
        self.reader = IncrementalNDJSON(self.raw_glob)
        self.started_at = _utc_now()

        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.last_top: TopState | None = None

        self.trade_times: deque[float] = deque()
        self.l2_times: deque[float] = deque()
        self.mid_window: deque[tuple[float, float]] = deque()
        self.last_mid: float | None = None
        self.last_l2_pressure = 0.0

        self.z_stats = {
            "tob": RollingStat(LOOKBACK_SEC),
            "micro": RollingStat(LOOKBACK_SEC),
            "pressure": RollingStat(LOOKBACK_SEC),
            "event_rate": RollingStat(LOOKBACK_SEC),
            "trade_count": RollingStat(LOOKBACK_SEC),
            "spread": RollingStat(LOOKBACK_SEC),
            "mid_vol": RollingStat(LOOKBACK_SEC),
        }
        self.rank = PercentileRank(CALIBRATION_SEC)

        self.raw_events_seen = 0
        self.signals_seen = 0
        self.trades_taken = 0
        self.trades_blocked = 0
        self.trades_blocked_by_reason: dict[str, int] = {}
        self.last_signal_ts = -1e18
        self.last_entry_ts = -1e18
        self.daily_counts: dict[str, int] = {}
        self.session_counts: dict[str, int] = {}

        self.open_trade: dict[str, Any] | None = None
        self.closed_trades: list[dict[str, Any]] = []
        self.realized_pnl_r = 0.0
        self.equity_curve: list[float] = []
        self.max_losing_streak = 0
        self._curr_losing_streak = 0
        self.opportunities_appended = 0
        self.execution_mode = str(os.environ.get("EXECUTION_MODE", "paper")).strip().lower()
        self.execution_dry_run_path = PAPER_TRADES_PATH.parent / "execution_dry_run.csv"
        self.last_execution_plan: dict[str, Any] | None = None
        self._bybit_execution: BybitDemoExecution | None = None
        if self.execution_mode == "bybit_demo_dry_run":
            self._bybit_execution = BybitDemoExecution(config=ExecutionConfig.from_env())

        _ensure_opportunities_csv_schema(OPPORTUNITIES_PATH)
        _write_csv_header(
            PAPER_TRADES_PATH,
            [
                "entry_ts",
                "exit_ts",
                "side",
                "entry_price",
                "exit_price",
                "outcome",
                "r_value",
                "regime",
            ],
        )
        _write_csv_header(self.execution_dry_run_path, EXECUTION_DRY_RUN_HEADER)

    def _expire_windows(self, ts: float) -> None:
        lo = ts - LOOKBACK_SEC
        while self.trade_times and self.trade_times[0] < lo:
            self.trade_times.popleft()
        while self.l2_times and self.l2_times[0] < lo:
            self.l2_times.popleft()
        while self.mid_window and self.mid_window[0][0] < lo:
            self.mid_window.popleft()

    def _top_from_book(self, ts: float, ts_text: str) -> TopState | None:
        if not self.bids or not self.asks:
            return None
        bb = max(self.bids)
        ba = min(self.asks)
        if bb >= ba:
            return None
        bsz = self.bids.get(bb, 0.0)
        asz = self.asks.get(ba, 0.0)
        if bsz <= 0 or asz <= 0:
            return None
        spread = ba - bb
        mid = (bb + ba) / 2.0
        denom = bsz + asz
        micro = (ba * bsz + bb * asz) / denom
        return TopState(
            ts=ts,
            ts_text=ts_text,
            best_bid=bb,
            best_ask=ba,
            best_bid_sz=bsz,
            best_ask_sz=asz,
            spread=spread,
            mid=mid,
            micro_dev=(micro - mid),
            tob_imb=((bsz - asz) / denom),
        )

    def _regime(self, z_event: float | None, z_spread: float | None, z_mid_vol: float | None, z_trade: float | None) -> str:
        vals = [v for v in (z_event, z_spread, z_mid_vol, z_trade) if v is not None]
        if not vals:
            return "normal"
        if any(v > 2.0 for v in vals):
            return "chaotic"
        if all(v < -0.5 for v in vals):
            return "calm"
        if any(v > 1.0 for v in vals):
            return "volatile"
        return "normal"

    def _regime_params(self, regime: str) -> tuple[float, float | None]:
        if regime == "calm":
            return 1.5, 0.75
        if regime == "normal":
            return 2.0, 1.0
        if regime == "volatile":
            return 2.5, 1.5
        return 2.0, None

    def _metrics(self) -> tuple[float | None, float | None, float]:
        if not self.closed_trades:
            return None, None, 0.0
        wins = sum(1 for t in self.closed_trades if t["r_value"] > 0)
        avg_r = mean(t["r_value"] for t in self.closed_trades)
        peak = self.equity_curve[0] if self.equity_curve else 0.0
        mdd = 0.0
        for x in self.equity_curve:
            peak = max(peak, x)
            mdd = max(mdd, peak - x)
        return wins / len(self.closed_trades), avg_r, mdd

    def _write_status(self, current_regime: str, anom_pct: float | None, current_mid: float | None) -> None:
        win_rate, avg_r, mdd = self._metrics()
        last10 = self.closed_trades[-10:]
        status = {
            "started_at": self.started_at,
            "last_update": _utc_now(),
            "symbol": ("BTCUSDT" if self.data_source == "bybit" else "BTC-USD"),
            "data_source": self.data_source,
            "exchange": self.exchange,
            "execution_mode": self.execution_mode,
            "raw_events_seen": self.raw_events_seen,
            "current_mid_price": current_mid,
            "current_regime": current_regime,
            "current_anomaly_percentile": anom_pct,
            "signals_seen": self.signals_seen,
            "trades_taken": self.trades_taken,
            "trades_blocked": self.trades_blocked_by_reason,
            "daily_trades": self.daily_counts,
            "session_trades": self.session_counts,
            "open_trade": self.open_trade,
            "realized_pnl_r": self.realized_pnl_r,
            "win_rate": win_rate,
            "avg_r": avg_r,
            "max_drawdown_r": mdd,
            "max_losing_streak": self.max_losing_streak,
            "last_10_trades": last10,
            "last_execution_plan": self.last_execution_plan,
        }
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(status, indent=2, allow_nan=False), encoding="utf-8")

    def feed_status(self) -> tuple[str, str | None, int, float | None]:
        files = [p for p in RAW_DATA_DIR.glob(self.raw_glob) if p.is_file()]
        if not files:
            return "NO_FEED", None, 0, None
        latest = max(files, key=lambda p: p.stat().st_mtime)
        age_sec = time.time() - latest.stat().st_mtime
        if age_sec > 120.0:
            return "STALE", latest.name, len(files), age_sec
        return "OK", latest.name, len(files), age_sec

    def describe_idle_reason(self) -> str:
        st, _name, _n, age_sec = self.feed_status()
        if st == "NO_FEED":
            return "no_ndjson_feed_recorder_must_write_coinbase_ndjson_to_data_raw"
        if st == "STALE":
            return f"feed_stale_age_sec={age_sec:.0f}_check_recorder_process"
        if self.open_trade is not None:
            return f"position_open_side={self.open_trade.get('side', '')}"
        if self.signals_seen == 0:
            return "awaiting_bearish_anomaly_pct99_no_signal_yet"
        if self.trades_blocked_by_reason:
            parts = [
                f"{k}:{v}"
                for k, v in sorted(self.trades_blocked_by_reason.items(), key=lambda x: -x[1])[:6]
            ]
            return "signals_seen_blocks=" + ";".join(parts)
        return "strategy_active_no_opportunity_row_this_tick"

    def _close_trade(self, exit_ts: str, exit_price: float, outcome: str, r_value: float) -> None:
        t = self.open_trade
        if t is None:
            return
        rec = {
            "entry_ts": t["entry_ts"],
            "exit_ts": exit_ts,
            "side": t["side"],
            "entry_price": t["entry_price"],
            "exit_price": exit_price,
            "outcome": outcome,
            "r_value": r_value,
            "regime": t["regime"],
        }
        self.closed_trades.append(rec)
        self.realized_pnl_r += r_value
        self.equity_curve.append(self.realized_pnl_r)
        if r_value < 0:
            self._curr_losing_streak += 1
            self.max_losing_streak = max(self.max_losing_streak, self._curr_losing_streak)
        else:
            self._curr_losing_streak = 0
        _append_csv(
            PAPER_TRADES_PATH,
            [
                rec["entry_ts"],
                rec["exit_ts"],
                rec["side"],
                rec["entry_price"],
                rec["exit_price"],
                rec["outcome"],
                rec["r_value"],
                rec["regime"],
            ],
        )
        self.open_trade = None

    def _maybe_manage_open_trade(self, top: TopState) -> None:
        t = self.open_trade
        if t is None:
            return
        entry = float(t["entry_price"])
        sl_price = float(t["sl_price"])
        tp_price = float(t["tp_price"])
        be_trigger = t.get("be_trigger_r")
        if be_trigger is not None and top.mid <= entry - float(be_trigger):
            t["sl_price"] = min(t["sl_price"], entry)  # move to breakeven for short
            sl_price = float(t["sl_price"])

        # Conservative: stop first on same tick.
        if top.mid >= sl_price:
            r = (entry - sl_price)
            self._close_trade(top.ts_text, top.mid, "sl_or_be", r)
            return
        if top.mid <= tp_price:
            r = (entry - tp_price)
            self._close_trade(top.ts_text, top.mid, "tp", r)
            return
        if (top.ts - float(t["entry_ts_unix"])) >= TIMEOUT_SEC:
            r = (entry - top.mid)
            self._close_trade(top.ts_text, top.mid, "timeout", r)

    def _process_trade_msg(self, obj: dict[str, Any]) -> None:
        if self.data_source == "bybit":
            self._process_trade_msg_bybit(obj)
            return
        if obj.get("channel") != "market_trades":
            return
        for ev in obj.get("events", []) if isinstance(obj.get("events"), list) else []:
            for tr in ev.get("trades", []) if isinstance(ev, dict) and isinstance(ev.get("trades"), list) else []:
                ts = tr.get("time")
                if ts is None:
                    continue
                try:
                    t = parse_ts_to_unix(str(ts))
                except ValueError:
                    continue
                self.trade_times.append(t)

    def _process_trade_msg_bybit(self, obj: dict[str, Any]) -> None:
        payload = obj.get("payload")
        if not isinstance(payload, dict):
            return
        if str(payload.get("topic", "")) != "publicTrade.BTCUSDT":
            return
        trades = payload.get("data")
        if not isinstance(trades, list):
            return
        for tr in trades:
            if not isinstance(tr, dict):
                continue
            t_ms = tr.get("T")
            try:
                t = float(int(t_ms)) / 1000.0
            except (TypeError, ValueError):
                continue
            self.trade_times.append(t)

    def _process_l2_msg(self, obj: dict[str, Any]) -> tuple[TopState | None, float]:
        if self.data_source == "bybit":
            return self._process_l2_msg_bybit(obj)
        if obj.get("channel") != "l2_data":
            return None, 0.0
        pressure = 0.0
        for ev in obj.get("events", []) if isinstance(obj.get("events"), list) else []:
            if not isinstance(ev, dict):
                continue
            ev_type = str(ev.get("type", "")).lower()
            if ev_type not in {"snapshot", "update"}:
                continue
            if ev_type == "snapshot":
                self.bids.clear()
                self.asks.clear()
            for up in ev.get("updates", []) if isinstance(ev.get("updates"), list) else []:
                if not isinstance(up, dict):
                    continue
                ts_text = str(up.get("event_time", ""))
                try:
                    ts = parse_ts_to_unix(ts_text)
                except ValueError:
                    continue
                side_raw = str(up.get("side", "")).lower()
                side = "ask" if side_raw == "offer" else side_raw
                px = _safe_float(up.get("price_level"))
                nq = _safe_float(up.get("new_quantity"))
                if side not in {"bid", "ask"} or px is None or nq is None:
                    continue
                self.l2_times.append(ts)
                book = self.bids if side == "bid" else self.asks
                old = book.get(px, 0.0)
                delta = nq - old
                pressure += delta if side == "bid" else -delta
                if nq <= 0:
                    book.pop(px, None)
                else:
                    book[px] = nq
                top = self._top_from_book(ts, ts_text)
                if top is not None:
                    self.last_top = top
        return self.last_top, pressure

    def _process_l2_msg_bybit(self, obj: dict[str, Any]) -> tuple[TopState | None, float]:
        payload = obj.get("payload")
        if not isinstance(payload, dict):
            return None, 0.0
        if str(payload.get("topic", "")) != "orderbook.50.BTCUSDT":
            return None, 0.0
        msg_type = str(payload.get("type", "")).lower()
        if msg_type not in {"snapshot", "delta"}:
            return None, 0.0
        data = payload.get("data")
        if not isinstance(data, dict):
            return None, 0.0
        if str(data.get("s", "")) != "BTCUSDT":
            return None, 0.0
        ts_ms = payload.get("ts")
        try:
            ts = float(int(ts_ms)) / 1000.0
        except (TypeError, ValueError):
            return None, 0.0
        ts_text = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

        if msg_type == "snapshot":
            self.bids.clear()
            self.asks.clear()

        pressure = 0.0
        for lv in data.get("b", []) if isinstance(data.get("b"), list) else []:
            if not isinstance(lv, list) or len(lv) < 2:
                continue
            px = _safe_float(lv[0])
            nq = _safe_float(lv[1])
            if px is None or nq is None:
                continue
            self.l2_times.append(ts)
            old = self.bids.get(px, 0.0)
            delta = nq - old
            pressure += delta
            if nq <= 0:
                self.bids.pop(px, None)
            else:
                self.bids[px] = nq

        for lv in data.get("a", []) if isinstance(data.get("a"), list) else []:
            if not isinstance(lv, list) or len(lv) < 2:
                continue
            px = _safe_float(lv[0])
            nq = _safe_float(lv[1])
            if px is None or nq is None:
                continue
            self.l2_times.append(ts)
            old = self.asks.get(px, 0.0)
            delta = nq - old
            pressure += -delta
            if nq <= 0:
                self.asks.pop(px, None)
            else:
                self.asks[px] = nq

        top = self._top_from_book(ts, ts_text)
        if top is not None:
            self.last_top = top
        return self.last_top, pressure

    def _can_take_trade(self, ts: float, regime: str) -> tuple[bool, str]:
        if regime == "chaotic":
            return False, "chaotic_regime"
        if (ts - self.last_entry_ts) < COOLDOWN_SEC:
            return False, "cooldown"
        if (ts - self.last_signal_ts) < CLUSTER_WINDOW_SEC:
            return False, "cluster_window"
        d = _day_key(ts)
        s = f"{d}|{_session_label(ts)}"
        if self.daily_counts.get(d, 0) >= MAX_TRADES_PER_DAY:
            return False, "daily_limit"
        if self.session_counts.get(s, 0) >= MAX_TRADES_PER_SESSION:
            return False, "session_limit"
        return True, ""

    def _take_trade(self, top: TopState, regime: str, tp_r: float, be_trigger: float | None) -> None:
        entry = top.mid
        self.open_trade = {
            "entry_ts": top.ts_text,
            "entry_ts_unix": top.ts,
            "side": "short",
            "entry_price": entry,
            "tp_price": entry - tp_r,
            "sl_price": entry + 1.0,
            "regime": regime,
            "be_trigger_r": be_trigger,
        }
        self.trades_taken += 1
        self.last_entry_ts = top.ts
        d = _day_key(top.ts)
        s = f"{d}|{_session_label(top.ts)}"
        self.daily_counts[d] = self.daily_counts.get(d, 0) + 1
        self.session_counts[s] = self.session_counts.get(s, 0) + 1

    def _plan_bybit_dry_run_entry(self, top: TopState, tp_r: float) -> tuple[bool, str]:
        if self.execution_mode != "bybit_demo_dry_run":
            self.last_execution_plan = None
            return True, ""
        if self.data_source != "bybit":
            self.last_execution_plan = None
            return True, ""
        if self._bybit_execution is None:
            self.last_execution_plan = None
            return True, ""

        stop_price = top.mid + 1.0
        tp_price = top.mid - tp_r
        balance_override = str(os.environ.get("BYBIT_BALANCE_OVERRIDE", "")).strip()
        try:
            account_balance = float(balance_override) if balance_override else self._bybit_execution.client.get_account_balance()
        except ValueError:
            account_balance = 0.0

        try:
            res = self._bybit_execution.build_entry_short(
                account_balance=account_balance,
                entry_price=top.mid,
                stop_price=stop_price,
            )
        except RuntimeError as exc:
            res = {"ok": False, "reject_reason": str(exc), "sizing": None}
        ok = bool(res.get("ok"))
        reason = "" if ok else str(res.get("reject_reason", "execution_dry_run_rejected"))

        sizing = res.get("sizing")
        qty = float(getattr(sizing, "qty", 0.0)) if sizing is not None else 0.0
        risk_usd = float(getattr(sizing, "risk_usd", 0.0)) if sizing is not None else 0.0
        notional = float(getattr(sizing, "notional", 0.0)) if sizing is not None else 0.0
        leverage = float(getattr(sizing, "leverage", 1.0)) if sizing is not None else 1.0
        self.last_execution_plan = {
            "ok": ok,
            "reason": reason,
            "symbol": self._bybit_execution.config.bybit_symbol,
            "entry_price": top.mid,
            "stop_price": stop_price,
            "tp_price": tp_price,
            "qty": qty,
            "risk_usd": risk_usd,
            "notional": notional,
            "leverage": leverage,
            "payloads": {
                "set_leverage": res.get("set_leverage"),
                "entry_order": res.get("entry_order"),
            },
        }
        _append_csv(
            self.execution_dry_run_path,
            [
                top.ts_text,
                self.data_source,
                self.execution_mode,
                self._bybit_execution.config.bybit_symbol,
                top.mid,
                stop_price,
                tp_price,
                qty,
                risk_usd,
                notional,
                leverage,
                ok,
                reason,
            ],
        )
        return ok, reason

    def loop_once(self) -> None:
        objs = self.reader.read_new_objects()
        if not objs:
            self._write_status("normal", None, self.last_mid)
            return

        current_regime = "normal"
        current_pct: float | None = None
        for obj in objs:
            self.raw_events_seen += 1
            self._process_trade_msg(obj)
            top, pressure = self._process_l2_msg(obj)
            if top is None:
                continue

            self._expire_windows(top.ts)
            self.mid_window.append((top.ts, top.mid))
            self.last_mid = top.mid
            self._maybe_manage_open_trade(top)

            event_rate = len(self.l2_times) / max(LOOKBACK_SEC, EPS)
            trade_count = float(len(self.trade_times))
            mid_vals = [m for _, m in self.mid_window]
            mid_vol = 0.0
            if len(mid_vals) >= 2:
                mu = sum(mid_vals) / len(mid_vals)
                var = sum((x - mu) ** 2 for x in mid_vals) / (len(mid_vals) - 1)
                mid_vol = math.sqrt(max(var, 0.0))

            z_tob = self.z_stats["tob"].zscore_before(top.ts, top.tob_imb)
            z_micro = self.z_stats["micro"].zscore_before(top.ts, top.micro_dev)
            z_pressure = self.z_stats["pressure"].zscore_before(top.ts, pressure)
            z_event = self.z_stats["event_rate"].zscore_before(top.ts, event_rate)
            z_trade = self.z_stats["trade_count"].zscore_before(top.ts, trade_count)
            z_spread = self.z_stats["spread"].zscore_before(top.ts, top.spread)
            z_mid_vol = self.z_stats["mid_vol"].zscore_before(top.ts, mid_vol)

            for k, v in (
                ("tob", top.tob_imb),
                ("micro", top.micro_dev),
                ("pressure", pressure),
                ("event_rate", event_rate),
                ("trade_count", trade_count),
                ("spread", top.spread),
                ("mid_vol", mid_vol),
            ):
                self.z_stats[k].add(top.ts, v)

            abs_parts = [abs(z) for z in (z_tob, z_micro, z_pressure, z_event) if z is not None]
            if not abs_parts:
                continue
            anomaly_score = max(abs_parts)
            bullish = max([z for z in (z_tob, z_micro, z_pressure) if z is not None] + [0.0])
            bearish = max([(-z) for z in (z_tob, z_micro, z_pressure) if z is not None] + [0.0])
            direction = "bearish" if bearish > bullish else ("bullish" if bullish > bearish else "neutral")

            pct = self.rank.rank_before(top.ts, anomaly_score)
            self.rank.add(top.ts, anomaly_score)
            current_pct = pct
            current_regime = self._regime(z_event, z_spread, z_mid_vol, z_trade)

            if pct is None:
                continue
            if direction == "bearish" and pct >= 0.99:
                self.signals_seen += 1
                can, reason = self._can_take_trade(top.ts, current_regime)
                action = "blocked"
                if can and self.open_trade is None:
                    tp_r, be_trigger = self._regime_params(current_regime)
                    ok_exec, exec_reason = self._plan_bybit_dry_run_entry(top, tp_r)
                    if ok_exec:
                        self._take_trade(top, current_regime, tp_r, be_trigger)
                        self.last_signal_ts = top.ts
                        action = "entered_short"
                        reason = ""
                    else:
                        action = "blocked"
                        reason = exec_reason
                if action == "blocked" and reason:
                    self.trades_blocked += 1
                    self.trades_blocked_by_reason[reason] = self.trades_blocked_by_reason.get(reason, 0) + 1
                _append_opportunity_row(
                    ts=top.ts_text,
                    anomaly_percentile=pct,
                    anomaly_score=anomaly_score,
                    direction=direction,
                    regime=current_regime,
                    action=action,
                    reason=reason,
                )
                self.opportunities_appended += 1

        self._write_status(current_regime, current_pct, self.last_mid)


def _csv_body_row_count(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            n = sum(1 for _ in f)
    except OSError:
        return 0
    return max(0, n - 1)


def _log_periodic_diagnostic(logger: logging.Logger, engine: LivePaperEngine) -> None:
    state, latest_name, nfiles, age_sec = engine.feed_status()
    if state == "NO_FEED":
        logger.warning(
            "NO LIVE DATA FEED DETECTED — no %s files under %s (start the NDJSON recorder writing here).",
            engine.raw_glob,
            RAW_DATA_DIR,
        )
    elif state == "STALE":
        logger.warning(
            "FEED STALE — latest_ndjson=%s age_sec=%.0f (recorder may be stopped)",
            latest_name,
            age_sec or -1.0,
        )
    op_file = _csv_body_row_count(OPPORTUNITIES_PATH)
    tr_file = _csv_body_row_count(PAPER_TRADES_PATH)
    logger.info(
        "DIAGNOSTIC ts=%s transport=file_ndjson feed=%s ndjson_files=%s latest_file=%s latest_age_sec=%s "
        "raw_events=%s signals=%s opps_session_appends=%s opps_csv_rows=%s trades_closed_session=%s trades_csv_rows=%s "
        "last_mid=%s idle_reason=%s",
        _utc_now(),
        state,
        nfiles,
        latest_name or "-",
        f"{age_sec:.1f}" if age_sec is not None else "-",
        engine.raw_events_seen,
        engine.signals_seen,
        engine.opportunities_appended,
        op_file,
        len(engine.closed_trades),
        tr_file,
        engine.last_mid if engine.last_mid is not None else "-",
        engine.describe_idle_reason(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live paper trading loop on Coinbase BTCUSD raw NDJSON")
    parser.add_argument("--sleep-sec", type=float, default=1.0)
    parser.add_argument(
        "--heartbeat-sec",
        type=float,
        default=float(os.environ.get("TJTB_HEARTBEAT_SEC", "30")),
        help="Write ISO UTC timestamp to logs/heartbeat.txt at this interval.",
    )
    parser.add_argument(
        "--diag-interval-sec",
        type=float,
        default=float(os.environ.get("TJTB_DIAG_INTERVAL_SEC", "30")),
        help="Periodic DIAGNOSTIC log interval (feed, counts, idle reason).",
    )
    parser.add_argument(
        "--data-source",
        choices=("coinbase", "bybit"),
        default=str(os.environ.get("DATA_SOURCE", "coinbase")).strip().lower(),
        help="Raw feed source: coinbase (default) or bybit",
    )
    args = parser.parse_args(argv)

    logger = _setup_logger()
    lock_fh = _try_acquire_singleton_lock(logger)
    if lock_fh is None:
        return 0

    stop = threading.Event()
    hb_interval = max(args.heartbeat_sec, 1.0)
    HEARTBEAT_PATH.write_text(_utc_now() + "\n", encoding="utf-8")
    _start_heartbeat(hb_interval, stop)

    shutdown = threading.Event()

    def _request_shutdown(signum: int, frame: object | None) -> None:
        shutdown.set()

    signal.signal(signal.SIGTERM, _request_shutdown)
    signal.signal(signal.SIGINT, _request_shutdown)

    engine = LivePaperEngine(logger, data_source=args.data_source)
    logger.info("live paper loop started (pid=%s data_source=%s)", os.getpid(), engine.data_source)
    diag_interval = max(args.diag_interval_sec, 5.0)
    last_diag = time.monotonic()
    _log_periodic_diagnostic(logger, engine)
    try:
        while not shutdown.is_set():
            engine.loop_once()
            now = time.monotonic()
            if now - last_diag >= diag_interval:
                last_diag = now
                _log_periodic_diagnostic(logger, engine)
            if shutdown.wait(timeout=max(args.sleep_sec, 0.1)):
                break
    except KeyboardInterrupt:
        shutdown.set()
    finally:
        stop.set()
        try:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        lock_fh.close()

    logger.info("live paper loop stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
