"""
Build microstructure feature matrix aligned to book_state timestamps.

Reads trades and L2 book_updates once (numpy arrays) for efficient window queries;
streams book_state rows. Market-agnostic price units; tick_size optional for
downstream conversion to ticks (CME).
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

import numpy as np

logger = logging.getLogger(__name__)

BOOK_PATH_DEFAULT = Path("data/parsed/book_state.csv")
TRADES_PATH_DEFAULT = Path("data/parsed/trades.csv")
BOOK_UPDATES_DEFAULT = Path("data/parsed/book_updates.csv")
OUT_DEFAULT = Path("data/parsed/feature_matrix.csv")
READ_BUFFER = 1024 * 1024
WRITE_BUFFER = 1024 * 1024
SIZE_EPS = 1e-12

# ISO-8601-like timestamps with variable fractional length (stdlib fromisoformat is strict).
_ISO_TS_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})(?:\.(\d+))?(Z|[+-]\d{2}:\d{2}(?::\d{2})?)?$"
)

FEATURE_HEADER = (
    "ts",
    "tob_imbalance",
    "microprice_dev",
    "spread",
    "ofi_k",
    "trade_count_k",
    "mid_vol_w",
    "return_lag",
    "del_bid_w",
    "del_ask_w",
    "l2_update_count_w",
    "abs_size_change_w",
    "signed_book_pressure_w",
    "l2_event_rate_hz",
    "avg_inter_event_sec_w",
    "trade_book_ratio_w",
)

NUMERIC_FEATURE_NAMES = FEATURE_HEADER[1:]


def parse_ts_to_unix(s: str) -> float:
    """
    Parse ISO-like timestamps to UTC unix seconds (deterministic).

    Accepts naive timestamps (interpreted as UTC), trailing ``Z``, and offsets
    like ``+00:00``. Fractional seconds may be any length; normalized to
    microseconds (pad/truncate to 6 digits) before ``datetime.fromisoformat``.
    """
    t = s.strip()
    if not t:
        raise ValueError("empty timestamp")
    m = _ISO_TS_RE.match(t)
    if not m:
        raise ValueError(f"unrecognized timestamp format: {s!r}")
    date_part, time_part, frac, tz = m.group(1), m.group(2), m.group(3), m.group(4)
    body = f"{date_part}T{time_part}"
    if frac is not None:
        micro = (frac + "000000")[:6]
        body += f".{micro}"
    if tz == "Z":
        body += "+00:00"
    elif tz:
        body += tz
    else:
        body += "+00:00"
    dt = datetime.fromisoformat(body)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return float(dt.timestamp())


def load_trades_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (trade_ts_unix, signed_size) sorted by trade_ts_unix."""
    ts_list: list[float] = []
    sgn_list: list[float] = []
    with path.open("r", encoding="utf-8", newline="", buffering=READ_BUFFER) as f:
        r = csv.reader(f)
        header = next(r, None)
        if header is None:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        for row in r:
            if len(row) < 4:
                continue
            try:
                ts = parse_ts_to_unix(row[0])
                size = float(row[2])
                side = str(row[3]).strip().lower()
                if side == "buy":
                    sgn = size
                elif side == "sell":
                    sgn = -size
                else:
                    continue
                ts_list.append(ts)
                sgn_list.append(sgn)
            except (ValueError, OSError):
                continue
    if not ts_list:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    order = np.argsort(np.asarray(ts_list, dtype=np.float64), kind="mergesort")
    t_arr = np.asarray(ts_list, dtype=np.float64)[order]
    v_arr = np.asarray(sgn_list, dtype=np.float64)[order]
    return t_arr, v_arr


def load_book_updates_arrays(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return sorted arrays: ts_unix, is_bid (uint8), price, new_qty, is_snapshot (uint8), sequence_num (int64).

    Supports legacy ``ts, side, price, size`` and current
    ``ts, event_type, sequence_num, side, price, size`` from parse_coinbase.

    Missing file -> empty arrays.
    """
    empty = (
        np.array([], dtype=np.float64),
        np.array([], dtype=np.uint8),
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
        np.array([], dtype=np.uint8),
        np.array([], dtype=np.int64),
    )
    if not path.is_file():
        return empty

    ts_list: list[float] = []
    bid_list: list[int] = []
    px_list: list[float] = []
    sz_list: list[float] = []
    snap_list: list[int] = []
    seq_list: list[int] = []

    with path.open("r", encoding="utf-8", newline="", buffering=READ_BUFFER) as f:
        r = csv.reader(f)
        header = next(r, None)
        if header is None:
            return empty
        hdr = [h.strip().lower() for h in header]
        v2 = "event_type" in hdr and "sequence_num" in hdr
        if v2:
            try:
                i_ts = hdr.index("ts")
                i_et = hdr.index("event_type")
                i_sq = hdr.index("sequence_num")
                i_sd = hdr.index("side")
                i_px = hdr.index("price")
                i_sz = hdr.index("size")
            except ValueError:
                v2 = False

        for row in r:
            if not row or (row[0].strip().lower() == "ts" and len(row) == len(hdr)):
                continue
            try:
                if v2:
                    if len(row) <= max(i_ts, i_et, i_sq, i_sd, i_px, i_sz):
                        continue
                    ts = parse_ts_to_unix(row[i_ts])
                    ev = str(row[i_et]).strip().lower()
                    seq = int(row[i_sq])
                    side = str(row[i_sd]).strip().lower()
                    is_snap = 1 if ev == "snapshot" else 0
                    if ev not in ("snapshot", "update"):
                        continue
                else:
                    if len(row) < 4:
                        continue
                    ts = parse_ts_to_unix(row[0])
                    side = str(row[1]).strip().lower()
                    seq = 0
                    is_snap = 0
                    i_px, i_sz = 2, 3

                if side == "bid":
                    is_bid = 1
                elif side in ("ask", "offer"):
                    is_bid = 0
                else:
                    continue
                px = float(row[i_px] if v2 else row[2])
                qty = float(row[i_sz] if v2 else row[3])
                if qty < -SIZE_EPS:
                    continue
                ts_list.append(ts)
                bid_list.append(is_bid)
                px_list.append(px)
                sz_list.append(qty)
                snap_list.append(is_snap)
                seq_list.append(seq)
            except (ValueError, OSError, IndexError):
                continue

    if not ts_list:
        return empty

    order = np.argsort(np.asarray(ts_list, dtype=np.float64), kind="mergesort")
    return (
        np.asarray(ts_list, dtype=np.float64)[order],
        np.asarray(bid_list, dtype=np.uint8)[order],
        np.asarray(px_list, dtype=np.float64)[order],
        np.asarray(sz_list, dtype=np.float64)[order],
        np.asarray(snap_list, dtype=np.uint8)[order],
        np.asarray(seq_list, dtype=np.int64)[order],
    )


@dataclass
class _WindowTotals:
    n: int = 0
    del_bid: int = 0
    del_ask: int = 0
    abs_sum: float = 0.0
    press: float = 0.0


@dataclass
class L2WindowAccumulator:
    """
    Apply L2 updates in timestamp order up to each book snapshot ``bt`` and
    maintain rolling window statistics over ``(bt - window_sec, bt]``.
    """

    window_sec: float
    ts: np.ndarray
    is_bid: np.ndarray
    price: np.ndarray
    new_qty: np.ndarray
    is_snapshot: np.ndarray
    sequence_num: np.ndarray
    bid_lv: dict[float, float] = field(default_factory=dict)
    ask_lv: dict[float, float] = field(default_factory=dict)
    idx: int = 0
    last_snapshot_seq: int | None = None
    evq: deque[dict[str, float | int]] = field(default_factory=deque)
    tot: _WindowTotals = field(default_factory=_WindowTotals)

    def _maybe_reset_for_snapshot(self, i: int) -> None:
        if int(self.is_snapshot[i]) != 1:
            return
        seq = int(self.sequence_num[i])
        if self.last_snapshot_seq is None or seq != self.last_snapshot_seq:
            self.bid_lv.clear()
            self.ask_lv.clear()
            self.last_snapshot_seq = seq

    def _ingest_one(self, i: int) -> dict[str, float | int]:
        self._maybe_reset_for_snapshot(i)
        t = float(self.ts[i])
        side_bid = bool(int(self.is_bid[i]))
        px = float(self.price[i])
        nq = float(self.new_qty[i])
        book = self.bid_lv if side_bid else self.ask_lv
        old = float(book.get(px, 0.0))
        delta = nq - old
        abs_d = abs(delta)
        del_bid = 1 if (side_bid and nq <= SIZE_EPS) else 0
        del_ask = 1 if ((not side_bid) and nq <= SIZE_EPS) else 0
        if nq <= SIZE_EPS:
            book.pop(px, None)
        else:
            book[px] = nq
        press = delta if side_bid else (-delta)
        return {"ts": t, "del_bid": del_bid, "del_ask": del_ask, "abs_d": abs_d, "press": press}

    def _push(self, ev: dict[str, float | int]) -> None:
        self.evq.append(ev)
        self.tot.n += 1
        self.tot.del_bid += int(ev["del_bid"])
        self.tot.del_ask += int(ev["del_ask"])
        self.tot.abs_sum += float(ev["abs_d"])
        self.tot.press += float(ev["press"])

    def _pop(self, ev: dict[str, float | int]) -> None:
        self.tot.n -= 1
        self.tot.del_bid -= int(ev["del_bid"])
        self.tot.del_ask -= int(ev["del_ask"])
        self.tot.abs_sum -= float(ev["abs_d"])
        self.tot.press -= float(ev["press"])

    def stats_at(self, bt: float) -> dict[str, float]:
        nmax = int(self.ts.shape[0])
        while self.idx < nmax and float(self.ts[self.idx]) <= bt:
            ev = self._ingest_one(self.idx)
            self.idx += 1
            self._push(ev)
        w = float(self.window_sec)
        lo = bt - w
        while self.evq and float(self.evq[0]["ts"]) < lo:
            old = self.evq.popleft()
            self._pop(old)
        n = self.tot.n
        if n >= 2:
            t0 = float(self.evq[0]["ts"])
            t1 = float(self.evq[-1]["ts"])
            span = max(t1 - t0, 0.0)
            avg_inter = span / float(n - 1)
        elif n == 1:
            lone = float(self.evq[0]["ts"])
            span_in_win = max(min(bt - lone, w), 0.0)
            avg_inter = span_in_win if span_in_win > 0 else w
        else:
            avg_inter = 0.0
        if n > 0 and self.evq:
            t_first = float(self.evq[0]["ts"])
            t_last = float(self.evq[-1]["ts"])
            span = max(t_last - t_first, 0.0)
            denom = span if span > SIZE_EPS else max(w, SIZE_EPS)
            rate = float(n) / denom
        else:
            rate = 0.0
        return {
            "del_bid_w": float(self.tot.del_bid),
            "del_ask_w": float(self.tot.del_ask),
            "l2_update_count_w": float(n),
            "abs_size_change_w": float(self.tot.abs_sum),
            "signed_book_pressure_w": float(self.tot.press),
            "l2_event_rate_hz": rate,
            "avg_inter_event_sec_w": float(avg_inter),
        }


def trade_window_stats(
    trade_ts: np.ndarray,
    signed_vol: np.ndarray,
    book_ts: float,
    k_trades: int,
) -> tuple[float, int]:
    """OFI sum and count over last ``k_trades`` trades with time <= ``book_ts``."""
    if trade_ts.size == 0 or k_trades <= 0:
        return 0.0, 0
    idx = int(np.searchsorted(trade_ts, book_ts, side="right"))
    if idx <= 0:
        return 0.0, 0
    start = max(0, idx - k_trades)
    sl = signed_vol[start:idx]
    return float(np.sum(sl)), int(sl.size)


def trade_count_time_window(trade_ts: np.ndarray, t_lo: float, t_hi: float) -> int:
    """Count trades with ``t_lo`` <= ts <= ``t_hi`` (inclusive, aligned with L2 window)."""
    if trade_ts.size == 0:
        return 0
    i0 = int(np.searchsorted(trade_ts, t_lo, side="left"))
    i1 = int(np.searchsorted(trade_ts, t_hi, side="right"))
    return max(0, i1 - i0)


def rolling_std(deq: deque[float]) -> float:
    if len(deq) < 2:
        return 0.0
    a = np.fromiter(deq, dtype=np.float64, count=len(deq))
    return float(np.std(a, ddof=1)) if a.size > 1 else 0.0


@dataclass
class FeatureStreamStats:
    """Online sums for mean/std and nonzero counts (one pass, no pandas)."""

    n_rows: int = 0
    sum_v: np.ndarray = field(default_factory=lambda: np.zeros(len(NUMERIC_FEATURE_NAMES), dtype=np.float64))
    sumsq_v: np.ndarray = field(default_factory=lambda: np.zeros(len(NUMERIC_FEATURE_NAMES), dtype=np.float64))
    nonzero: np.ndarray = field(default_factory=lambda: np.zeros(len(NUMERIC_FEATURE_NAMES), dtype=np.int64))

    def observe_row(self, values: list[float]) -> None:
        self.n_rows += 1
        arr = np.asarray(values, dtype=np.float64)
        self.sum_v += arr
        self.sumsq_v += arr * arr
        self.nonzero += (np.abs(arr) > SIZE_EPS).astype(np.int64)

    def log_summary(self) -> None:
        n = self.n_rows
        if n <= 0:
            logger.info("feature stats: no rows written")
            return
        mean = self.sum_v / float(n)
        var_pop = np.maximum(self.sumsq_v / float(n) - mean * mean, 0.0)
        std_pop = np.sqrt(var_pop)
        for j, name in enumerate(NUMERIC_FEATURE_NAMES):
            logger.info(
                "feature_stats name=%s mean=%s std=%s nonzero_rows=%s/%s",
                name,
                float(mean[j]),
                float(std_pop[j]),
                int(self.nonzero[j]),
                n,
            )


def stream_features(
    book_in: TextIO,
    out: TextIO,
    trade_ts: np.ndarray,
    signed_vol: np.ndarray,
    l2_acc: L2WindowAccumulator | None,
    fstats: FeatureStreamStats,
    *,
    k_trades: int,
    vol_window: int,
    lag: int,
    l2_window_sec: float,
) -> int:
    w = csv.writer(out)
    w.writerow(FEATURE_HEADER)
    n = 0
    vol_buf: deque[float] = deque(maxlen=max(vol_window, 2))
    lag_buf: deque[float] = deque(maxlen=max(lag + 1, 1) if lag > 0 else 1)
    r = csv.reader(book_in)
    _ = next(r, None)
    for row in r:
        if len(row) < 9:
            continue
        if row[0].strip().lower() == "ts":
            continue
        try:
            ts_s = str(row[0]).strip()
            _ = float(row[1]), float(row[2])
            best_bid_sz = float(row[3])
            best_ask_sz = float(row[4])
            spread = float(row[5])
            mid = float(row[6])
            micro = float(row[7])
            tob_col = float(row[8])
        except (ValueError, IndexError):
            continue
        denom = best_bid_sz + best_ask_sz
        if denom > 0:
            tob_imb = (best_bid_sz - best_ask_sz) / denom
        else:
            tob_imb = tob_col
        micro_dev = micro - mid
        bt = parse_ts_to_unix(ts_s)
        ofi, tcnt = trade_window_stats(trade_ts, signed_vol, bt, k_trades)
        vol_buf.append(mid)
        vol_proxy = rolling_std(vol_buf) if len(vol_buf) >= 2 else 0.0
        lag_buf.append(mid)
        if lag > 0 and len(lag_buf) == lag + 1:
            ret_lag = mid - lag_buf[0]
        else:
            ret_lag = 0.0

        t_lo = bt - float(l2_window_sec)
        t_cnt = trade_count_time_window(trade_ts, t_lo, bt)
        if l2_acc is not None:
            st = l2_acc.stats_at(bt)
            l2n = int(st["l2_update_count_w"])
            ratio = float(t_cnt) / float(l2n) if l2n > 0 else 0.0
        else:
            st = {
                "del_bid_w": 0.0,
                "del_ask_w": 0.0,
                "l2_update_count_w": 0.0,
                "abs_size_change_w": 0.0,
                "signed_book_pressure_w": 0.0,
                "l2_event_rate_hz": 0.0,
                "avg_inter_event_sec_w": 0.0,
            }
            ratio = 0.0

        numeric_vals = [
            tob_imb,
            micro_dev,
            spread,
            ofi,
            float(tcnt),
            vol_proxy,
            ret_lag,
            st["del_bid_w"],
            st["del_ask_w"],
            st["l2_update_count_w"],
            st["abs_size_change_w"],
            st["signed_book_pressure_w"],
            st["l2_event_rate_hz"],
            st["avg_inter_event_sec_w"],
            ratio,
        ]
        fstats.observe_row(numeric_vals)

        w.writerow(
            [
                ts_s,
                *numeric_vals,
            ]
        )
        n += 1
    return n


def build_feature_matrix(
    book_path: Path,
    trades_path: Path,
    out_path: Path,
    *,
    book_updates_path: Path | None,
    k_trades: int,
    vol_window: int,
    lag: int,
    l2_window_sec: float,
) -> int:
    t_arr, v_arr = load_trades_arrays(trades_path)
    fstats = FeatureStreamStats()
    if book_updates_path is not None:
        bu_ts, bu_bid, bu_px, bu_sz, bu_snap, bu_seq = load_book_updates_arrays(book_updates_path)
    else:
        bu_ts = np.array([], dtype=np.float64)
        bu_bid = np.array([], dtype=np.uint8)
        bu_px = np.array([], dtype=np.float64)
        bu_sz = np.array([], dtype=np.float64)
        bu_snap = np.array([], dtype=np.uint8)
        bu_seq = np.array([], dtype=np.int64)

    l2_acc: L2WindowAccumulator | None
    if bu_ts.size > 0:
        l2_acc = L2WindowAccumulator(
            l2_window_sec,
            bu_ts,
            bu_bid,
            bu_px,
            bu_sz,
            bu_snap,
            bu_seq,
        )
        snap_n = int(np.sum(bu_snap))
        logger.info(
            "loaded book_updates rows=%s snapshot_rows=%s update_rows=%s window_sec=%s",
            bu_ts.size,
            snap_n,
            int(bu_ts.size) - snap_n,
            l2_window_sec,
        )
    else:
        l2_acc = None
        logger.warning("no book_updates loaded; L2 window features zeroed")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with book_path.open("r", encoding="utf-8", newline="", buffering=READ_BUFFER) as binp, out_path.open(
        "w", newline="", encoding="utf-8", buffering=WRITE_BUFFER
    ) as o:
        n = stream_features(
            binp,
            o,
            t_arr,
            v_arr,
            l2_acc,
            fstats,
            k_trades=k_trades,
            vol_window=vol_window,
            lag=lag,
            l2_window_sec=l2_window_sec,
        )
    fstats.log_summary()
    logger.info("feature rows=%s trades_loaded=%s -> %s", n, t_arr.size, out_path)
    return n


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    p = argparse.ArgumentParser(description="Build microstructure feature matrix (book + trades + L2).")
    p.add_argument("--book", type=Path, default=BOOK_PATH_DEFAULT)
    p.add_argument("--trades", type=Path, default=TRADES_PATH_DEFAULT)
    p.add_argument("--book-updates", type=Path, default=BOOK_UPDATES_DEFAULT)
    p.add_argument("--output", type=Path, default=OUT_DEFAULT)
    p.add_argument("--k-trades", type=int, default=200)
    p.add_argument("--vol-window", type=int, default=50)
    p.add_argument("--lag", type=int, default=5)
    p.add_argument("--l2-window-sec", type=float, default=5.0)
    args = p.parse_args(argv)
    if not args.book.is_file():
        logger.error("Missing book_state: %s", args.book)
        return 1
    if not args.trades.is_file():
        logger.error("Missing trades: %s", args.trades)
        return 1
    bu_path = args.book_updates if args.book_updates.is_file() else None
    build_feature_matrix(
        args.book,
        args.trades,
        args.output,
        book_updates_path=bu_path,
        k_trades=args.k_trades,
        vol_window=args.vol_window,
        lag=args.lag,
        l2_window_sec=args.l2_window_sec,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
