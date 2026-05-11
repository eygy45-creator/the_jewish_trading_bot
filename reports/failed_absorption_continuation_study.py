#!/usr/bin/env python3
"""
Failed Absorption → Immediate Continuation Study (research).

Replays the same Bybit NDJSON source and ETH geometry signal path as
`tjtb.research.eth_geometry_runner` (EthGeometryEngine, $2 stop focus).

Outputs JSON to reports/failed_absorption_continuation_study.json by default.
"""

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Callable, Optional

# Allow `python reports/failed_absorption_continuation_study.py` without PYTHONPATH tweaks.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tjtb.live.live_paper_crypto import BYBIT_RAW_GLOB, LivePaperEngine, TopState
from tjtb.research.eth_geometry_runner import ROUND_TRIP_TAKER_FEE, EthGeometryEngine, _per_trade_net_r
from tjtb.research.stop_grid_runner import _iter_objects
from tjtb.runtime_paths import RAW_DATA_DIR, REPORTS_DIR

LOGGER = logging.getLogger("tjtb.research.failed_absorption_continuation")


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


STOP_FOCUS = 2.0
PATH_HORIZON_SEC = 900.0
EPS = 1e-9


def _sf(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _sb(x: Any, default: bool = False) -> bool:
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    s = str(x).strip().lower()
    if s in ("1", "true", "yes", "y"):
        return True
    if s in ("0", "false", "no", "n", ""):
        return False
    return default


def _session_bucket_for_summary(entry_session: str) -> str:
    s = str(entry_session or "").strip().lower()
    if s in ("asia", "london", "london_ny_overlap"):
        return s
    return "off_hours"


def _reference_level_short(trade: dict[str, Any]) -> tuple[float, bool]:
    """Absorption ceiling proxy from stall geometry; else entry fallback."""
    entry = _sf(trade.get("entry_price"), float("nan"))
    mid = _sf(trade.get("entry_mid_price"), float("nan"))
    rr = _sf(trade.get("entry_mid_range_ratio"), float("nan"))
    if math.isfinite(mid) and math.isfinite(rr) and mid > 0 and rr > 0:
        ceiling = mid * (1.0 + 0.5 * rr)
        if math.isfinite(ceiling):
            return float(ceiling), False
    if math.isfinite(entry):
        return float(entry), True
    return float("nan"), True


def _favorable_r_short(entry: float, mid: float, stop: float) -> float:
    return (entry - mid) / stop if stop > 0 else float("nan")


def _adverse_r_short(entry: float, mid: float, stop: float) -> float:
    return (mid - entry) / stop if stop > 0 else float("nan")


def _interp_mid(t: float, t1: float, m1: float, t2: float, m2: float) -> float:
    if t2 <= t1 + EPS:
        return m2
    w = (t - t1) / (t2 - t1)
    return m1 + w * (m2 - m1)


def _gross_dollars_short(entry: float, exit_px: float) -> float:
    return entry - exit_px


def _net_r(entry: float, gross_usd: float, stop: float) -> float:
    if stop <= 0 or not math.isfinite(entry):
        return float("nan")
    net_usd = _per_trade_net_r(entry, gross_usd, ROUND_TRIP_TAKER_FEE)
    return net_usd / stop


def _cross_time_first_down(t1: float, m1: float, t2: float, m2: float, level: float) -> Optional[float]:
    """First time in (t1,t2] where mid reaches `level` moving from m1→m2 (short TP / favorable)."""
    if t2 < t1 - EPS:
        return None
    if m1 <= level + EPS:
        return t1
    if m2 <= level + EPS and abs(m2 - m1) > EPS:
        s = (level - m1) / (m2 - m1)
        if s < -EPS or s > 1.0 + EPS:
            return None
        s = min(1.0, max(0.0, s))
        return t1 + s * (t2 - t1)
    return None


def _cross_time_first_up(t1: float, m1: float, t2: float, m2: float, level: float) -> Optional[float]:
    """First time in (t1,t2] where mid reaches `level` (stop / reclaim for shorts)."""
    if t2 < t1 - EPS:
        return None
    if m1 >= level - EPS:
        return t1
    if m2 >= level - EPS and abs(m2 - m1) > EPS:
        s = (level - m1) / (m2 - m1)
        if s < -EPS or s > 1.0 + EPS:
            return None
        s = min(1.0, max(0.0, s))
        return t1 + s * (t2 - t1)
    return None


def _min_time(cur: Optional[float], cand: Optional[float]) -> Optional[float]:
    if cand is None:
        return cur
    if cur is None:
        return cand
    return cand if cand < cur - EPS else cur


@dataclass
class PathSimResult:
    continuation_5s_r: float = float("nan")
    continuation_10s_r: float = float("nan")
    continuation_20s_r: float = float("nan")
    continuation_30s_r: float = float("nan")
    continuation_60s_r: float = float("nan")
    reclaim_10s: bool = False
    reclaim_20s: bool = False
    reclaim_30s: bool = False
    time_to_reclaim_sec: Optional[float] = None
    no_reclaim_20s: bool = True
    no_reclaim_30s: bool = True
    hit_0_5r: bool = False
    hit_1r: bool = False
    hit_1_5r: bool = False
    hit_2r: bool = False
    max_favorable_excursion_r: float = 0.0
    max_adverse_excursion_r: float = 0.0
    time_to_0_5r_sec: Optional[float] = None
    time_to_1r_sec: Optional[float] = None
    time_to_1_5r_sec: Optional[float] = None
    time_to_2r_sec: Optional[float] = None
    max_pullback_before_1r_r: float = 0.0
    returned_to_entry_before_1r: bool = False
    fast_escape_10s: bool = False
    fast_escape_20s: bool = False
    net_r_oco_1r: float = float("nan")
    net_r_oco_1_5r: float = float("nan")
    net_r_oco_2r: float = float("nan")


class MidReplay(LivePaperEngine):
    """Book/trade replay only: rebuild mids without opening geometry trades."""

    def __init__(self, logger: logging.Logger, data_source: str) -> None:
        super().__init__(logger, data_source=data_source)
        self.execution_mode = "paper"
        self._bybit_execution = None

    def process_object(self, obj: dict[str, Any]) -> Optional[TopState]:
        self.raw_events_seen += 1
        self._process_trade_msg(obj)
        top, _pressure = self._process_l2_msg(obj)
        if top is None:
            return None
        self._expire_windows(top.ts)
        self.mid_window.append((top.ts, top.mid))
        self.last_mid = top.mid
        return top


def _mid_at_time(mids: list[tuple[float, float]], t_query: float) -> Optional[float]:
    """Interpolate mid at t_query; None if query precedes first sample."""
    if not mids or t_query < mids[0][0] - EPS:
        return None
    prev_t, prev_m = mids[0]
    if t_query <= prev_t + EPS:
        return prev_m
    for t, m in mids[1:]:
        if t_query <= t + EPS:
            return _interp_mid(t_query, prev_t, prev_m, t, m)
        prev_t, prev_m = t, m
    return prev_m


def _trim_mids_from_entry(
    mids: list[tuple[float, float]],
    entry_ts: float,
    entry_price: float,
) -> list[tuple[float, float]]:
    """Series starting at entry_ts with interpolated mid (stable path sim)."""
    if not mids:
        return [(entry_ts, entry_price), (entry_ts + 1e-3, entry_price)]
    m_at = _mid_at_time(mids, entry_ts)
    if m_at is None or not math.isfinite(m_at):
        m_at = entry_price
    trimmed: list[tuple[float, float]] = [(float(entry_ts), float(m_at))]
    for t, m in mids:
        if t > entry_ts + EPS:
            trimmed.append((float(t), float(m)))
    if len(trimmed) < 2:
        trimmed.append((float(entry_ts) + 1e-3, float(m_at)))
    return trimmed


def _simulate_short_path(
    mids: list[tuple[float, float]],
    *,
    entry_ts: float,
    entry_price: float,
    stop: float,
    ref_level: float,
    horizon: float,
) -> PathSimResult:
    out = PathSimResult()
    t_hor = entry_ts + horizon
    if stop <= 0 or not math.isfinite(entry_price) or not math.isfinite(ref_level) or not mids:
        return out

    path = _trim_mids_from_entry(mids, entry_ts, entry_price)

    sl_px = entry_price + stop
    tp05 = entry_price - 0.5 * stop
    tp1 = entry_price - 1.0 * stop
    tp15 = entry_price - 1.5 * stop
    tp2 = entry_price - 2.0 * stop

    t_stop: Optional[float] = None
    t_reclaim: Optional[float] = None
    t_tp05: Optional[float] = None
    t_tp1: Optional[float] = None
    t_tp15: Optional[float] = None
    t_tp2: Optional[float] = None

    cont_targets = (5.0, 10.0, 20.0, 30.0, 60.0)
    cont_done = {k: False for k in cont_targets}
    cont_mid: dict[float, float] = {}

    prev_t, prev_m = path[0]
    for t2, m2 in path[1:]:
        t1, m1 = prev_t, prev_m
        seg_lo = max(t1, entry_ts)
        seg_hi = min(t2, t_hor)
        if seg_hi < seg_lo - EPS:
            prev_t, prev_m = t2, m2
            continue
        for dt in cont_targets:
            if cont_done[dt]:
                continue
            tq = entry_ts + dt
            if tq < seg_lo - EPS:
                continue
            if tq > seg_hi + EPS:
                break
            cont_mid[dt] = _interp_mid(tq, t1, m1, t2, m2)
            cont_done[dt] = True

        t_stop = _min_time(t_stop, _cross_time_first_up(t1, m1, t2, m2, sl_px))
        t_reclaim = _min_time(t_reclaim, _cross_time_first_up(t1, m1, t2, m2, ref_level))
        t_tp05 = _min_time(t_tp05, _cross_time_first_down(t1, m1, t2, m2, tp05))
        t_tp1 = _min_time(t_tp1, _cross_time_first_down(t1, m1, t2, m2, tp1))
        t_tp15 = _min_time(t_tp15, _cross_time_first_down(t1, m1, t2, m2, tp15))
        t_tp2 = _min_time(t_tp2, _cross_time_first_down(t1, m1, t2, m2, tp2))

        prev_t, prev_m = t2, m2

    for dt in cont_targets:
        if cont_done.get(dt):
            mm = cont_mid[dt]
            r = _favorable_r_short(entry_price, mm, stop)
            if dt == 5.0:
                out.continuation_5s_r = r
            elif dt == 10.0:
                out.continuation_10s_r = r
            elif dt == 20.0:
                out.continuation_20s_r = r
            elif dt == 30.0:
                out.continuation_30s_r = r
            elif dt == 60.0:
                out.continuation_60s_r = r

    if t_reclaim is not None:
        out.time_to_reclaim_sec = max(0.0, t_reclaim - entry_ts)
        out.reclaim_10s = t_reclaim <= entry_ts + 10.0 + EPS
        out.reclaim_20s = t_reclaim <= entry_ts + 20.0 + EPS
        out.reclaim_30s = t_reclaim <= entry_ts + 30.0 + EPS
        out.no_reclaim_20s = t_reclaim > entry_ts + 20.0 + EPS
        out.no_reclaim_30s = t_reclaim > entry_ts + 30.0 + EPS
    else:
        out.no_reclaim_20s = True
        out.no_reclaim_30s = True

    def _before_bad(tp_t: Optional[float]) -> bool:
        if tp_t is None:
            return False
        bads = [x for x in (t_stop, t_reclaim, t_hor) if x is not None]
        if not bads:
            return True
        return tp_t + EPS < min(bads)

    out.hit_0_5r = _before_bad(t_tp05)
    out.hit_1r = _before_bad(t_tp1)
    out.hit_1_5r = _before_bad(t_tp15)
    out.hit_2r = _before_bad(t_tp2)

    def _rel(tp_t: Optional[float]) -> Optional[float]:
        if tp_t is None:
            return None
        return max(0.0, tp_t - entry_ts)

    out.time_to_0_5r_sec = _rel(t_tp05) if out.hit_0_5r else None
    out.time_to_1r_sec = _rel(t_tp1) if out.hit_1r else None
    out.time_to_1_5r_sec = _rel(t_tp15) if out.hit_1_5r else None
    out.time_to_2r_sec = _rel(t_tp2) if out.hit_2r else None

    t_explore = t_hor
    for x in (t_stop, t_reclaim):
        if x is not None and x + EPS < t_explore:
            t_explore = x

    mfe = 0.0
    mae = 0.0
    prev_t, prev_m = path[0]
    for t2, m2 in path[1:]:
        t1, m1 = prev_t, prev_m
        seg_lo = max(t1, entry_ts)
        seg_hi = min(t2, t_explore)
        if seg_hi < seg_lo - EPS:
            prev_t, prev_m = t2, m2
            continue
        a = _interp_mid(seg_lo, t1, m1, t2, m2)
        b = _interp_mid(seg_hi, t1, m1, t2, m2)
        mfe = max(mfe, _favorable_r_short(entry_price, a, stop), _favorable_r_short(entry_price, b, stop))
        mae = min(mae, -_adverse_r_short(entry_price, a, stop), -_adverse_r_short(entry_price, b, stop))
        prev_t, prev_m = t2, m2
    out.max_favorable_excursion_r = float(mfe)
    out.max_adverse_excursion_r = float(mae)

    t_win_end = t_hor
    if t_tp1 is not None and t_tp1 + EPS < t_win_end:
        t_win_end = t_tp1
    for x in (t_stop, t_reclaim):
        if x is not None and x + EPS < t_win_end:
            t_win_end = x

    max_pull = 0.0
    ret_entry = False
    prev_t, prev_m = path[0]
    for t2, m2 in path[1:]:
        t1, m1 = prev_t, prev_m
        seg_lo = max(t1, entry_ts)
        seg_hi = min(t2, t_win_end)
        if seg_hi < seg_lo - EPS:
            prev_t, prev_m = t2, m2
            continue
        a = _interp_mid(seg_lo, t1, m1, t2, m2)
        b = _interp_mid(seg_hi, t1, m1, t2, m2)
        max_pull = max(max_pull, _adverse_r_short(entry_price, a, stop), _adverse_r_short(entry_price, b, stop))
        if a >= entry_price - EPS or b >= entry_price - EPS:
            ret_entry = True
        elif a < entry_price - EPS and b > entry_price + EPS:
            tc = _cross_time_first_up(seg_lo, a, seg_hi, b, entry_price)
            if tc is not None:
                ret_entry = True
        prev_t, prev_m = t2, m2
    out.max_pullback_before_1r_r = float(max_pull)
    out.returned_to_entry_before_1r = bool(ret_entry)

    out.fast_escape_10s = (
        math.isfinite(out.continuation_10s_r) and out.continuation_10s_r >= 0.25 and out.no_reclaim_20s
    )
    out.fast_escape_20s = (
        math.isfinite(out.continuation_20s_r) and out.continuation_20s_r >= 0.50 and out.no_reclaim_30s
    )

    out.net_r_oco_1r = _oco_net_r_first_exit(entry_price, stop, ref_level, mids, entry_ts, t_hor, tp_mult=1.0)
    out.net_r_oco_1_5r = _oco_net_r_first_exit(entry_price, stop, ref_level, mids, entry_ts, t_hor, tp_mult=1.5)
    out.net_r_oco_2r = _oco_net_r_first_exit(entry_price, stop, ref_level, mids, entry_ts, t_hor, tp_mult=2.0)

    return out


def _oco_net_r_first_exit(
    entry: float,
    stop: float,
    ref: float,
    mids: list[tuple[float, float]],
    entry_ts: float,
    t_hor: float,
    *,
    tp_mult: float,
) -> float:
    """Single TP at tp_mult*R vs 1R stop, reclaim scratch, or timeout at horizon mid."""
    sl_px = entry + stop
    tp_px = entry - tp_mult * stop
    path = _trim_mids_from_entry(mids, entry_ts, entry)
    best: Optional[tuple[float, int, str]] = None  # (time, tie_priority, kind) lower priority wins ties
    prev_t, prev_m = path[0]
    for t2, m2 in path[1:]:
        t1, m1 = prev_t, prev_m
        seg_lo = max(t1, entry_ts)
        seg_hi = min(t2, t_hor)
        if seg_hi < seg_lo - EPS:
            prev_t, prev_m = t2, m2
            continue
        ta = _cross_time_first_up(t1, m1, t2, m2, sl_px)
        if ta is not None and seg_lo - EPS <= ta <= seg_hi + EPS:
            tt = max(ta, seg_lo)
            cand = (tt, 0, "stop")
            if best is None or cand[0] < best[0] - EPS or (abs(cand[0] - best[0]) <= EPS and cand[1] < best[1]):
                best = cand
        tb = _cross_time_first_up(t1, m1, t2, m2, ref)
        if tb is not None and seg_lo - EPS <= tb <= seg_hi + EPS:
            tt = max(tb, seg_lo)
            cand = (tt, 1, "reclaim")
            if best is None or cand[0] < best[0] - EPS or (abs(cand[0] - best[0]) <= EPS and cand[1] < best[1]):
                best = cand
        tc = _cross_time_first_down(t1, m1, t2, m2, tp_px)
        if tc is not None and seg_lo - EPS <= tc <= seg_hi + EPS:
            tt = max(tc, seg_lo)
            cand = (tt, 2, "tp")
            if best is None or cand[0] < best[0] - EPS or (abs(cand[0] - best[0]) <= EPS and cand[1] < best[1]):
                best = cand
        prev_t, prev_m = t2, m2
    if best is not None:
        kind = best[2]
        if kind == "stop":
            return _net_r(entry, _gross_dollars_short(entry, sl_px), stop)
        if kind == "reclaim":
            return _net_r(entry, _gross_dollars_short(entry, ref), stop)
        return _net_r(entry, _gross_dollars_short(entry, tp_px), stop)
    mx = _mid_at_time(mids, t_hor)
    if mx is None:
        mx = entry
    return _net_r(entry, _gross_dollars_short(entry, mx), stop)


def mean_safe(xs: list[float]) -> float:
    ys = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return sum(ys) / len(ys) if ys else float("nan")


def median_safe(xs: list[Optional[float]]) -> Optional[float]:
    ys = sorted(float(x) for x in xs if x is not None and math.isfinite(float(x)))
    if not ys:
        return None
    return float(median(ys))


def _summarize(rows: list[dict[str, Any]], *, fee_adjusted: bool) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {
            "signal_count": 0,
            "winrate_0_5r": float("nan"),
            "winrate_1r": float("nan"),
            "winrate_1_5r": float("nan"),
            "winrate_2r": float("nan"),
            "avg_mfe_r": float("nan"),
            "avg_mae_r": float("nan"),
            "expectancy_at_1r_target": float("nan"),
            "expectancy_at_1_5r_target": float("nan"),
            "expectancy_at_2r_target": float("nan"),
            "median_time_to_1r_sec": None,
            "median_time_to_2r_sec": None,
            "fee_adjusted": fee_adjusted,
        }

    def wr(hit_key: str) -> float:
        return sum(1 for x in rows if _sb(x.get(hit_key))) / n

    return {
        "signal_count": n,
        "winrate_0_5r": wr("hit_0_5r"),
        "winrate_1r": wr("hit_1r"),
        "winrate_1_5r": wr("hit_1_5r"),
        "winrate_2r": wr("hit_2r"),
        "avg_mfe_r": mean_safe([_sf(x.get("max_favorable_excursion_r")) for x in rows]),
        "avg_mae_r": mean_safe([_sf(x.get("max_adverse_excursion_r")) for x in rows]),
        "expectancy_at_1r_target": mean_safe([_sf(x.get("net_r_oco_1r")) for x in rows]),
        "expectancy_at_1_5r_target": mean_safe([_sf(x.get("net_r_oco_1_5r")) for x in rows]),
        "expectancy_at_2r_target": mean_safe([_sf(x.get("net_r_oco_2r")) for x in rows]),
        "median_time_to_1r_sec": median_safe([x.get("time_to_1r_sec") for x in rows]),
        "median_time_to_2r_sec": median_safe([x.get("time_to_2r_sec") for x in rows]),
        "fee_adjusted": fee_adjusted,
    }


def _filter_named() -> dict[str, Callable[[dict[str, Any]], bool]]:
    return {
        "continuation_5s_r_ge_0_10": lambda r: math.isfinite(_sf(r.get("continuation_5s_r"))) and _sf(r.get("continuation_5s_r")) >= 0.10,
        "continuation_10s_r_ge_0_25": lambda r: math.isfinite(_sf(r.get("continuation_10s_r"))) and _sf(r.get("continuation_10s_r")) >= 0.25,
        "continuation_20s_r_ge_0_50": lambda r: math.isfinite(_sf(r.get("continuation_20s_r"))) and _sf(r.get("continuation_20s_r")) >= 0.50,
        "no_reclaim_20s": lambda r: _sb(r.get("no_reclaim_20s")),
        "no_reclaim_30s": lambda r: _sb(r.get("no_reclaim_30s")),
        "fast_escape_10s": lambda r: _sb(r.get("fast_escape_10s")),
        "fast_escape_20s": lambda r: _sb(r.get("fast_escape_20s")),
        "cont10_ge025_and_no_reclaim_20s": lambda r: (
            math.isfinite(_sf(r.get("continuation_10s_r"))) and _sf(r.get("continuation_10s_r")) >= 0.25 and _sb(r.get("no_reclaim_20s"))
        ),
        "cont20_ge050_and_no_reclaim_30s": lambda r: (
            math.isfinite(_sf(r.get("continuation_20s_r"))) and _sf(r.get("continuation_20s_r")) >= 0.50 and _sb(r.get("no_reclaim_30s"))
        ),
        "strict_fast_escape_combo": lambda r: _sb(r.get("fast_escape_10s")) and _sb(r.get("fast_escape_20s")),
    }


def run_study(
    *,
    data_source: str = "bybit",
    raw_dir: Path = RAW_DATA_DIR,
    symbol: str = "ETHUSDT",
    json_output: Path = REPORTS_DIR / "failed_absorption_continuation_study.json",
) -> dict[str, Any]:
    glob_pat = BYBIT_RAW_GLOB if data_source == "bybit" else "*.ndjson"
    objs = list(_iter_objects(raw_dir, glob_pat))
    prev_sym = os.environ.get("BYBIT_SYMBOL")
    os.environ["BYBIT_SYMBOL"] = str(symbol).strip().upper()
    try:
        eng = EthGeometryEngine(LOGGER, data_source=data_source, stop_size=STOP_FOCUS, timeout_sec=PATH_HORIZON_SEC)
        for obj in objs:
            eng.process_object(obj)
        eng.finalize_excursions()
        candidates_src = [dict(t) for t in eng.closed_trades]
    finally:
        if prev_sym is None:
            os.environ.pop("BYBIT_SYMBOL", None)
        else:
            os.environ["BYBIT_SYMBOL"] = prev_sym

    replay = MidReplay(LOGGER, data_source=data_source)
    mids: list[tuple[float, float]] = []
    for obj in objs:
        top = replay.process_object(obj)
        if top is not None:
            mids.append((float(top.ts), float(top.mid)))

    rows: list[dict[str, Any]] = []
    for t in candidates_src:
        entry_ts_u = _sf(t.get("entry_ts_unix"))
        ref, fb = _reference_level_short(t)
        if not math.isfinite(ref):
            ref = _sf(t.get("entry_price"))
            fb = True
        row: dict[str, Any] = {
            "timestamp": t.get("entry_ts"),
            "timestamp_unix": entry_ts_u,
            "session": t.get("entry_session"),
            "direction": str(t.get("side", "short")).lower(),
            "symbol": str(symbol).strip().upper(),
            "source_file": None,
            "source_day": None,
            "is_repeated_signal_30s": _sb(t.get("is_repeated_signal_30s")),
            "is_repeated_signal_60s": _sb(t.get("is_repeated_signal_60s")),
            "failed_absorption_medium": _sb(t.get("failed_absorption_medium")),
            "failed_absorption_strict": _sb(t.get("failed_absorption_strict")),
            "anomaly_percentile": t.get("entry_anomaly_percentile", t.get("anomaly_percentile")),
            "regime": t.get("regime"),
            "failed_absorption_reference_price": ref,
            "fallback_level_used": fb,
            "entry_price": t.get("entry_price"),
            "entry_signal_key": t.get("entry_signal_key"),
            "entry_mid_price": t.get("entry_mid_price"),
            "entry_mid_range_ratio": t.get("entry_mid_range_ratio"),
        }
        try:
            et = str(t.get("entry_ts") or "")
            if et:
                row["source_day"] = et.split("T")[0].split(" ")[0]
        except Exception:
            row["source_day"] = None

        try:
            sim = _simulate_short_path(
                mids,
                entry_ts=entry_ts_u,
                entry_price=_sf(t.get("entry_price")),
                stop=STOP_FOCUS,
                ref_level=ref,
                horizon=PATH_HORIZON_SEC,
            )
        except Exception as exc:
            LOGGER.warning("path_sim_failed key=%s err=%s", t.get("entry_signal_key"), exc)
            sim = PathSimResult()

        row.update(
            {
                "continuation_5s_r": sim.continuation_5s_r,
                "continuation_10s_r": sim.continuation_10s_r,
                "continuation_20s_r": sim.continuation_20s_r,
                "continuation_30s_r": sim.continuation_30s_r,
                "continuation_60s_r": sim.continuation_60s_r,
                "reclaim_10s": sim.reclaim_10s,
                "reclaim_20s": sim.reclaim_20s,
                "reclaim_30s": sim.reclaim_30s,
                "time_to_reclaim_sec": sim.time_to_reclaim_sec,
                "no_reclaim_20s": sim.no_reclaim_20s,
                "no_reclaim_30s": sim.no_reclaim_30s,
                "hit_0_5r": sim.hit_0_5r,
                "hit_1r": sim.hit_1r,
                "hit_1_5r": sim.hit_1_5r,
                "hit_2r": sim.hit_2r,
                "max_favorable_excursion_r": sim.max_favorable_excursion_r,
                "max_adverse_excursion_r": sim.max_adverse_excursion_r,
                "time_to_0_5r_sec": sim.time_to_0_5r_sec,
                "time_to_1r_sec": sim.time_to_1r_sec,
                "time_to_1_5r_sec": sim.time_to_1_5r_sec,
                "time_to_2r_sec": sim.time_to_2r_sec,
                "max_pullback_before_1r_r": sim.max_pullback_before_1r_r,
                "returned_to_entry_before_1r": sim.returned_to_entry_before_1r,
                "fast_escape_10s": sim.fast_escape_10s,
                "fast_escape_20s": sim.fast_escape_20s,
                "net_r_oco_1r": sim.net_r_oco_1r,
                "net_r_oco_1_5r": sim.net_r_oco_1_5r,
                "net_r_oco_2r": sim.net_r_oco_2r,
            }
        )
        rows.append(row)

    fee_adj = True
    all_sum = _summarize(rows, fee_adjusted=fee_adj)

    def core_fa(r: dict[str, Any]) -> bool:
        if str(r.get("direction", "")).lower() != "short":
            return False
        rep = _sb(r.get("is_repeated_signal_30s")) or _sb(r.get("is_repeated_signal_60s"))
        fa = _sb(r.get("failed_absorption_medium")) or _sb(r.get("failed_absorption_strict"))
        return rep and fa

    strict_only = [r for r in rows if core_fa(r) and _sb(r.get("failed_absorption_strict"))]
    medium_not_strict = [
        r for r in rows if core_fa(r) and _sb(r.get("failed_absorption_medium")) and not _sb(r.get("failed_absorption_strict"))
    ]
    core_rows = [r for r in rows if core_fa(r)]

    by_session: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        b = _session_bucket_for_summary(str(r.get("session") or ""))
        by_session.setdefault(b, []).append(r)

    continuation_filters: dict[str, Any] = {"all_named_filters": {}, "combinations": {}}
    named = _filter_named()
    for name, pred in named.items():
        continuation_filters["all_named_filters"][name] = _summarize([x for x in rows if pred(x)], fee_adjusted=fee_adj)

    continuation_filters["combinations"]["cont5_ge010_and_no_reclaim_20"] = _summarize(
        [x for x in rows if named["continuation_5s_r_ge_0_10"](x) and named["no_reclaim_20s"](x)],
        fee_adjusted=fee_adj,
    )

    candidates_for_best: list[tuple[str, dict[str, Any]]] = [("all_candidates", all_sum)]
    candidates_for_best.append(("core_failed_absorption", _summarize(core_rows, fee_adjusted=fee_adj)))
    candidates_for_best.append(("strict_only", _summarize(strict_only, fee_adjusted=fee_adj)))
    candidates_for_best.append(("medium_not_strict", _summarize(medium_not_strict, fee_adjusted=fee_adj)))
    for name, pred in named.items():
        candidates_for_best.append((name, _summarize([x for x in rows if pred(x)], fee_adjusted=fee_adj)))

    def _finite_or_neg_inf(x: Any) -> float:
        try:
            v = float(x)
            return v if math.isfinite(v) else float("-inf")
        except (TypeError, ValueError):
            return float("-inf")

    def score_key(item: tuple[str, dict[str, Any]]) -> tuple[float, float, int]:
        _, s = item
        ev = _finite_or_neg_inf(s.get("expectancy_at_1_5r_target"))
        wr = _finite_or_neg_inf(s.get("winrate_1r"))
        n = int(s.get("signal_count") or 0)
        return (ev, wr, n)

    best_name, best_stats = max(candidates_for_best, key=score_key)

    def _diversity_ok(subset: list[dict[str, Any]]) -> bool:
        days = {str(x.get("source_day") or "") for x in subset}
        days.discard("")
        sess = {str(x.get("session") or "") for x in subset}
        sess.discard("")
        return len(days) >= 2 or len(sess) >= 2

    best_subset_rows = rows
    if best_name == "core_failed_absorption":
        best_subset_rows = core_rows
    elif best_name == "strict_only":
        best_subset_rows = strict_only
    elif best_name == "medium_not_strict":
        best_subset_rows = medium_not_strict
    elif best_name != "all_candidates":
        pred = named.get(best_name)
        if pred:
            best_subset_rows = [x for x in rows if pred(x)]

    n_best = int(best_stats.get("signal_count") or 0)
    wr1 = _sf(best_stats.get("winrate_1r"))
    ev15 = _sf(best_stats.get("expectancy_at_1_5r_target"))
    diverse = _diversity_ok(best_subset_rows)
    sample_warn = n_best < 20
    production_candidate = (
        n_best >= 10
        and (wr1 >= 0.60 or ev15 > 0.0)
        and diverse
        and not sample_warn
    )

    if production_candidate:
        verdict = "CONTINUATION_EDGE_FOUND"
    elif n_best >= 10 and (ev15 > -0.05 or wr1 >= 0.52):
        verdict = "WEAK_EDGE_NEEDS_MORE_DATA"
    else:
        verdict = "NO_CONTINUATION_EDGE_REDESIGN_REQUIRED"

    doc: dict[str, Any] = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "study": "failed_absorption_continuation",
        "parameters": {
            "stop_size_usd": STOP_FOCUS,
            "path_horizon_sec": PATH_HORIZON_SEC,
            "round_trip_taker_fee_rate": ROUND_TRIP_TAKER_FEE,
            "fee_adjusted": fee_adj,
            "symbol": str(symbol).strip().upper(),
            "data_source": data_source,
            "raw_dir": str(raw_dir),
            "reference_level_note": "Proxy ceiling = entry_mid_price*(1+0.5*entry_mid_range_ratio); else entry_price.",
        },
        "signals": rows,
        "aggregates": {
            "all_candidates": all_sum,
            "core_failed_absorption": _summarize(core_rows, fee_adjusted=fee_adj),
            "strict_only": _summarize(strict_only, fee_adjusted=fee_adj),
            "medium_not_strict": _summarize(medium_not_strict, fee_adjusted=fee_adj),
            "by_session": {k: _summarize(v, fee_adjusted=fee_adj) for k, v in sorted(by_session.items())},
            "continuation_filters": continuation_filters,
        },
        "decision": {
            "best_filter_name": best_name,
            "best_filter_signal_count": n_best,
            "best_filter_winrate_1r": wr1,
            "best_filter_winrate_1_5r": _sf(best_stats.get("winrate_1_5r")),
            "best_filter_winrate_2r": _sf(best_stats.get("winrate_2r")),
            "best_filter_expectancy_1_5r": ev15,
            "production_candidate": production_candidate,
            "verdict": verdict,
            "diversity_sessions_or_days_ok": diverse,
            "sample_size_warning": sample_warn,
        },
    }

    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(_json_safe(doc), indent=2, allow_nan=False), encoding="utf-8")
    return doc


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Failed absorption → continuation study")
    p.add_argument("--data-source", choices=("bybit", "coinbase"), default="bybit")
    p.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--json-output", type=Path, default=REPORTS_DIR / "failed_absorption_continuation_study.json")
    args = p.parse_args(argv)
    try:
        doc = run_study(
            data_source=str(args.data_source),
            raw_dir=args.raw_dir,
            symbol=str(args.symbol).strip().upper(),
            json_output=args.json_output,
        )
    except Exception as e:
        LOGGER.exception("study_failed: %s", e)
        return 1
    dec = doc.get("decision") or {}
    signals = doc.get("signals") or []
    print("--- Failed Absorption → Continuation Study ---")
    print(f"total_candidates: {len(signals)}")
    print(f"best_filter: {dec.get('best_filter_name')}")
    print(f"best_filter_signal_count: {dec.get('best_filter_signal_count')}")
    def _fmt4(v: Any) -> str:
        try:
            x = float(v)
            return f"{x:.4f}" if math.isfinite(x) else "nan"
        except (TypeError, ValueError):
            return "nan"

    print(
        "winrate_1r / 1.5r / 2r: "
        f"{_fmt4(dec.get('best_filter_winrate_1r'))} / "
        f"{_fmt4(dec.get('best_filter_winrate_1_5r'))} / "
        f"{_fmt4(dec.get('best_filter_winrate_2r'))}"
    )
    print(f"expectancy_1_5r_net: {_fmt4(dec.get('best_filter_expectancy_1_5r'))}")
    print(f"verdict: {dec.get('verdict')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
