"""
True failed-absorption sequence study (research only).

Auction-structure path: aggression → absorption → failed retest → continuation,
for both long and short, with structure-based invalidation stops (not fixed $2).

Outputs: reports/true_failed_absorption_study.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Optional

from tjtb.live.live_paper_crypto import BYBIT_RAW_GLOB, LOOKBACK_SEC, LivePaperEngine, TopState
from tjtb.research.eth_geometry_runner import ROUND_TRIP_TAKER_FEE, _entry_session_label, _per_trade_net_r
from tjtb.research.stop_grid_runner import _iter_objects, _profit_factor
from tjtb.runtime_paths import RAW_DATA_DIR, REPORTS_DIR

LOGGER = logging.getLogger("tjtb.research.true_failed_absorption")

EPS = 1e-9
PATH_HORIZON_SEC = 900.0
COOLDOWN_SEC = 55.0
FIXED_REF_STOP_USD = 2.0
LIMIT_FILL_WINDOW_SEC = 120.0
MAKER_ENTRY_REBATE = -0.0001
FEE_MAKER_ENTRY_TAKER_EXIT = max(0.0, MAKER_ENTRY_REBATE + ROUND_TRIP_TAKER_FEE / 2.0)

MIN_SIGNALS_FOR_VERDICT = 25


def _sf(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(x) for x in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _interp_mid(t: float, t1: float, m1: float, t2: float, m2: float) -> float:
    if t2 <= t1 + EPS:
        return m2
    w = (t - t1) / (t2 - t1)
    return m1 + w * (m2 - m1)


def _mid_at_time(mids: list[tuple[float, float]], t_query: float) -> Optional[float]:
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


def _cross_time_first_up(t1: float, m1: float, t2: float, m2: float, level: float) -> Optional[float]:
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


def _cross_time_first_down(t1: float, m1: float, t2: float, m2: float, level: float) -> Optional[float]:
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


def _min_time(cur: Optional[float], cand: Optional[float]) -> Optional[float]:
    if cand is None:
        return cur
    if cur is None:
        return cand
    return cand if cand < cur - EPS else cur


def _gross_r_short(entry: float, exit_px: float) -> float:
    return entry - exit_px


def _gross_r_long(entry: float, exit_px: float) -> float:
    return exit_px - entry


def _oco_first_exit(
    *,
    side: str,
    mids: list[tuple[float, float]],
    entry_ts: float,
    entry: float,
    stop_dist: float,
    inv_level: float,
    horizon: float,
    tp_mult: float,
    fee_rate: float,
) -> dict[str, Any]:
    """First exit among structural stop, invalidation scratch, TP at tp_mult*R, or horizon mid."""
    side = str(side).lower()
    t_hor = entry_ts + horizon
    if stop_dist <= 0 or not math.isfinite(entry):
        return {"net_r": float("nan"), "exit_kind": "invalid", "exit_ts": None}

    if side == "short":
        sl_px = entry + stop_dist
        tp_px = entry - tp_mult * stop_dist
        gross_tp = _gross_r_short(entry, tp_px)
        gross_sl = _gross_r_short(entry, sl_px)
        gross_inv = _gross_r_short(entry, inv_level)
        gross_horiz = _gross_r_short(entry, _mid_at_time(mids, t_hor) or entry)

        def favorable(seg_m: float) -> float:
            return _gross_r_short(entry, seg_m) / stop_dist

        def hit_stop_up(t1: float, m1: float, t2: float, m2: float) -> Optional[float]:
            return _cross_time_first_up(t1, m1, t2, m2, sl_px)

        def hit_inv(t1: float, m1: float, t2: float, m2: float) -> Optional[float]:
            return _cross_time_first_up(t1, m1, t2, m2, inv_level)

        def hit_tp(t1: float, m1: float, t2: float, m2: float) -> Optional[float]:
            return _cross_time_first_down(t1, m1, t2, m2, tp_px)
    else:
        sl_px = entry - stop_dist
        tp_px = entry + tp_mult * stop_dist
        gross_tp = _gross_r_long(entry, tp_px)
        gross_sl = _gross_r_long(entry, sl_px)
        gross_inv = _gross_r_long(entry, inv_level)
        gross_horiz = _gross_r_long(entry, _mid_at_time(mids, t_hor) or entry)

        def favorable(seg_m: float) -> float:
            return _gross_r_long(entry, seg_m) / stop_dist

        def hit_stop_up(t1: float, m1: float, t2: float, m2: float) -> Optional[float]:
            return _cross_time_first_down(t1, m1, t2, m2, sl_px)

        def hit_inv(t1: float, m1: float, t2: float, m2: float) -> Optional[float]:
            return _cross_time_first_down(t1, m1, t2, m2, inv_level)

        def hit_tp(t1: float, m1: float, t2: float, m2: float) -> Optional[float]:
            return _cross_time_first_up(t1, m1, t2, m2, tp_px)

    path = _trim_mids_from_entry(mids, entry_ts, entry)
    best: Optional[tuple[float, int, str]] = None
    prev_t, prev_m = path[0]
    mfe = 0.0
    mae = 0.0
    for t2, m2 in path[1:]:
        t1, m1 = prev_t, prev_m
        seg_lo = max(t1, entry_ts)
        seg_hi = min(t2, t_hor)
        if seg_hi >= seg_lo - EPS:
            a = _interp_mid(seg_lo, t1, m1, t2, m2)
            b = _interp_mid(seg_hi, t1, m1, t2, m2)
            mfe = max(mfe, favorable(a), favorable(b))
            mae = min(mae, -max(0.0, -favorable(a)), -max(0.0, -favorable(b)))
        for fn, kind_pri in ((hit_stop_up, 0), (hit_inv, 1), (hit_tp, 2)):
            ta = fn(t1, m1, t2, m2)
            if ta is not None and seg_lo - EPS <= ta <= seg_hi + EPS:
                tt = max(ta, seg_lo)
                cand = (tt, kind_pri, ("stop", "invalidation", "tp")[kind_pri])
                if best is None or cand[0] < best[0] - EPS or (
                    abs(cand[0] - best[0]) <= EPS and cand[1] < best[1]
                ):
                    best = cand
        prev_t, prev_m = t2, m2

    if best is None:
        net = _per_trade_net_r(entry, gross_horiz, fee_rate) / stop_dist
        return {
            "net_r": float(net),
            "exit_kind": "timeout",
            "exit_ts": t_hor,
            "mfe_r": float(mfe),
            "mae_r": float(mae),
        }
    kind = best[2]
    if kind == "stop":
        net = _per_trade_net_r(entry, gross_sl, fee_rate) / stop_dist
    elif kind == "invalidation":
        net = _per_trade_net_r(entry, gross_inv, fee_rate) / stop_dist
    else:
        net = _per_trade_net_r(entry, gross_tp, fee_rate) / stop_dist
    return {
        "net_r": float(net),
        "exit_kind": kind,
        "exit_ts": float(best[0]),
        "mfe_r": float(mfe),
        "mae_r": float(mae),
    }


def _synthetic_limit_fill_short(
    t: dict[str, Any],
    *,
    limit_above_entry_usd: float,
    max_px_field: str = "max_price_reached",
) -> tuple[bool, float, float]:
    entry = _sf(t.get("entry_price"))
    stop_d = _sf(t.get("structural_stop_distance"))
    seconds_to_mae = _sf(t.get("seconds_to_mae"), 9e9)
    max_px = _sf(t.get(max_px_field), entry)
    if seconds_to_mae > LIMIT_FILL_WINDOW_SEC or stop_d <= 0:
        return False, 0.0, 0.0
    level = entry + limit_above_entry_usd
    if max_px >= level - EPS:
        imp = limit_above_entry_usd
        return True, imp / stop_d, min(seconds_to_mae, LIMIT_FILL_WINDOW_SEC)
    return False, 0.0, 0.0


def _synthetic_limit_fill_long(
    t: dict[str, Any],
    *,
    limit_below_entry_usd: float,
    min_px_field: str = "min_price_reached",
) -> tuple[bool, float, float]:
    entry = _sf(t.get("entry_price"))
    stop_d = _sf(t.get("structural_stop_distance"))
    seconds_to_mae = _sf(t.get("seconds_to_mae"), 9e9)
    min_px = _sf(t.get(min_px_field), entry)
    if seconds_to_mae > LIMIT_FILL_WINDOW_SEC or stop_d <= 0:
        return False, 0.0, 0.0
    level = entry - limit_below_entry_usd
    if min_px <= level + EPS:
        imp = -limit_below_entry_usd
        return True, imp / stop_d, min(seconds_to_mae, LIMIT_FILL_WINDOW_SEC)
    return False, 0.0, 0.0


def _excursion_from_path(
    *,
    side: str,
    mids: list[tuple[float, float]],
    entry_ts: float,
    entry: float,
    stop_dist: float,
    horizon: float,
) -> dict[str, Any]:
    side = str(side).lower()
    t_hor = entry_ts + horizon
    if stop_dist <= 0:
        return {}
    path = _trim_mids_from_entry(mids, entry_ts, entry)
    max_p = entry
    min_p = entry
    prev_t, prev_m = path[0]
    for t2, m2 in path[1:]:
        t1, m1 = prev_t, prev_m
        seg_lo = max(t1, entry_ts)
        seg_hi = min(t2, t_hor)
        if seg_hi < seg_lo - EPS:
            prev_t, prev_m = t2, m2
            continue
        for tq in (seg_lo, seg_hi):
            mm = _interp_mid(tq, t1, m1, t2, m2)
            max_p = max(max_p, mm)
            min_p = min(min_p, mm)
        prev_t, prev_m = t2, m2

    if side == "short":
        mfe_r = (entry - min_p) / stop_dist
        mae_r = -(max_p - entry) / stop_dist
        seconds_to_mae = float("inf")
        prev_t, prev_m = path[0]
        for t2, m2 in path[1:]:
            t1, m1 = prev_t, prev_m
            seg_lo = max(t1, entry_ts)
            seg_hi = min(t2, t_hor)
            if seg_hi < seg_lo - EPS:
                prev_t, prev_m = t2, m2
                continue
            for tq in (seg_lo, seg_hi):
                mm = _interp_mid(tq, t1, m1, t2, m2)
                adv = (mm - entry) / stop_dist
                if adv > 0.01 and math.isfinite(seconds_to_mae) and seconds_to_mae > 1e8:
                    seconds_to_mae = max(0.0, tq - entry_ts)
            prev_t, prev_m = t2, m2
    else:
        mfe_r = (max_p - entry) / stop_dist
        mae_r = -(entry - min_p) / stop_dist
        seconds_to_mae = float("inf")
        prev_t, prev_m = path[0]
        for t2, m2 in path[1:]:
            t1, m1 = prev_t, prev_m
            seg_lo = max(t1, entry_ts)
            seg_hi = min(t2, t_hor)
            if seg_hi < seg_lo - EPS:
                prev_t, prev_m = t2, m2
                continue
            for tq in (seg_lo, seg_hi):
                mm = _interp_mid(tq, t1, m1, t2, m2)
                adv = (entry - mm) / stop_dist
                if adv > 0.01 and math.isfinite(seconds_to_mae) and seconds_to_mae > 1e8:
                    seconds_to_mae = max(0.0, tq - entry_ts)
            prev_t, prev_m = t2, m2

    if not math.isfinite(seconds_to_mae):
        seconds_to_mae = horizon

    t05 = t1r = t2r = t3r = None
    prev_t, prev_m = path[0]
    for t2, m2 in path[1:]:
        t1, m1 = prev_t, prev_m
        seg_lo = max(t1, entry_ts)
        seg_hi = min(t2, t_hor)
        if seg_hi < seg_lo - EPS:
            prev_t, prev_m = t2, m2
            continue
        if side == "short":
            for lab, mult, setter in (
                ("05", 0.5, "t05"),
                ("1", 1.0, "t1r"),
                ("2", 2.0, "t2r"),
                ("3", 3.0, "t3r"),
            ):
                lev = entry - mult * stop_dist
                tx = _cross_time_first_down(t1, m1, t2, m2, lev)
                if tx is not None and seg_lo - EPS <= tx <= seg_hi + EPS:
                    tt = max(tx, seg_lo)
                    if lab == "05" and t05 is None:
                        t05 = tt - entry_ts
                    if lab == "1" and t1r is None:
                        t1r = tt - entry_ts
                    if lab == "2" and t2r is None:
                        t2r = tt - entry_ts
                    if lab == "3" and t3r is None:
                        t3r = tt - entry_ts
        else:
            for lab, mult, _ in (
                ("05", 0.5, "t05"),
                ("1", 1.0, "t1r"),
                ("2", 2.0, "t2r"),
                ("3", 3.0, "t3r"),
            ):
                lev = entry + mult * stop_dist
                tx = _cross_time_first_up(t1, m1, t2, m2, lev)
                if tx is not None and seg_lo - EPS <= tx <= seg_hi + EPS:
                    tt = max(tx, seg_lo)
                    if lab == "05" and t05 is None:
                        t05 = tt - entry_ts
                    if lab == "1" and t1r is None:
                        t1r = tt - entry_ts
                    if lab == "2" and t2r is None:
                        t2r = tt - entry_ts
                    if lab == "3" and t3r is None:
                        t3r = tt - entry_ts
        prev_t, prev_m = t2, m2

    return {
        "mfe_r": float(mfe_r),
        "mae_r": float(mae_r),
        "max_price_reached": float(max_p),
        "min_price_reached": float(min_p),
        "seconds_to_mae": float(seconds_to_mae),
        "time_to_first_0_5R": float(t05) if t05 is not None else None,
        "time_to_first_1R": float(t1r) if t1r is not None else None,
        "time_to_first_2R": float(t2r) if t2r is not None else None,
        "time_to_first_3R": float(t3r) if t3r is not None else None,
    }


@dataclass
class _Det:
    """Per-side auction sequence detector (primary: book + mid geometry, not anomaly percentile)."""

    side: str
    last_emit_ts: float = -1e18

    # phase: idle | aggr | absorb | broke | retest
    phase: str = "idle"
    aggr_anchor_ts: float = 0.0
    aggr_ref_extreme: float = 0.0
    pressure_sum_aggr: float = 0.0
    absorb_start_ts: float = 0.0
    box_low: float = 0.0
    box_high: float = 0.0
    box_start_bb: float = 0.0
    bb_peak_absorb: float = 0.0
    saw_depletion: bool = False
    max_mid_pre_break: float = 0.0
    break_ts: float = 0.0
    break_extreme: float = 0.0
    retest_max: float = 0.0
    retest_armed: bool = False
    rh: float = 0.0

    def reset(self) -> None:
        self.phase = "idle"
        self.aggr_anchor_ts = 0.0
        self.aggr_ref_extreme = 0.0
        self.pressure_sum_aggr = 0.0
        self.absorb_start_ts = 0.0
        self.box_low = 0.0
        self.box_high = 0.0
        self.box_start_bb = 0.0
        self.bb_peak_absorb = 0.0
        self.saw_depletion = False
        self.max_mid_pre_break = 0.0
        self.break_ts = 0.0
        self.break_extreme = 0.0
        self.retest_max = 0.0
        self.retest_armed = False
        self.rh = 0.0


class AuctionStructureReplay(LivePaperEngine):
    """Replay book/trades; maintain z_stats + tick history; detect true absorption sequences."""

    def __init__(self, logger: logging.Logger, data_source: str) -> None:
        super().__init__(logger, data_source=data_source)
        self.execution_mode = "paper"
        self._bybit_execution = None
        self.tick_history: deque[tuple[float, float, float, float, float, float, float]] = deque(maxlen=4000)
        self._bb_snapshots: deque[tuple[float, float]] = deque(maxlen=512)
        self.signals: list[dict[str, Any]] = []
        self.d_short = _Det("short")
        self.d_long = _Det("long")

    def _mean_tob(self, ts: float, win: float) -> Optional[float]:
        lo = ts - win
        xs = [x[2] for x in self.tick_history if x[0] >= lo]
        if len(xs) < 2:
            return None
        return float(mean(xs))

    def _pressure_sum(self, ts: float, win: float) -> float:
        lo = ts - win
        return float(sum(x[6] for x in self.tick_history if x[0] >= lo))

    def _mid_slope(self, ts: float, win: float) -> Optional[float]:
        lo = ts - win
        pts = [(x[0], x[1]) for x in self.tick_history if x[0] >= lo]
        if len(pts) < 2:
            return None
        return float(pts[-1][1] - pts[0][1])

    def _bb_peak(self, ts: float, win: float) -> float:
        lo = ts - win
        return max((x[4] for x in self.tick_history if x[0] >= lo), default=0.0)

    def _aa_peak(self, ts: float, win: float) -> float:
        lo = ts - win
        return max((x[5] for x in self.tick_history if x[0] >= lo), default=0.0)

    def _avg_spread(self, ts: float, win: float) -> float:
        lo = ts - win
        xs = [x[3] for x in self.tick_history if x[0] >= lo]
        if not xs:
            return 0.0
        return float(mean(xs))

    def _update_detector_short(self, top: TopState, pressure: float) -> None:
        d = self.d_short
        ts, mid = float(top.ts), float(top.mid)
        if d.phase == "idle" and ts - d.last_emit_ts < COOLDOWN_SEC:
            return
        mtob = self._mean_tob(ts, 3.0)
        msl = self._mid_slope(ts, 3.5)
        ps = self._pressure_sum(ts, 4.0)
        asp = self._avg_spread(ts, 6.0)

        aggr = (
            mtob is not None
            and mtob <= -0.56
            and msl is not None
            and msl < -max(asp * 0.35, mid * 1.2e-6)
            and ps < 0.0
        )

        if d.phase == "idle":
            if aggr:
                d.phase = "aggr"
                d.aggr_anchor_ts = ts
                recent_hi = [x[1] for x in self.tick_history if x[0] >= ts - 4.0]
                d.aggr_ref_extreme = max(mid, max(recent_hi) if recent_hi else mid)
                d.pressure_sum_aggr = ps
            return

        if d.phase == "aggr":
            if not aggr and ts - d.aggr_anchor_ts > 6.0:
                d.reset()
                return
            if ts - d.aggr_anchor_ts >= 1.2 and aggr:
                d.phase = "absorb"
                d.absorb_start_ts = ts
                d.box_low = d.box_high = mid
                d.box_start_bb = float(top.best_bid_sz)
                d.bb_peak_absorb = float(top.best_bid_sz)
                d.saw_depletion = False
                d.max_mid_pre_break = mid
            return

        if d.phase == "absorb":
            d.box_low = min(d.box_low, mid)
            d.box_high = max(d.box_high, mid)
            d.bb_peak_absorb = max(d.bb_peak_absorb, float(top.best_bid_sz))
            if float(top.best_bid_sz) < 0.88 * d.bb_peak_absorb:
                d.saw_depletion = True
            d.max_mid_pre_break = max(d.max_mid_pre_break, mid)
            rng = d.box_high - d.box_low
            rel = rng / mid if mid > 0 else 1.0
            tight = rng < max(2.2 * asp, mid * 3.8e-5) and rel < 5.5e-4
            if ts - d.absorb_start_ts > 12.0 or (tight and ts - d.absorb_start_ts >= 2.4):
                if d.box_start_bb < 1e-9:
                    d.reset()
                    return
                if d.box_start_bb / max(self._bb_peak(ts, 12.0), 1e-9) < 0.62:
                    d.reset()
                    return
                if d.max_mid_pre_break > d.box_high + 0.55 * asp:
                    d.reset()
                    return
                d.phase = "broke_wait"
            if ts - d.absorb_start_ts > 14.0:
                d.reset()
            return

        if d.phase == "broke_wait":
            d.max_mid_pre_break = max(d.max_mid_pre_break, mid)
            buf = max(0.45 * asp, mid * 8e-7)
            if mid < d.box_low - buf:
                d.phase = "broke"
                d.break_ts = ts
                d.break_extreme = mid
                d.retest_max = mid
                d.retest_armed = False
                d.rh = mid
            elif ts - d.absorb_start_ts > 28.0:
                d.reset()
            return

        if d.phase == "broke":
            d.retest_max = max(d.retest_max, mid)
            box_rng = max(d.box_high - d.box_low, asp * 0.5)
            if mid > d.box_low + 0.22 * box_rng:
                d.retest_armed = True
            d.rh = max(d.rh, mid)
            reclaim_fail = d.rh < d.box_high - 0.12 * max(asp, box_rng * 0.25)
            if d.retest_armed and reclaim_fail and mid < d.rh - 0.11 * max(d.rh - d.box_low, asp):
                ctrl_hi = max(d.aggr_ref_extreme, d.box_high)
                stop_px = ctrl_hi + 0.65 * asp
                inv = stop_px
                retest_score = min(
                    100.0,
                    max(
                        0.0,
                        50.0 * (d.box_high - d.rh) / max(box_rng, asp)
                        + 30.0 * float(d.saw_depletion)
                        + 20.0 * min(1.0, max(0.0, -d.pressure_sum_aggr)),
                    ),
                )
                sig = {
                    "side": "short",
                    "entry_ts_unix": ts,
                    "entry_ts": top.ts_text,
                    "entry_price": mid,
                    "entry_session": _entry_session_label(ts),
                    "regime": getattr(self, "_last_regime", "normal"),
                    "structural_stop_price": float(stop_px),
                    "structural_invalidation_price": float(inv),
                    "control_reference_extreme": float(ctrl_hi),
                    "absorption_box_low": float(d.box_low),
                    "absorption_box_high": float(d.box_high),
                    "failed_retest_high": float(d.rh),
                    "retest_score": float(retest_score),
                    "supporting_z_pressure_at_entry": self._last_z_pressure,
                    "supporting_z_tob_at_entry": self._last_z_tob,
                }
                self.signals.append(sig)
                d.last_emit_ts = ts
                d.reset()
            elif ts - d.break_ts > 26.0:
                d.reset()
            return

    def _update_detector_long(self, top: TopState, pressure: float) -> None:
        d = self.d_long
        ts, mid = float(top.ts), float(top.mid)
        if d.phase == "idle" and ts - d.last_emit_ts < COOLDOWN_SEC:
            return
        mtob = self._mean_tob(ts, 3.0)
        msl = self._mid_slope(ts, 3.5)
        ps = self._pressure_sum(ts, 4.0)
        asp = self._avg_spread(ts, 6.0)

        aggr = (
            mtob is not None
            and mtob >= 0.56
            and msl is not None
            and msl > max(asp * 0.35, mid * 1.2e-6)
            and ps > 0.0
        )

        if d.phase == "idle":
            if aggr:
                d.phase = "aggr"
                d.aggr_anchor_ts = ts
                recent_lo = [x[1] for x in self.tick_history if x[0] >= ts - 4.0]
                d.aggr_ref_extreme = min(mid, min(recent_lo) if recent_lo else mid)
                d.pressure_sum_aggr = ps
            return

        if d.phase == "aggr":
            if not aggr and ts - d.aggr_anchor_ts > 6.0:
                d.reset()
                return
            if ts - d.aggr_anchor_ts >= 1.2 and aggr:
                d.phase = "absorb"
                d.absorb_start_ts = ts
                d.box_low = d.box_high = mid
                d.box_start_bb = float(top.best_ask_sz)
                d.bb_peak_absorb = float(top.best_ask_sz)
                d.saw_depletion = False
                d.max_mid_pre_break = mid
            return

        if d.phase == "absorb":
            d.box_low = min(d.box_low, mid)
            d.box_high = max(d.box_high, mid)
            d.bb_peak_absorb = max(d.bb_peak_absorb, float(top.best_ask_sz))
            if float(top.best_ask_sz) < 0.88 * d.bb_peak_absorb:
                d.saw_depletion = True
            d.max_mid_pre_break = min(d.max_mid_pre_break, mid)
            rng = d.box_high - d.box_low
            rel = rng / mid if mid > 0 else 1.0
            tight = rng < max(2.2 * asp, mid * 3.8e-5) and rel < 5.5e-4
            if ts - d.absorb_start_ts > 12.0 or (tight and ts - d.absorb_start_ts >= 2.4):
                if d.box_start_bb < 1e-9:
                    d.reset()
                    return
                if d.box_start_bb / max(self._aa_peak(ts, 12.0), 1e-9) < 0.62:
                    d.reset()
                    return
                if d.max_mid_pre_break < d.box_low - 0.55 * asp:
                    d.reset()
                    return
                d.phase = "broke_wait"
            if ts - d.absorb_start_ts > 14.0:
                d.reset()
            return

        if d.phase == "broke_wait":
            d.max_mid_pre_break = min(d.max_mid_pre_break, mid)
            buf = max(0.45 * asp, mid * 8e-7)
            if mid > d.box_high + buf:
                d.phase = "broke"
                d.break_ts = ts
                d.break_extreme = mid
                d.retest_armed = False
                d.rh = mid
            elif ts - d.absorb_start_ts > 28.0:
                d.reset()
            return

        if d.phase == "broke":
            box_rng = max(d.box_high - d.box_low, asp * 0.5)
            if mid < d.box_high - 0.22 * box_rng:
                d.retest_armed = True
            d.rh = min(d.rh, mid)
            reclaim_fail = d.rh > d.box_low + 0.12 * max(asp, box_rng * 0.25)
            if d.retest_armed and reclaim_fail and mid > d.rh + 0.11 * max(d.box_high - d.rh, asp):
                ctrl_lo = min(d.aggr_ref_extreme, d.box_low)
                stop_px = ctrl_lo - 0.65 * asp
                inv = stop_px
                retest_score = min(
                    100.0,
                    max(
                        0.0,
                        50.0 * (d.rh - d.box_low) / max(box_rng, asp)
                        + 30.0 * float(d.saw_depletion)
                        + 20.0 * min(1.0, max(0.0, d.pressure_sum_aggr)),
                    ),
                )
                sig = {
                    "side": "long",
                    "entry_ts_unix": ts,
                    "entry_ts": top.ts_text,
                    "entry_price": mid,
                    "entry_session": _entry_session_label(ts),
                    "regime": getattr(self, "_last_regime", "normal"),
                    "structural_stop_price": float(stop_px),
                    "structural_invalidation_price": float(inv),
                    "control_reference_extreme": float(ctrl_lo),
                    "absorption_box_low": float(d.box_low),
                    "absorption_box_high": float(d.box_high),
                    "failed_retest_low": float(d.rh),
                    "retest_score": float(retest_score),
                    "supporting_z_pressure_at_entry": self._last_z_pressure,
                    "supporting_z_tob_at_entry": self._last_z_tob,
                }
                self.signals.append(sig)
                d.last_emit_ts = ts
                d.reset()
            elif ts - d.break_ts > 26.0:
                d.reset()
            return

    def process_object(self, obj: dict[str, Any]) -> Optional[TopState]:
        self.raw_events_seen += 1
        self._process_trade_msg(obj)
        top, pressure = self._process_l2_msg(obj)
        if top is None:
            return None

        self._expire_windows(top.ts)
        while self._bb_snapshots and self._bb_snapshots[0][0] < top.ts - LOOKBACK_SEC:
            self._bb_snapshots.popleft()
        self._bb_snapshots.append((top.ts, float(top.best_bid_sz)))
        self.mid_window.append((top.ts, top.mid))
        self.last_mid = top.mid

        event_rate = len(self.l2_times) / max(15.0, 1e-9)
        trade_count = float(len(self.trade_times))
        mid_vals = [m for _, m in self.mid_window]
        mid_vol = 0.0
        if len(mid_vals) >= 2:
            mu = sum(mid_vals) / len(mid_vals)
            var = sum((x - mu) ** 2 for x in mid_vals) / (len(mid_vals) - 1)
            mid_vol = (var if var > 0 else 0.0) ** 0.5

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

        self._last_regime = self._regime(z_event, z_spread, z_mid_vol, z_trade)
        self._last_z_pressure = z_pressure
        self._last_z_tob = z_tob

        self.tick_history.append(
            (float(top.ts), float(top.mid), float(top.tob_imb), float(top.spread), float(top.best_bid_sz), float(top.best_ask_sz), float(pressure))
        )

        self._update_detector_short(top, pressure)
        self._update_detector_long(top, pressure)
        return top


def _build_rows_for_signals(
    signals: list[dict[str, Any]],
    mids: list[tuple[float, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for s in signals:
        side = str(s["side"]).lower()
        entry = float(s["entry_price"])
        entry_ts = float(s["entry_ts_unix"])
        stop_px = float(s["structural_stop_price"])
        inv = float(s["structural_invalidation_price"])
        if side == "short":
            stop_dist = max(stop_px - entry, EPS)
        else:
            stop_dist = max(entry - stop_px, EPS)

        ex = _excursion_from_path(
            side=side,
            mids=mids,
            entry_ts=entry_ts,
            entry=entry,
            stop_dist=stop_dist,
            horizon=PATH_HORIZON_SEC,
        )
        base = {
            **s,
            "structural_stop_distance": float(stop_dist),
            "fixed_reference_stop_distance": float(FIXED_REF_STOP_USD),
            **ex,
        }

        oco1 = _oco_first_exit(
            side=side,
            mids=mids,
            entry_ts=entry_ts,
            entry=entry,
            stop_dist=stop_dist,
            inv_level=inv,
            horizon=PATH_HORIZON_SEC,
            tp_mult=1.0,
            fee_rate=ROUND_TRIP_TAKER_FEE,
        )
        oco2 = _oco_first_exit(
            side=side,
            mids=mids,
            entry_ts=entry_ts,
            entry=entry,
            stop_dist=stop_dist,
            inv_level=inv,
            horizon=PATH_HORIZON_SEC,
            tp_mult=2.0,
            fee_rate=ROUND_TRIP_TAKER_FEE,
        )
        oco3 = _oco_first_exit(
            side=side,
            mids=mids,
            entry_ts=entry_ts,
            entry=entry,
            stop_dist=stop_dist,
            inv_level=inv,
            horizon=PATH_HORIZON_SEC,
            tp_mult=3.0,
            fee_rate=ROUND_TRIP_TAKER_FEE,
        )
        base["oco_market_net_r_1R"] = oco1["net_r"]
        base["oco_market_net_r_2R"] = oco2["net_r"]
        base["oco_market_net_r_3R"] = oco3["net_r"]

        fixed_dist = float(FIXED_REF_STOP_USD)
        fx1 = _oco_first_exit(
            side=side,
            mids=mids,
            entry_ts=entry_ts,
            entry=entry,
            stop_dist=fixed_dist,
            inv_level=entry + fixed_dist if side == "short" else entry - fixed_dist,
            horizon=PATH_HORIZON_SEC,
            tp_mult=1.0,
            fee_rate=ROUND_TRIP_TAKER_FEE,
        )
        base["oco_fixed2usd_net_r_1R"] = fx1["net_r"]
        rows.append(base)
    return rows


def _evaluate_entry_models(r: dict[str, Any], mids: list[tuple[float, float]]) -> dict[str, Any]:
    """Returns per-model OCO 1R/2R net R (after fees) for structural stop distance."""
    side = str(r["side"]).lower()
    entry0 = float(r["entry_price"])
    ex_ts = float(r["entry_ts_unix"])
    sd0 = float(r["structural_stop_distance"])
    inv = float(r["structural_invalidation_price"])

    def oc(entry_px: float, stop_d: float, fee_rt: float) -> dict[str, float]:
        return {
            "1R": _oco_first_exit(
                side=side,
                mids=mids,
                entry_ts=ex_ts,
                entry=entry_px,
                stop_dist=stop_d,
                inv_level=inv,
                horizon=PATH_HORIZON_SEC,
                tp_mult=1.0,
                fee_rate=fee_rt,
            )["net_r"],
            "2R": _oco_first_exit(
                side=side,
                mids=mids,
                entry_ts=ex_ts,
                entry=entry_px,
                stop_dist=stop_d,
                inv_level=inv,
                horizon=PATH_HORIZON_SEC,
                tp_mult=2.0,
                fee_rate=fee_rt,
            )["net_r"],
            "3R": _oco_first_exit(
                side=side,
                mids=mids,
                entry_ts=ex_ts,
                entry=entry_px,
                stop_dist=stop_d,
                inv_level=inv,
                horizon=PATH_HORIZON_SEC,
                tp_mult=3.0,
                fee_rate=fee_rt,
            )["net_r"],
        }

    # A: market
    a = oc(entry0, sd0, ROUND_TRIP_TAKER_FEE)

    # B / C: maker-style improved entry if synthetic fill within window
    tcopy = dict(r)
    ok_b = ok_c = False
    if side == "short":
        ok_b, imp_b, _ = _synthetic_limit_fill_short(tcopy, limit_above_entry_usd=0.25 * sd0)
        ok_c, imp_c, _ = _synthetic_limit_fill_short(
            tcopy,
            limit_above_entry_usd=max(float(r.get("absorption_box_high", entry0)) - entry0, 0.15 * sd0),
        )
        entry_b = entry0 + 0.25 * sd0 if ok_b else entry0
        entry_c = entry0 + max(float(r.get("absorption_box_high", entry0)) - entry0, 0.15 * sd0) if ok_c else entry0
        stop_b = max(inv - entry_b, EPS) if ok_b else sd0
        stop_c = max(inv - entry_c, EPS) if ok_c else sd0
    else:
        ok_b, imp_b, _ = _synthetic_limit_fill_long(tcopy, limit_below_entry_usd=0.25 * sd0)
        box_lo = float(r.get("absorption_box_low", entry0))
        ok_c, imp_c, _ = _synthetic_limit_fill_long(
            tcopy,
            limit_below_entry_usd=max(entry0 - box_lo, 0.15 * sd0),
        )
        entry_b = entry0 - 0.25 * sd0 if ok_b else entry0
        entry_c = entry0 - max(entry0 - box_lo, 0.15 * sd0) if ok_c else entry0
        stop_b = max(entry_b - inv, EPS) if ok_b else sd0
        stop_c = max(entry_c - inv, EPS) if ok_c else sd0

    b = oc(entry_b, stop_b, FEE_MAKER_ENTRY_TAKER_EXIT)
    c = oc(entry_c, stop_c, FEE_MAKER_ENTRY_TAKER_EXIT)

    return {
        "A_market_after_failed_retest": a,
        "B_maker_pullback_0_25R_zone": {**b, "synthetic_filled": ok_b},
        "C_maker_absorption_retest_level": {**c, "synthetic_filled": ok_c},
    }


def _mean(xs: list[float]) -> float:
    ys = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return float(mean(ys)) if ys else float("nan")


def _median(xs: list[float]) -> Optional[float]:
    ys = sorted(x for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x)))
    if not ys:
        return None
    return float(median(ys))


def _summarize_trades(rows: list[dict[str, Any]], net_key: str) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {
            "n": 0,
            "win_rate": 0.0,
            "reach_1R": 0.0,
            "reach_2R": 0.0,
            "reach_3R": 0.0,
            "net_expectancy_after_fees": 0.0,
            "profit_factor_net": 0.0,
        }
    nets = [_sf(r.get(net_key)) for r in rows]
    wins = sum(1 for x in nets if x > 0)
    r1 = sum(1 for r in rows if _sf(r.get("mfe_r")) >= 1.0 - 1e-9) / n
    r2 = sum(1 for r in rows if _sf(r.get("mfe_r")) >= 2.0 - 1e-9) / n
    r3 = sum(1 for r in rows if _sf(r.get("mfe_r")) >= 3.0 - 1e-9) / n
    pf = float(_profit_factor(nets))
    if not math.isfinite(pf):
        pf = 0.0
    return {
        "n": n,
        "win_rate": wins / n,
        "reach_1R": r1,
        "reach_2R": r2,
        "reach_3R": r3,
        "net_expectancy_after_fees": _mean(nets),
        "profit_factor_net": pf,
    }


def _by_key(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        k = str(r.get(key) or "unknown")
        out.setdefault(k, []).append(r)
    return out


def _adverse_block(rows: list[dict[str, Any]], per_row_models: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"note": "no_rows", "baseline_loser_rate_1R_market": 0.0}
    base_losers = sum(1 for r in rows if _sf(r.get("oco_market_net_r_1R")) < 0) / n
    filled_idx = [
        i
        for i, m in enumerate(per_row_models)
        if m.get("B_maker_pullback_0_25R_zone", {}).get("synthetic_filled")
    ]
    if filled_idx:
        mk = sum(1 for i in filled_idx if _sf(per_row_models[i]["B_maker_pullback_0_25R_zone"]["1R"]) < 0) / len(
            filled_idx
        )
        delta = mk - base_losers
    else:
        mk = float("nan")
        delta = float("nan")
    return {
        "baseline_loser_rate_1R_market": float(base_losers),
        "filled_maker_B_loser_rate_1R": mk if filled_idx else None,
        "delta_filled_maker_B_minus_market": delta if filled_idx else None,
        "definition": "Adverse selection: maker-B loser rate minus market 1R OCO loser rate on same signals.",
    }


def run_true_failed_absorption_study(
    *,
    data_source: str = "bybit",
    raw_dir: Path = RAW_DATA_DIR,
    symbol: str = "ETHUSDT",
    json_output: Path = REPORTS_DIR / "true_failed_absorption_study.json",
) -> dict[str, Any]:
    glob_pat = BYBIT_RAW_GLOB if data_source == "bybit" else "*.ndjson"
    objs = list(_iter_objects(raw_dir, glob_pat))
    doc: dict[str, Any] = {
        "study": "true_failed_absorption_sequence",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "parameters": {
            "path_horizon_sec": PATH_HORIZON_SEC,
            "cooldown_sec": COOLDOWN_SEC,
            "fixed_reference_stop_usd": FIXED_REF_STOP_USD,
            "limit_fill_window_sec": LIMIT_FILL_WINDOW_SEC,
            "primary_signal": "auction_microstructure_state_machine_tob_pressure_mid_box",
            "supporting_features": "z_pressure_z_tob_recorded_at_entry_not_used_as_primary_trigger",
            "fee_round_trip_taker": ROUND_TRIP_TAKER_FEE,
            "fee_maker_entry_taker_exit_effective": FEE_MAKER_ENTRY_TAKER_EXIT,
        },
        "data_source": data_source,
        "symbol": str(symbol).strip().upper(),
        "raw_event_count_estimate": len(objs),
    }

    if not objs:
        doc["verdict"] = _final_verdict(
            [],
            [],
            {
                "best_side_1R_ev": "insufficient_data",
                "best_session": "insufficient_data",
                "best_entry_model": "insufficient_data",
                "struct_vs_fixed": {},
                "production_candidate": False,
            },
        )
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(_sanitize(doc), indent=2), encoding="utf-8")
        return doc

    prev_sym = os.environ.get("BYBIT_SYMBOL")
    os.environ["BYBIT_SYMBOL"] = str(symbol).strip().upper()
    try:
        replay = AuctionStructureReplay(LOGGER, data_source=data_source)
        mids: list[tuple[float, float]] = []
        for obj in objs:
            top = replay.process_object(obj)
            if top is not None:
                mids.append((float(top.ts), float(top.mid)))

        signals = list(replay.signals)
        rows = _build_rows_for_signals(signals, mids)

        per_row_models: list[dict[str, Any]] = []
        for r in rows:
            per_row_models.append(_evaluate_entry_models(r, mids))

        struct_dists = [_sf(r.get("structural_stop_distance")) for r in rows]
        fixed_nets = [_sf(r.get("oco_fixed2usd_net_r_1R")) for r in rows]
        struct_nets = [_sf(r.get("oco_market_net_r_1R")) for r in rows]

        short_r = [r for r in rows if str(r.get("side")) == "short"]
        long_r = [r for r in rows if str(r.get("side")) == "long"]

        ev_s = _mean([_sf(r.get("oco_market_net_r_1R")) for r in short_r])
        ev_l = _mean([_sf(r.get("oco_market_net_r_1R")) for r in long_r])
        if not short_r and not long_r:
            best_side = "insufficient_data"
        elif ev_s > ev_l:
            best_side = "short"
        else:
            best_side = "long"

        session_summary = {
            sess: _summarize_trades(sub, "oco_market_net_r_1R")
            for sess, sub in sorted(_by_key(rows, "entry_session").items())
        }
        regime_summary = {
            reg: _summarize_trades(sub, "oco_market_net_r_1R")
            for reg, sub in sorted(_by_key(rows, "regime").items())
        }

        struct_vs_fixed = {
            "mean_net_r_1R_market_structure_stop": _mean(struct_nets),
            "mean_net_r_1R_market_fixed_2usd_stop": _mean(fixed_nets),
            "structure_stop_better_mean_1R_ev": bool(_mean(struct_nets) > _mean(fixed_nets) + 1e-12),
        }

        entry_compare = {
            "A_market_1R_ev": float("nan"),
            "B_maker_1R_ev": float("nan"),
            "C_maker_1R_ev": float("nan"),
        }
        if per_row_models:
            b_sel = [
                _sf(m["B_maker_pullback_0_25R_zone"]["1R"])
                for m in per_row_models
                if m["B_maker_pullback_0_25R_zone"].get("synthetic_filled")
            ]
            c_sel = [
                _sf(m["C_maker_absorption_retest_level"]["1R"])
                for m in per_row_models
                if m["C_maker_absorption_retest_level"].get("synthetic_filled")
            ]
            entry_compare = {
                "A_market_1R_ev": _mean([_sf(m["A_market_after_failed_retest"]["1R"]) for m in per_row_models]),
                "B_maker_1R_ev": _mean(b_sel) if b_sel else float("nan"),
                "C_maker_1R_ev": _mean(c_sel) if c_sel else float("nan"),
            }

        positive_edge = bool(len(rows) >= MIN_SIGNALS_FOR_VERDICT and _mean(struct_nets) > 0)
        ev_a = float(entry_compare.get("A_market_1R_ev", float("nan")))
        pf_s = float(_profit_factor(struct_nets))
        if not math.isfinite(pf_s):
            pf_s = 0.0
        best_entry = max(
            ("A_market", entry_compare.get("A_market_1R_ev", float("nan"))),
            ("B_maker_pullback", entry_compare.get("B_maker_1R_ev", float("nan"))),
            ("C_maker_absorption", entry_compare.get("C_maker_1R_ev", float("nan"))),
            key=lambda x: x[1] if math.isfinite(x[1]) else -1e18,
        )

        if session_summary:
            best_session = max(
                session_summary.items(),
                key=lambda kv: float(kv[1].get("net_expectancy_after_fees") or -1e18),
            )[0]
        else:
            best_session = "insufficient_data"

        production_candidate = bool(
            len(rows) >= MIN_SIGNALS_FOR_VERDICT
            and positive_edge
            and math.isfinite(ev_a)
            and ev_a > 0
            and pf_s >= 1.05
        )

        doc.update(
            {
                "signals_detected": len(signals),
                "executed_rows": len(rows),
                "structural_stop_usd_stats": {
                    "mean": _mean(struct_dists),
                    "median": _median(struct_dists),
                },
                "overall_market_structure_stop": _summarize_trades(rows, "oco_market_net_r_1R"),
                "by_side": {
                    "short": _summarize_trades(short_r, "oco_market_net_r_1R"),
                    "long": _summarize_trades(long_r, "oco_market_net_r_1R"),
                },
                "session_attribution": session_summary,
                "regime_attribution": regime_summary,
                "entry_model_ev_1R": entry_compare,
                "per_signal_entry_models": per_row_models,
                "structural_vs_fixed_stop": struct_vs_fixed,
                "failed_retest_quality": {
                    "mean_retest_score": _mean([_sf(r.get("retest_score")) for r in rows]),
                    "median_retest_score": _median([_sf(r.get("retest_score")) for r in rows]),
                },
                "adverse_selection_analysis": _adverse_block(rows, per_row_models),
                "signals_sample": rows[:40],
            }
        )

        doc["verdict"] = _final_verdict(
            rows,
            struct_nets,
            {
                "positive_edge": positive_edge,
                "best_side_1R_ev": best_side,
                "best_session": best_session,
                "best_entry_model": best_entry[0],
                "struct_vs_fixed": struct_vs_fixed,
                "production_candidate": production_candidate,
            },
        )
    finally:
        if prev_sym is None:
            os.environ.pop("BYBIT_SYMBOL", None)
        else:
            os.environ["BYBIT_SYMBOL"] = prev_sym

    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(_sanitize(doc), indent=2), encoding="utf-8")
    LOGGER.info("wrote %s (signals=%s rows=%s)", json_output, doc.get("signals_detected", 0), doc.get("executed_rows", 0))
    return doc


def _final_verdict(
    rows: list[dict[str, Any]],
    struct_nets: list[float],
    hints: dict[str, Any],
) -> dict[str, Any]:
    n = len(rows)
    q1 = bool(n >= MIN_SIGNALS_FOR_VERDICT and _mean(struct_nets) > 0)
    svf = hints.get("struct_vs_fixed") if isinstance(hints.get("struct_vs_fixed"), dict) else {}
    struct_beats_fixed = bool(svf.get("structure_stop_better_mean_1R_ev", False))
    out = {
        "1_true_failed_absorption_positive_edge": q1,
        "2_long_or_short_superior_on_mean_1R_ev": hints.get("best_side_1R_ev", "insufficient_data"),
        "3_best_session_bucket": hints.get("best_session", "insufficient_data"),
        "4_best_entry_model": hints.get("best_entry_model", "insufficient_data"),
        "5_structure_based_stop_outperforms_fixed_2usd_on_mean_1R_ev": struct_beats_fixed,
        "6_real_production_candidate": bool(hints.get("production_candidate", False)),
        "7_if_not_production_failed_assumption": "",
    }
    if not q1:
        if n < MIN_SIGNALS_FOR_VERDICT:
            out["7_if_not_production_failed_assumption"] = f"insufficient_signals_need_at_least_{MIN_SIGNALS_FOR_VERDICT}"
        elif n == 0:
            out["7_if_not_production_failed_assumption"] = "no_sequence_signals_detected_in_replay"
        else:
            out["7_if_not_production_failed_assumption"] = "mean_net_expectancy_at_1R_market_with_structure_stop_not_positive"
    elif not hints.get("production_candidate"):
        out["7_if_not_production_failed_assumption"] = (
            "positive_mean_ev_but_fails_production_gates_check_pf_or_sample_size_or_entry_model_A"
        )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="True failed absorption sequence study (long+short)")
    p.add_argument("--data-source", choices=("bybit", "coinbase"), default="bybit")
    p.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--json-output", type=Path, default=REPORTS_DIR / "true_failed_absorption_study.json")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_true_failed_absorption_study(
        data_source=args.data_source,
        raw_dir=args.raw_dir,
        symbol=str(args.symbol).strip().upper(),
        json_output=args.json_output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
