"""
Stop-management simulation on a fixed anomaly signal (research only).

Baseline signal (unchanged):
- anomaly_direction == bearish
- anomaly_percentile_rank >= 0.99
- trade direction = short

Path-level sequencing from ``book_state.csv`` is used for exact first-touch logic
within a forward horizon, with conservative same-timestamp tie handling (SL first).
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from tjtb.features.build_features import parse_ts_to_unix

logger = logging.getLogger(__name__)

DEFAULT_SETUPS = Path("data/parsed/anomaly_setups.csv")
DEFAULT_BOOK = Path("data/parsed/book_state.csv")
DEFAULT_MATRIX = Path("data/parsed/anomaly_matrix.csv")
DEFAULT_OUTPUT = Path("reports/stop_management_simulation.json")
READ_BUFFER = 1024 * 1024
EPS = 1e-9
DEFAULT_FORWARD_HORIZON_SEC = 2.0

COST_LEVELS_R: tuple[float, ...] = (0.00, 0.05, 0.10, 0.25, 0.50)


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    s = value.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="", buffering=READ_BUFFER) as f:
        return list(csv.DictReader(f))


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    q = min(max(q, 0.0), 1.0)
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    idx = q * (n - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


@dataclass(frozen=True)
class BookTsPoint:
    ts: float
    mid_min: float
    mid_max: float
    mid_last: float


@dataclass(frozen=True)
class SignalRow:
    ts: float
    entry_price: float
    event_rate_hz: float | None
    trade_count_k: float | None
    mid_vol_w: float | None
    spread: float | None


@dataclass(frozen=True)
class StopUpdateRule:
    trigger_fav_r: float
    new_stop_r: float


@dataclass(frozen=True)
class StopMgmtConfig:
    name: str
    tp_r: float | None  # None -> per-trade regime-derived TP
    initial_stop_r: float = -1.0  # -1R loss stop at entry+1R for shorts
    updates: tuple[StopUpdateRule, ...] = ()
    regime_based: bool = False


@dataclass(frozen=True)
class TradeOutcome:
    outcome: str  # tp | sl | be | locked | timeout | skipped
    gross_r: float


def _load_book_path(path: Path) -> tuple[list[float], list[BookTsPoint]]:
    grouped: list[BookTsPoint] = []
    current_ts: float | None = None
    cur_min = 0.0
    cur_max = 0.0
    cur_last = 0.0

    with path.open("r", encoding="utf-8", newline="", buffering=READ_BUFFER) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_s = (row.get("ts") or "").strip()
            mid = _safe_float(row.get("mid_price"))
            if not ts_s or mid is None or math.isnan(mid):
                continue
            try:
                ts = parse_ts_to_unix(ts_s)
            except ValueError:
                continue

            if current_ts is None:
                current_ts = ts
                cur_min = mid
                cur_max = mid
                cur_last = mid
                continue
            if ts == current_ts:
                cur_min = min(cur_min, mid)
                cur_max = max(cur_max, mid)
                cur_last = mid
                continue

            grouped.append(BookTsPoint(ts=current_ts, mid_min=cur_min, mid_max=cur_max, mid_last=cur_last))
            current_ts = ts
            cur_min = mid
            cur_max = mid
            cur_last = mid

    if current_ts is not None:
        grouped.append(BookTsPoint(ts=current_ts, mid_min=cur_min, mid_max=cur_max, mid_last=cur_last))
    return [x.ts for x in grouped], grouped


def _horizon_slice(ts_vec: list[float], points: list[BookTsPoint], *, signal_ts: float, horizon_sec: float) -> list[BookTsPoint]:
    lo = bisect.bisect_left(ts_vec, signal_ts)
    hi = bisect.bisect_right(ts_vec, signal_ts + horizon_sec)
    return points[lo:hi]


def _matrix_index_by_ts(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    out: dict[str, dict[str, str]] = {}
    for row in _load_csv(path):
        ts = (row.get("ts") or "").strip()
        if ts:
            out[ts] = row
    return out


def _build_baseline_rows(setups_path: Path, matrix_idx: dict[str, dict[str, str]]) -> list[SignalRow]:
    rows: list[SignalRow] = []
    for r in _load_csv(setups_path):
        direction = (r.get("anomaly_direction") or "").strip().lower()
        pctl = _safe_float(r.get("anomaly_percentile_rank"))
        ts_s = (r.get("ts") or "").strip()
        entry = _safe_float(r.get("mid_price"))
        if direction != "bearish":
            continue
        if pctl is None or pctl < 0.99:
            continue
        if not ts_s or entry is None or math.isnan(entry):
            continue
        try:
            ts = parse_ts_to_unix(ts_s)
        except ValueError:
            continue
        m = matrix_idx.get(ts_s, {})
        rows.append(
            SignalRow(
                ts=ts,
                entry_price=entry,
                event_rate_hz=_safe_float(m.get("l2_event_rate_hz")),
                trade_count_k=_safe_float(m.get("trade_count_k")),
                mid_vol_w=_safe_float(m.get("mid_vol_w")),
                spread=_safe_float(m.get("spread")),
            )
        )
    return rows


def _classify_regime(row: SignalRow, q: dict[str, float]) -> str:
    ev = row.event_rate_hz
    tc = row.trade_count_k
    mv = row.mid_vol_w
    sp = row.spread
    if ev is None or tc is None or mv is None or sp is None:
        return "normal"

    if sp >= q["spread_q90"] or mv >= q["mid_vol_q90"] or ev >= q["event_q90"]:
        return "chaotic"
    if sp <= q["spread_q33"] and mv <= q["mid_vol_q33"] and ev <= q["event_q33"] and tc <= q["trade_count_q33"]:
        return "calm"
    if mv >= q["mid_vol_q66"] or ev >= q["event_q66"] or tc >= q["trade_count_q66"]:
        return "volatile"
    return "normal"


def _regime_quantiles(rows: list[SignalRow]) -> dict[str, float]:
    def vals(getter: str) -> list[float]:
        out: list[float] = []
        for r in rows:
            v = getattr(r, getter)
            if v is None or math.isnan(v):
                continue
            out.append(v)
        out.sort()
        return out

    ev = vals("event_rate_hz")
    tc = vals("trade_count_k")
    mv = vals("mid_vol_w")
    sp = vals("spread")
    return {
        "event_q33": _quantile(ev, 0.33) if ev else float("nan"),
        "event_q66": _quantile(ev, 0.66) if ev else float("nan"),
        "event_q90": _quantile(ev, 0.90) if ev else float("nan"),
        "trade_count_q33": _quantile(tc, 0.33) if tc else float("nan"),
        "trade_count_q66": _quantile(tc, 0.66) if tc else float("nan"),
        "mid_vol_q33": _quantile(mv, 0.33) if mv else float("nan"),
        "mid_vol_q66": _quantile(mv, 0.66) if mv else float("nan"),
        "mid_vol_q90": _quantile(mv, 0.90) if mv else float("nan"),
        "spread_q33": _quantile(sp, 0.33) if sp else float("nan"),
        "spread_q90": _quantile(sp, 0.90) if sp else float("nan"),
    }


def _price_from_stop_r_short(entry: float, sl_dist: float, stop_r: float) -> float:
    # stop_r = -1.0 -> entry + sl_dist (loss stop)
    # stop_r = 0.0  -> entry (breakeven)
    # stop_r = 0.25 -> entry - 0.25*sl_dist (locked profit)
    return entry - stop_r * sl_dist


def _outcome_label_from_stop_r(stop_r: float) -> str:
    if stop_r < 0.0:
        return "sl"
    if abs(stop_r) <= EPS:
        return "be"
    return "locked"


def _simulate_trade_short(
    row: SignalRow,
    path: list[BookTsPoint],
    *,
    tp_r: float,
    updates: tuple[StopUpdateRule, ...],
    initial_stop_r: float,
) -> TradeOutcome:
    if not path:
        return TradeOutcome(outcome="timeout", gross_r=0.0)

    sl_dist = 1.0  # strategy defines SL = 1R
    entry = row.entry_price
    tp_price = entry - tp_r * sl_dist
    current_stop_r = initial_stop_r
    current_stop_price = _price_from_stop_r_short(entry, sl_dist, current_stop_r)

    updates_sorted = sorted(updates, key=lambda x: x.trigger_fav_r)

    for p in path:
        # Conservative ordering: stop check before TP/updates within timestamp.
        if p.mid_max >= current_stop_price:
            return TradeOutcome(outcome=_outcome_label_from_stop_r(current_stop_r), gross_r=current_stop_r)

        if p.mid_min <= tp_price:
            return TradeOutcome(outcome="tp", gross_r=tp_r)

        # Apply favorable-triggered stop moves.
        for u in updates_sorted:
            trig_price = entry - u.trigger_fav_r * sl_dist
            if p.mid_min <= trig_price:
                current_stop_r = max(current_stop_r, u.new_stop_r)
        current_stop_price = _price_from_stop_r_short(entry, sl_dist, current_stop_r)

        # After update, conservative immediate stop check in same timestamp.
        if p.mid_max >= current_stop_price:
            return TradeOutcome(outcome=_outcome_label_from_stop_r(current_stop_r), gross_r=current_stop_r)

    timeout_r = (entry - path[-1].mid_last) / max(sl_dist, EPS)
    return TradeOutcome(outcome="timeout", gross_r=timeout_r)


def _simulate_config_paths(
    rows: list[SignalRow],
    cfg: StopMgmtConfig,
    *,
    ts_vec: list[float],
    book_points: list[BookTsPoint],
    forward_horizon_sec: float,
    regime_q: dict[str, float],
) -> dict[str, Any]:
    gross_r_vals: list[float] = []
    tp_count = 0
    sl_count = 0
    be_count = 0
    lock_count = 0
    timeout_count = 0
    skipped_count = 0

    for row in rows:
        path = _horizon_slice(ts_vec, book_points, signal_ts=row.ts, horizon_sec=forward_horizon_sec)
        tp_r = cfg.tp_r
        updates = cfg.updates
        if cfg.regime_based:
            regime = _classify_regime(row, regime_q)
            if regime == "chaotic":
                skipped_count += 1
                continue
            if regime == "calm":
                tp_r = 1.5
                updates = (StopUpdateRule(0.75, 0.0),)
            elif regime == "normal":
                tp_r = 2.0
                updates = (StopUpdateRule(1.0, 0.0),)
            else:  # volatile
                tp_r = 2.5
                updates = (StopUpdateRule(1.5, 0.0),)

        if tp_r is None:
            raise ValueError(f"tp_r unresolved for config {cfg.name}")

        tr = _simulate_trade_short(
            row,
            path,
            tp_r=tp_r,
            updates=updates,
            initial_stop_r=cfg.initial_stop_r,
        )
        gross_r_vals.append(tr.gross_r)
        if tr.outcome == "tp":
            tp_count += 1
        elif tr.outcome == "sl":
            sl_count += 1
        elif tr.outcome == "be":
            be_count += 1
        elif tr.outcome == "locked":
            lock_count += 1
        else:
            timeout_count += 1

    return {
        "gross_r_vals": gross_r_vals,
        "tp_count": tp_count,
        "sl_count": sl_count,
        "be_count": be_count,
        "lock_count": lock_count,
        "timeout_count": timeout_count,
        "skipped_count": skipped_count,
    }


def _profit_factor(r_vals: list[float]) -> float | None:
    gp = sum(x for x in r_vals if x > 0)
    gl = -sum(x for x in r_vals if x < 0)
    if gl <= 0:
        return None
    return gp / gl


def _max_drawdown(curve: list[float]) -> float:
    if not curve:
        return 0.0
    peak = curve[0]
    mdd = 0.0
    for x in curve:
        peak = max(peak, x)
        mdd = max(mdd, peak - x)
    return mdd


def _max_losing_streak(r_vals: list[float]) -> int:
    cur = 0
    mx = 0
    for x in r_vals:
        if x < 0:
            cur += 1
            mx = max(mx, cur)
        else:
            cur = 0
    return mx


def _metrics_for_cost(path_res: dict[str, Any], *, cost_r: float) -> dict[str, Any]:
    gross = path_res["gross_r_vals"]
    n = len(gross)
    if n == 0:
        return {
            "n_trades": 0,
            "win_rate": None,
            "gross_avg_r": None,
            "net_avg_r": None,
            "profit_factor": None,
            "max_drawdown": None,
            "max_losing_streak": 0,
            "timeout_rate": None,
            "tp_count": 0,
            "sl_count": 0,
            "be_exit_count": 0,
            "locked_profit_exit_count": 0,
            "timeout_count": 0,
        }

    net = [x - cost_r for x in gross]
    wins = sum(1 for x in net if x > 0)

    curve: list[float] = []
    c = 0.0
    for x in net:
        c += x
        curve.append(c)

    return {
        "n_trades": n,
        "win_rate": wins / float(n),
        "gross_avg_r": float(mean(gross)),
        "net_avg_r": float(mean(net)),
        "profit_factor": _profit_factor(net),
        "max_drawdown": _max_drawdown(curve),
        "max_losing_streak": _max_losing_streak(net),
        "timeout_rate": path_res["timeout_count"] / float(n),
        "tp_count": path_res["tp_count"],
        "sl_count": path_res["sl_count"],
        "be_exit_count": path_res["be_count"],
        "locked_profit_exit_count": path_res["lock_count"],
        "timeout_count": path_res["timeout_count"],
    }


def _run_config_cost_sweep(
    rows: list[SignalRow],
    *,
    cfg: StopMgmtConfig,
    ts_vec: list[float],
    book_points: list[BookTsPoint],
    forward_horizon_sec: float,
    regime_q: dict[str, float],
) -> dict[str, Any]:
    path_res = _simulate_config_paths(
        rows,
        cfg,
        ts_vec=ts_vec,
        book_points=book_points,
        forward_horizon_sec=forward_horizon_sec,
        regime_q=regime_q,
    )
    by_cost: dict[str, Any] = {}
    for c in COST_LEVELS_R:
        by_cost[f"{c:.2f}"] = _metrics_for_cost(path_res, cost_r=c)
    return {
        "cost_levels_r": by_cost,
        "skipped_count": path_res["skipped_count"],
    }


def build_stop_management_simulation(
    setups_path: Path,
    book_path: Path,
    matrix_path: Path,
    output_path: Path,
    *,
    forward_horizon_sec: float,
    high_event_only: bool,
) -> dict[str, Any]:
    ts_vec, book_points = _load_book_path(book_path)
    matrix_idx = _matrix_index_by_ts(matrix_path)
    rows = _build_baseline_rows(setups_path, matrix_idx)

    # Optional event-rate filter remains available but defaults OFF.
    if high_event_only:
        ev = sorted(v.event_rate_hz for v in rows if v.event_rate_hz is not None and not math.isnan(v.event_rate_hz))
        thr = _quantile(ev, 0.75) if ev else float("nan")
        rows = [r for r in rows if r.event_rate_hz is not None and not math.isnan(r.event_rate_hz) and r.event_rate_hz > thr]
        event_info: dict[str, Any] = {"enabled": True, "threshold": None if math.isnan(thr) else thr}
    else:
        event_info = {"enabled": False, "threshold": None}

    regime_q = _regime_quantiles(rows)

    configs = [
        StopMgmtConfig(name="fixed_baseline_tp2_sl1", tp_r=2.0, updates=()),
        StopMgmtConfig(name="breakeven_at_plus_1r", tp_r=2.0, updates=(StopUpdateRule(1.0, 0.0),)),
        StopMgmtConfig(
            name="lock_partial_profit",
            tp_r=2.0,
            updates=(StopUpdateRule(1.0, 0.25), StopUpdateRule(1.5, 0.5)),
        ),
        StopMgmtConfig(name="regime_based_stop_management", tp_r=None, regime_based=True),
    ]

    out_cfg: dict[str, Any] = {}
    for cfg in configs:
        out_cfg[cfg.name] = _run_config_cost_sweep(
            rows,
            cfg=cfg,
            ts_vec=ts_vec,
            book_points=book_points,
            forward_horizon_sec=forward_horizon_sec,
            regime_q=regime_q,
        )

    report: dict[str, Any] = {
        "simulation_only": True,
        "signal_rule": {
            "anomaly_direction": "bearish",
            "min_anomaly_percentile": 0.99,
            "trade_direction": "short",
            "high_event_only": high_event_only,
            "high_event_filter_info": event_info,
        },
        "cost_levels_r": list(COST_LEVELS_R),
        "forward_horizon_sec": forward_horizon_sec,
        "regime_definition": {
            "features": ["trade_count_k", "l2_event_rate_hz", "mid_vol_w", "spread"],
            "calm": "spread, mid_vol_w, l2_event_rate_hz, trade_count_k all <= ~33rd percentile",
            "normal": "neither calm nor volatile/chaotic",
            "volatile": "mid_vol_w or l2_event_rate_hz or trade_count_k >= ~66th percentile",
            "chaotic": "spread or mid_vol_w or l2_event_rate_hz >= ~90th percentile (skip trade)",
        },
        "sample_sizes": {
            "n_signals_after_filters": len(rows),
            "n_book_timestamps": len(ts_vec),
        },
        "configs": out_cfg,
        "notes": [
            "Path-level sequencing uses grouped timestamp min/max/last from book_state.csv.",
            "Same-timestamp TP/SL ambiguity resolved conservatively in favor of SL.",
            "No signal optimization: only stop-management variants compared against baseline.",
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    logger.info("wrote %s signals=%s", output_path, len(rows))
    return report


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Stop-management simulation on fixed bearish anomaly signal")
    p.add_argument("--setups", type=Path, default=DEFAULT_SETUPS)
    p.add_argument("--book-state", type=Path, default=DEFAULT_BOOK)
    p.add_argument("--anomaly-matrix", type=Path, default=DEFAULT_MATRIX)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--forward-horizon-sec", type=float, default=DEFAULT_FORWARD_HORIZON_SEC)
    p.add_argument("--high-event-only", action="store_true")
    args = p.parse_args(argv)

    if not args.setups.is_file():
        logger.error("missing setups file: %s", args.setups)
        return 1
    if not args.book_state.is_file():
        logger.error("missing book_state file: %s", args.book_state)
        return 1
    if args.forward_horizon_sec <= 0:
        logger.error("--forward-horizon-sec must be > 0")
        return 1

    build_stop_management_simulation(
        args.setups,
        args.book_state,
        args.anomaly_matrix,
        args.output,
        forward_horizon_sec=args.forward_horizon_sec,
        high_event_only=args.high_event_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------
# Realistic prop execution layer
# ---------------------------

from datetime import datetime, timezone


def _session_label(ts: float) -> str:
    h = datetime.fromtimestamp(ts, tz=timezone.utc).hour
    if 0 <= h < 8:
        return "asia"
    if 8 <= h < 16:
        return "europe"
    return "us"


def _day_key(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def _cluster_deduplicate(
    rows: list[SignalRow],
    *,
    cluster_window_sec: float,
    choose_by: str,
) -> tuple[list[SignalRow], int]:
    if not rows:
        return [], 0
    key_mode = choose_by.strip().lower()
    if key_mode not in {"highest_anomaly_percentile", "highest_anomaly_score"}:
        raise ValueError("choose_by must be highest_anomaly_percentile or highest_anomaly_score")

    # Extend accessor with fallbacks for backward compatibility.
    def score(r: SignalRow) -> tuple[float, float]:
        p = float(getattr(r, "anomaly_percentile_rank", float("nan")))
        a = float(getattr(r, "anomaly_score", float("nan")))
        if math.isnan(p):
            p = -1.0
        if math.isnan(a):
            a = -1.0
        return p, a

    sorted_rows = sorted(rows, key=lambda r: r.ts)
    out: list[SignalRow] = []
    i = 0
    skipped = 0
    n = len(sorted_rows)
    while i < n:
        cluster = [sorted_rows[i]]
        start = sorted_rows[i].ts
        j = i + 1
        while j < n and (sorted_rows[j].ts - start) <= cluster_window_sec:
            cluster.append(sorted_rows[j])
            j += 1
        if key_mode == "highest_anomaly_percentile":
            keep = max(cluster, key=lambda r: (score(r)[0], score(r)[1], -r.ts))
        else:
            keep = max(cluster, key=lambda r: (score(r)[1], score(r)[0], -r.ts))
        out.append(keep)
        skipped += max(0, len(cluster) - 1)
        i = j
    return out, skipped


@dataclass(frozen=True)
class ThrottleResult:
    executed: list[SignalRow]
    skipped_by_cooldown: int
    skipped_by_daily_limit: int
    skipped_by_session_limit: int
    day_counts: dict[str, int]
    session_counts: dict[str, int]


def _apply_prop_throttles(
    rows: list[SignalRow],
    *,
    cooldown_sec: float,
    max_trades_per_day: int,
    max_trades_per_session: int,
) -> ThrottleResult:
    executed: list[SignalRow] = []
    skipped_cooldown = 0
    skipped_day = 0
    skipped_session = 0
    last_exec_ts: float | None = None
    day_counts: dict[str, int] = {}
    session_counts: dict[str, int] = {}

    for r in sorted(rows, key=lambda x: x.ts):
        if last_exec_ts is not None and (r.ts - last_exec_ts) < cooldown_sec:
            skipped_cooldown += 1
            continue
        dkey = _day_key(r.ts)
        skey = f"{dkey}|{_session_label(r.ts)}"
        dcount = day_counts.get(dkey, 0)
        scount = session_counts.get(skey, 0)
        if dcount >= max_trades_per_day:
            skipped_day += 1
            continue
        if scount >= max_trades_per_session:
            skipped_session += 1
            continue
        executed.append(r)
        last_exec_ts = r.ts
        day_counts[dkey] = dcount + 1
        session_counts[skey] = scount + 1

    return ThrottleResult(
        executed=executed,
        skipped_by_cooldown=skipped_cooldown,
        skipped_by_daily_limit=skipped_day,
        skipped_by_session_limit=skipped_session,
        day_counts=day_counts,
        session_counts=session_counts,
    )


def _summary_counts(values: dict[str, int]) -> dict[str, Any]:
    if not values:
        return {"avg": 0.0, "max": 0, "details": {}}
    nums = list(values.values())
    return {
        "avg": float(mean(nums)),
        "max": int(max(nums)),
        "details": values,
    }


def _metrics_from_path_res(path_res: dict[str, Any]) -> dict[str, Any]:
    gross = path_res["gross_r_vals"]
    n = len(gross)
    if n == 0:
        return {
            "executed_trade_count": 0,
            "avg_r": None,
            "profit_factor": None,
            "max_drawdown": None,
            "max_losing_streak": 0,
            "win_rate": None,
            "timeout_rate": None,
            "tp_count": 0,
            "sl_count": 0,
            "be_exit_count": 0,
            "locked_profit_exit_count": 0,
            "timeout_count": 0,
        }
    wins = sum(1 for x in gross if x > 0)
    curve: list[float] = []
    c = 0.0
    for x in gross:
        c += x
        curve.append(c)
    return {
        "executed_trade_count": n,
        "avg_r": float(mean(gross)),
        "profit_factor": _profit_factor(gross),
        "max_drawdown": _max_drawdown(curve),
        "max_losing_streak": _max_losing_streak(gross),
        "win_rate": wins / float(n),
        "timeout_rate": path_res["timeout_count"] / float(n),
        "tp_count": path_res["tp_count"],
        "sl_count": path_res["sl_count"],
        "be_exit_count": path_res["be_count"],
        "locked_profit_exit_count": path_res["lock_count"],
        "timeout_count": path_res["timeout_count"],
    }


def build_realistic_prop_simulation(
    setups_path: Path,
    book_path: Path,
    matrix_path: Path,
    output_path: Path,
    *,
    forward_horizon_sec: float = DEFAULT_FORWARD_HORIZON_SEC,
    max_trades_per_day: int = 20,
    max_trades_per_session: int = 5,
    anomaly_cluster_window_sec: float = 30.0,
    cooldown_tests_sec: tuple[float, ...] = (30.0, 60.0, 120.0, 300.0),
    choose_modes: tuple[str, ...] = ("highest_anomaly_percentile", "highest_anomaly_score"),
) -> dict[str, Any]:
    ts_vec, book_points = _load_book_path(book_path)
    matrix_idx = _matrix_index_by_ts(matrix_path)
    raw_rows = _build_baseline_rows(setups_path, matrix_idx)  # bearish >=0.99 already
    regime_q = _regime_quantiles(raw_rows)

    configs = [
        StopMgmtConfig(name="fixed_baseline_tp2_sl1", tp_r=2.0, updates=()),
        StopMgmtConfig(name="regime_based_stop_management", tp_r=None, regime_based=True),
    ]

    scenarios: dict[str, Any] = {}
    for choose_by in choose_modes:
        dedup_rows, skipped_cluster = _cluster_deduplicate(
            raw_rows,
            cluster_window_sec=anomaly_cluster_window_sec,
            choose_by=choose_by,
        )
        for cd in cooldown_tests_sec:
            throttle = _apply_prop_throttles(
                dedup_rows,
                cooldown_sec=cd,
                max_trades_per_day=max_trades_per_day,
                max_trades_per_session=max_trades_per_session,
            )
            key = f"choose_{choose_by}__cooldown_{int(cd)}s"
            cfg_block: dict[str, Any] = {}
            for cfg in configs:
                path_res = _simulate_config_paths(
                    throttle.executed,
                    cfg,
                    ts_vec=ts_vec,
                    book_points=book_points,
                    forward_horizon_sec=forward_horizon_sec,
                    regime_q=regime_q,
                )
                met = _metrics_from_path_res(path_res)
                met.update(
                    {
                        "raw_signal_count": len(raw_rows),
                        "deduplicated_signal_count": len(dedup_rows),
                        "executed_trade_count": len(throttle.executed),
                        "skipped_by_cooldown": throttle.skipped_by_cooldown,
                        "skipped_by_daily_limit": throttle.skipped_by_daily_limit,
                        "skipped_by_session_limit": throttle.skipped_by_session_limit,
                        "skipped_by_cluster_dedup": skipped_cluster,
                        "trades_per_day": _summary_counts(throttle.day_counts),
                        "trades_per_session": _summary_counts(throttle.session_counts),
                        "skipped_by_regime_chaotic": path_res.get("skipped_count", 0),
                    }
                )
                cfg_block[cfg.name] = met
            scenarios[key] = {
                "choose_cluster_signal_by": choose_by,
                "cooldown_sec": cd,
                "max_trades_per_day": max_trades_per_day,
                "max_trades_per_session": max_trades_per_session,
                "anomaly_cluster_window_sec": anomaly_cluster_window_sec,
                "configs": cfg_block,
            }

    report: dict[str, Any] = {
        "simulation_only": True,
        "signal_rule": {
            "anomaly_direction": "bearish",
            "min_anomaly_percentile": 0.99,
            "trade_direction": "short",
            "tp_r": 2.0,
            "sl_r": 1.0,
        },
        "execution_realism_layer": {
            "cooldown_tests_sec": list(cooldown_tests_sec),
            "max_trades_per_day": max_trades_per_day,
            "max_trades_per_session": max_trades_per_session,
            "anomaly_cluster_window_sec": anomaly_cluster_window_sec,
            "choose_cluster_signal_by_tests": list(choose_modes),
        },
        "scenarios": scenarios,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    logger.info("wrote %s raw_signals=%s", output_path, len(raw_rows))
    return report
