"""
Standalone realistic prop-firm throttling simulation.

Runs the fixed candidate strategy under realistic execution limits:
- bearish anomaly direction
- anomaly percentile >= 0.99
- short direction
- TP=2R, SL=1R
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from tjtb.features.build_features import parse_ts_to_unix

logger = logging.getLogger(__name__)

DEFAULT_SETUPS = Path("data/parsed/anomaly_setups.csv")
DEFAULT_MATRIX = Path("data/parsed/anomaly_matrix.csv")
DEFAULT_BOOK = Path("data/parsed/book_state.csv")
DEFAULT_OUTPUT = Path("reports/realistic_prop_simulation.json")

READ_BUFFER = 1024 * 1024
EPS = 1e-9
FORWARD_HORIZON_SEC = 2.0
CLUSTER_WINDOW_SEC = 30.0
COOLDOWN_SWEEP_SEC: tuple[float, ...] = (30.0, 60.0, 120.0, 300.0)
MAX_TRADES_PER_DAY = 20
MAX_TRADES_PER_SESSION = 5


@dataclass(frozen=True)
class Signal:
    ts_text: str
    ts_unix: float
    entry_price: float
    anomaly_percentile_rank: float


@dataclass(frozen=True)
class BookTsPoint:
    ts_unix: float
    mid_min: float
    mid_max: float
    mid_last: float


@dataclass(frozen=True)
class TradeOutcome:
    outcome: str  # tp | sl | timeout
    r_value: float


@dataclass(frozen=True)
class ThrottleResult:
    executed: list[Signal]
    skipped_by_cooldown: int
    skipped_by_daily_limit: int
    skipped_by_session_limit: int
    day_counts: dict[str, int]
    session_counts: dict[str, int]


def _safe_float(s: str | None) -> float | None:
    if s is None:
        return None
    t = s.strip()
    if not t:
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="", buffering=READ_BUFFER) as f:
        return list(csv.DictReader(f))


def _load_signals(setups_path: Path) -> list[Signal]:
    out: list[Signal] = []
    for row in _load_csv(setups_path):
        direction = (row.get("anomaly_direction") or "").strip().lower()
        if direction != "bearish":
            continue
        pctl = _safe_float(row.get("anomaly_percentile_rank"))
        entry = _safe_float(row.get("mid_price"))
        ts_text = (row.get("ts") or "").strip()
        if pctl is None or entry is None or not ts_text:
            continue
        if pctl < 0.99:
            continue
        try:
            ts_unix = parse_ts_to_unix(ts_text)
        except ValueError:
            continue
        out.append(
            Signal(
                ts_text=ts_text,
                ts_unix=ts_unix,
                entry_price=entry,
                anomaly_percentile_rank=pctl,
            )
        )
    out.sort(key=lambda x: x.ts_unix)
    return out


def _load_book_path(book_path: Path) -> tuple[list[float], list[BookTsPoint]]:
    points: list[BookTsPoint] = []
    cur_ts: float | None = None
    cur_min = 0.0
    cur_max = 0.0
    cur_last = 0.0

    with book_path.open("r", encoding="utf-8", newline="", buffering=READ_BUFFER) as f:
        r = csv.DictReader(f)
        for row in r:
            ts_s = (row.get("ts") or "").strip()
            mid = _safe_float(row.get("mid_price"))
            if not ts_s or mid is None or math.isnan(mid):
                continue
            try:
                ts = parse_ts_to_unix(ts_s)
            except ValueError:
                continue

            if cur_ts is None:
                cur_ts = ts
                cur_min = mid
                cur_max = mid
                cur_last = mid
                continue

            if ts == cur_ts:
                cur_min = min(cur_min, mid)
                cur_max = max(cur_max, mid)
                cur_last = mid
                continue

            points.append(BookTsPoint(ts_unix=cur_ts, mid_min=cur_min, mid_max=cur_max, mid_last=cur_last))
            cur_ts = ts
            cur_min = mid
            cur_max = mid
            cur_last = mid

    if cur_ts is not None:
        points.append(BookTsPoint(ts_unix=cur_ts, mid_min=cur_min, mid_max=cur_max, mid_last=cur_last))

    return [p.ts_unix for p in points], points


def _horizon_slice(ts_vec: list[float], points: list[BookTsPoint], *, ts: float, horizon_sec: float) -> list[BookTsPoint]:
    lo = bisect.bisect_left(ts_vec, ts)
    hi = bisect.bisect_right(ts_vec, ts + horizon_sec)
    return points[lo:hi]


def _simulate_short_trade(entry: float, path: list[BookTsPoint]) -> TradeOutcome:
    """
    Path-level first touch for fixed TP=2R SL=1R short.
    """
    if not path:
        return TradeOutcome(outcome="timeout", r_value=0.0)

    tp_price = entry - 2.0
    sl_price = entry + 1.0

    for p in path:
        hit_tp = p.mid_min <= tp_price
        hit_sl = p.mid_max >= sl_price
        if hit_tp and hit_sl:
            return TradeOutcome(outcome="sl", r_value=-1.0)
        if hit_sl:
            return TradeOutcome(outcome="sl", r_value=-1.0)
        if hit_tp:
            return TradeOutcome(outcome="tp", r_value=2.0)

    timeout_r = (entry - path[-1].mid_last) / max(1.0, EPS)
    return TradeOutcome(outcome="timeout", r_value=timeout_r)


def _cluster_deduplicate(signals: list[Signal], *, window_sec: float) -> tuple[list[Signal], int]:
    if not signals:
        return [], 0
    deduped: list[Signal] = []
    skipped = 0
    i = 0
    n = len(signals)
    while i < n:
        cluster = [signals[i]]
        t0 = signals[i].ts_unix
        j = i + 1
        while j < n and (signals[j].ts_unix - t0) <= window_sec:
            cluster.append(signals[j])
            j += 1
        keep = max(cluster, key=lambda s: (s.anomaly_percentile_rank, -s.ts_unix))
        deduped.append(keep)
        skipped += max(0, len(cluster) - 1)
        i = j
    return deduped, skipped


def _session_label(ts_unix: float) -> str:
    h = datetime.fromtimestamp(ts_unix, tz=timezone.utc).hour
    if 0 <= h < 6:
        return "asia"
    if 6 <= h < 12:
        return "london"
    if 12 <= h < 20:
        return "ny"
    return "off_hours"


def _day_key(ts_unix: float) -> str:
    return datetime.fromtimestamp(ts_unix, tz=timezone.utc).strftime("%Y-%m-%d")


def _apply_throttles(
    signals: list[Signal],
    *,
    cooldown_sec: float,
    max_trades_per_day: int,
    max_trades_per_session: int,
) -> ThrottleResult:
    executed: list[Signal] = []
    skipped_cd = 0
    skipped_day = 0
    skipped_session = 0
    day_counts: dict[str, int] = {}
    session_counts: dict[str, int] = {}
    last_exec_ts: float | None = None

    for s in signals:
        if last_exec_ts is not None and (s.ts_unix - last_exec_ts) < cooldown_sec:
            skipped_cd += 1
            continue
        d = _day_key(s.ts_unix)
        sess = f"{d}|{_session_label(s.ts_unix)}"
        dc = day_counts.get(d, 0)
        sc = session_counts.get(sess, 0)
        if dc >= max_trades_per_day:
            skipped_day += 1
            continue
        if sc >= max_trades_per_session:
            skipped_session += 1
            continue
        executed.append(s)
        last_exec_ts = s.ts_unix
        day_counts[d] = dc + 1
        session_counts[sess] = sc + 1

    return ThrottleResult(
        executed=executed,
        skipped_by_cooldown=skipped_cd,
        skipped_by_daily_limit=skipped_day,
        skipped_by_session_limit=skipped_session,
        day_counts=day_counts,
        session_counts=session_counts,
    )


def _profit_factor(r_vals: list[float]) -> float | None:
    gp = sum(x for x in r_vals if x > 0)
    gl = -sum(x for x in r_vals if x < 0)
    if gl <= 0:
        return None
    return gp / gl


def _max_drawdown(r_vals: list[float]) -> float:
    if not r_vals:
        return 0.0
    eq: list[float] = []
    c = 0.0
    for x in r_vals:
        c += x
        eq.append(c)
    peak = eq[0]
    mdd = 0.0
    for x in eq:
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


def _summary_counts(counts: dict[str, int]) -> dict[str, Any]:
    if not counts:
        return {"avg": 0.0, "max": 0, "details": {}}
    vals = list(counts.values())
    return {
        "avg": float(mean(vals)),
        "max": int(max(vals)),
        "details": counts,
    }


def build_realistic_prop_simulation(
    setups_path: Path,
    matrix_path: Path,
    book_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    _ = _load_csv(matrix_path) if matrix_path.is_file() else []
    raw_signals = _load_signals(setups_path)
    ts_vec, book_points = _load_book_path(book_path)

    deduped, skipped_cluster = _cluster_deduplicate(raw_signals, window_sec=CLUSTER_WINDOW_SEC)

    scenarios: dict[str, Any] = {}
    for cooldown in COOLDOWN_SWEEP_SEC:
        throttled = _apply_throttles(
            deduped,
            cooldown_sec=cooldown,
            max_trades_per_day=MAX_TRADES_PER_DAY,
            max_trades_per_session=MAX_TRADES_PER_SESSION,
        )
        executed = throttled.executed
        r_vals: list[float] = []
        tp = 0
        sl = 0
        timeout = 0
        for s in executed:
            path = _horizon_slice(ts_vec, book_points, ts=s.ts_unix, horizon_sec=FORWARD_HORIZON_SEC)
            out = _simulate_short_trade(s.entry_price, path)
            r_vals.append(out.r_value)
            if out.outcome == "tp":
                tp += 1
            elif out.outcome == "sl":
                sl += 1
            else:
                timeout += 1

        n = len(r_vals)
        if n == 0:
            win_rate = None
            avg_r = None
            pf = None
            mdd = None
            mls = 0
            timeout_rate = None
        else:
            win_rate = sum(1 for x in r_vals if x > 0) / float(n)
            avg_r = float(mean(r_vals))
            pf = _profit_factor(r_vals)
            mdd = _max_drawdown(r_vals)
            mls = _max_losing_streak(r_vals)
            timeout_rate = timeout / float(n)

        scenarios[f"cooldown_{int(cooldown)}s"] = {
            "raw_signal_count": len(raw_signals),
            "deduplicated_signal_count": len(deduped),
            "executed_trade_count": len(executed),
            "skipped_by_cluster_dedup": skipped_cluster,
            "skipped_by_cooldown": throttled.skipped_by_cooldown,
            "skipped_by_daily_limit": throttled.skipped_by_daily_limit,
            "skipped_by_session_limit": throttled.skipped_by_session_limit,
            "trades_per_day": _summary_counts(throttled.day_counts),
            "trades_per_session": _summary_counts(throttled.session_counts),
            "win_rate": win_rate,
            "avg_r": avg_r,
            "profit_factor": pf,
            "max_drawdown": mdd,
            "max_losing_streak": mls,
            "timeout_rate": timeout_rate,
            "tp_count": tp,
            "sl_count": sl,
            "timeout_count": timeout,
        }

    report: dict[str, Any] = {
        "simulation_only": True,
        "strategy_rule": {
            "anomaly_direction": "bearish",
            "min_anomaly_percentile": 0.99,
            "direction": "short",
            "tp_r": 2.0,
            "sl_r": 1.0,
            "cluster_window_sec": CLUSTER_WINDOW_SEC,
            "choose_cluster_signal_by": "highest anomaly_percentile",
            "max_trades_per_day": MAX_TRADES_PER_DAY,
            "max_trades_per_session": MAX_TRADES_PER_SESSION,
            "sessions": ["asia", "london", "ny", "off_hours"],
        },
        "cooldown_sweep_sec": [int(x) for x in COOLDOWN_SWEEP_SEC],
        "scenarios": scenarios,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    logger.info("wrote reports/realistic_prop_simulation.json")
    return report


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Realistic prop-firm constrained simulation")
    p.add_argument("--setups", type=Path, default=DEFAULT_SETUPS)
    p.add_argument("--anomaly-matrix", type=Path, default=DEFAULT_MATRIX)
    p.add_argument("--book-state", type=Path, default=DEFAULT_BOOK)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args(argv)

    if not args.setups.is_file():
        logger.error("missing setups file: %s", args.setups)
        return 1
    if not args.book_state.is_file():
        logger.error("missing book_state file: %s", args.book_state)
        return 1

    build_realistic_prop_simulation(args.setups, args.anomaly_matrix, args.book_state, args.output)
    return 0


__all__ = ["build_realistic_prop_simulation", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
