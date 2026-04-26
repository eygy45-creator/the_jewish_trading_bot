"""
Label extreme anomaly events by forward mid-price path (time-based horizon).

Label definition (first-touch within horizon):
- +1: profit-up barrier hit before adverse-down barrier
- -1: profit-down barrier hit before adverse-up barrier
-  0: neither barrier hit within horizon
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from tjtb.features.build_features import parse_ts_to_unix

logger = logging.getLogger(__name__)

DEFAULT_ANOMALY = Path("data/parsed/anomaly_matrix.csv")
DEFAULT_BOOK = Path("data/parsed/book_state.csv")
DEFAULT_OUTPUT = Path("data/parsed/anomaly_setups.csv")
READ_BUFFER = 1024 * 1024
WRITE_BUFFER = 1024 * 1024


@dataclass
class SetupLabel:
    label: int
    max_favorable_move: float
    max_adverse_move: float
    time_to_outcome: float


def _safe_float(value: str | None) -> float:
    if value is None:
        return float("nan")
    s = value.strip()
    if not s:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _fmt_float(v: float) -> str:
    if math.isnan(v):
        return ""
    return f"{v:.12g}"


def _load_book_rows(path: Path) -> tuple[list[str], list[float], list[float], list[float]]:
    ts_text: list[str] = []
    ts_unix: list[float] = []
    mids: list[float] = []
    with path.open("r", encoding="utf-8", newline="", buffering=READ_BUFFER) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("book_state has no header")
        if "ts" not in reader.fieldnames or "mid_price" not in reader.fieldnames:
            raise ValueError("book_state must include `ts` and `mid_price`")
        for row in reader:
            ts_s = (row.get("ts") or "").strip()
            if not ts_s:
                continue
            try:
                ts = parse_ts_to_unix(ts_s)
            except ValueError:
                continue
            mid = _safe_float(row.get("mid_price"))
            if math.isnan(mid):
                continue
            ts_text.append(ts_s)
            ts_unix.append(ts)
            mids.append(mid)
    return ts_text, ts_unix, mids


def _label_event(
    *,
    idx: int,
    ts_unix: list[float],
    mids: list[float],
    direction: str,
    forward_horizon_sec: float,
    profit_threshold: float,
    adverse_threshold: float,
) -> SetupLabel:
    t0 = ts_unix[idx]
    p0 = mids[idx]
    t_end = t0 + forward_horizon_sec

    max_up = 0.0
    max_down = 0.0
    first_up_hit_time: float | None = None
    first_down_hit_time: float | None = None

    j = idx + 1
    n = len(ts_unix)
    while j < n and ts_unix[j] <= t_end:
        dp = mids[j] - p0
        if dp > max_up:
            max_up = dp
        if -dp > max_down:
            max_down = -dp
        if first_up_hit_time is None and dp >= profit_threshold:
            first_up_hit_time = ts_unix[j] - t0
        if first_down_hit_time is None and (-dp) >= profit_threshold:
            first_down_hit_time = ts_unix[j] - t0
        j += 1

    neutral = SetupLabel(label=0, max_favorable_move=0.0, max_adverse_move=0.0, time_to_outcome=forward_horizon_sec)

    if direction == "bullish":
        if first_up_hit_time is not None and (first_down_hit_time is None or first_up_hit_time < first_down_hit_time):
            return SetupLabel(label=1, max_favorable_move=max_up, max_adverse_move=max_down, time_to_outcome=first_up_hit_time)
        if max_down >= adverse_threshold:
            t_adverse = first_down_hit_time if first_down_hit_time is not None else forward_horizon_sec
            return SetupLabel(label=-1, max_favorable_move=max_up, max_adverse_move=max_down, time_to_outcome=t_adverse)
        neutral.max_favorable_move = max_up
        neutral.max_adverse_move = max_down
        return neutral

    if direction == "bearish":
        if first_down_hit_time is not None and (first_up_hit_time is None or first_down_hit_time < first_up_hit_time):
            return SetupLabel(label=1, max_favorable_move=max_down, max_adverse_move=max_up, time_to_outcome=first_down_hit_time)
        if max_up >= adverse_threshold:
            t_adverse = first_up_hit_time if first_up_hit_time is not None else forward_horizon_sec
            return SetupLabel(label=-1, max_favorable_move=max_down, max_adverse_move=max_up, time_to_outcome=t_adverse)
        neutral.max_favorable_move = max_down
        neutral.max_adverse_move = max_up
        return neutral

    # Neutral direction: symmetric first-touch race.
    if first_up_hit_time is not None and (first_down_hit_time is None or first_up_hit_time < first_down_hit_time):
        return SetupLabel(label=1, max_favorable_move=max_up, max_adverse_move=max_down, time_to_outcome=first_up_hit_time)
    if first_down_hit_time is not None and (first_up_hit_time is None or first_down_hit_time < first_up_hit_time):
        return SetupLabel(label=-1, max_favorable_move=max_down, max_adverse_move=max_up, time_to_outcome=first_down_hit_time)
    neutral.max_favorable_move = max(max_up, max_down)
    neutral.max_adverse_move = min(max_up, max_down)
    return neutral


def label_anomaly_setups(
    anomaly_path: Path,
    book_state_path: Path,
    output_path: Path,
    *,
    extreme_column: str,
    forward_horizon_sec: float,
    profit_threshold: float,
    adverse_threshold: float,
    tick_size: float | None,
) -> int:
    if forward_horizon_sec <= 0:
        raise ValueError("forward_horizon_sec must be > 0")
    if profit_threshold <= 0:
        raise ValueError("profit_threshold must be > 0")
    if adverse_threshold <= 0:
        raise ValueError("adverse_threshold must be > 0")

    if tick_size is not None:
        if tick_size <= 0:
            raise ValueError("tick_size must be > 0 when provided")
        profit = profit_threshold * tick_size
        adverse = adverse_threshold * tick_size
    else:
        profit = profit_threshold
        adverse = adverse_threshold

    book_ts_text, book_ts_unix, mids = _load_book_rows(book_state_path)
    if not book_ts_unix:
        raise ValueError("book_state has no usable rows")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_out = 0
    with anomaly_path.open("r", encoding="utf-8", newline="", buffering=READ_BUFFER) as ain, output_path.open(
        "w", encoding="utf-8", newline="", buffering=WRITE_BUFFER
    ) as out:
        ar = csv.DictReader(ain)
        if ar.fieldnames is None:
            raise ValueError("anomaly_matrix has no header")
        if extreme_column not in ar.fieldnames:
            raise ValueError(f"anomaly_matrix missing extreme column: {extreme_column}")
        if "anomaly_score" not in ar.fieldnames or "anomaly_percentile_rank" not in ar.fieldnames:
            raise ValueError("anomaly_matrix missing required anomaly score columns")

        out_cols = [
            "ts",
            "anomaly_score",
            "anomaly_percentile_rank",
            "anomaly_direction",
            "bullish_anomaly_score",
            "bearish_anomaly_score",
            "mid_price",
            "label",
            "max_favorable_move",
            "max_adverse_move",
            "time_to_outcome",
        ]
        w = csv.DictWriter(out, fieldnames=out_cols)
        w.writeheader()

        idx = 0
        for row in ar:
            if idx >= len(book_ts_unix):
                break
            extreme_flag = (row.get(extreme_column) or "").strip().lower()
            is_extreme = extreme_flag in {"1", "true", "t", "yes", "y"}
            if not is_extreme:
                idx += 1
                continue

            direction = (row.get("anomaly_direction") or "neutral").strip().lower()
            if direction not in {"bullish", "bearish", "neutral"}:
                direction = "neutral"

            lbl = _label_event(
                idx=idx,
                ts_unix=book_ts_unix,
                mids=mids,
                direction=direction,
                forward_horizon_sec=forward_horizon_sec,
                profit_threshold=profit,
                adverse_threshold=adverse,
            )
            w.writerow(
                {
                    "ts": book_ts_text[idx],
                    "anomaly_score": row.get("anomaly_score", ""),
                    "anomaly_percentile_rank": row.get("anomaly_percentile_rank", ""),
                    "anomaly_direction": direction,
                    "bullish_anomaly_score": row.get("bullish_anomaly_score", ""),
                    "bearish_anomaly_score": row.get("bearish_anomaly_score", ""),
                    "mid_price": _fmt_float(mids[idx]),
                    "label": str(lbl.label),
                    "max_favorable_move": _fmt_float(lbl.max_favorable_move),
                    "max_adverse_move": _fmt_float(lbl.max_adverse_move),
                    "time_to_outcome": _fmt_float(lbl.time_to_outcome),
                }
            )
            n_out += 1
            idx += 1

        # Consume remaining anomaly rows if book rows were longer/shorter mismatch already handled by break.

    logger.info(
        "labeled anomaly setups rows=%s extreme_col=%s horizon_sec=%s profit=%s adverse=%s tick_size=%s -> %s",
        n_out,
        extreme_column,
        forward_horizon_sec,
        profit_threshold,
        adverse_threshold,
        tick_size,
        output_path,
    )
    return n_out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    p = argparse.ArgumentParser(description="Label extreme anomaly setups via forward first-touch outcomes.")
    p.add_argument("--anomaly-matrix", type=Path, default=DEFAULT_ANOMALY)
    p.add_argument("--book-state", type=Path, default=DEFAULT_BOOK)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--extreme-column", type=str, default="is_extreme_10pct")
    p.add_argument("--forward-horizon-sec", type=float, default=2.0)
    p.add_argument("--profit-threshold", type=float, required=True)
    p.add_argument("--adverse-threshold", type=float, required=True)
    p.add_argument("--tick-size", type=float, default=None)
    args = p.parse_args(argv)

    if not args.anomaly_matrix.is_file():
        logger.error("missing anomaly matrix: %s", args.anomaly_matrix)
        return 1
    if not args.book_state.is_file():
        logger.error("missing book_state: %s", args.book_state)
        return 1

    label_anomaly_setups(
        args.anomaly_matrix,
        args.book_state,
        args.output,
        extreme_column=args.extreme_column,
        forward_horizon_sec=args.forward_horizon_sec,
        profit_threshold=args.profit_threshold,
        adverse_threshold=args.adverse_threshold,
        tick_size=args.tick_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
