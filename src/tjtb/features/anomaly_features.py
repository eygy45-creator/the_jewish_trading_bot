"""
Build statistical anomaly features from microstructure feature matrix.

Research-only output:
- past-only rolling Z-scores on time windows
- trailing percentile rank for anomaly score
- extreme-tail candidate flags (top 10% and top 5%)
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from bisect import bisect_left, bisect_right, insort
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from tjtb.features.build_features import parse_ts_to_unix

logger = logging.getLogger(__name__)

DEFAULT_INPUT = Path("data/parsed/feature_matrix.csv")
DEFAULT_OUTPUT = Path("data/parsed/anomaly_matrix.csv")
READ_BUFFER = 1024 * 1024
WRITE_BUFFER = 1024 * 1024
MIN_STD = 1e-12

Z_COLUMN_MAP: tuple[tuple[str, str], ...] = (
    ("tob_imbalance", "z_tob_imbalance"),
    ("microprice_dev", "z_microprice_dev"),
    ("ofi_k", "z_ofi_k"),
    ("trade_count_k", "z_trade_count_k"),
    ("del_bid_w", "z_del_bid_w"),
    ("del_ask_w", "z_del_ask_w"),
    ("l2_update_count_w", "z_l2_update_count_w"),
    ("signed_book_pressure_w", "z_signed_book_pressure_w"),
    ("l2_event_rate_hz", "z_l2_event_rate_hz"),
)

DIRECTIONAL_Z_COLUMNS = (
    "z_tob_imbalance",
    "z_microprice_dev",
    "z_ofi_k",
    "z_signed_book_pressure_w",
)

ANOMALY_COMPONENTS = (
    "z_tob_imbalance",
    "z_microprice_dev",
    "z_ofi_k",
    "z_signed_book_pressure_w",
    "z_l2_event_rate_hz",
)

OUTPUT_ADDED_COLUMNS: tuple[str, ...] = (
    "z_tob_imbalance",
    "z_microprice_dev",
    "z_ofi_k",
    "z_trade_count_k",
    "z_del_bid_w",
    "z_del_ask_w",
    "z_l2_update_count_w",
    "z_signed_book_pressure_w",
    "z_l2_event_rate_hz",
    "abs_z_max",
    "anomaly_score",
    "bullish_anomaly_score",
    "bearish_anomaly_score",
    "anomaly_direction",
    "anomaly_percentile_rank",
    "is_extreme_10pct",
    "is_extreme_5pct",
)


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


@dataclass
class RollingWindowStats:
    window_sec: float
    values: deque[tuple[float, float]] = field(default_factory=deque)
    n: int = 0
    sum_v: float = 0.0
    sumsq_v: float = 0.0

    def _expire(self, ts_now: float) -> None:
        cutoff = ts_now - self.window_sec
        while self.values and self.values[0][0] < cutoff:
            _, old_v = self.values.popleft()
            self.n -= 1
            self.sum_v -= old_v
            self.sumsq_v -= old_v * old_v

    def stats_before_update(self, ts_now: float) -> tuple[float, float, int]:
        self._expire(ts_now)
        if self.n <= 0:
            return float("nan"), float("nan"), 0
        mean = self.sum_v / float(self.n)
        if self.n <= 1:
            return mean, float("nan"), self.n
        var = max((self.sumsq_v - self.sum_v * self.sum_v / float(self.n)) / float(self.n - 1), 0.0)
        std = math.sqrt(var)
        return mean, std, self.n

    def add(self, ts: float, value: float) -> None:
        if math.isnan(value):
            return
        self.values.append((ts, value))
        self.n += 1
        self.sum_v += value
        self.sumsq_v += value * value


@dataclass
class TrailingPercentile:
    window_sec: float
    queue: deque[tuple[float, float]] = field(default_factory=deque)
    sorted_values: list[float] = field(default_factory=list)

    def _expire(self, ts_now: float) -> None:
        cutoff = ts_now - self.window_sec
        while self.queue and self.queue[0][0] < cutoff:
            _, old_v = self.queue.popleft()
            j = bisect_left(self.sorted_values, old_v)
            if j < len(self.sorted_values) and self.sorted_values[j] == old_v:
                self.sorted_values.pop(j)

    def rank_before_update(self, ts_now: float, value: float) -> tuple[float, int]:
        self._expire(ts_now)
        n = len(self.sorted_values)
        if n == 0 or math.isnan(value):
            return float("nan"), n
        right = bisect_right(self.sorted_values, value)
        return float(right) / float(n), n

    def add(self, ts: float, value: float) -> None:
        if math.isnan(value):
            return
        self.queue.append((ts, value))
        insort(self.sorted_values, value)


def _directional_scores(z_map: dict[str, float]) -> tuple[float, float]:
    bullish_parts: list[float] = []
    bearish_parts: list[float] = []
    for key in DIRECTIONAL_Z_COLUMNS:
        z = z_map.get(key, float("nan"))
        if math.isnan(z):
            continue
        bullish_parts.append(max(z, 0.0))
        bearish_parts.append(max(-z, 0.0))
    bullish = max(bullish_parts) if bullish_parts else float("nan")
    bearish = max(bearish_parts) if bearish_parts else float("nan")
    return bullish, bearish


def _direction_from_scores(bullish: float, bearish: float) -> str:
    if math.isnan(bullish) and math.isnan(bearish):
        return "neutral"
    b = 0.0 if math.isnan(bullish) else bullish
    s = 0.0 if math.isnan(bearish) else bearish
    if b > s:
        return "bullish"
    if s > b:
        return "bearish"
    return "neutral"


def build_anomaly_matrix(
    input_path: Path,
    output_path: Path,
    *,
    lookback_sec: float,
    calibration_sec: float,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8", newline="", buffering=READ_BUFFER) as inp, output_path.open(
        "w", encoding="utf-8", newline="", buffering=WRITE_BUFFER
    ) as out:
        reader = csv.DictReader(inp)
        if reader.fieldnames is None:
            raise ValueError("feature matrix has no header")

        in_cols = tuple(reader.fieldnames)
        missing = [src for src, _ in Z_COLUMN_MAP if src not in in_cols]
        if missing:
            logger.warning("missing expected feature columns: %s", ", ".join(missing))
        if "ts" not in in_cols:
            raise ValueError("input is missing required `ts` column")

        out_cols = [*in_cols, *OUTPUT_ADDED_COLUMNS]
        writer = csv.DictWriter(out, fieldnames=out_cols)
        writer.writeheader()

        stat_windows = {src: RollingWindowStats(window_sec=lookback_sec) for src, _ in Z_COLUMN_MAP}
        tail_cal = TrailingPercentile(window_sec=calibration_sec)

        n_written = 0
        for row in reader:
            ts_s = (row.get("ts") or "").strip()
            if not ts_s:
                continue
            try:
                ts = parse_ts_to_unix(ts_s)
            except ValueError:
                continue

            z_by_outcol: dict[str, float] = {}
            abs_z_vals: list[float] = []
            for src_col, out_col in Z_COLUMN_MAP:
                y = _safe_float(row.get(src_col))
                mean, std, n_hist = stat_windows[src_col].stats_before_update(ts)
                if n_hist <= 1 or math.isnan(y) or math.isnan(mean) or math.isnan(std) or std <= MIN_STD:
                    z = float("nan")
                else:
                    z = (y - mean) / std
                    abs_z_vals.append(abs(z))
                z_by_outcol[out_col] = z

            abs_z_max = max(abs_z_vals) if abs_z_vals else float("nan")
            anomaly_parts: list[float] = []
            for z_col in ANOMALY_COMPONENTS:
                z = z_by_outcol.get(z_col, float("nan"))
                if not math.isnan(z):
                    anomaly_parts.append(abs(z))
            anomaly_score = max(anomaly_parts) if anomaly_parts else float("nan")

            bullish, bearish = _directional_scores(z_by_outcol)
            direction = _direction_from_scores(bullish, bearish)

            pct_rank, pct_n = tail_cal.rank_before_update(ts, anomaly_score)
            is_10 = bool((not math.isnan(pct_rank)) and pct_n > 0 and pct_rank >= 0.90)
            is_5 = bool((not math.isnan(pct_rank)) and pct_n > 0 and pct_rank >= 0.95)

            out_row = dict(row)
            for col in OUTPUT_ADDED_COLUMNS:
                out_row[col] = ""
            for z_col, z_val in z_by_outcol.items():
                out_row[z_col] = _fmt_float(z_val)
            out_row["abs_z_max"] = _fmt_float(abs_z_max)
            out_row["anomaly_score"] = _fmt_float(anomaly_score)
            out_row["bullish_anomaly_score"] = _fmt_float(bullish)
            out_row["bearish_anomaly_score"] = _fmt_float(bearish)
            out_row["anomaly_direction"] = direction
            out_row["anomaly_percentile_rank"] = _fmt_float(pct_rank)
            out_row["is_extreme_10pct"] = "1" if is_10 else "0"
            out_row["is_extreme_5pct"] = "1" if is_5 else "0"
            writer.writerow(out_row)
            n_written += 1

            for src_col, _ in Z_COLUMN_MAP:
                stat_windows[src_col].add(ts, _safe_float(row.get(src_col)))
            tail_cal.add(ts, anomaly_score)

    logger.info(
        "anomaly matrix rows=%s lookback_sec=%s calibration_sec=%s -> %s",
        n_written,
        lookback_sec,
        calibration_sec,
        output_path,
    )
    return n_written


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    p = argparse.ArgumentParser(description="Build rolling statistical anomaly feature matrix.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--lookback-sec", type=float, default=15.0)
    p.add_argument("--calibration-sec", type=float, default=300.0)
    args = p.parse_args(argv)

    if args.lookback_sec <= 0:
        logger.error("--lookback-sec must be > 0")
        return 1
    if args.calibration_sec <= 0:
        logger.error("--calibration-sec must be > 0")
        return 1
    if not args.input.is_file():
        logger.error("missing input: %s", args.input)
        return 1

    build_anomaly_matrix(
        args.input,
        args.output,
        lookback_sec=args.lookback_sec,
        calibration_sec=args.calibration_sec,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
