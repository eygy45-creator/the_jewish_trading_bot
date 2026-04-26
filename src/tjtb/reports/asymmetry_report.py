"""
Asymmetry analysis: favorable vs adverse excursion (not hit/miss classification).

Reads anomaly_setups.csv; optionally joins anomaly_matrix.csv on ts for spread
and l2_event_rate_hz for conditional slices.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_SETUPS = Path("data/parsed/anomaly_setups.csv")
DEFAULT_MATRIX = Path("data/parsed/anomaly_matrix.csv")
DEFAULT_OUTPUT = Path("reports/asymmetry_report.json")
READ_BUFFER = 1024 * 1024
EPS = 1e-6

# Default conditional slices (quantiles computed on rows with finite joined features).
DEFAULT_HIGH_EVENT_QUANTILE = 0.75
DEFAULT_LOW_SPREAD_QUANTILE = 0.25
TOP_ANOMALY_PCT_THRESHOLD = 0.97  # top 3%: percentile rank >= 0.97


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


def _ratio(fav: float | None, adv: float | None) -> float | None:
    if fav is None or adv is None:
        return None
    if math.isnan(fav) or math.isnan(adv):
        return None
    return float(fav) / (float(adv) + EPS)


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    q = max(0.0, min(1.0, q))
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


def _load_setups(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="", buffering=READ_BUFFER) as f:
        return list(csv.DictReader(f))


def _matrix_by_ts(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    out: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="", buffering=READ_BUFFER) as f:
        for row in csv.DictReader(f):
            ts = (row.get("ts") or "").strip()
            if ts:
                out[ts] = row
    return out


def _histogram_ratios(ratios: list[float], edges: list[float]) -> list[dict[str, Any]]:
    """edges strictly increasing; last bucket [edges[-2], inf)."""
    counts = [0] * (len(edges) - 1)
    for r in ratios:
        if math.isnan(r) or math.isinf(r):
            continue
        placed = False
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            if i == len(edges) - 2:
                if r >= lo:
                    counts[i] += 1
                    placed = True
                    break
            if lo <= r < hi:
                counts[i] += 1
                placed = True
                break
        if not placed and r < edges[0]:
            counts[0] += 1
    out: list[dict[str, Any]] = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        label = f"[{lo:g},{hi:g})" if i < len(edges) - 2 else f"[{lo:g},inf)"
        out.append({"bucket": label, "n": counts[i]})
    return out


def _pct_gt(ratios: list[float], threshold: float) -> float | None:
    finite = [r for r in ratios if not math.isnan(r) and not math.isinf(r)]
    if not finite:
        return None
    return float(sum(1 for r in finite if r > threshold)) / float(len(finite))


def _avg_ratio(ratios: list[float]) -> float | None:
    finite = [r for r in ratios if not math.isnan(r) and not math.isinf(r)]
    if not finite:
        return None
    return float(mean(finite))


def _stats_block(ratios: list[float]) -> dict[str, Any]:
    finite = [r for r in ratios if not math.isnan(r) and not math.isinf(r)]
    if not finite:
        return {
            "n": 0,
            "avg_ratio": None,
            "median_ratio": None,
            "min_ratio": None,
            "max_ratio": None,
            "percent_ratio_gt_1_5": None,
            "percent_ratio_gt_2_0": None,
        }
    finite.sort()
    return {
        "n": len(finite),
        "avg_ratio": float(mean(finite)),
        "median_ratio": float(median(finite)),
        "min_ratio": finite[0],
        "max_ratio": finite[-1],
        "percent_ratio_gt_1_5": _pct_gt(finite, 1.5),
        "percent_ratio_gt_2_0": _pct_gt(finite, 2.0),
    }


def _filter_indices(
    rows: list[dict[str, str]],
    matrix: dict[str, dict[str, str]],
    *,
    top_anomaly_only: bool,
    high_event_only: bool,
    low_spread_only: bool,
    event_threshold: float | None,
    spread_threshold: float | None,
) -> list[int]:
    idx_out: list[int] = []
    for i, r in enumerate(rows):
        ts = (r.get("ts") or "").strip()
        p = _safe_float(r.get("anomaly_percentile_rank"))
        if top_anomaly_only:
            if p is None or p < TOP_ANOMALY_PCT_THRESHOLD:
                continue
        m = matrix.get(ts, {})
        if high_event_only:
            er = _safe_float(m.get("l2_event_rate_hz"))
            if er is None or math.isnan(er) or event_threshold is None or math.isnan(event_threshold):
                continue
            if er < event_threshold:
                continue
        if low_spread_only:
            sp = _safe_float(m.get("spread"))
            if sp is None or math.isnan(sp) or spread_threshold is None or math.isnan(spread_threshold):
                continue
            if sp > spread_threshold:
                continue
        idx_out.append(i)
    return idx_out


def build_asymmetry_report(
    setups_path: Path,
    matrix_path: Path,
    output_path: Path,
    *,
    high_event_quantile: float,
    low_spread_quantile: float,
) -> dict[str, Any]:
    rows = _load_setups(setups_path)
    matrix = _matrix_by_ts(matrix_path)

    ratios: list[float | None] = []
    directions: list[str] = []
    event_rates: list[float] = []
    spreads: list[float] = []

    for r in rows:
        fav = _safe_float(r.get("max_favorable_move"))
        adv = _safe_float(r.get("max_adverse_move"))
        ratios.append(_ratio(fav, adv))
        directions.append((r.get("anomaly_direction") or "neutral").strip().lower())
        ts = (r.get("ts") or "").strip()
        m = matrix.get(ts, {})
        er = _safe_float(m.get("l2_event_rate_hz"))
        if er is not None and not math.isnan(er):
            event_rates.append(er)
        sp = _safe_float(m.get("spread"))
        if sp is not None and not math.isnan(sp) and sp >= 0:
            spreads.append(sp)

    event_rates.sort()
    spreads.sort()
    event_thr = _quantile(event_rates, high_event_quantile) if event_rates else float("nan")
    spread_thr = _quantile(spreads, low_spread_quantile) if spreads else float("nan")

    ratio_list = [x for x in ratios if x is not None]
    ratio_finite = [x for x in ratio_list if not math.isnan(x) and not math.isinf(x)]

    hist_edges_num = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 1e18]
    dist_buckets = _histogram_ratios(ratio_finite, hist_edges_num)

    bullish_ratios = [ratios[i] for i in range(len(rows)) if directions[i] == "bullish" and ratios[i] is not None]
    bearish_ratios = [ratios[i] for i in range(len(rows)) if directions[i] == "bearish" and ratios[i] is not None]
    neutral_ratios = [ratios[i] for i in range(len(rows)) if directions[i] == "neutral" and ratios[i] is not None]

    def block_for_indices(indices: list[int]) -> dict[str, Any]:
        rs = [ratios[i] for i in indices if ratios[i] is not None]
        rs_f = [x for x in rs if not math.isnan(x) and not math.isinf(x)]
        st = _stats_block(rs_f)
        st["distribution"] = _histogram_ratios(rs_f, hist_edges_num)
        return st

    idx_top3 = _filter_indices(
        rows,
        matrix,
        top_anomaly_only=True,
        high_event_only=False,
        low_spread_only=False,
        event_threshold=None,
        spread_threshold=None,
    )
    idx_high_ev = _filter_indices(
        rows,
        matrix,
        top_anomaly_only=False,
        high_event_only=True,
        low_spread_only=False,
        event_threshold=event_thr,
        spread_threshold=None,
    )
    idx_low_spread = _filter_indices(
        rows,
        matrix,
        top_anomaly_only=False,
        high_event_only=False,
        low_spread_only=True,
        event_threshold=None,
        spread_threshold=spread_thr,
    )

    percentile_buckets = [
        (0.95, 0.97, "[0.95,0.97)"),
        (0.97, 0.99, "[0.97,0.99)"),
        (0.99, 1.0000001, "[0.99,1.00]"),
    ]
    bucket_rows: list[dict[str, Any]] = []
    for lo, hi, label in percentile_buckets:
        idxs = []
        for i, r in enumerate(rows):
            p = _safe_float(r.get("anomaly_percentile_rank"))
            if p is None or math.isnan(p):
                continue
            if p >= lo and p < hi:
                idxs.append(i)
        rs = [ratios[i] for i in idxs if ratios[i] is not None]
        rs_f = [x for x in rs if not math.isnan(x) and not math.isinf(x)]
        st = _stats_block(rs_f)
        st["anomaly_percentile_bucket"] = label
        bucket_rows.append(st)

    report: dict[str, Any] = {
        "input": {"setups": str(setups_path), "anomaly_matrix": str(matrix_path)},
        "n_rows": len(rows),
        "n_matrix_rows_indexed": len(matrix),
        "favorable_to_adverse_ratio": {
            "definition": "max_favorable_move / (max_adverse_move + 1e-6)",
            "overall": _stats_block(ratio_finite),
            "distribution_histogram": dist_buckets,
        },
        "avg_ratio_by_anomaly_direction": {
            "bullish": _avg_ratio([x for x in bullish_ratios if x is not None]),
            "bearish": _avg_ratio([x for x in bearish_ratios if x is not None]),
            "neutral": _avg_ratio([x for x in neutral_ratios if x is not None]),
            "n_bullish": len([x for x in bullish_ratios if x is not None]),
            "n_bearish": len([x for x in bearish_ratios if x is not None]),
            "n_neutral": len([x for x in neutral_ratios if x is not None]),
        },
        "percent_ratio_gt_thresholds": {
            "overall_gt_1_5": _pct_gt(ratio_finite, 1.5),
            "overall_gt_2_0": _pct_gt(ratio_finite, 2.0),
            "n_denominator": len(ratio_finite),
        },
        "conditional_slices": {
            "top_3pct_anomalies": {
                "definition": f"anomaly_percentile_rank >= {TOP_ANOMALY_PCT_THRESHOLD}",
                **block_for_indices(idx_top3),
            },
            "high_event_rate": {
                "definition": f"l2_event_rate_hz >= quantile({high_event_quantile}) over joined rows with finite rate",
                "threshold_l2_event_rate_hz": None if math.isnan(event_thr) else event_thr,
                "n_rows_with_finite_event_rate_in_matrix": len(event_rates),
                **block_for_indices(idx_high_ev),
            },
            "low_spread": {
                "definition": f"spread <= quantile({low_spread_quantile}) over joined rows with finite non-negative spread",
                "threshold_spread": None if math.isnan(spread_thr) else spread_thr,
                "n_rows_with_finite_spread_in_matrix": len(spreads),
                **block_for_indices(idx_low_spread),
            },
        },
        "bucket_by_anomaly_percentile": bucket_rows,
        "notes": [
            "Asymmetry focus: favorable vs adverse excursion size, not directional hit rate.",
            "Conditional slices require ts join to anomaly_matrix for spread and l2_event_rate_hz.",
            "Goal: identify conditions where favorable_to_adverse_ratio is consistently high.",
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    logger.info("wrote %s rows=%s", output_path, len(rows))
    return report


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Asymmetry report (favorable vs adverse move).")
    p.add_argument("--setups", type=Path, default=DEFAULT_SETUPS)
    p.add_argument("--anomaly-matrix", type=Path, default=DEFAULT_MATRIX)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument(
        "--high-event-quantile",
        type=float,
        default=DEFAULT_HIGH_EVENT_QUANTILE,
        help="Rows with l2_event_rate_hz >= this quantile (joined matrix) count as high event rate.",
    )
    p.add_argument(
        "--low-spread-quantile",
        type=float,
        default=DEFAULT_LOW_SPREAD_QUANTILE,
        help="Rows with spread <= this quantile (joined matrix) count as low spread.",
    )
    args = p.parse_args(argv)

    if not (0.0 < args.high_event_quantile < 1.0):
        logger.error("--high-event-quantile must be in (0,1)")
        return 1
    if not (0.0 < args.low_spread_quantile < 1.0):
        logger.error("--low-spread-quantile must be in (0,1)")
        return 1
    if not args.setups.is_file():
        logger.error("missing setups: %s", args.setups)
        return 1

    build_asymmetry_report(
        args.setups,
        args.anomaly_matrix,
        args.output,
        high_event_quantile=args.high_event_quantile,
        low_spread_quantile=args.low_spread_quantile,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
