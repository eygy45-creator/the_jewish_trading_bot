"""
Summarize microstructure dataset, model metrics, and EV distribution (research-only).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from tjtb.signals.expected_value import MicrostructureEVConfig, expected_value_microstructure_vector

logger = logging.getLogger(__name__)

DATASET_DEFAULT = Path("data/parsed/dataset.csv")
FEATURE_MATRIX_DEFAULT = Path("data/parsed/feature_matrix.csv")
METRICS_DEFAULT = Path("runs/microstructure/metrics.json")
PROBS_DEFAULT = Path("runs/microstructure/test_probs.csv")
OUT_DEFAULT = Path("reports/microstructure_summary.json")
READ_BUFFER = 1024 * 1024

WINDOW_FEATURE_NAMES: tuple[str, ...] = (
    "del_bid_w",
    "del_ask_w",
    "l2_update_count_w",
    "abs_size_change_w",
    "signed_book_pressure_w",
    "l2_event_rate_hz",
    "avg_inter_event_sec_w",
    "trade_book_ratio_w",
)


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8", newline="", buffering=READ_BUFFER) as f:
        r = csv.reader(f)
        rows = list(r)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def label_distribution(header: list[str], body: list[list[str]]) -> dict[str, Any]:
    if "label" not in header:
        return {}
    j = header.index("label")
    vals = [int(float(row[j])) for row in body if len(row) > j]
    if not vals:
        return {}
    a = np.asarray(vals, dtype=np.int64)
    out: dict[str, Any] = {}
    for k in (-1, 0, 1):
        out[str(k)] = int(np.sum(a == k))
    out["n"] = int(a.size)
    return out


def window_feature_statistics(header: list[str], body: list[list[str]]) -> dict[str, Any]:
    """Summary stats for L2 rolling-window features (subset of columns)."""
    all_stats = feature_statistics(header, body)
    return {k: all_stats[k] for k in WINDOW_FEATURE_NAMES if k in all_stats}


def feature_statistics(header: list[str], body: list[list[str]]) -> dict[str, Any]:
    skip = {"ts", "label", "mid_price", "up_move", "down_move"}
    stats: dict[str, Any] = {}
    for name in header:
        if name in skip:
            continue
        j = header.index(name)
        col: list[float] = []
        for row in body:
            if len(row) <= j:
                continue
            try:
                col.append(float(row[j]))
            except ValueError:
                continue
        if not col:
            continue
        a = np.asarray(col, dtype=np.float64)
        stats[name] = {
            "mean": float(np.mean(a)),
            "std": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
            "min": float(np.min(a)),
            "max": float(np.max(a)),
        }
    return stats


def ev_block_from_probs(path: Path, cfg: MicrostructureEVConfig) -> dict[str, Any]:
    if not path.is_file():
        return {}
    _, body = _read_csv(path)
    if not body:
        return {}
    # header y_true,y_pred,p_neg,p_zero,p_plus
    p_plus = np.asarray([float(r[4]) for r in body], dtype=np.float64)
    p_minus = np.asarray([float(r[2]) for r in body], dtype=np.float64)
    y_true = np.asarray([int(float(r[0])) for r in body], dtype=np.int64)
    y_pred = np.asarray([int(float(r[1])) for r in body], dtype=np.int64)
    ev = expected_value_microstructure_vector(p_plus, p_minus, cfg)
    mask = ev > 0
    hit = float(np.mean(y_true == y_pred)) if y_true.size else 0.0
    pos_n = int(np.sum(mask))
    avg_ret_signal = float(np.mean(y_true[mask])) if pos_n > 0 else 0.0
    return {
        "n": int(ev.size),
        "hit_rate_accuracy": hit,
        "ev_mean": float(np.mean(ev)) if ev.size else 0.0,
        "ev_std": float(np.std(ev, ddof=1)) if ev.size > 1 else 0.0,
        "ev_positive_count": pos_n,
        "ev_positive_fraction": float(pos_n) / float(ev.size) if ev.size else 0.0,
        "avg_outcome_on_ev_positive": avg_ret_signal,
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    p = argparse.ArgumentParser(description="Microstructure research summary.")
    p.add_argument("--dataset", type=Path, default=DATASET_DEFAULT)
    p.add_argument("--feature-matrix", type=Path, default=FEATURE_MATRIX_DEFAULT)
    p.add_argument("--metrics", type=Path, default=METRICS_DEFAULT)
    p.add_argument("--test-probs", type=Path, default=PROBS_DEFAULT)
    p.add_argument("--reward", type=float, required=True)
    p.add_argument("--risk", type=float, required=True)
    p.add_argument("--costs", type=float, required=True)
    p.add_argument("--tick-size", type=float, default=None)
    p.add_argument("--output", type=Path, default=OUT_DEFAULT)
    args = p.parse_args(argv)

    cfg = MicrostructureEVConfig(reward=args.reward, risk=args.risk, costs=args.costs, tick_size=args.tick_size)

    summary: dict[str, Any] = {"ev_config": asdict(cfg)}

    if args.dataset.is_file():
        h, b = _read_csv(args.dataset)
        summary["label_distribution"] = label_distribution(h, b)
        summary["feature_statistics"] = feature_statistics(h, b)
    else:
        logger.warning("dataset missing: %s", args.dataset)

    if args.feature_matrix.is_file():
        fh, fb = _read_csv(args.feature_matrix)
        summary["l2_window_feature_stats"] = window_feature_statistics(fh, fb)
    else:
        summary["l2_window_feature_stats"] = {}
        logger.warning("feature_matrix missing: %s", args.feature_matrix)

    if args.metrics.is_file():
        summary["model_metrics"] = json.loads(args.metrics.read_text(encoding="utf-8"))
    else:
        logger.warning("metrics missing: %s", args.metrics)

    summary["ev_on_test"] = ev_block_from_probs(args.test_probs, cfg)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
