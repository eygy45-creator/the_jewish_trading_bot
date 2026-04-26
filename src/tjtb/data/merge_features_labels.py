"""
Merge feature_matrix.csv with labeled_book.csv on nearest timestamp (within tolerance).

Deterministic: stable sort by ts, then nearest-neighbor assignment without randomness.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Sequence

from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)

FEAT_DEFAULT = Path("data/parsed/feature_matrix.csv")
LAB_DEFAULT = Path("data/parsed/labeled_book.csv")
OUT_DEFAULT = Path("data/parsed/dataset.csv")
READ_BUFFER = 1024 * 1024
WRITE_BUFFER = 1024 * 1024


def _read_csv_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8", newline="", buffering=READ_BUFFER) as f:
        r = csv.reader(f)
        rows = list(r)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _parse_ts_col(ts: str) -> float:
    t = ts.strip()
    if t.endswith("Z"):
        dt = datetime.fromisoformat(t[:-1]).replace(tzinfo=timezone.utc)
    else:
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
    return float(dt.timestamp())


def merge_nearest(
    feature_header: Sequence[str],
    feature_rows: list[list[str]],
    label_header: Sequence[str],
    label_rows: list[list[str]],
    *,
    tolerance_sec: float,
) -> tuple[list[str], list[list[str]], int, int]:
    """Return (out_header, out_rows, n_matched, n_unmatched_labels)."""
    if not feature_rows or not label_rows:
        return [], [], 0, len(label_rows)

    fi_ts = feature_header.index("ts")
    li_ts = label_header.index("ts")
    li_mid = label_header.index("mid_price")
    li_lab = label_header.index("label")
    li_up = label_header.index("up_move")
    li_dn = label_header.index("down_move")

    f_times: list[float] = []
    f_good: list[list[str]] = []
    for fr in feature_rows:
        if len(fr) <= fi_ts:
            continue
        try:
            f_times.append(_parse_ts_col(fr[fi_ts]))
            f_good.append(fr)
        except (ValueError, OSError):
            continue
    if not f_good:
        return [], [], 0, len(label_rows)

    order = np.argsort(np.asarray(f_times, dtype=np.float64), kind="mergesort")
    f_times_arr = np.asarray(f_times, dtype=np.float64)[order]
    f_rows_sorted = [f_good[i] for i in order.tolist()]

    feat_cols = [c for c in feature_header if c != "ts"]
    out_header = ["ts"] + feat_cols + ["mid_price", "label", "up_move", "down_move"]
    out_rows: list[list[str]] = []
    n_match = 0
    n_miss = 0

    for lr in label_rows:
        if len(lr) <= max(li_ts, li_mid, li_lab, li_up, li_dn):
            n_miss += 1
            continue
        try:
            lt = _parse_ts_col(lr[li_ts])
        except (ValueError, OSError):
            n_miss += 1
            continue
        idx = int(np.searchsorted(f_times_arr, lt, side="left"))
        candidates: list[int] = []
        if 0 <= idx < len(f_times_arr):
            candidates.append(idx)
        if idx - 1 >= 0:
            candidates.append(idx - 1)
        if not candidates:
            n_miss += 1
            continue
        best_i = min(candidates, key=lambda i: abs(f_times_arr[i] - lt))
        if abs(f_times_arr[best_i] - lt) > tolerance_sec:
            n_miss += 1
            continue
        fr = f_rows_sorted[best_i]
        row_out: list[str] = [lr[li_ts]]
        for c in feat_cols:
            j = feature_header.index(c)
            row_out.append(fr[j] if j < len(fr) else "")
        row_out.extend([lr[li_mid], lr[li_lab], lr[li_up], lr[li_dn]])
        out_rows.append(row_out)
        n_match += 1

    return out_header, out_rows, n_match, n_miss


def write_dataset(out_path: Path, header: list[str], rows: list[list[str]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8", buffering=WRITE_BUFFER) as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    p = argparse.ArgumentParser(description="Merge features and labels by nearest ts.")
    p.add_argument("--features", type=Path, default=FEAT_DEFAULT)
    p.add_argument("--labels", type=Path, default=LAB_DEFAULT)
    p.add_argument("--output", type=Path, default=OUT_DEFAULT)
    p.add_argument("--tolerance-sec", type=float, default=0.25)
    args = p.parse_args(argv)
    if not args.features.is_file() or not args.labels.is_file():
        logger.error("Missing inputs.")
        return 1
    fh, frs = _read_csv_rows(args.features)
    lh, lrs = _read_csv_rows(args.labels)
    header, rows, n_m, n_x = merge_nearest(fh, frs, lh, lrs, tolerance_sec=args.tolerance_sec)
    write_dataset(args.output, list(header), rows)
    logger.info("dataset rows=%s unmatched_labels=%s -> %s", n_m, n_x, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
