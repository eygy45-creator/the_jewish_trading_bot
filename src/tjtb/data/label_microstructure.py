"""
Binary setup labels on mid-price path from reconstructed book state.

Label = 1 if either ``threshold_up`` (upward excursion from current mid) or
``threshold_down`` (downward excursion) is reached at any time within the next
``horizon`` rows; otherwise 0. No directional multiclass.

Streaming with bounded memory: sliding deque of (ts, mid) up to horizon+1 rows.
Market-agnostic (price units); optional tick conversion only in CLI metadata.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, TextIO

logger = logging.getLogger(__name__)

DEFAULT_INPUT = Path("data/parsed/book_state.csv")
DEFAULT_OUTPUT = Path("data/parsed/labeled_book.csv")
READ_BUFFER = 1024 * 1024
WRITE_BUFFER = 1024 * 1024


@dataclass
class LabelStats:
    n_positive: int = 0
    n_negative: int = 0
    n_rows_written: int = 0
    n_rows_skipped: int = 0


def _parse_mid_row(row: list[str]) -> tuple[str, float] | None:
    if len(row) < 7:
        return None
    if row[0].strip().lower() == "ts":
        return None
    try:
        ts = str(row[0]).strip()
        mid = float(row[6])
    except (IndexError, ValueError):
        return None
    return ts, mid


def label_from_futures(
    current_mid: float,
    future_mids: list[float],
    th_up: float,
    th_down: float,
) -> tuple[int, float, float]:
    """
    Over the lookahead path, max/min mid vs ``current_mid``.

    Returns (label, up_move, down_move) where label is 1 if either excursion
    reaches its threshold within the scanned window, else 0.
    """
    max_p = current_mid
    min_p = current_mid
    for m in future_mids:
        max_p = max(max_p, m)
        min_p = min(min_p, m)
    up_move = max_p - current_mid
    down_move = current_mid - min_p
    hit = 1 if (up_move >= th_up or down_move >= th_down) else 0
    return hit, up_move, down_move


def stream_label_book(
    inp: TextIO,
    out: TextIO,
    *,
    horizon: int,
    threshold_up: float,
    threshold_down: float,
) -> LabelStats:
    stats = LabelStats()
    dq: Deque[tuple[str, float]] = deque()
    w = csv.writer(out)
    w.writerow(["ts", "mid_price", "label", "up_move", "down_move"])

    def emit_one() -> None:
        nonlocal stats
        if len(dq) < 2:
            return
        ts_cur, mid_cur = dq[0]
        futures = [m for _, m in list(dq)[1 : horizon + 1]]
        lab, up_m, dn_m = label_from_futures(mid_cur, futures, threshold_up, threshold_down)
        if lab == 1:
            stats.n_positive += 1
        else:
            stats.n_negative += 1
        w.writerow([ts_cur, mid_cur, lab, up_m, dn_m])
        stats.n_rows_written += 1

    for row in csv.reader(inp):
        if not row or all(not c.strip() for c in row):
            stats.n_rows_skipped += 1
            continue
        parsed = _parse_mid_row(row)
        if parsed is None:
            stats.n_rows_skipped += 1
            continue
        dq.append(parsed)
        if len(dq) == horizon + 1:
            emit_one()
            dq.popleft()

    while len(dq) >= 2:
        emit_one()
        dq.popleft()
    if len(dq) == 1:
        stats.n_rows_skipped += 1
    return stats


def label_file(
    input_path: Path,
    output_path: Path,
    *,
    horizon: int,
    threshold_up: float,
    threshold_down: float,
) -> LabelStats:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8", newline="", buffering=READ_BUFFER) as inp, output_path.open(
        "w", newline="", encoding="utf-8", buffering=WRITE_BUFFER
    ) as out:
        return stream_label_book(inp, out, horizon=horizon, threshold_up=threshold_up, threshold_down=threshold_down)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    p = argparse.ArgumentParser(description="Binary setup labels on mid-price (streaming).")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--threshold-up", type=float, required=True)
    p.add_argument("--threshold-down", type=float, required=True)
    args = p.parse_args(argv)
    if not args.input.is_file():
        logger.error("Missing input: %s", args.input)
        return 1
    st = label_file(
        args.input,
        args.output,
        horizon=args.horizon,
        threshold_up=args.threshold_up,
        threshold_down=args.threshold_down,
    )
    pct_pos = (100.0 * st.n_positive / st.n_rows_written) if st.n_rows_written else 0.0
    logger.info(
        "labels written=%s skipped=%s positive=%s negative=%s pct_positive=%.4f%% -> %s",
        st.n_rows_written,
        st.n_rows_skipped,
        st.n_positive,
        st.n_negative,
        pct_pos,
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
