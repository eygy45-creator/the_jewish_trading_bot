"""
Stream-reconstruct L2 order book from normalized Coinbase book_updates.csv.

Standard library only; streaming I/O; deterministic given identical input CSV.
"""

from __future__ import annotations

import csv
import logging
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

INPUT_PATH = Path("data/parsed/book_updates.csv")
OUTPUT_PATH = Path("data/parsed/book_state.csv")
READ_BUFFER = 1024 * 1024
WRITE_BUFFER = 1
PROGRESS_EVERY = 50_000
MAX_CROSSED_EXAMPLES = 20
INVALID_MAJORITY_THRESHOLD = 0.90
INVALID_MIN_ROWS_FOR_ABORT = 50_000

HEADER_IN = ("ts", "side", "price", "size")
HEADER_IN_V2 = ("ts", "event_type", "sequence_num", "side", "price", "size")
HEADER_OUT = (
    "ts",
    "best_bid",
    "best_ask",
    "best_bid_size",
    "best_ask_size",
    "spread",
    "mid_price",
    "microprice",
    "top_of_book_imbalance",
)


@dataclass
class ReconstructStats:
    rows_processed: int = 0
    rows_skipped: int = 0
    deleted_bid_levels: int = 0
    deleted_ask_levels: int = 0
    crossed_or_invalid: int = 0
    rows_emitted: int = 0
    duplicate_top_of_book_rows_skipped: int = 0
    snapshot_rows_parsed: int = 0
    updates_before_first_valid_book: int = 0
    first_valid_seen: bool = False
    crossed_examples: list[tuple[str, float, float]] = field(default_factory=list)


@dataclass
class BookState:
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    best_bid: float | None = None
    best_ask: float | None = None


def _parse_book_row(row: list[str]) -> tuple[str, str, int | None, str, float, float] | None:
    if len(row) == 4:
        ts, side, price_s, size_s = row
        event_type = "update"
        sequence_num: int | None = None
    elif len(row) == 6:
        ts, event_type, sequence_num_s, side, price_s, size_s = row
        event_type = event_type.strip().lower()
        if event_type not in ("snapshot", "update"):
            return None
        try:
            sequence_num = int(sequence_num_s)
        except ValueError:
            return None
    else:
        return None
    side_l = side.strip().lower()
    if side_l not in ("bid", "ask"):
        return None
    try:
        price = float(price_s)
        size = float(size_s)
    except ValueError:
        return None
    if size < 0:
        return None
    return ts, event_type, sequence_num, side_l, price, size


def apply_update(state: BookState, side: str, price: float, size: float, stats: ReconstructStats) -> tuple[bool, bool]:
    recompute_best_bid = False
    recompute_best_ask = False
    if side == "bid":
        if size == 0.0:
            if price in state.bids:
                del state.bids[price]
                stats.deleted_bid_levels += 1
                if state.best_bid == price:
                    recompute_best_bid = True
        else:
            state.bids[price] = size
            if state.best_bid is None or price > state.best_bid:
                state.best_bid = price
    elif side == "ask":
        if size == 0.0:
            if price in state.asks:
                del state.asks[price]
                stats.deleted_ask_levels += 1
                if state.best_ask == price:
                    recompute_best_ask = True
        else:
            state.asks[price] = size
            if state.best_ask is None or price < state.best_ask:
                state.best_ask = price
    return recompute_best_bid, recompute_best_ask


def is_crossed_or_touching(state: BookState) -> bool:
    if state.best_bid is None or state.best_ask is None:
        return False
    return state.best_bid >= state.best_ask


def compute_top_of_book(state: BookState) -> dict[str, float] | None:
    if state.best_bid is None or state.best_ask is None:
        return None
    best_bid = state.best_bid
    best_ask = state.best_ask
    if best_bid >= best_ask:
        return None
    best_bid_size = state.bids.get(best_bid, 0.0)
    best_ask_size = state.asks.get(best_ask, 0.0)
    if best_bid_size <= 0 or best_ask_size <= 0:
        return None
    spread = best_ask - best_bid
    mid_price = (best_bid + best_ask) / 2.0
    denom = best_bid_size + best_ask_size
    microprice = (best_ask * best_bid_size + best_bid * best_ask_size) / denom
    tob_imb = (best_bid_size - best_ask_size) / denom
    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "best_bid_size": best_bid_size,
        "best_ask_size": best_ask_size,
        "spread": spread,
        "mid_price": mid_price,
        "microprice": microprice,
        "top_of_book_imbalance": tob_imb,
    }


def _recompute_best_bid(state: BookState) -> None:
    state.best_bid = max(state.bids) if state.bids else None


def _recompute_best_ask(state: BookState) -> None:
    state.best_ask = min(state.asks) if state.asks else None


def _reset_book(state: BookState) -> None:
    state.bids.clear()
    state.asks.clear()
    state.best_bid = None
    state.best_ask = None


def iter_book_updates(path: Path) -> Iterator[list[str]]:
    with path.open("r", encoding="utf-8", newline="", buffering=READ_BUFFER) as f:
        reader = csv.reader(f)
        try:
            first = next(reader)
        except StopIteration:
            return
        if first and tuple(h.strip() for h in first) in (HEADER_IN, HEADER_IN_V2):
            pass
        else:
            yield first
        for row in reader:
            yield row


def reconstruct_stream(input_path: Path, output_path: Path) -> ReconstructStats:
    stats = ReconstructStats()
    state = BookState()
    last_emitted_tob: tuple[float, float, float, float] | None = None
    last_snapshot_sequence: int | None = None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8", buffering=WRITE_BUFFER) as out_f:
        writer = csv.writer(out_f)
        writer.writerow(HEADER_OUT)
        for row in iter_book_updates(input_path):
            try:
                if not row or all(not c.strip() for c in row):
                    stats.rows_skipped += 1
                    continue
                parsed = _parse_book_row([c.strip() for c in row])
                if parsed is None:
                    stats.rows_skipped += 1
                    continue
                ts, event_type, sequence_num, side, price, size = parsed
                if event_type == "snapshot":
                    stats.snapshot_rows_parsed += 1
                    if sequence_num is not None and sequence_num != last_snapshot_sequence:
                        _reset_book(state)
                        last_emitted_tob = None
                        last_snapshot_sequence = sequence_num
                recompute_best_bid, recompute_best_ask = apply_update(state, side, price, size, stats)
                if recompute_best_bid:
                    _recompute_best_bid(state)
                if recompute_best_ask:
                    _recompute_best_ask(state)
                stats.rows_processed += 1
                if stats.rows_processed % PROGRESS_EVERY == 0:
                    logger.info(
                        "progress rows_processed=%s rows_emitted=%s duplicate_rows_skipped=%s "
                        "deleted_bid_levels=%s deleted_ask_levels=%s",
                        stats.rows_processed,
                        stats.rows_emitted,
                        stats.duplicate_top_of_book_rows_skipped,
                        stats.deleted_bid_levels,
                        stats.deleted_ask_levels,
                    )
                    out_f.flush()
                top = compute_top_of_book(state)
                if top is None:
                    if is_crossed_or_touching(state):
                        stats.crossed_or_invalid += 1
                        if len(stats.crossed_examples) < MAX_CROSSED_EXAMPLES:
                            stats.crossed_examples.append((ts, state.best_bid or 0.0, state.best_ask or 0.0))
                    if not stats.first_valid_seen:
                        stats.updates_before_first_valid_book += 1
                    if (
                        stats.rows_processed >= INVALID_MIN_ROWS_FOR_ABORT
                        and stats.crossed_or_invalid / stats.rows_processed >= INVALID_MAJORITY_THRESHOLD
                    ):
                        raise RuntimeError(
                            "Aborting reconstruction: invalid/crossed book for majority of rows. "
                            "Likely missing or mishandled initial snapshot."
                        )
                    continue
                tob_key = (
                    top["best_bid"],
                    top["best_ask"],
                    top["best_bid_size"],
                    top["best_ask_size"],
                )
                if not stats.first_valid_seen:
                    stats.first_valid_seen = True
                if last_emitted_tob == tob_key:
                    stats.duplicate_top_of_book_rows_skipped += 1
                    continue
                last_emitted_tob = tob_key
                stats.rows_emitted += 1
                writer.writerow(
                    [
                        ts,
                        top["best_bid"],
                        top["best_ask"],
                        top["best_bid_size"],
                        top["best_ask_size"],
                        top["spread"],
                        top["mid_price"],
                        top["microprice"],
                        top["top_of_book_imbalance"],
                    ]
                )
                if stats.rows_emitted == 1:
                    out_f.flush()
            except RuntimeError:
                raise
            except Exception:
                stats.rows_skipped += 1
                continue
    if stats.snapshot_rows_parsed == 0:
        raise RuntimeError("Initial snapshot is missing in book_updates.csv; cannot reconstruct order book.")
    if not stats.first_valid_seen:
        raise RuntimeError("No valid uncrossed top-of-book found after applying snapshot and updates.")
    if stats.rows_processed and stats.crossed_or_invalid / stats.rows_processed >= INVALID_MAJORITY_THRESHOLD:
        raise RuntimeError(
            "Aborting reconstruction: invalid/crossed book for majority of rows. "
            "Likely missing or inconsistent snapshot handling."
        )
    return stats


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )
    if not INPUT_PATH.is_file():
        logger.error("Input not found: %s", INPUT_PATH)
        return 1
    try:
        st = reconstruct_stream(INPUT_PATH, OUTPUT_PATH)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1
    if st.crossed_examples:
        for idx, (ts, best_bid, best_ask) in enumerate(st.crossed_examples, start=1):
            logger.warning(
                "crossed_example_%s ts=%s best_bid=%s best_ask=%s",
                idx,
                ts,
                best_bid,
                best_ask,
            )
    logger.info(
        "rows_processed=%s rows_skipped=%s deleted_bid_levels=%s deleted_ask_levels=%s "
        "crossed_or_invalid=%s rows_emitted=%s duplicate_top_of_book_rows_skipped=%s "
        "snapshot_rows_parsed=%s updates_before_first_valid_book=%s -> %s",
        st.rows_processed,
        st.rows_skipped,
        st.deleted_bid_levels,
        st.deleted_ask_levels,
        st.crossed_or_invalid,
        st.rows_emitted,
        st.duplicate_top_of_book_rows_skipped,
        st.snapshot_rows_parsed,
        st.updates_before_first_valid_book,
        OUTPUT_PATH,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
