"""
Realistic prop-firm execution wrapper for stop-management simulations.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tjtb.reports.trade_simulation import (
    DEFAULT_BOOK,
    DEFAULT_MATRIX,
    DEFAULT_SETUPS,
    build_realistic_prop_simulation,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path("reports/realistic_prop_simulation.json")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Realistic prop-firm throttled stop-management simulation")
    p.add_argument("--setups", type=Path, default=DEFAULT_SETUPS)
    p.add_argument("--book-state", type=Path, default=DEFAULT_BOOK)
    p.add_argument("--anomaly-matrix", type=Path, default=DEFAULT_MATRIX)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--forward-horizon-sec", type=float, default=2.0)
    p.add_argument("--max-trades-per-day", type=int, default=20)
    p.add_argument("--max-trades-per-session", type=int, default=5)
    p.add_argument("--cluster-window-sec", type=float, default=30.0)
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
    if args.max_trades_per_day <= 0 or args.max_trades_per_session <= 0:
        logger.error("--max-trades-per-day and --max-trades-per-session must be > 0")
        return 1
    if args.cluster_window_sec < 0:
        logger.error("--cluster-window-sec must be >= 0")
        return 1

    build_realistic_prop_simulation(
        args.setups,
        args.book_state,
        args.anomaly_matrix,
        args.output,
        forward_horizon_sec=args.forward_horizon_sec,
        max_trades_per_day=args.max_trades_per_day,
        max_trades_per_session=args.max_trades_per_session,
        anomaly_cluster_window_sec=args.cluster_window_sec,
    )
    return 0


__all__ = ["main", "build_realistic_prop_simulation"]


if __name__ == "__main__":
    raise SystemExit(main())
