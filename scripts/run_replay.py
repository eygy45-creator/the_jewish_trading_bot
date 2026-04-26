#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from tjtb.backtest.engine import run_replay_count
from tjtb.data.replay import load_market_events_csv
from tjtb.monitoring.logging import configure_logging, get_logger


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--max-events", type=int, default=None)
    args = p.parse_args()

    configure_logging()
    log = get_logger("replay")
    stream = load_market_events_csv(args.csv, max_events=args.max_events)
    stats = run_replay_count(stream)
    log.info("replay_complete", **stats)


if __name__ == "__main__":
    main()
