#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tjtb.config.model_settings import ModelSettings
from tjtb.config.session_research_settings import SessionResearchSettings
from tjtb.models.baseline_logistic import train_walk_forward_baseline
from tjtb.monitoring.logging import configure_logging, get_logger
from tjtb.reports.session_research import build_session_research_report, write_session_research_json


def _synthetic_trades(n: int) -> pd.DataFrame:
    t = pd.date_range("2026-02-01", periods=n, freq="h")
    pattern = ["asia", "europe_london_overlap", "us_cash_open", "midday"]
    sessions = (pattern * (n // len(pattern) + 1))[:n]
    net = np.where(np.arange(n) % 2 == 0, 12.5, -12.5)
    return pd.DataFrame({"ts": t, "session_bucket": sessions, "net_pnl": net})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("runs/session_report_demo.json"))
    args = p.parse_args()

    configure_logging()
    log = get_logger("session_report")

    trades = _synthetic_trades(120)
    sr = SessionResearchSettings()
    wf = train_walk_forward_baseline(
        pd.DataFrame(
            {
                "ts": pd.date_range("2026-01-01", periods=400, freq="min"),
                "f1": pd.Series(range(400), dtype=float) % 7.0,
                "f2": pd.Series(range(400), dtype=float) % 5.0,
                "label": [0, 1] * 200,
            }
        ),
        feature_cols=["f1", "f2"],
        label_col="label",
        time_col="ts",
        settings=ModelSettings(),
        n_splits=4,
    )
    report = build_session_research_report(trades, sr, walk_forward_fold_stats=wf)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_session_research_json(report, args.output)
    log.info("session_report_written", path=str(args.output))


if __name__ == "__main__":
    main()
