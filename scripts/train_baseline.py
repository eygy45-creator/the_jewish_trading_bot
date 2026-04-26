#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tjtb.config.model_settings import ModelSettings
from tjtb.models.baseline_logistic import train_walk_forward_baseline
from tjtb.monitoring.logging import configure_logging, get_logger


def _synthetic_frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range("2026-01-01", periods=n, freq="min")
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    logits = 0.7 * x1 - 0.4 * x2
    p = 1 / (1 + np.exp(-logits))
    y = (rng.random(n) < p).astype(int)
    return pd.DataFrame(
        {
            "ts": t,
            "f1": x1,
            "f2": x2,
            "label": y,
        }
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=Path("runs/baseline_demo"))
    args = p.parse_args()

    configure_logging()
    log = get_logger("train_baseline")
    df = _synthetic_frame(args.rows, args.seed)
    settings = ModelSettings()
    metrics = train_walk_forward_baseline(
        df,
        feature_cols=["f1", "f2"],
        label_col="label",
        time_col="ts",
        settings=settings,
        n_splits=4,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(args.output_dir / "walk_forward_metrics.csv", index=False)
    log.info("training_done", folds=len(metrics))


if __name__ == "__main__":
    main()
