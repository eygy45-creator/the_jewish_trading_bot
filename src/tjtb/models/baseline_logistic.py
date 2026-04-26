"""
Baseline logistic regression with probability calibration.

Walk-forward uses `time_based_splits` — no random shuffling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tjtb.backtest.walk_forward import time_based_splits
from tjtb.config.model_settings import ModelSettings


@dataclass
class BaselineLogisticPipeline:
    """Thin wrapper for persistence / inference later."""

    inner: Pipeline

    def predict_proba(self, X):
        return self.inner.predict_proba(X)


def train_walk_forward_baseline(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    time_col: str,
    settings: ModelSettings,
    n_splits: int = 4,
) -> list[dict]:
    """
    Expanding train / held-out test per fold; inside each train block, a time tail is used as val
    only for diagnostics (model still fit on full train block to keep the API small for M1).
    """
    results: list[dict] = []
    for fold, (train_all, test) in enumerate(time_based_splits(df, time_col, n_splits)):
        if len(test) < 5:
            continue
        m = max(10, int(len(train_all) * settings.walk_forward_train_fraction))
        train = train_all.iloc[:m]
        val = train_all.iloc[m:]
        if len(train) < 10 or len(val) < 5:
            continue

        X_train = train[feature_cols].to_numpy()
        y_train = train[label_col].to_numpy()
        X_val = val[feature_cols].to_numpy()
        y_val = val[label_col].to_numpy()
        X_test = test[feature_cols].to_numpy()
        y_test = test[label_col].to_numpy()

        base = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(max_iter=200, random_state=settings.random_state),
                ),
            ]
        )
        model = CalibratedClassifierCV(base, method=settings.calibration_method, cv=3)
        model.fit(X_train, y_train)
        proba_val = model.predict_proba(X_val)[:, 1]
        proba_test = model.predict_proba(X_test)[:, 1]
        auc_val = roc_auc_score(y_val, proba_val) if len(np.unique(y_val)) > 1 else float("nan")
        auc_test = roc_auc_score(y_test, proba_test) if len(np.unique(y_test)) > 1 else float("nan")
        results.append(
            {
                "fold": fold,
                "auc_val": float(auc_val),
                "auc_test": float(auc_test),
                "n_train": len(train),
                "n_val": len(val),
                "n_test": len(test),
            }
        )
    return results


def fit_full_calibrated_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    settings: ModelSettings,
) -> CalibratedClassifierCV:
    """Fit on full frame (use only after walk-forward research)."""
    X = df[feature_cols].to_numpy()
    y = df[label_col].to_numpy()
    base = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200, random_state=settings.random_state)),
        ]
    )
    model = CalibratedClassifierCV(base, method=settings.calibration_method, cv=3)
    model.fit(X, y)
    return model
