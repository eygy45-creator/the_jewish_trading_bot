"""
Train a calibrated multinomial logistic regression on merged microstructure dataset.

Strict time-based splits (no shuffle). Class imbalance handled via ``class_weight``.
Calibration: Platt scaling (``sigmoid``) on a held-out temporal validation slice.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

DATASET_DEFAULT = Path("data/parsed/dataset.csv")
OUT_DIR_DEFAULT = Path("runs/microstructure")
READ_BUFFER = 1024 * 1024


def load_dataset(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="", buffering=READ_BUFFER) as f:
        r = csv.reader(f)
        header = next(r)
        rows = [row for row in r if row]
    if not header or not rows:
        raise ValueError("empty dataset")
    label_name = "label"
    exclude = {"ts", label_name, "mid_price", "up_move", "down_move"}
    feat_idx = [i for i, h in enumerate(header) if h not in exclude]
    li = header.index(label_name)
    X = np.asarray([[float(row[i]) for i in feat_idx] for row in rows], dtype=np.float64)
    y = np.asarray([int(float(row[li])) for row in rows], dtype=np.int64)
    feats = [header[i] for i in feat_idx]
    return feats, X, y


def time_splits(n: int, train_frac: float = 0.6, val_frac: float = 0.2) -> tuple[slice, slice, slice]:
    if n < 30:
        raise ValueError("dataset too small for reliable splits")
    i1 = max(1, min(int(n * train_frac), n - 2))
    i2 = max(i1 + 1, min(int(n * (train_frac + val_frac)), n - 1))
    return slice(0, i1), slice(i1, i2), slice(i2, n)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    p = argparse.ArgumentParser(description="Train calibrated multinomial logistic model.")
    p.add_argument("--dataset", type=Path, default=DATASET_DEFAULT)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--val-frac", type=float, default=0.2)
    args = p.parse_args(argv)
    if not args.dataset.is_file():
        logger.error("Missing dataset: %s", args.dataset)
        return 1
    feats, X, y = load_dataset(args.dataset)
    n = X.shape[0]
    tr_sl, va_sl, te_sl = time_splits(n, train_frac=args.train_frac, val_frac=args.val_frac)
    X_train, y_train = X[tr_sl], y[tr_sl]
    X_val, y_val = X[va_sl], y[va_sl]
    X_test, y_test = X[te_sl], y[te_sl]

    base = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    multi_class="multinomial",
                    solver="lbfgs",
                    random_state=0,
                ),
            ),
        ]
    )
    base.fit(X_train, y_train)
    cal = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
    cal.fit(X_val, y_val)

    classes_list = [int(c) for c in cal.classes_.tolist()]
    cls_to_j = {int(c): j for j, c in enumerate(classes_list)}

    y_pred = cal.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=[-1, 0, 1]).tolist()
    report = classification_report(y_test, y_pred, labels=[-1, 0, 1], zero_division=0)

    proba = cal.predict_proba(X_test)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "model.joblib"
    joblib.dump({"model": cal, "feature_names": feats, "classes": classes_list}, model_path)

    probs_path = args.out_dir / "test_probs.csv"
    with probs_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["y_true", "y_pred", "p_neg", "p_zero", "p_plus"])
        for i in range(len(y_test)):
            rowp = proba[i]
            p_neg = float(rowp[cls_to_j[-1]]) if -1 in cls_to_j else 0.0
            p_zero = float(rowp[cls_to_j[0]]) if 0 in cls_to_j else 0.0
            p_plus = float(rowp[cls_to_j[1]]) if 1 in cls_to_j else 0.0
            w.writerow([int(y_test[i]), int(y_pred[i]), p_neg, p_zero, p_plus])

    metrics: dict[str, Any] = {
        "n_total": n,
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_test": int(X_test.shape[0]),
        "accuracy_test": acc,
        "confusion_matrix_test": cm,
        "classification_report_test": report,
        "feature_names": feats,
        "classes": classes_list,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    logger.info("saved model=%s metrics accuracy=%.6f", model_path, acc)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
