"""
Helpers for the Streamlit research dashboard: safe CSV loading and sampling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_FILES: dict[str, Path] = {
    "book_state": Path("data/parsed/book_state.csv"),
    "labeled_book": Path("data/parsed/labeled_book.csv"),
    "feature_matrix": Path("data/parsed/feature_matrix.csv"),
    "dataset": Path("data/parsed/dataset.csv"),
}


def try_read_csv(path: Path) -> pd.DataFrame | None:
    """Load CSV if file exists; return None on missing path or read errors."""
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:  # noqa: BLE001 — research UI must not crash
        logger.warning("Failed to read %s: %s", path, exc)
        return None


def coerce_datetime_columns(df: pd.DataFrame, columns: tuple[str, ...] = ("ts",)) -> pd.DataFrame:
    out = df.copy()
    for c in columns:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], utc=True, errors="coerce")
    return out


def apply_plot_sample(df: pd.DataFrame, max_rows: int, mode: str) -> pd.DataFrame:
    """Deterministic subsample for plotting."""
    if max_rows <= 0 or len(df) <= max_rows:
        return df
    mode_l = mode.lower()
    if mode_l == "head":
        return df.head(max_rows).copy()
    if mode_l == "random":
        samp = df.sample(n=max_rows, random_state=0)
        if "ts" in samp.columns:
            return samp.sort_values("ts").copy()
        return samp.sort_index().copy()
    return df.tail(max_rows).copy()


def first_existing_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def numeric_columns(df: pd.DataFrame, exclude: set[str] | None = None) -> list[str]:
    ex = exclude or set()
    cols: list[str] = []
    for c in df.columns:
        if c in ex:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def summarize_ts_range(df: pd.DataFrame, ts_col: str = "ts") -> tuple[Any, Any] | None:
    if ts_col not in df.columns:
        return None
    s = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    s = s.dropna()
    if s.empty:
        return None
    return s.min(), s.max()


def file_loading_report(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    """Per-key status: exists, rows, ts_min, ts_max."""
    out: dict[str, dict[str, Any]] = {}
    for key, p in paths.items():
        df = try_read_csv(p)
        if df is None:
            out[key] = {"path": str(p), "found": False, "rows": 0, "ts_range": None}
            continue
        rng = summarize_ts_range(df, "ts")
        out[key] = {"path": str(p), "found": True, "rows": int(len(df)), "ts_range": rng}
    return out
