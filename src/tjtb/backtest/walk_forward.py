"""Walk-forward splitting utilities (time-ordered, no random split)."""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd


def time_based_splits(df: pd.DataFrame, time_col: str, n_splits: int) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield (train, test) expanding windows sorted by `time_col`."""
    d = df.sort_values(time_col).reset_index(drop=True)
    n = len(d)
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    edges = [int(n * i / n_splits) for i in range(n_splits + 1)]
    for k in range(1, n_splits):
        train_end = edges[k]
        test_end = edges[k + 1]
        if train_end < 10 or test_end - train_end < 5:
            continue
        yield d.iloc[:train_end], d.iloc[train_end:test_end]
