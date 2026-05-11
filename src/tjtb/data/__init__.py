"""Data helpers and recorders.

Heavy replay helpers (pandas) are lazy-loaded so lightweight modules such as
``tjtb.data.bybit_recorder`` run without requiring pandas at import time.
"""

from __future__ import annotations

from tjtb.data.normalization import normalize_price

__all__ = ["HistoricalReplay", "load_market_events_csv", "normalize_price"]


def __getattr__(name: str):
    if name == "HistoricalReplay":
        from tjtb.data.replay import HistoricalReplay as HistoricalReplay_cls

        return HistoricalReplay_cls
    if name == "load_market_events_csv":
        from tjtb.data.replay import load_market_events_csv as fn

        return fn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
