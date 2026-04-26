"""
Historical replay from CSV/Parquet.

CSV columns (Milestone 1):
  ts, kind, instrument,
  bid_px, bid_sz, ask_px, ask_sz,
  trade_px, trade_sz, aggressor, is_sweep (optional)

kind is 'book' or 'trade'.

TODO: Parquet column mapping, venue-specific normalization, nanosecond timestamps,
      L3/delta book reconstruction, compressed feeds.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd

from tjtb.schemas.market import BookLevel, OrderBookSnapshot, TradeEvent


def load_market_events_csv(path: str | Path, max_events: int | None = None) -> Iterator[OrderBookSnapshot | TradeEvent]:
    df = pd.read_csv(path, parse_dates=["ts"])
    if max_events is not None:
        df = df.head(max_events)
    has_sweep = "is_sweep" in df.columns
    for _, row in df.iterrows():
        kind = str(row["kind"]).lower()
        inst = str(row["instrument"])
        ts: datetime = row["ts"].to_pydatetime() if hasattr(row["ts"], "to_pydatetime") else row["ts"]
        if kind == "book":
            bids = []
            asks = []
            if pd.notna(row.get("bid_px")) and pd.notna(row.get("bid_sz")):
                bids.append(BookLevel(price=float(row["bid_px"]), size=int(row["bid_sz"])))
            if pd.notna(row.get("ask_px")) and pd.notna(row.get("ask_sz")):
                asks.append(BookLevel(price=float(row["ask_px"]), size=int(row["ask_sz"])))
            yield OrderBookSnapshot(ts=ts, instrument=inst, bids=bids, asks=asks)
        elif kind == "trade":
            agg: Literal["buy", "sell", "unknown"] = "unknown"
            if pd.notna(row.get("aggressor")):
                a = str(row["aggressor"]).lower()
                if a in ("buy", "sell"):
                    agg = a  # type: ignore[assignment]
            is_sweep = bool(row["is_sweep"]) if has_sweep and pd.notna(row.get("is_sweep")) else False
            yield TradeEvent(
                ts=ts,
                instrument=inst,
                price=float(row["trade_px"]),
                size=int(row["trade_sz"]),
                aggressor=agg,
                is_sweep=is_sweep,
            )
        else:
            continue


class HistoricalReplay:
    """Thin iterator wrapper for tests and the simple backtest loop."""

    def __init__(self, events: Iterator[OrderBookSnapshot | TradeEvent]) -> None:
        self._events = events

    def __iter__(self):
        return self._events
