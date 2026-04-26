"""
Minimal event-driven replay driver (Milestone 1).

TODO: fill simulation, partial fills, latency model, order queue position, venue specifics.
"""

from __future__ import annotations

from collections.abc import Iterator

from tjtb.schemas.market import OrderBookSnapshot, TradeEvent


def run_replay_count(events: Iterator[OrderBookSnapshot | TradeEvent]) -> dict[str, int]:
    """Sanity helper: count event types in a stream."""
    n_book = 0
    n_trade = 0
    for ev in events:
        if isinstance(ev, OrderBookSnapshot):
            n_book += 1
        elif isinstance(ev, TradeEvent):
            n_trade += 1
    return {"books": n_book, "trades": n_trade}
