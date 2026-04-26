"""Price / size normalization hooks."""

from __future__ import annotations


def normalize_price(price: float, tick_size: float) -> float:
    """Round to nearest tick (simple grid snap)."""
    if tick_size <= 0:
        return price
    steps = round(price / tick_size)
    return steps * tick_size
