"""
Barrier-based labeling for long/short continuation targets.

Labels are explicit and testable: hit +X ticks before -Y within max_horizon steps.
Triple-barrier extensions can reuse the same path with an extra neutral barrier.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class BarrierOutcome(str, Enum):
    HIT_FAVORABLE = "favorable_first"
    HIT_ADVERSE = "adverse_first"
    NEITHER = "neither"


class BarrierLabelConfig(BaseModel):
    favorable_ticks: int = Field(..., ge=1)
    adverse_ticks: int = Field(..., ge=1)
    max_horizon: int = Field(..., ge=1)


def _long_barrier_walk(
    entry_price: float,
    future_mid_prices: list[float],
    tick_size: float,
    cfg: BarrierLabelConfig,
) -> tuple[int, BarrierOutcome]:
    up = cfg.favorable_ticks * tick_size
    down = cfg.adverse_ticks * tick_size
    for px in future_mid_prices[: cfg.max_horizon]:
        hit_up = px >= entry_price + up
        hit_dn = px <= entry_price - down
        if hit_up and hit_dn:
            return 0, BarrierOutcome.NEITHER
        if hit_up:
            return 1, BarrierOutcome.HIT_FAVORABLE
        if hit_dn:
            return 0, BarrierOutcome.HIT_ADVERSE
    return 0, BarrierOutcome.NEITHER


def _short_barrier_walk(
    entry_price: float,
    future_mid_prices: list[float],
    tick_size: float,
    cfg: BarrierLabelConfig,
) -> tuple[int, BarrierOutcome]:
    up = cfg.favorable_ticks * tick_size
    down = cfg.adverse_ticks * tick_size
    for px in future_mid_prices[: cfg.max_horizon]:
        hit_up = px <= entry_price - up  # favorable for short is lower prices
        hit_dn = px >= entry_price + down
        if hit_up and hit_dn:
            return 0, BarrierOutcome.NEITHER
        if hit_up:
            return 1, BarrierOutcome.HIT_FAVORABLE
        if hit_dn:
            return 0, BarrierOutcome.HIT_ADVERSE
    return 0, BarrierOutcome.NEITHER


def barrier_hit_label_long_short(
    entry_price: float,
    future_mid_prices: list[float],
    tick_size: float,
    cfg: BarrierLabelConfig,
) -> tuple[int, int, BarrierOutcome, BarrierOutcome]:
    """
    Return (y_long, y_short, outcome_long, outcome_short).

    Each side is evaluated with its own first-touch ordering on the same path.
    """
    y_long, o_long = _long_barrier_walk(entry_price, future_mid_prices, tick_size, cfg)
    y_short, o_short = _short_barrier_walk(entry_price, future_mid_prices, tick_size, cfg)
    return y_long, y_short, o_long, o_short
