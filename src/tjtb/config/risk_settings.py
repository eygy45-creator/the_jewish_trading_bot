"""Risk engine parameters (ticks + currency tracking)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RiskSettings(BaseModel):
    max_losing_trades_per_day: int = Field(default=2, ge=0)
    max_daily_loss_currency: float = Field(default=500.0, ge=0)
    loss_cooldown_seconds: int = Field(default=600, ge=0)
    max_open_positions: int = Field(default=1, ge=1)
    kill_switch: bool = Field(default=False)
    max_daily_loss_ticks: float | None = Field(
        default=None,
        ge=0,
        description="Optional cap in ticks (converted with tick_value at runtime)",
    )
