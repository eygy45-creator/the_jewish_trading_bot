"""Feature engineering hyperparameters (config-driven, no magic in formulas)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FeatureSettings(BaseModel):
    queue_top_k: int = Field(default=5, ge=1, le=50)
    queue_level_weights: list[float] | None = Field(
        default=None,
        description="If set, length must equal queue_top_k; else uniform weights",
    )
    order_flow_window_trades: int = Field(default=50, ge=1)
    volatility_window_ticks: int = Field(default=100, ge=2)
    microprice_eps: float = Field(default=1e-9, gt=0)
