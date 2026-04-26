"""Session research: buckets, stability thresholds (research outputs drive tradability)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SessionResearchSettings(BaseModel):
    hour_bucket: str = Field(default="1H", description="Pandas offset alias for hourly buckets")
    min_trades_per_bucket: int = Field(default=30, ge=1)
    min_positive_expectancy_samples: int = Field(default=20, ge=1)
    stability_metric: str = Field(
        default="sign_consistency",
        description="Heuristic for walk-forward stability scaffolding",
    )
    max_drawdown_currency_threshold: float = Field(
        default=250.0,
        ge=0,
        description="Report flag if bucket max DD exceeds this (research signal, not a hard ban)",
    )
