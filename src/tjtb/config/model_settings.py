"""Baseline model training configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelSettings(BaseModel):
    random_state: int = Field(default=42)
    calibration_method: str = Field(
        default="sigmoid",
        description="sklearn CalibratedClassifierCV method; 'isotonic' or 'sigmoid'",
    )
    walk_forward_train_fraction: float = Field(
        default=0.7,
        gt=0,
        lt=1,
        description="Fraction of sorted-by-time rows for train in each fold (rest = val)",
    )
    probability_threshold: float = Field(default=0.55, gt=0, lt=1)
