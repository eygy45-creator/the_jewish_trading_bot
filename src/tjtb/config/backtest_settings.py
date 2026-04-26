"""Backtest / simulation defaults."""

from __future__ import annotations

from pydantic import BaseModel, Field


class BacktestSettings(BaseModel):
    slippage_ticks: float = Field(default=1.0, ge=0)
    commission_per_contract: float = Field(default=0.62, ge=0)
    max_position_contracts: int = Field(default=1, ge=1)
    walk_forward_n_splits: int = Field(default=3, ge=1)
    min_rows_per_fold: int = Field(default=200, ge=10)
