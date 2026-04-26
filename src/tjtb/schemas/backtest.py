"""Aggregated backtest / walk-forward outputs."""

from __future__ import annotations

from pydantic import BaseModel, Field


class BacktestResultSummary(BaseModel):
    gross_pnl: float
    net_pnl: float
    slippage_adjusted_pnl: float
    hit_rate: float
    expectancy: float
    profit_factor: float
    max_drawdown: float
    mean_mae_ticks: float
    mean_mfe_ticks: float
    avg_holding_bars: float
    n_trades: int
