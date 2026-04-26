"""Risk engine observable state."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field


class RiskState(BaseModel):
    as_of: datetime
    trading_day: date
    open_positions: int = 0
    losing_trades_today: int = 0
    consecutive_losses: int = 0
    realized_pnl_today_currency: float = 0.0
    unrealized_pnl_currency: float = 0.0
    peak_equity_currency: float = 0.0
    drawdown_currency: float = 0.0
    drawdown_ticks: float = 0.0
    kill_switch_active: bool = False
    cooldown_until: datetime | None = None
