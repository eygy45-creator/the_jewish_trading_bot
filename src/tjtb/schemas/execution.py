"""Execution domain: orders, fills, positions."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    MARKETABLE_LIMIT = "marketable_limit"
    LIMIT = "limit"


class TimeInForce(str, Enum):
    DAY = "day"
    IOC = "ioc"
    FOK = "fok"
    GTC = "gtc"


class OrderRequest(BaseModel):
    client_order_id: str
    instrument: str
    side: OrderSide
    quantity: int = Field(..., ge=1)
    order_type: OrderType = OrderType.MARKETABLE_LIMIT
    limit_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.IOC
    reason: str = Field(default="", description="Explainable decision text for logs")


class FillEvent(BaseModel):
    ts: datetime
    instrument: str
    side: OrderSide
    quantity: int = Field(..., ge=1)
    fill_price: float
    expected_price: float | None = None
    slippage_ticks: float | None = None
    commission: float = Field(default=0.0, ge=0)
    order_id: str | None = None


class PositionState(BaseModel):
    instrument: str
    quantity: int = Field(default=0)
    avg_price: float | None = None
    unrealized_pnl_currency: float = 0.0
    realized_pnl_currency: float = 0.0
