"""Market data schemas: order book and trades."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class MarketEventType(str, Enum):
    BOOK = "book"
    TRADE = "trade"


class BookLevel(BaseModel):
    price: float
    size: int = Field(..., ge=0)


class OrderBookSnapshot(BaseModel):
    """Top-of-book plus optional depth."""

    ts: datetime
    instrument: str
    bids: list[BookLevel] = Field(default_factory=list)
    asks: list[BookLevel] = Field(default_factory=list)

    @property
    def best_bid(self) -> BookLevel | None:
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> BookLevel | None:
        return self.asks[0] if self.asks else None


class TradeEvent(BaseModel):
    ts: datetime
    instrument: str
    price: float
    size: int = Field(..., gt=0)
    aggressor: Literal["buy", "sell", "unknown"] = "unknown"
    is_sweep: bool = False
