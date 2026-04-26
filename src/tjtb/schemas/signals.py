"""Signal and direction enums."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class TradeDirection(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class SignalEvent(BaseModel):
    ts: datetime
    instrument: str
    direction: TradeDirection
    probability: float = Field(..., ge=0, le=1)
    expected_value_net: float
    feature_vector: dict[str, float] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
