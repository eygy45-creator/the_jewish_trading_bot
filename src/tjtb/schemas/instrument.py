"""Runtime instrument reference (links to InstrumentSpec in config)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class InstrumentRef(BaseModel):
    symbol: str
    exchange: str = Field(default="CME")
