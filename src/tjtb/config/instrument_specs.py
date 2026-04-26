"""Configurable instrument specification (MNQ default; swappable later)."""

from __future__ import annotations

from datetime import time

from pydantic import BaseModel, Field


class SessionWindow(BaseModel):
    """Candidate observation/tradable window in exchange-local time (not final permission)."""

    name: str = Field(..., description="Human-readable bucket name, e.g. 'us_cash_open'")
    start: time
    end: time


class InstrumentSpec(BaseModel):
    """All prices and risk in ticks; currency via tick_value."""

    symbol: str = Field(..., description="Primary contract symbol, e.g. MNQ")
    exchange: str = Field(default="CME", description="Exchange id for routing/logging")
    tick_size: float = Field(..., gt=0, description="Minimum price increment")
    tick_value: float = Field(..., gt=0, description="USD per tick per contract")
    point_value: float | None = Field(
        default=None,
        description="If set, USD per index point (else derived from tick_size/tick_value)",
    )
    currency: str = Field(default="USD")
    default_commission_per_contract: float = Field(default=0.62, ge=0)
    default_slippage_ticks: float = Field(default=1.0, ge=0)
    # TODO: wire full CME Globex + maintenance calendar; use exchange calendar provider later.
    exchange_timezone: str = Field(default="America/Chicago", description="IANA TZ for sessions")
    candidate_observation_windows: list[SessionWindow] = Field(
        default_factory=list,
        description="Hours where the system may observe/score (research-defined)",
    )
    candidate_tradable_windows: list[SessionWindow] = Field(
        default_factory=list,
        description="Candidate execution windows; final eligibility from OOS research only",
    )

    def ticks_from_price_diff(self, price_a: float, price_b: float) -> float:
        return abs(price_a - price_b) / self.tick_size


MNQ_DEFAULT_SPEC = InstrumentSpec(
    symbol="MNQ",
    exchange="CME",
    tick_size=0.25,
    tick_value=0.50,
    point_value=None,
    default_commission_per_contract=0.62,
    default_slippage_ticks=1.0,
    exchange_timezone="America/Chicago",
    candidate_observation_windows=[
        SessionWindow(name="asia", start=time(17, 0), end=time(2, 0)),
        SessionWindow(name="europe_london_overlap", start=time(2, 0), end=time(7, 0)),
        SessionWindow(name="us_pre_cash", start=time(7, 0), end=time(8, 30)),
        SessionWindow(name="us_cash_open", start=time(8, 30), end=time(10, 0)),
        SessionWindow(name="midday", start=time(10, 0), end=time(14, 0)),
        SessionWindow(name="late_session", start=time(14, 0), end=time(16, 0)),
    ],
    candidate_tradable_windows=[],
)
