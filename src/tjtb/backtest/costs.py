"""Transaction cost helpers."""

from __future__ import annotations

from tjtb.config.instrument_specs import InstrumentSpec


def roundtrip_cost_currency(
    spec: InstrumentSpec,
    *,
    slippage_ticks: float,
    commission_per_contract: float,
    contracts: int = 1,
) -> float:
    slip = slippage_ticks * spec.tick_value * contracts
    return slip + commission_per_contract * contracts
