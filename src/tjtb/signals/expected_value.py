"""Explicit expected value after fees and slippage (currency per contract)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field

from tjtb.config.instrument_specs import InstrumentSpec


class ExpectedValueInputs(BaseModel):
    probability_success: float = Field(..., ge=0, le=1)
    expected_favorable_ticks: float = Field(..., ge=0)
    expected_adverse_ticks: float = Field(..., ge=0)
    expected_slippage_ticks: float = Field(default=1.0, ge=0)
    commission_currency: float = Field(default=0.62, ge=0)


def expected_value_net(p: ExpectedValueInputs, spec: InstrumentSpec) -> float:
    """
    EV = p * favorable_ticks * tick_value - (1-p) * adverse_ticks * tick_value
         - roundtrip costs in ticks * tick_value - commission (per open/close as modeled).

    Milestone 1 uses a single commission charge here; extend to per-leg later.
    """
    gross = (
        p.probability_success * p.expected_favorable_ticks * spec.tick_value
        - (1.0 - p.probability_success) * p.expected_adverse_ticks * spec.tick_value
    )
    slip = p.expected_slippage_ticks * spec.tick_value
    return gross - slip - p.commission_currency


@dataclass(frozen=True)
class MicrostructureEVConfig:
    """
    Directional EV in **price units** (crypto/futures-agnostic).

    If ``tick_size`` is set, ``reward`` and ``risk`` are interpreted as **ticks**
    and converted to price via ``ticks * tick_size`` for EV math.
    ``costs`` is always in the same final unit as the reward/risk leg (e.g. USD
    or index points), supplied by the caller.
    """

    reward: float
    risk: float
    costs: float
    tick_size: float | None = None


def reward_risk_in_price_units(cfg: MicrostructureEVConfig) -> tuple[float, float]:
    if cfg.tick_size is None or cfg.tick_size <= 0:
        return cfg.reward, cfg.risk
    return cfg.reward * cfg.tick_size, cfg.risk * cfg.tick_size


def expected_value_microstructure(
    p_plus: float,
    p_minus: float,
    cfg: MicrostructureEVConfig,
) -> float:
    """
    EV = P(+1) * reward - P(-1) * risk - costs (flat ``P(0)`` has no directional PnL).

    ``p_plus`` / ``p_minus`` should be calibrated marginal probabilities for labels +1 / -1.
    """
    rw, rk = reward_risk_in_price_units(cfg)
    return float(p_plus) * rw - float(p_minus) * rk - float(cfg.costs)


def expected_value_microstructure_vector(
    p_plus: np.ndarray,
    p_minus: np.ndarray,
    cfg: MicrostructureEVConfig,
) -> np.ndarray:
    """Vectorized EV for reporting / thresholding."""
    rw, rk = reward_risk_in_price_units(cfg)
    return np.asarray(p_plus, dtype=np.float64) * rw - np.asarray(p_minus, dtype=np.float64) * rk - float(cfg.costs)
