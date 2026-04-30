"""Position sizing helpers for linear USDT perpetual contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SizingResult:
    qty: float
    risk_usd: float
    notional: float
    stop_distance: float
    leverage: float
    reject_reason: str | None = None

    @property
    def accepted(self) -> bool:
        return self.reject_reason is None


def _floor_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return math.floor(value / step) * step


def compute_linear_usdt_position_size(
    *,
    account_balance: float,
    entry_price: float,
    stop_price: float,
    qty_step: float,
    min_qty: float,
    risk_per_trade: float = 0.0025,
    max_order_notional: float | None = None,
    max_risk_usd: float | None = None,
    max_leverage: float = 10.0,
) -> SizingResult:
    """
    Compute quantity from fixed risk fraction.

    risk_usd = account_balance * risk_per_trade
    stop_distance = abs(stop_price - entry_price)
    qty = risk_usd / stop_distance
    """
    if account_balance <= 0:
        return SizingResult(0.0, 0.0, 0.0, 0.0, 1.0, "invalid_account_balance")
    stop_distance = abs(float(stop_price) - float(entry_price))
    if stop_distance <= 0:
        return SizingResult(0.0, 0.0, 0.0, stop_distance, 1.0, "invalid_stop_distance")

    risk_usd = float(account_balance) * float(risk_per_trade)
    if max_risk_usd is not None and risk_usd > float(max_risk_usd):
        return SizingResult(0.0, risk_usd, 0.0, stop_distance, 1.0, "max_risk_usd_exceeded")

    raw_qty = risk_usd / stop_distance
    qty = _floor_to_step(raw_qty, float(qty_step))
    if qty <= 0 or qty < float(min_qty):
        return SizingResult(qty, risk_usd, 0.0, stop_distance, 1.0, "qty_below_min_qty")

    notional = qty * float(entry_price)
    if max_order_notional is not None and notional > float(max_order_notional):
        return SizingResult(qty, risk_usd, notional, stop_distance, 1.0, "max_order_notional_exceeded")

    # Choose minimal leverage required for this notional against account balance.
    leverage = notional / float(account_balance) if account_balance > 0 else float(max_leverage)
    leverage = max(1.0, leverage)
    if leverage > float(max_leverage):
        return SizingResult(qty, risk_usd, notional, stop_distance, leverage, "max_leverage_exceeded")

    return SizingResult(qty, risk_usd, notional, stop_distance, leverage, None)

