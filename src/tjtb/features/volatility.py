"""Short-horizon realized volatility in ticks."""

from __future__ import annotations

import math

from tjtb.config.instrument_specs import InstrumentSpec


def realized_vol_ticks(mid_prices: list[float], spec: InstrumentSpec, window: int) -> float:
    """Std of tick returns over trailing `window` mids (requires len >= window)."""
    if len(mid_prices) < window or window < 2:
        return 0.0
    segment = mid_prices[-window:]
    rets: list[float] = []
    for a, b in zip(segment, segment[1:]):
        rets.append(spec.ticks_from_price_diff(a, b))
    if not rets:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / max(len(rets) - 1, 1)
    return math.sqrt(var)
