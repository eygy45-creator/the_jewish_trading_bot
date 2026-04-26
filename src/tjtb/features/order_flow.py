"""Order-flow rolling features: aggression, delta, cancellation proxy, sweeps."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from tjtb.config.feature_settings import FeatureSettings
from tjtb.features.queue_imbalance import queue_imbalance_l1
from tjtb.schemas.market import TradeEvent


@dataclass
class OrderFlowRolling:
    """Fixed-window trade statistics for aggressive buy/sell imbalance."""

    settings: FeatureSettings
    _trades: deque[TradeEvent] = field(default_factory=deque)

    def push(self, trade: TradeEvent) -> None:
        self._trades.append(trade)
        while len(self._trades) > self.settings.order_flow_window_trades:
            self._trades.popleft()

    def aggressive_buy_volume(self) -> float:
        return float(sum(t.size for t in self._trades if t.aggressor == "buy"))

    def aggressive_sell_volume(self) -> float:
        return float(sum(t.size for t in self._trades if t.aggressor == "sell"))

    def sweep_count(self) -> int:
        return sum(1 for t in self._trades if t.is_sweep)

    def trade_intensity_imbalance(self) -> float:
        b = self.aggressive_buy_volume()
        s = self.aggressive_sell_volume()
        return queue_imbalance_l1(b, s)

    def cancellation_pressure_proxy(self) -> float:
        """Milestone 1 proxy: unknown aggressor volume / total (real cancels need book deltas)."""
        unknown = float(sum(t.size for t in self._trades if t.aggressor == "unknown"))
        total = float(sum(t.size for t in self._trades)) or 1.0
        return unknown / total


def cumulative_delta(trades: list[TradeEvent]) -> float:
    delta = 0.0
    for t in trades:
        if t.aggressor == "buy":
            delta += t.size
        elif t.aggressor == "sell":
            delta -= t.size
    return delta
