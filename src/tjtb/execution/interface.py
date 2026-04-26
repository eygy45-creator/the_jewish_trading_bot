"""Broker adapter interface (Tradovate later)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from tjtb.schemas.execution import FillEvent, OrderRequest


class BrokerAdapter(ABC):
    @abstractmethod
    def place_order(self, req: OrderRequest) -> str:
        raise NotImplementedError

    @abstractmethod
    def cancel_all(self, instrument: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def stream_fills(self):
        raise NotImplementedError
