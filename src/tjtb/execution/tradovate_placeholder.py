"""Tradovate integration placeholder."""

from __future__ import annotations

from tjtb.execution.interface import BrokerAdapter
from tjtb.schemas.execution import OrderRequest


class TradovateAdapterPlaceholder(BrokerAdapter):
    # TODO: OAuth / API keys, websocket market data, REST order routing, reconnect/backoff.
    def place_order(self, req: OrderRequest) -> str:
        raise NotImplementedError(f"Tradovate not wired yet: {req.client_order_id}")

    def cancel_all(self, instrument: str) -> None:
        raise NotImplementedError(instrument)

    def stream_fills(self):
        raise NotImplementedError
