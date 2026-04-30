"""Bybit client skeleton (dry-run payload generation only)."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BybitCredentials:
    api_key: str
    api_secret: str


@dataclass(frozen=True)
class SymbolSpecs:
    symbol: str
    qty_step: float
    min_qty: float
    tick_size: float


class BybitClient:
    """
    Dry-run only by default.

    This class intentionally avoids network side effects in this PR.
    """

    def __init__(self, *, testnet: bool = True, dry_run: bool = True) -> None:
        self.testnet = bool(testnet)
        self.dry_run = bool(dry_run)

    @staticmethod
    def load_credentials_from_env() -> BybitCredentials:
        key = str(os.environ.get("BYBIT_API_KEY", "")).strip()
        secret = str(os.environ.get("BYBIT_API_SECRET", "")).strip()
        if not key or not secret:
            raise RuntimeError("missing_bybit_credentials")
        return BybitCredentials(api_key=key, api_secret=secret)

    def get_account_balance(self) -> float:
        """Execution-layer can override by reading live account later."""
        env_bal = str(os.environ.get("BYBIT_BALANCE_OVERRIDE", "")).strip()
        if env_bal:
            try:
                return float(env_bal)
            except ValueError:
                pass
        return 0.0

    def get_symbol_specs(self, symbol: str) -> SymbolSpecs:
        """
        Minimal, configurable specs for linear USDT symbols.
        Env overrides supported for tests/smoke.
        """
        qty_step = float(os.environ.get("BYBIT_QTY_STEP", "0.001"))
        min_qty = float(os.environ.get("BYBIT_MIN_QTY", "0.001"))
        tick_size = float(os.environ.get("BYBIT_TICK_SIZE", "0.1"))
        return SymbolSpecs(symbol=symbol, qty_step=qty_step, min_qty=min_qty, tick_size=tick_size)

    def set_leverage(self, *, symbol: str, leverage: float) -> dict[str, Any]:
        payload = {
            "op": "set_leverage",
            "symbol": symbol,
            "buyLeverage": f"{float(leverage):.8f}".rstrip("0").rstrip("."),
            "sellLeverage": f"{float(leverage):.8f}".rstrip("0").rstrip("."),
        }
        return {"dry_run": self.dry_run, "payload": payload}

    def place_market_short(self, *, symbol: str, qty: float, client_order_id: str | None = None) -> dict[str, Any]:
        payload = {
            "category": "linear",
            "symbol": symbol,
            "side": "Sell",
            "orderType": "Market",
            "qty": f"{float(qty):.8f}".rstrip("0").rstrip("."),
            "timeInForce": "IOC",
            "orderLinkId": client_order_id or f"tjtb_short_{int(time.time() * 1000)}",
        }
        return {"dry_run": self.dry_run, "payload": payload}

    def close_position_reduce_only(
        self,
        *,
        symbol: str,
        qty: float,
        side: str = "Buy",
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": "Market",
            "qty": f"{float(qty):.8f}".rstrip("0").rstrip("."),
            "timeInForce": "IOC",
            "reduceOnly": True,
            "orderLinkId": client_order_id or f"tjtb_close_{int(time.time() * 1000)}",
        }
        return {"dry_run": self.dry_run, "payload": payload}

    def get_open_position(self, *, symbol: str) -> dict[str, Any]:
        return {"dry_run": self.dry_run, "symbol": symbol, "size": 0.0}

    def reconcile_state(self, *, symbol: str) -> dict[str, Any]:
        return {"symbol": symbol, "position": self.get_open_position(symbol=symbol), "ok": True}

