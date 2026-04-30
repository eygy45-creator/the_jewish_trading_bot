"""Bybit demo execution skeleton with strict safety guards."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from tjtb.exchanges.bybit.client import BybitClient
from tjtb.risk.position_sizing import SizingResult, compute_linear_usdt_position_size


def _env_bool(name: str, default: bool = False) -> bool:
    v = str(os.environ.get(name, "")).strip().lower()
    if not v:
        return default
    return v in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ExecutionConfig:
    execution_mode: str
    bybit_testnet: bool
    bybit_symbol: str
    risk_per_trade: float
    max_order_notional: float | None
    max_risk_usd: float | None
    max_leverage: float
    kill_switch: bool

    @staticmethod
    def from_env() -> "ExecutionConfig":
        def _opt_float(name: str) -> float | None:
            raw = str(os.environ.get(name, "")).strip()
            if not raw:
                return None
            return float(raw)

        return ExecutionConfig(
            execution_mode=str(os.environ.get("EXECUTION_MODE", "paper")).strip().lower(),
            bybit_testnet=_env_bool("BYBIT_TESTNET", default=False),
            bybit_symbol=str(os.environ.get("BYBIT_SYMBOL", "BTCUSDT")).strip().upper(),
            risk_per_trade=float(os.environ.get("RISK_PER_TRADE", "0.0025")),
            max_order_notional=_opt_float("MAX_ORDER_NOTIONAL"),
            max_risk_usd=_opt_float("MAX_RISK_USD"),
            max_leverage=float(os.environ.get("MAX_LEVERAGE", "10")),
            kill_switch=_env_bool("KILL_SWITCH", default=True),
        )


def validate_execution_guards(cfg: ExecutionConfig) -> tuple[bool, str | None]:
    if cfg.execution_mode != "bybit_demo":
        return False, "execution_mode_not_bybit_demo"
    if not cfg.bybit_testnet:
        return False, "bybit_testnet_not_true"
    if cfg.kill_switch:
        return False, "kill_switch_active"
    return True, None


class BybitDemoExecution:
    """Dry-run payload builder with pre-trade safety checks."""

    def __init__(self, client: BybitClient | None = None, config: ExecutionConfig | None = None) -> None:
        self.client = client or BybitClient(testnet=True, dry_run=True)
        self.config = config or ExecutionConfig.from_env()

    def _guard_or_raise(self) -> None:
        ok, reason = validate_execution_guards(self.config)
        if not ok:
            raise RuntimeError(reason or "execution_guard_failed")

    def _load_credentials_if_needed(self) -> None:
        # Must fail only when execution is attempted.
        BybitClient.load_credentials_from_env()

    def build_entry_short(
        self,
        *,
        account_balance: float,
        entry_price: float,
        stop_price: float,
    ) -> dict[str, Any]:
        self._guard_or_raise()
        self._load_credentials_if_needed()

        specs = self.client.get_symbol_specs(self.config.bybit_symbol)
        sizing: SizingResult = compute_linear_usdt_position_size(
            account_balance=account_balance,
            entry_price=entry_price,
            stop_price=stop_price,
            qty_step=specs.qty_step,
            min_qty=specs.min_qty,
            risk_per_trade=self.config.risk_per_trade,
            max_order_notional=self.config.max_order_notional,
            max_risk_usd=self.config.max_risk_usd,
            max_leverage=self.config.max_leverage,
        )
        if not sizing.accepted:
            return {"ok": False, "reject_reason": sizing.reject_reason, "sizing": sizing}

        lev_payload = self.client.set_leverage(symbol=self.config.bybit_symbol, leverage=sizing.leverage)
        ord_payload = self.client.place_market_short(symbol=self.config.bybit_symbol, qty=sizing.qty)
        return {
            "ok": True,
            "mode": self.config.execution_mode,
            "symbol": self.config.bybit_symbol,
            "sizing": sizing,
            "set_leverage": lev_payload,
            "entry_order": ord_payload,
        }

    def build_close_position(self, *, qty: float) -> dict[str, Any]:
        self._guard_or_raise()
        self._load_credentials_if_needed()
        payload = self.client.close_position_reduce_only(symbol=self.config.bybit_symbol, qty=qty, side="Buy")
        return {"ok": True, "mode": self.config.execution_mode, "symbol": self.config.bybit_symbol, "close_order": payload}

