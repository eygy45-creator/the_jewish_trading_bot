from __future__ import annotations

import pytest

from tjtb.exchanges.bybit.client import BybitClient
from tjtb.exchanges.bybit.execution import BybitDemoExecution, ExecutionConfig, validate_execution_guards


def test_imports_require_no_credentials() -> None:
    cfg = ExecutionConfig.from_env()
    client = BybitClient(testnet=True, dry_run=True)
    assert isinstance(cfg.execution_mode, str)
    assert client.dry_run is True


def test_execution_refuses_when_mode_not_bybit_demo() -> None:
    cfg = ExecutionConfig(
        execution_mode="paper",
        bybit_testnet=True,
        bybit_symbol="BTCUSDT",
        risk_per_trade=0.0025,
        max_order_notional=None,
        max_risk_usd=None,
        max_leverage=10.0,
        kill_switch=False,
    )
    ok, reason = validate_execution_guards(cfg)
    assert ok is False
    assert reason == "execution_mode_not_bybit_demo"


def test_execution_refuses_when_testnet_not_true() -> None:
    cfg = ExecutionConfig(
        execution_mode="bybit_demo",
        bybit_testnet=False,
        bybit_symbol="BTCUSDT",
        risk_per_trade=0.0025,
        max_order_notional=None,
        max_risk_usd=None,
        max_leverage=10.0,
        kill_switch=False,
    )
    ok, reason = validate_execution_guards(cfg)
    assert ok is False
    assert reason == "bybit_testnet_not_true"


def test_execution_refuses_when_kill_switch_true() -> None:
    cfg = ExecutionConfig(
        execution_mode="bybit_demo",
        bybit_testnet=True,
        bybit_symbol="BTCUSDT",
        risk_per_trade=0.0025,
        max_order_notional=None,
        max_risk_usd=None,
        max_leverage=10.0,
        kill_switch=True,
    )
    ok, reason = validate_execution_guards(cfg)
    assert ok is False
    assert reason == "kill_switch_active"


def test_execution_build_raises_without_credentials_even_when_guards_pass() -> None:
    cfg = ExecutionConfig(
        execution_mode="bybit_demo",
        bybit_testnet=True,
        bybit_symbol="BTCUSDT",
        risk_per_trade=0.0025,
        max_order_notional=1_000_000.0,
        max_risk_usd=1_000.0,
        max_leverage=10.0,
        kill_switch=False,
    )
    exe = BybitDemoExecution(client=BybitClient(testnet=True, dry_run=True), config=cfg)
    with pytest.raises(RuntimeError):
        exe.build_entry_short(account_balance=10_000.0, entry_price=100.0, stop_price=101.0)


def test_execution_build_generates_dry_run_payload(monkeypatch) -> None:
    monkeypatch.setenv("BYBIT_API_KEY", "k")
    monkeypatch.setenv("BYBIT_API_SECRET", "s")
    cfg = ExecutionConfig(
        execution_mode="bybit_demo",
        bybit_testnet=True,
        bybit_symbol="BTCUSDT",
        risk_per_trade=0.0025,
        max_order_notional=1_000_000.0,
        max_risk_usd=1_000.0,
        max_leverage=10.0,
        kill_switch=False,
    )
    exe = BybitDemoExecution(client=BybitClient(testnet=True, dry_run=True), config=cfg)
    res = exe.build_entry_short(account_balance=10_000.0, entry_price=100.0, stop_price=101.0)
    assert res["ok"] is True
    order = res["entry_order"]["payload"]
    assert order["category"] == "linear"
    assert order["symbol"] == "BTCUSDT"
    assert order["side"] == "Sell"
    assert order["orderType"] == "Market"

