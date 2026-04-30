from tjtb.risk.position_sizing import compute_linear_usdt_position_size


def test_sizing_normal_case() -> None:
    r = compute_linear_usdt_position_size(
        account_balance=10_000.0,
        entry_price=100.0,
        stop_price=101.0,
        qty_step=0.001,
        min_qty=0.001,
        risk_per_trade=0.0025,
        max_order_notional=100_000.0,
        max_risk_usd=1000.0,
        max_leverage=10.0,
    )
    assert r.accepted
    assert r.qty > 0
    assert r.stop_distance == 1.0
    assert abs(r.risk_usd - 25.0) < 1e-9


def test_zero_stop_reject() -> None:
    r = compute_linear_usdt_position_size(
        account_balance=10_000.0,
        entry_price=100.0,
        stop_price=100.0,
        qty_step=0.001,
        min_qty=0.001,
    )
    assert not r.accepted
    assert r.reject_reason == "invalid_stop_distance"


def test_min_qty_reject() -> None:
    r = compute_linear_usdt_position_size(
        account_balance=100.0,
        entry_price=1000.0,
        stop_price=1010.0,
        qty_step=0.001,
        min_qty=1.0,
    )
    assert not r.accepted
    assert r.reject_reason == "qty_below_min_qty"


def test_max_notional_reject() -> None:
    r = compute_linear_usdt_position_size(
        account_balance=10_000.0,
        entry_price=100.0,
        stop_price=99.9,
        qty_step=0.001,
        min_qty=0.001,
        max_order_notional=1000.0,
    )
    assert not r.accepted
    assert r.reject_reason == "max_order_notional_exceeded"


def test_max_risk_reject() -> None:
    r = compute_linear_usdt_position_size(
        account_balance=10_000.0,
        entry_price=100.0,
        stop_price=99.0,
        qty_step=0.001,
        min_qty=0.001,
        max_risk_usd=10.0,
    )
    assert not r.accepted
    assert r.reject_reason == "max_risk_usd_exceeded"

