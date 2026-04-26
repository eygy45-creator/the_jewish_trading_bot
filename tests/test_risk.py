from datetime import datetime, timedelta, timezone

from tjtb.config.instrument_specs import MNQ_DEFAULT_SPEC
from tjtb.config.risk_settings import RiskSettings
from tjtb.risk.engine import RiskEngine


def test_risk_blocks_after_max_losses() -> None:
    rs = RiskSettings(max_losing_trades_per_day=2, loss_cooldown_seconds=0)
    eng = RiskEngine(rs, MNQ_DEFAULT_SPEC)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    eng.set_open_positions(0, now)
    eng.on_fill_pnl(now, -50.0, is_loss=True)
    eng.on_fill_pnl(now, -50.0, is_loss=True)
    ok, reasons = eng.allow_new_trade(now)
    assert ok is False
    assert "max_losing_trades_per_day" in reasons


def test_cooldown_blocks() -> None:
    rs = RiskSettings(max_losing_trades_per_day=10, loss_cooldown_seconds=3600)
    eng = RiskEngine(rs, MNQ_DEFAULT_SPEC)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    eng.on_fill_pnl(now, -1.0, is_loss=True)
    ok, _ = eng.allow_new_trade(now + timedelta(seconds=10))
    assert ok is False
