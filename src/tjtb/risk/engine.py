"""Risk engine: independent of signal generation."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from tjtb.config.instrument_specs import InstrumentSpec
from tjtb.config.risk_settings import RiskSettings
from tjtb.schemas.risk import RiskState


class RiskEngine:
    """Enforces per-day limits, cooldowns, drawdown caps, and kill switch."""

    def __init__(self, settings: RiskSettings, instrument: InstrumentSpec) -> None:
        self.settings = settings
        self.instrument = instrument
        self._state = RiskState(
            as_of=datetime.now(timezone.utc),
            trading_day=date.today(),
            peak_equity_currency=0.0,
        )

    @property
    def state(self) -> RiskState:
        return self._state

    def reset_for_new_day(self, as_of: datetime, trading_day: date) -> None:
        self._state = RiskState(
            as_of=as_of,
            trading_day=trading_day,
            peak_equity_currency=self._state.realized_pnl_today_currency
            + self._state.unrealized_pnl_currency,
        )

    def activate_kill_switch(self, reason: str) -> None:
        _ = reason
        self._state.kill_switch_active = True

    def on_fill_pnl(
        self,
        as_of: datetime,
        realized_delta_currency: float,
        is_loss: bool,
    ) -> None:
        self._state.as_of = as_of
        self._state.realized_pnl_today_currency += realized_delta_currency
        if is_loss:
            self._state.losing_trades_today += 1
            self._state.consecutive_losses += 1
            self._state.cooldown_until = as_of + timedelta(seconds=self.settings.loss_cooldown_seconds)
        else:
            self._state.consecutive_losses = 0

        equity = self._state.realized_pnl_today_currency + self._state.unrealized_pnl_currency
        self._state.peak_equity_currency = max(self._state.peak_equity_currency, equity)
        self._state.drawdown_currency = self._state.peak_equity_currency - equity
        self._state.drawdown_ticks = (
            self._state.drawdown_currency / self.instrument.tick_value
            if self.instrument.tick_value > 0
            else 0.0
        )

    def update_unrealized(self, unrealized_currency: float, as_of: datetime) -> None:
        self._state.unrealized_pnl_currency = unrealized_currency
        self._state.as_of = as_of
        equity = self._state.realized_pnl_today_currency + unrealized_currency
        self._state.peak_equity_currency = max(self._state.peak_equity_currency, equity)
        self._state.drawdown_currency = self._state.peak_equity_currency - equity
        self._state.drawdown_ticks = (
            self._state.drawdown_currency / self.instrument.tick_value
            if self.instrument.tick_value > 0
            else 0.0
        )

    def set_open_positions(self, n: int, as_of: datetime) -> None:
        self._state.open_positions = n
        self._state.as_of = as_of

    def allow_new_trade(self, now: datetime) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if self.settings.kill_switch or self._state.kill_switch_active:
            return False, ["kill_switch"]

        if self._state.cooldown_until and now < self._state.cooldown_until:
            return False, ["cooldown_active"]

        if self._state.open_positions >= self.settings.max_open_positions:
            return False, ["max_open_positions"]

        if self._state.losing_trades_today >= self.settings.max_losing_trades_per_day:
            return False, ["max_losing_trades_per_day"]

        if self._state.realized_pnl_today_currency <= -self.settings.max_daily_loss_currency:
            return False, ["max_daily_loss_currency"]

        if self.settings.max_daily_loss_ticks is not None:
            if self._state.drawdown_ticks >= self.settings.max_daily_loss_ticks:
                return False, ["max_daily_loss_ticks"]

        return True, reasons
