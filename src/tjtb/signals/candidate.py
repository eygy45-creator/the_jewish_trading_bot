"""Trade candidate evaluation: gates + logging reasons."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from tjtb.config.instrument_specs import InstrumentSpec
from tjtb.news.interface import NewsCalendarService
from tjtb.risk.engine import RiskEngine
from tjtb.schemas.signals import TradeDirection
from tjtb.signals.expected_value import ExpectedValueInputs, expected_value_net
from tjtb.utils.time import session_flags_for_timestamp


class TradeCandidate(BaseModel):
    ts: datetime
    instrument: str
    direction: TradeDirection
    probability: float
    ev_inputs: ExpectedValueInputs
    spread_ticks: float
    top_depth_contracts: float


def evaluate_trade_candidate(
    candidate: TradeCandidate,
    spec: InstrumentSpec,
    risk: RiskEngine,
    news: NewsCalendarService,
    *,
    max_spread_ticks: float,
    min_depth: float,
    min_probability: float,
) -> tuple[bool, list[str]]:
    """
    All gates must pass for a trade to be opened (Milestone 1 skeleton).

    TODO: plug real book liquidity metrics, regime confidence degradation, model calibration checks.
    """
    reasons: list[str] = []

    if candidate.instrument != spec.symbol:
        return False, ["instrument_mismatch"]

    if spec.candidate_tradable_windows:
        _, trad = session_flags_for_timestamp(candidate.ts, spec)
        if not trad:
            reasons.append("outside_candidate_tradable_window")

    if news.is_news_lockout(candidate.ts):
        reasons.append("news_lockout")

    if candidate.spread_ticks > max_spread_ticks:
        reasons.append("spread_too_wide")

    if candidate.top_depth_contracts < min_depth:
        reasons.append("insufficient_depth")

    if candidate.probability < min_probability:
        reasons.append("probability_below_threshold")

    ev = expected_value_net(candidate.ev_inputs, spec)
    if ev <= 0:
        reasons.append("ev_non_positive")

    ok_risk, risk_reasons = risk.allow_new_trade(candidate.ts)
    if not ok_risk:
        reasons.extend(risk_reasons)

    return len(reasons) == 0, reasons
