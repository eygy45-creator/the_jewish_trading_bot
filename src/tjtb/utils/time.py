"""Time/session helpers (exchange-local time for session bucket labeling)."""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

from tjtb.config.instrument_specs import InstrumentSpec, SessionWindow


def _local_time(dt: datetime, spec: InstrumentSpec) -> time:
    if dt.tzinfo is None:
        return time(dt.hour, dt.minute, dt.second)
    tz = ZoneInfo(spec.exchange_timezone)
    local = dt.astimezone(tz)
    return time(local.hour, local.minute, local.second)


def _in_window_local(local_t: time, window: SessionWindow) -> bool:
    """Return True if local_t falls in [start, end), allowing wrap past midnight."""
    s, e = window.start, window.end
    if s <= e:
        return s <= local_t < e
    return local_t >= s or local_t < e


def in_named_session_bucket(dt: datetime, spec: InstrumentSpec) -> str:
    local_t = _local_time(dt, spec)
    for w in spec.candidate_observation_windows:
        if _in_window_local(local_t, w):
            return w.name
    return "unknown"


def session_flags_for_timestamp(dt: datetime, spec: InstrumentSpec) -> tuple[bool, bool]:
    """Return (is_candidate_observation, is_candidate_tradable)."""
    local_t = _local_time(dt, spec)
    obs = any(_in_window_local(local_t, w) for w in spec.candidate_observation_windows)
    trad = any(_in_window_local(local_t, w) for w in spec.candidate_tradable_windows)
    return obs, trad
