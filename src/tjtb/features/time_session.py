"""Time-of-day and session bucket features."""

from __future__ import annotations

import math
from datetime import datetime

from tjtb.config.instrument_specs import InstrumentSpec
from tjtb.utils.time import in_named_session_bucket


def time_of_day_features(dt: datetime) -> dict[str, float]:
    """Simple cyclical encoding (UTC); extend with exchange-local later."""
    h = dt.hour + dt.minute / 60.0
    return {
        "tod_sin": math.sin(2 * math.pi * h / 24.0),
        "tod_cos": math.cos(2 * math.pi * h / 24.0),
        "hour_utc": float(dt.hour),
    }


def session_bucket_feature(dt: datetime, spec: InstrumentSpec) -> dict[str, float]:
    name = in_named_session_bucket(dt, spec)
    # One-hot style numeric map for tree models; expand as needed
    buckets = [w.name for w in spec.candidate_observation_windows] + ["unknown"]
    return {f"session_{b}": 1.0 if name == b else 0.0 for b in buckets}
