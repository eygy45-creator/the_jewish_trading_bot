"""Session / clock state for observation vs tradable research."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class SessionState(BaseModel):
    """Named buckets for research features (not final permission)."""

    ts: datetime
    hour_utc: int = Field(..., ge=0, le=23)
    named_session_bucket: str = Field(
        default="unknown",
        description="Maps to candidate window name when inside a configured bucket",
    )
    is_candidate_tradable: bool = False
    is_candidate_observation: bool = False
