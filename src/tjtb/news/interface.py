"""News / economic calendar abstraction (dependency-injected)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import IntEnum

from pydantic import BaseModel, Field


class EventImportance(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class CalendarEvent(BaseModel):
    title: str
    start: datetime
    end: datetime | None = None
    importance: EventImportance = EventImportance.MEDIUM
    affected_asset_classes: list[str] = Field(default_factory=list)
    affected_symbols: list[str] = Field(default_factory=list)


class NewsCalendarService(ABC):
    """Implementations: MockNewsProvider, TradingEconomics (TODO), FXStreet (TODO)."""

    lockout_before_minutes: int = 15
    lockout_after_minutes: int = 10
    min_importance: EventImportance = EventImportance.MEDIUM

    @abstractmethod
    def get_upcoming_events(self, start_time: datetime, end_time: datetime) -> list[CalendarEvent]:
        raise NotImplementedError

    def is_news_lockout(self, now: datetime) -> bool:
        for ev in self._iter_relevant_events(now):
            if ev.importance < self.min_importance:
                continue
            start = ev.start - timedelta(minutes=self.lockout_before_minutes)
            end = (ev.end or ev.start) + timedelta(minutes=self.lockout_after_minutes)
            if start <= now <= end:
                return True
        return False

    def _iter_relevant_events(self, now: datetime):
        """Default scan window around `now` for upcoming/nearby events."""
        horizon = timedelta(hours=6)
        return self.get_upcoming_events(now - horizon, now + horizon)
