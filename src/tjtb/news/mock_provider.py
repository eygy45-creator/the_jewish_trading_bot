"""In-memory calendar for tests and local runs."""

from __future__ import annotations

from datetime import datetime

from tjtb.news.interface import CalendarEvent, NewsCalendarService


class MockNewsProvider(NewsCalendarService):
    def __init__(self, events: list[CalendarEvent] | None = None) -> None:
        self._events = events or []

    def add(self, event: CalendarEvent) -> None:
        self._events.append(event)

    def get_upcoming_events(self, start_time: datetime, end_time: datetime) -> list[CalendarEvent]:
        return [e for e in self._events if e.start <= end_time and (e.end or e.start) >= start_time]
