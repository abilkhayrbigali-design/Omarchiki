"""Timestamps with fixed UTC offsets.

Episodes use fixed offsets ("+05:00"), never IANA zones: tz databases change between versions
and would break reproducibility. Every timestamp is ISO-8601 with an explicit offset and whole
seconds, e.g. "2026-03-31T23:30:00+05:00".
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta, timezone
from typing import Literal

PeriodClass = Literal["before", "in", "after"]

_OFFSET_RE = re.compile(r"^([+-])(\d{2}):(\d{2})$")
_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}([+-]\d{2}:\d{2}|Z)$")


class TimeError(ValueError):
    """Invalid timestamp, date or offset."""


def parse_offset(text: str) -> timezone:
    """'+05:00' -> timezone(timedelta(hours=5))."""
    m = _OFFSET_RE.match(text)
    if not m:
        raise TimeError(f"invalid UTC offset {text!r}, expected like '+05:00'")
    sign = 1 if m.group(1) == "+" else -1
    hours, minutes = int(m.group(2)), int(m.group(3))
    if hours > 14 or minutes >= 60:
        raise TimeError(f"UTC offset out of range: {text!r}")
    return timezone(sign * timedelta(hours=hours, minutes=minutes))


def format_offset(tz: timezone) -> str:
    delta = tz.utcoffset(None)
    assert delta is not None
    total = int(delta.total_seconds())
    sign = "+" if total >= 0 else "-"
    total = abs(total)
    return f"{sign}{total // 3600:02d}:{(total % 3600) // 60:02d}"


def parse_ts(text: str) -> datetime:
    """Parse an ISO-8601 timestamp with an explicit offset. Naive timestamps are rejected."""
    if not isinstance(text, str) or not _TS_RE.match(text):
        raise TimeError(f"invalid timestamp {text!r}, expected YYYY-MM-DDTHH:MM:SS+HH:MM")
    try:
        value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TimeError(f"invalid timestamp {text!r}") from exc
    if value.tzinfo is None:
        raise TimeError(f"timestamp without offset: {text!r}")
    return value


def format_ts(value: datetime, tz: timezone | None = None) -> str:
    """Render an aware datetime as ISO-8601 with offset and whole seconds.

    If `tz` is given, the instant is first expressed in that offset.
    """
    if value.tzinfo is None:
        raise TimeError("cannot format a naive datetime")
    if tz is not None:
        value = value.astimezone(tz)
    return value.replace(microsecond=0).isoformat(timespec="seconds")


def to_instant(value: str | datetime) -> datetime:
    return parse_ts(value) if isinstance(value, str) else value


def local_date(value: str | datetime, tz: timezone) -> date:
    """Calendar date of the instant in the given fixed offset."""
    return to_instant(value).astimezone(tz).date()


def utc_date(value: str | datetime) -> date:
    return to_instant(value).astimezone(UTC).date()


def period_class(value: str | datetime, start: str | datetime, end: str | datetime) -> PeriodClass:
    """Position of an instant relative to the half-open period [start, end)."""
    t, s, e = to_instant(value), to_instant(start), to_instant(end)
    if t < s:
        return "before"
    if t < e:
        return "in"
    return "after"


def parse_date(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except (TypeError, ValueError) as exc:
        raise TimeError(f"invalid date {text!r}, expected YYYY-MM-DD") from exc


def format_date(value: date) -> str:
    return value.isoformat()


def daterange(first: date, last: date) -> Iterator[date]:
    """Dates from `first` to `last`, both included."""
    day = first
    while day <= last:
        yield day
        day += timedelta(days=1)
