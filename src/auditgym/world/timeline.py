"""Calendar of one episode: company offset, reporting period, data range and time draws.

All datetimes here are aware and expressed in the company's fixed offset. Comparisons are by
instant, so a timestamp written in another offset compares correctly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from auditgym.util.rng import Stream
from auditgym.util.timeutil import parse_offset

SECOND = timedelta(seconds=1)
MINUTE = timedelta(minutes=1)
HOUR = timedelta(hours=1)
DAY = timedelta(days=1)

# Documents within this distance before a period boundary can be posted across it (period_cutoff).
BOUNDARY_WINDOW = timedelta(days=3)
BUSINESS_START = 8 * 3600
BUSINESS_END = 20 * 3600  # exclusive


def _next_month(year: int, month: int) -> tuple[int, int]:
    return (year + 1, 1) if month == 12 else (year, month + 1)


@dataclass(frozen=True)
class Timeline:
    tz: timezone
    tz_text: str
    period_start: datetime
    period_end: datetime
    range_start: datetime
    range_end: datetime

    @classmethod
    def build(cls, tz_text: str, year: int, month: int, margin_days: int) -> Timeline:
        tz = parse_offset(tz_text)
        start = datetime(year, month, 1, tzinfo=tz)
        ny, nm = _next_month(year, month)
        end = datetime(ny, nm, 1, tzinfo=tz)
        margin = timedelta(days=margin_days)
        return cls(tz, tz_text, start, end, start - margin, end + margin)

    # ------------------------------------------------------------------ calendar
    def local(self, t: datetime) -> datetime:
        return t.astimezone(self.tz)

    def local_date(self, t: datetime) -> date:
        return t.astimezone(self.tz).date()

    def midnight(self, d: date) -> datetime:
        return datetime(d.year, d.month, d.day, tzinfo=self.tz)

    def period_class(self, t: datetime) -> str:
        if t < self.period_start:
            return "before"
        if t < self.period_end:
            return "in"
        return "after"

    def next_month_start(self, t: datetime) -> datetime:
        local = self.local(t)
        ny, nm = _next_month(local.year, local.month)
        return datetime(ny, nm, 1, tzinfo=self.tz)

    def same_month(self, a: datetime, b: datetime) -> bool:
        la, lb = self.local(a), self.local(b)
        return (la.year, la.month) == (lb.year, lb.month)

    def boundaries(self) -> tuple[datetime, datetime]:
        return (self.period_start, self.period_end)

    def boundary_before(self, t: datetime) -> datetime | None:
        """The period boundary b with b - BOUNDARY_WINDOW <= t < b, if any."""
        for b in self.boundaries():
            if b - BOUNDARY_WINDOW <= t < b:
                return b
        return None

    # ------------------------------------------------------------------ draws
    def at_seconds(self, d: date, seconds: int) -> datetime:
        return self.midnight(d) + timedelta(seconds=seconds)

    def business_time(self, s: Stream, d: date) -> datetime:
        return self.at_seconds(d, s.randint(BUSINESS_START, BUSINESS_END - 1))

    def any_time(self, s: Stream, d: date) -> datetime:
        return self.at_seconds(d, s.randint(0, 86399))

    def instant_between(self, s: Stream, lo: datetime, hi: datetime) -> datetime:
        """Uniform whole-second instant in [lo, hi)."""
        span = int((hi - lo).total_seconds())
        if span <= 0:
            raise ValueError("empty time interval")
        return self.local(lo) + timedelta(seconds=s.randint(0, span - 1))

    def business_between(self, s: Stream, lo: datetime, hi: datetime) -> datetime:
        """Instant in [lo, hi) in business hours when possible, else any instant in [lo, hi)."""
        first, last = self.local_date(lo), self.local_date(hi)
        days = (last - first).days
        for _ in range(12):
            d = first + timedelta(days=s.randint(0, days))
            t = self.business_time(s, d)
            if lo <= t < hi:
                return t
        return self.instant_between(s, lo, hi)

    def payment_between(self, s: Stream, lo: datetime, hi: datetime) -> datetime:
        """Bank payments arrive mostly in business hours, sometimes at any time of day."""
        if s.chance(3, 4):
            return self.business_between(s, lo, hi)
        return self.instant_between(s, lo, hi)
