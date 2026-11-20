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


# Kinds of reporting period and how often each is drawn. Boundaries are always local midnights.
PERIOD_KINDS = ("month", "quarter", "fiscal_4w", "fiscal_5w")
PERIOD_WEIGHTS = (6, 2, 1, 1)


def _next_month(year: int, month: int) -> tuple[int, int]:
    return (year + 1, 1) if month == 12 else (year, month + 1)


def _add_months(year: int, month: int, n: int) -> tuple[int, int]:
    total = year * 12 + (month - 1) + n
    return total // 12, total % 12 + 1


def draw_period(s: Stream, year: int) -> tuple[str, date, date]:
    """(kind, first day, day after the last day) of a reporting period in `year`.

    month: a calendar month; quarter: a calendar quarter; fiscal_4w / fiscal_5w: 28 or 35 days
    starting on any day of the year.
    """
    kind = s.weighted_choice(list(PERIOD_KINDS), list(PERIOD_WEIGHTS))
    if kind == "month":
        month = s.randint(1, 12)
        ny, nm = _next_month(year, month)
        return kind, date(year, month, 1), date(ny, nm, 1)
    if kind == "quarter":
        month = 3 * s.randint(0, 3) + 1
        ny, nm = _add_months(year, month, 3)
        return kind, date(year, month, 1), date(ny, nm, 1)
    first = date(year, 1, 1) + timedelta(days=s.randint(0, 364))
    return kind, first, first + timedelta(days=28 if kind == "fiscal_4w" else 35)


@dataclass(frozen=True)
class Timeline:
    tz: timezone
    tz_text: str
    period_start: datetime
    period_end: datetime
    range_start: datetime
    range_end: datetime
    period_kind: str = "month"

    @classmethod
    def build(
        cls, tz_text: str, first: date, end: date, margin_days: int, kind: str = "month"
    ) -> Timeline:
        """Period [first, end) as local midnights, data range widened by margin_days each side."""
        tz = parse_offset(tz_text)
        start = datetime(first.year, first.month, first.day, tzinfo=tz)
        stop = datetime(end.year, end.month, end.day, tzinfo=tz)
        margin = timedelta(days=margin_days)
        return cls(tz, tz_text, start, stop, start - margin, stop + margin, kind)

    @property
    def period_days(self) -> int:
        return (self.local_date(self.period_end) - self.local_date(self.period_start)).days

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

    def in_period(self, t: datetime) -> bool:
        return self.period_start <= t < self.period_end

    def class_end(self, t: datetime) -> datetime | None:
        """First instant of the next period class after t (None after the period)."""
        if t < self.period_start:
            return self.period_start
        if t < self.period_end:
            return self.period_end
        return None

    def same_calendar_month(self, a: datetime, b: datetime) -> bool:
        la, lb = self.local(a), self.local(b)
        return (la.year, la.month) == (lb.year, lb.month)

    def month_starts(self) -> list[datetime]:
        """Local midnights starting a calendar month inside the data range."""
        local = self.local(self.range_start)
        y, m = local.year, local.month
        out = []
        while True:
            y, m = _next_month(y, m)
            t = datetime(y, m, 1, tzinfo=self.tz)
            if t >= self.range_end:
                return out
            out.append(t)

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
