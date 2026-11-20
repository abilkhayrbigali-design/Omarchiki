"""Distractors X1-X10 (docs/DOMAIN.md, section 8). Correct by construction.

Each kind has its own random stream. Records created here carry `tag = kind`, so injectors never
modify them and they stay correct. The tag is internal and never published.

Every builder takes `sizes`: one entry per instance with the number of invoice documents that
instance creates (auditgym.world.sizing), and creates exactly that many.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta, timezone

from auditgym.util.rng import Stream
from auditgym.util.timeutil import format_offset
from auditgym.world.amounts import draw_minor, split_part
from auditgym.world.docs import add_payment, new_invoice, pay_full, pay_time
from auditgym.world.model import DistractorRec, WInvoice, WLine, World, WorldError, WPayment
from auditgym.world.names import DESCRIPTIONS, SUBSCRIPTIONS
from auditgym.world.timeline import DAY, HOUR

DISTRACTOR_CODES = {
    "split_payment": "X1",
    "multi_invoice_payment": "X2",
    "partial_open": "X3",
    "credit_note_refund": "X4",
    "subscription": "X5",
    "same_amount_orders": "X6",
    "unapplied_payment": "X7",
    "boundary_offset": "X8",
    "fx_spike": "X9",
    "posting_lag": "X10",
}


def _record(world: World, kind: str, docs: list[str], note: str = "") -> None:
    world.distractors.append(DistractorRec(kind, DISTRACTOR_CODES[kind], docs, note))


def _issued_between(world: World, s: Stream, lo: datetime, hi: datetime) -> datetime:
    return world.claim(lambda: world.tl.business_between(s, lo, hi))


def _instalment(
    world: World, s: Stream, inv: WInvoice, kind: str, ts: datetime, amount: int, applied: int
) -> WPayment:
    return add_payment(
        world,
        s,
        payer=inv.customer,
        ts=ts,
        currency=inv.currency,
        amount=amount,
        refs=[inv.number],
        plan=[(inv.number, applied)],
        tag=kind,
    )


# ---------------------------------------------------------------------------------- X1
def split_payment(world: World, s: Stream, sizes: list[int]) -> None:
    """One invoice settled by 2-3 payments. Sometimes the last one pays more than is still
    open: it is applied up to the open balance and the rest is unapplied."""
    tl = world.tl
    kind = "split_payment"
    for _ in sizes:
        issued = _issued_between(world, s, tl.range_start, tl.range_end - 20 * DAY)
        inv = new_invoice(world, s, tag=kind, issued=issued)
        k = s.randint(2, 3)
        over = s.chance(1, 3)
        pays: list[WPayment] = []
        remaining, t = inv.total, inv.issued
        for i in range(k):
            last = i == k - 1
            part = remaining if last else split_part(s, remaining, 30, 70)
            if part is None:
                part, last = remaining, True
            ts = pay_time(world, s, t, max_days=6)
            if ts is None:
                break
            extra = draw_minor(s, inv.currency, 5, 300) if last and over else 0
            pays.append(_instalment(world, s, inv, kind, ts, part + extra, part))
            remaining -= part
            t = ts
            if last:
                break
        note = f"{len(pays)} payments"
        if over and remaining == 0 and pays:
            note += "; the last one overpays, the rest of it is unapplied"
        _record(world, kind, [inv.key] + [p.key for p in pays], note)


# ---------------------------------------------------------------------------------- X2
def multi_invoice_payment(world: World, s: Stream, sizes: list[int]) -> None:
    tl = world.tl
    kind = "multi_invoice_payment"
    for m in sizes:
        cust = s.choice(world.customers)
        t0 = tl.business_between(s, tl.range_start, tl.range_end - 16 * DAY)
        invs = [
            new_invoice(
                world,
                s,
                tag=kind,
                customer=cust,
                issued=_issued_between(world, s, t0, t0 + 4 * DAY),
            )
            for _ in range(m)
        ]
        order = s.shuffled(invs)
        plan = [(inv.number, inv.total) for inv in order]
        note = "settles all listed invoices"
        if s.chance(1, 2):
            part = split_part(s, order[-1].total)
            if part is not None:
                plan[-1] = (order[-1].number, part)
                note = "last listed invoice stays partly open"
        latest = max(inv.issued for inv in invs)
        ts = pay_time(world, s, latest, max_days=10)
        docs = [inv.key for inv in invs]
        if ts is not None:
            pay = add_payment(
                world,
                s,
                payer=cust.key,
                ts=ts,
                currency=cust.currency,
                amount=sum(p for _, p in plan),
                refs=[inv.number for inv in order],
                plan=plan,
                tag=kind,
            )
            docs.append(pay.key)
        _record(world, kind, docs, note)


# ---------------------------------------------------------------------------------- X3
def partial_open(world: World, s: Stream, sizes: list[int]) -> None:
    tl = world.tl
    kind = "partial_open"
    for _ in sizes:
        issued = _issued_between(world, s, tl.range_start, tl.range_end - 12 * DAY)
        inv = new_invoice(world, s, tag=kind, issued=issued)
        docs = [inv.key]
        part = split_part(s, inv.total)
        ts = pay_time(world, s, inv.issued, max_days=10) if part is not None else None
        if part is not None and ts is not None:
            pay = add_payment(
                world,
                s,
                payer=inv.customer,
                ts=ts,
                currency=inv.currency,
                amount=part,
                refs=[inv.number],
                plan=[(inv.number, part)],
                tag=kind,
            )
            docs.append(pay.key)
        _record(world, kind, docs)


# ---------------------------------------------------------------------------------- X4
def credit_note_refund(world: World, s: Stream, sizes: list[int]) -> None:
    tl = world.tl
    kind = "credit_note_refund"
    for _ in sizes:
        cust = s.choice(world.customers)
        issued = _issued_between(world, s, tl.range_start, tl.range_end - 20 * DAY)
        inv = new_invoice(world, s, tag=kind, customer=cust, issued=issued)
        docs = [inv.key]
        p1 = pay_full(world, s, inv, tag=kind, max_days=5)
        after = inv.issued if p1 is None else p1.ts
        if p1 is not None:
            docs.append(p1.key)
        line = inv.lines[0]
        returned = s.randint(1, line.quantity)
        cn_lines = [
            WLine(f"Return: {line.description}", -returned, line.unit_price, world.mint("ord"))
        ]
        cn_issued = _issued_between(world, s, after + HOUR, after + 5 * DAY)
        cn = new_invoice(world, s, tag=kind, customer=cust, issued=cn_issued, lines=cn_lines)
        docs.append(cn.key)
        ts = world.claim(
            lambda cn=cn: tl.payment_between(s, cn.issued + 2 * HOUR, cn.issued + 5 * DAY)
        )
        refund = add_payment(
            world,
            s,
            payer=cust.key,
            ts=ts,
            currency=cn.currency,
            amount=cn.total,
            refs=[cn.number],
            plan=[(cn.number, cn.total)],
            tag=kind,
        )
        docs.append(refund.key)
        _record(world, kind, docs, "credit note and refund")


# ---------------------------------------------------------------------------------- X5
def _subscription_slots(world: World) -> list[tuple[date, date]]:
    """Pairs of the same day-of-month in two consecutive months, both inside the data range."""
    tl = world.tl
    local = tl.local(tl.range_start)
    y, m = local.year, local.month
    out = []
    while datetime(y, m, 1, tzinfo=tl.tz) < tl.range_end:
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        for dd in range(1, 29):
            a, b = date(y, m, dd), date(ny, nm, dd)
            if tl.midnight(a) >= tl.range_start and tl.midnight(b) + DAY <= tl.range_end - DAY:
                out.append((a, b))
        y, m = ny, nm
    return out


def subscription(world: World, s: Stream, sizes: list[int]) -> None:
    """The same amount to the same customer in two consecutive months: two invoices with
    different service_period and order_ref."""
    tl = world.tl
    kind = "subscription"
    slots = _subscription_slots(world)
    if not slots:
        raise WorldError("subscription_slots")
    for _ in sizes:
        cust = s.choice(world.customers)
        price = draw_minor(s, cust.currency, 50, 2000)
        desc = s.choice(SUBSCRIPTIONS)
        docs = []
        for day in s.choice(slots):
            issued = world.claim(lambda day=day: tl.business_time(s, day))
            period = f"{day.year:04d}-{day.month:02d}"
            lines = [WLine(desc, 1, price, world.mint("ord"), period)]
            inv = new_invoice(world, s, tag=kind, customer=cust, issued=issued, lines=lines)
            docs.append(inv.key)
            if s.chance(3, 4):
                pay = pay_full(world, s, inv, tag=kind, max_days=10)
                if pay is not None:
                    docs.append(pay.key)
        _record(
            world, kind, docs, "same amount every month, different service_period and order_ref"
        )


# ---------------------------------------------------------------------------------- X6
def same_amount_orders(world: World, s: Stream, sizes: list[int]) -> None:
    tl = world.tl
    kind = "same_amount_orders"
    for _ in sizes:
        cust = s.choice(world.customers)
        desc = s.choice(DESCRIPTIONS)
        qty = s.randint(1, 5)
        price = draw_minor(s, cust.currency, 20, 3000)
        first = tl.local_date(tl.range_start)
        span = (tl.local_date(tl.range_end) - first).days - 3
        day = first + timedelta(days=s.randint(0, span))
        t1 = world.claim(lambda day=day: tl.business_time(s, day))

        def second(day: date = day, t1: datetime = t1) -> datetime:
            for _ in range(32):
                t = tl.business_time(s, day)
                if abs(t - t1) >= HOUR:
                    return t
            raise WorldError("same_amount_orders_time")

        t2 = world.claim(second)
        docs = []
        for t in (t1, t2):
            lines = [WLine(desc, qty, price, world.mint("ord"))]
            inv = new_invoice(world, s, tag=kind, customer=cust, issued=t, lines=lines)
            docs.append(inv.key)
            if s.chance(1, 2):
                pay = pay_full(world, s, inv, tag=kind)
                if pay is not None:
                    docs.append(pay.key)
        _record(world, kind, docs, "two orders, same customer, amount and day")


# ---------------------------------------------------------------------------------- X7
def unapplied_payment(world: World, s: Stream, sizes: list[int]) -> None:
    tl = world.tl
    kind = "unapplied_payment"
    for _ in sizes:
        payer = s.choice(world.customers)
        ts = world.claim(lambda: tl.payment_between(s, tl.range_start, tl.range_end - HOUR))
        pay = add_payment(
            world,
            s,
            payer=payer.key,
            ts=ts,
            currency=payer.currency,
            amount=draw_minor(s, payer.currency, 50, 5000),
            refs=None,
            plan=[],
            tag=kind,
        )
        _record(world, kind, [pay.key], "no invoice reference, booked to UNAPPLIED")


# ---------------------------------------------------------------------------------- X8
X8_MAX_DISTANCE = 3 * HOUR  # X8 documents lie within this distance of a boundary midnight


def _date_flipping_offsets(world: World, t: datetime) -> list[str]:
    local_d = world.tl.local_date(t)
    out = []
    for hours in range(-12, 15):
        tz = timezone(timedelta(hours=hours))
        text = format_offset(tz)
        if text != world.tl.tz_text and t.astimezone(tz).date() != local_d:
            out.append(text)
    return out


def boundary_offset(world: World, s: Stream, sizes: list[int]) -> None:
    tl = world.tl
    kind = "boundary_offset"
    for _ in sizes:
        b = s.choice(tl.boundaries())
        side = s.choice(["before", "after"])
        # Probe the farthest instant from midnight we may draw: an offset that flips the date
        # there flips it for every nearer instant too. One side always works for offsets in
        # [-12:00, +14:00].
        probe = b - X8_MAX_DISTANCE if side == "before" else b + X8_MAX_DISTANCE
        if not _date_flipping_offsets(world, probe):
            side = "after" if side == "before" else "before"

        def draw(b: datetime = b, side: str = side) -> datetime:
            secs = s.randint(60, int(X8_MAX_DISTANCE.total_seconds()) - 1)
            return b - timedelta(seconds=secs) if side == "before" else b + timedelta(seconds=secs)

        t = world.claim(draw)
        options = _date_flipping_offsets(world, t)
        if not options:
            raise WorldError("boundary_offset_no_offset")
        display = timezone(_offset_delta(s.choice(options)))
        if side == "before":
            # Post later the same local day, before the boundary: same period.
            post = t + timedelta(seconds=s.randint(0, int((b - t).total_seconds()) - 1))
        else:
            post = t + timedelta(seconds=s.randint(0, 8 * 3600))
        if s.chance(1, 2):
            inv = new_invoice(world, s, tag=kind, issued=t, display_tz=display, lag=False)
            inv.post_ts = post
            docs = [inv.key]
            if s.chance(1, 2):
                pay = pay_full(world, s, inv, tag=kind)
                if pay is not None:
                    docs.append(pay.key)
            note = f"invoice written in {format_offset(display)}"
        else:
            lo = max(tl.range_start, t - 15 * DAY)
            inv = new_invoice(world, s, tag=kind, issued=_issued_between(world, s, lo, t - 2 * DAY))
            pay = add_payment(
                world,
                s,
                payer=inv.customer,
                ts=t,
                currency=inv.currency,
                amount=inv.total,
                refs=[inv.number],
                plan=[(inv.number, inv.total)],
                tag=kind,
                display_tz=display,
                lag=False,
            )
            pay.post_ts = post
            docs = [inv.key, pay.key]
            note = f"payment written in {format_offset(display)}"
        _record(world, kind, docs, note)


def _offset_delta(text: str) -> timedelta:
    sign = 1 if text[0] == "+" else -1
    return sign * timedelta(hours=int(text[1:3]), minutes=int(text[4:6]))


# ---------------------------------------------------------------------------------- X9
def _spike_payment_time(world: World, s: Stream, day: date) -> datetime:
    """A payment instant whose policy rate date is `day`."""
    tl = world.tl
    if world.fx_rule == "payment_local_date":
        return tl.business_time(s, day)
    if world.fx_rule == "payment_local_date_minus_1":
        return tl.business_time(s, day + DAY)
    midnight_utc = datetime(day.year, day.month, day.day, tzinfo=UTC)
    return tl.local(midnight_utc + timedelta(seconds=s.randint(8 * 3600, 18 * 3600)))


def fx_spike(world: World, s: Stream, sizes: list[int]) -> None:
    """Foreign payments whose policy rate date is a spike day; converted correctly."""
    tl = world.tl
    kind = "fx_spike"
    del sizes  # the spikes were planned before the FX table was built (see build.plan_spikes)
    for spike in world.spikes:
        cur = spike["currency"]
        day = date.fromisoformat(spike["day"])
        cust = s.choice([c for c in world.customers if c.currency == cur])
        ts = world.claim(lambda day=day: _spike_payment_time(world, s, day))
        lo = max(tl.range_start, ts - 12 * DAY)
        inv = new_invoice(
            world, s, tag=kind, customer=cust, issued=_issued_between(world, s, lo, ts - DAY)
        )
        pay = add_payment(
            world,
            s,
            payer=cust.key,
            ts=ts,
            currency=cur,
            amount=inv.total,
            refs=[inv.number],
            plan=[(inv.number, inv.total)],
            tag=kind,
        )
        _record(
            world,
            kind,
            [inv.key, pay.key],
            f"{cur} rate jumps by factor {spike['factor']} on {spike['day']}",
        )


# ---------------------------------------------------------------------------------- X10
MAX_LAG_DAYS = 6


def _max_lag_days(world: World, t: datetime) -> int:
    """Longest lag in whole days that keeps the posting in the same period class as t."""
    end = world.tl.class_end(t)
    if end is None:
        return MAX_LAG_DAYS
    return min(MAX_LAG_DAYS, (world.tl.local_date(end) - world.tl.local_date(t)).days - 1)


def _month_crossing_starts(world: World) -> list[datetime]:
    """Calendar-month starts that a lag of up to MAX_LAG_DAYS can cross without leaving the
    period class (only possible for periods that are not calendar months, or in long margins)."""
    tl = world.tl
    out = []
    for m in tl.month_starts():
        lo, hi = m - 5 * DAY, m + (MAX_LAG_DAYS - 1) * DAY
        if lo >= tl.range_start + DAY and tl.period_class(lo) == tl.period_class(hi):
            out.append(m)
    return out


def posting_lag(world: World, s: Stream, sizes: list[int]) -> None:
    """Postings days after the document, in the same period class. Where the calendar allows,
    half of them cross a calendar-month start, so "another month" is not "another period"."""
    tl = world.tl
    kind = "posting_lag"
    for _ in sizes:
        crossings = _month_crossing_starts(world)
        if crossings and s.chance(1, 2):
            m = s.choice(crossings)
            t = _issued_between(world, s, m - 5 * DAY, m)
            first = (tl.local_date(m) - tl.local_date(t)).days
            lag_days = s.randint(first, min(_max_lag_days(world, t), first + 3))
        else:

            def draw() -> datetime:
                for _ in range(64):
                    t = tl.business_between(s, tl.range_start + DAY, tl.range_end - 2 * DAY)
                    if _max_lag_days(world, t) >= 1:
                        return t
                raise WorldError("posting_lag_time")

            t = world.claim(draw)
            lag_days = s.randint(1, _max_lag_days(world, t))
        post = tl.business_time(s, tl.local_date(t) + timedelta(days=lag_days))
        if tl.period_class(post) != tl.period_class(t) or post <= t:
            raise WorldError("posting_lag_class")
        if s.chance(1, 2):
            inv = new_invoice(world, s, tag=kind, issued=t, lag=False)
            inv.post_ts = post
            docs = [inv.key]
            note = f"invoice posted {lag_days} day(s) late, same period"
        else:
            lo = max(tl.range_start, t - 15 * DAY)
            inv = new_invoice(
                world, s, tag=kind, issued=_issued_between(world, s, lo, t - 2 * HOUR)
            )
            pay = add_payment(
                world,
                s,
                payer=inv.customer,
                ts=t,
                currency=inv.currency,
                amount=inv.total,
                refs=[inv.number],
                plan=[(inv.number, inv.total)],
                tag=kind,
                lag=False,
            )
            pay.post_ts = post
            docs = [inv.key, pay.key]
            note = f"payment posted {lag_days} day(s) late, same period"
        if not tl.same_calendar_month(t, post):
            note += ", another calendar month"
        _record(world, kind, docs, note)


BUILDERS: dict[str, Callable[[World, Stream, list[int]], None]] = {
    "split_payment": split_payment,
    "multi_invoice_payment": multi_invoice_payment,
    "partial_open": partial_open,
    "credit_note_refund": credit_note_refund,
    "subscription": subscription,
    "same_amount_orders": same_amount_orders,
    "unapplied_payment": unapplied_payment,
    "boundary_offset": boundary_offset,
    "fx_spike": fx_spike,
    "posting_lag": posting_lag,
}
