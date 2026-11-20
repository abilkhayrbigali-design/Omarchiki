"""Creating invoices and payments in the world. Shared by regular history, distractors, top-up."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from auditgym.util.rng import Stream
from auditgym.world.amounts import draw_minor, split_part
from auditgym.world.model import Customer, WInvoice, WLine, World, WPayment
from auditgym.world.names import DESCRIPTIONS
from auditgym.world.timeline import BOUNDARY_WINDOW, DAY, HOUR

PAYMENT_TERMS_DAYS = (14, 30, 30, 45, 60)


def small_lag(world: World, s: Stream, t: datetime) -> datetime:
    """Posting time of an ordinary document: the same instant, or a few hours later the same day."""
    if s.chance(1, 4):
        later = t + timedelta(seconds=s.randint(300, 6 * 3600))
        if world.tl.local_date(later) == world.tl.local_date(t):
            return later
    return t


def draw_lines(world: World, s: Stream, currency: str) -> list[WLine]:
    n = s.weighted_choice([1, 2, 3], [6, 3, 1])
    order = world.mint("ord")
    lines = []
    for desc in s.sample(DESCRIPTIONS, n):
        bucket = s.weighted_choice(["one", "few", "many"], [4, 4, 2])
        qty = 1 if bucket == "one" else s.randint(2, 5) if bucket == "few" else s.randint(6, 20)
        lines.append(WLine(desc, qty, draw_minor(s, currency, 20, 4000), order))
    return lines


def issue_time(world: World, s: Stream) -> datetime:
    """Ordinary issue time: mostly uniform over the data range, often near a period boundary."""
    tl = world.tl
    if s.chance(1, 4):
        b = s.choice(tl.boundaries())
        return tl.business_between(s, b - BOUNDARY_WINDOW, b + BOUNDARY_WINDOW)
    return tl.business_between(s, tl.range_start, tl.range_end - 2 * DAY)


def new_invoice(
    world: World,
    s: Stream,
    *,
    tag: str,
    customer: Customer | None = None,
    issued: datetime | None = None,
    lines: list[WLine] | None = None,
    display_tz: timezone | None = None,
    lag: bool = True,
) -> WInvoice:
    cust = customer if customer is not None else s.choice(world.customers)
    if issued is None:
        issued = world.claim(lambda: issue_time(world, s))
    if lines is None:
        lines = draw_lines(world, s, cust.currency)
    terms = timedelta(days=s.choice(PAYMENT_TERMS_DAYS))
    inv = WInvoice(
        key=world.mint("inv"),
        number=world.mint("invno"),
        customer=cust.key,
        issued=issued,
        due=issued + terms,
        currency=cust.currency,
        lines=lines,
        display_tz=display_tz if display_tz is not None else world.tl.tz,
        post_ts=issued,
        tag=tag,
    )
    if lag:
        inv.post_ts = small_lag(world, s, issued)
    world.invoices.append(inv)
    return inv


def add_payment(
    world: World,
    s: Stream,
    *,
    payer: str,
    ts: datetime,
    currency: str,
    amount: int,
    refs: list[str] | None,
    plan: list[tuple[str, int]],
    tag: str,
    display_tz: timezone | None = None,
    lag: bool = True,
) -> WPayment:
    pay = WPayment(
        key=world.mint("pay"),
        number=world.mint("payno"),
        payer=payer,
        ts=ts,
        currency=currency,
        amount=amount,
        refs=refs,
        plan=plan,
        display_tz=display_tz if display_tz is not None else world.tl.tz,
        post_ts=ts,
        tag=tag,
    )
    if lag:
        pay.post_ts = small_lag(world, s, ts)
    world.payments.append(pay)
    return pay


def pay_time(world: World, s: Stream, after: datetime, max_days: int = 25) -> datetime | None:
    """A payment instant after `after`, sometimes just before a period boundary."""
    tl = world.tl
    lo = after + 2 * HOUR
    end = tl.range_end - HOUR
    hi = min(after + timedelta(days=max_days), end)
    if s.chance(1, 4):
        bounds = [b for b in tl.boundaries() if lo < b <= hi]
        if bounds:
            b = s.choice(bounds)
            wlo = max(lo, b - BOUNDARY_WINDOW)
            return world.claim(lambda: tl.payment_between(s, wlo, b))
    if hi - lo < HOUR:
        return None
    return world.claim(lambda: tl.payment_between(s, lo, hi))


def pay_full(
    world: World, s: Stream, inv: WInvoice, *, tag: str, max_days: int = 25, lag: bool = True
) -> WPayment | None:
    ts = pay_time(world, s, inv.issued, max_days)
    if ts is None:
        return None
    return add_payment(
        world,
        s,
        payer=inv.customer,
        ts=ts,
        currency=inv.currency,
        amount=inv.total,
        refs=[inv.number],
        plan=[(inv.number, inv.total)],
        tag=tag,
        lag=lag,
    )


def pay_installments(world: World, s: Stream, inv: WInvoice, k: int, *, tag: str) -> list[WPayment]:
    """Up to k payments with a single reference each; together they settle the invoice."""
    out: list[WPayment] = []
    remaining = inv.total
    t = inv.issued
    for i in range(k):
        part = remaining if i == k - 1 else split_part(s, remaining, 30, 70)
        if part is None:
            part = remaining
        ts = pay_time(world, s, t, max_days=12)
        if ts is None:
            break
        out.append(
            add_payment(
                world,
                s,
                payer=inv.customer,
                ts=ts,
                currency=inv.currency,
                amount=part,
                refs=[inv.number],
                plan=[(inv.number, part)],
                tag=tag,
            )
        )
        t = ts
        remaining -= part
        if remaining == 0:
            break
    return out
