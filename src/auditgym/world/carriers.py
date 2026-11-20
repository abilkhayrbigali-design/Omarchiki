"""Carriers: ordinary records built so that every planned error has a place to go.

Each carrier is one regular invoice (plus its payments), drawn like any other regular record but
conditioned on what its injector needs, then verified against that injector's candidate pool and
reserved for it. Injectors pick uniformly among unreserved candidates and their own carriers, so a
carrier is neither guaranteed to receive the error nor distinguishable from other records.

When config `variants` restricts an injector, its carrier is built and verified for the allowed
variants (World.variants): a cut-off carrier is an invoice or a payment as allowed, an FX carrier
for wrong_pair alone is verified against the wrong-pair pool, an allocation carrier for
other_invoice alone comes with a second invoice of the same customer. With every variant allowed
the draws are unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timedelta
from decimal import Decimal

from auditgym.util.rng import Stream
from auditgym.world import candidates
from auditgym.world.amounts import draw_minor, split_part
from auditgym.world.docs import add_payment, new_invoice, pay_full
from auditgym.world.fx import rate_date
from auditgym.world.model import Customer, WInvoice, World, WorldError, WPayment
from auditgym.world.timeline import BOUNDARY_WINDOW, DAY, HOUR

MAX_TRIES = 40


def _foreign_customer(world: World, s: Stream) -> Customer | None:
    pool = [c for c in world.customers if c.currency != world.base]
    return s.choice(pool) if pool else None


def _issue(world: World, s: Stream, lo: datetime, hi: datetime) -> datetime | None:
    if hi - lo < HOUR:
        return None
    return world.claim(lambda: world.tl.business_between(s, lo, hi))


def _pay_at(world: World, s: Stream, lo: datetime, hi: datetime) -> datetime | None:
    if hi - lo < HOUR:
        return None
    return world.claim(lambda: world.tl.payment_between(s, lo, hi))


def _payment(
    world: World, s: Stream, inv: WInvoice, ts: datetime, amount: int, applied: int
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
        tag="regular",
        lag=False,
    )


def _in_period_payment_window(world: World, issued: datetime) -> tuple[datetime, datetime]:
    tl = world.tl
    return max(issued + 2 * HOUR, tl.period_start), min(issued + 20 * DAY, tl.period_end - HOUR)


# ---------------------------------------------------------------------------------- builders
def carrier_dup(world: World, s: Stream) -> list[str] | None:
    tl = world.tl
    issued = _issue(world, s, tl.period_start, tl.period_end - 2 * DAY)
    if issued is None:
        return None
    inv = new_invoice(world, s, tag="regular", issued=issued)
    if s.chance(1, 2):
        pay_full(world, s, inv, tag="regular")
    return [inv.key]


def _cutoff_invoice(world: World, s: Stream) -> bool:
    """Invoice or payment carrier: a fair coin, or the only document kind the variants allow."""
    kinds = world.allowed_variants("period_cutoff")
    if len(kinds) == 1:
        return kinds == ("invoice_posted_late",)
    return s.chance(1, 2)


def carrier_cutoff(world: World, s: Stream) -> list[str] | None:
    tl = world.tl
    b = s.choice(tl.boundaries())
    if _cutoff_invoice(world, s):
        issued = _issue(world, s, b - BOUNDARY_WINDOW, b)
        if issued is None:
            return None
        inv = new_invoice(world, s, tag="regular", issued=issued, lag=False)
        if s.chance(1, 2):
            pay_full(world, s, inv, tag="regular")
        return [inv.key]
    issued = _issue(world, s, max(tl.range_start, b - 20 * DAY), b - BOUNDARY_WINDOW - DAY)
    if issued is None:
        return None
    inv = new_invoice(world, s, tag="regular", issued=issued, lag=False)
    ts = _pay_at(world, s, b - BOUNDARY_WINDOW, b)
    if ts is None:
        return None
    return [_payment(world, s, inv, ts, inv.total, inv.total).key]


def _paid_in_period(world: World, s: Stream, cust: Customer) -> tuple[WInvoice, datetime] | None:
    tl = world.tl
    issued = _issue(
        world, s, max(tl.range_start, tl.period_start - 15 * DAY), tl.period_end - 2 * DAY
    )
    if issued is None:
        return None
    inv = new_invoice(world, s, tag="regular", customer=cust, issued=issued, lag=False)
    lo, hi = _in_period_payment_window(world, issued)
    ts = _pay_at(world, s, lo, hi)
    if ts is None:
        return None
    return inv, ts


def carrier_fx(world: World, s: Stream) -> list[str] | None:
    cust = _foreign_customer(world, s)
    if cust is None:
        raise WorldError("no_foreign_customer")
    got = _paid_in_period(world, s, cust)
    if got is None:
        return None
    inv, ts = got
    return [_payment(world, s, inv, ts, inv.total, inv.total).key]


def carrier_alloc(world: World, s: Stream) -> list[str] | None:
    """The payment/invoice pair below. When other_invoice is the only allowed allocation variant,
    also an ordinary open invoice of the same customer (not reserved, an invoice document like
    any other) for the payment to be misapplied to; auditgym.world.sizing counts it."""
    keys = _alloc_pair(world, s)
    if keys is not None and candidates.alloc_needs_other(
        world.allowed_variants("payment_allocation")
    ):
        inv = next(i for i in world.invoices if i.key == keys[1])
        cust = next(c for c in world.customers if c.key == inv.customer)
        new_invoice(world, s, tag="regular", customer=cust)
    return keys


def _alloc_pair(world: World, s: Stream) -> list[str] | None:
    """A payment in the period applied to its invoice: the first payment, a later instalment
    that settles the rest, or a later instalment that overpays the rest (remainder unapplied)."""
    tl = world.tl
    shape = s.weighted_choice(["first", "later_exact", "later_over"], [1, 1, 2])
    cust = s.choice(world.customers)
    if shape == "first":
        got = _paid_in_period(world, s, cust)
        if got is None:
            return None
        inv, ts = got
        return [_payment(world, s, inv, ts, inv.total, inv.total).key, inv.key]
    issued = _issue(
        world, s, max(tl.range_start, tl.period_start - 20 * DAY), tl.period_end - 4 * DAY
    )
    if issued is None:
        return None
    inv = new_invoice(world, s, tag="regular", customer=cust, issued=issued)
    first = split_part(s, inv.total, 30, 70)
    if first is None:
        return None
    t1 = _pay_at(world, s, issued + 2 * HOUR, min(issued + 15 * DAY, tl.period_end - 2 * DAY))
    if t1 is None:
        return None
    _payment(world, s, inv, t1, first, first)
    lo, hi = max(t1 + 2 * HOUR, tl.period_start), min(t1 + 15 * DAY, tl.period_end - HOUR)
    t2 = _pay_at(world, s, lo, hi)
    if t2 is None:
        return None
    rest = inv.total - first
    extra = draw_minor(s, inv.currency, 5, 300) if shape == "later_over" else 0
    return [_payment(world, s, inv, t2, rest + extra, rest).key, inv.key]


def carrier_dm(world: World, s: Stream) -> list[str] | None:
    tl = world.tl
    issued = _issue(world, s, tl.period_start, tl.period_end - 8 * DAY)
    if issued is None:
        return None
    inv = new_invoice(world, s, tag="regular", issued=issued, lag=False)
    ts = _pay_at(world, s, issued + DAY, issued + 6 * DAY)
    if ts is None:
        return None
    return [_payment(world, s, inv, ts, inv.total, inv.total).key, inv.key]


FXB_MIN_MOVE = Decimal("0.006")  # a bit above the 0.5% materiality, to absorb rounding


def _window_hours(world: World, b: datetime) -> list[tuple[datetime, date]]:
    """Whole hours in [b - BOUNDARY_WINDOW, b - 1h) and the policy rate date of every instant
    in that hour (hours where the rate date changes inside the hour are skipped)."""
    out = []
    for h in range(int(BOUNDARY_WINDOW / HOUR) - 1):
        start = b - BOUNDARY_WINDOW + h * HOUR
        d = rate_date(world, start)
        if rate_date(world, start + HOUR - timedelta(seconds=1)) == d:
            out.append((start, d))
    return out


def fxb_options(world: World) -> list[tuple[datetime, str, date]]:
    """(boundary, currency, policy rate date) for a payment just before the boundary such that
    the rate of some posting date 0-3 days after the boundary differs by FXB_MIN_MOVE."""
    tl = world.tl
    payable = [c for c in world.foreign if any(cu.currency == c for cu in world.customers)]
    out = []
    for b in tl.boundaries():
        dates = sorted({d for _, d in _window_hours(world, b)})
        posts = [tl.local_date(b) + timedelta(days=off) for off in candidates.POSTING_OFFSETS]
        for cur in payable:
            for d in dates:
                r1 = Decimal(world.fx[(d.isoformat(), cur)])
                if any(
                    abs(Decimal(world.fx[(p.isoformat(), cur)]) - r1) >= FXB_MIN_MOVE * r1
                    for p in posts
                ):
                    out.append((b, cur, d))
    return out


def carrier_fxb(world: World, s: Stream) -> list[str] | None:
    """A foreign payment just before a boundary whose rate date is one the FX table makes
    material for a posting just after the boundary."""
    tl = world.tl
    options = fxb_options(world)
    if not options:
        raise WorldError("fx_partial_infeasible")
    b, cur, d = s.choice(options)
    cust = s.choice([c for c in world.customers if c.currency == cur])
    hours = [start for start, day in _window_hours(world, b) if day == d]
    # Like other bank payments: mostly in business hours, sometimes at any time of day.
    business = [h for h in hours if 8 <= tl.local(h).hour < 20]
    if business and s.chance(3, 4):
        hours = business
    ts = world.claim(lambda: s.choice(hours) + timedelta(seconds=s.randint(0, 3599)))
    issued = _issue(world, s, max(tl.range_start, b - 25 * DAY), b - BOUNDARY_WINDOW - DAY)
    if issued is None:
        return None
    inv = new_invoice(world, s, tag="regular", customer=cust, issued=issued, lag=False)
    return [_payment(world, s, inv, ts, inv.total, inv.total).key, inv.key]


BUILDERS: dict[str, Callable[[World, Stream], list[str] | None]] = {
    "fxb": carrier_fxb,
    "dm": carrier_dm,
    "cutoff": carrier_cutoff,
    "fx": carrier_fx,
    "dup": carrier_dup,
    "alloc": carrier_alloc,
}


def build_carriers(world: World, s: Stream, carriers: dict[str, int]) -> None:
    """Exactly carriers[pool] verified carriers per pool, one invoice document each."""
    for pool in candidates.POOLS:
        for _ in range(carriers.get(pool, 0)):
            for _try in range(MAX_TRIES):
                mark = world.mark()
                keys = BUILDERS[pool](world, s)
                if keys is not None and keys[0] in candidates.pool_keys(world, pool):
                    for key in keys:
                        world.reserved[key] = pool
                    break
                world.rollback(mark)
            else:
                raise WorldError(f"carrier_{pool}")
