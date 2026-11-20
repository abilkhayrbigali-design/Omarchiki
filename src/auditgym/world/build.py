"""Build the hidden, consistent company history of one episode (pipeline step 1).

Streams: "world" (company, customers, FX, regular history, top-up, background), and one stream per
distractor kind ("distractor/<kind>"). The world never sees which errors will be injected; it only
receives the minimum number of candidates each injector pool needs (`needs`), and adds ordinary
records drawn from the same distributions when the random history has too few.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from typing import Any

from auditgym.contracts.constants import DISTRACTOR_KINDS, FX_RATE_DATE_RULES, ROUNDING_RULES
from auditgym.util.rng import Stream
from auditgym.world import candidates
from auditgym.world.amounts import draw_minor
from auditgym.world.distractors import BUILDERS, INVOICES_PER_INSTANCE
from auditgym.world.docs import add_payment, new_invoice, pay_full, pay_installments, pay_time
from auditgym.world.fx import build_fx
from auditgym.world.model import Customer, WExpense, World, WorldError
from auditgym.world.names import company_name, customer_name
from auditgym.world.timeline import BOUNDARY_WINDOW, DAY, HOUR, Timeline

# Account code ranges per role. Codes are drawn per episode (G06).
ACCOUNT_RANGES = {
    "CASH": (1000, 1099),
    "AR": (1100, 1299),
    "FX_CLEARING": (1800, 1999),
    "UNAPPLIED": (2100, 2399),
    "REVENUE": (4000, 4199),
    "EXPENSE": (6000, 6999),
}
FX_MARGIN_DAYS = 10  # FX rows extend this far beyond the data range on both sides
MAX_TOPUP = 60


def _accounts(s: Stream) -> dict[str, str]:
    style = s.choice(["plain", "long", "dotted"])
    out = {}
    for role in sorted(ACCOUNT_RANGES):
        lo, hi = ACCOUNT_RANGES[role]
        code = s.randint(lo, hi)
        if style == "plain":
            out[role] = str(code)
        elif style == "long":
            out[role] = f"{code}{s.randint(0, 9)}"
        else:
            out[role] = f"{code // 100}.{code % 100:02d}"
    return out


def _customers(world: World, s: Stream, n: int) -> None:
    curs = [
        world.base if not world.foreign or s.chance(11, 20) else s.choice(world.foreign)
        for _ in range(n)
    ]
    # Every foreign currency gets a customer, and at least one customer pays in base currency,
    # as far as the number of customers allows.
    for cur in [*world.foreign, world.base]:
        if cur not in curs:
            spare = [i for i, c in enumerate(curs) if curs.count(c) > 1]
            if spare:
                curs[s.choice(spare)] = cur
    names: list[str] = []
    for cur in curs:
        name = customer_name(s)
        while name in names:
            name = customer_name(s)
        names.append(name)
        world.customers.append(Customer(world.mint("cust"), name, cur))


def plan_spikes(world: World, s: Stream, n: int) -> dict[tuple[str, Any], Decimal]:
    """Spike days for X9. Drawn before the FX table, from the fx_spike stream."""
    tl = world.tl
    payable = [c for c in world.foreign if any(cu.currency == c for cu in world.customers)]
    spikes: dict[tuple[str, Any], Decimal] = {}
    if not payable:
        return spikes
    first = tl.local_date(tl.period_start)
    days = (tl.local_date(tl.period_end) - first).days
    for _ in range(n):
        cur = s.choice(payable)
        day = first + timedelta(days=s.randint(3, days - 4))
        factor = s.decimal("1.03", "1.08", 3) if s.chance(1, 2) else s.decimal("0.92", "0.97", 3)
        if (cur, day) in spikes:
            continue
        spikes[(cur, day)] = factor
        world.spikes.append({"currency": cur, "day": day.isoformat(), "factor": str(factor)})
    return spikes


REGULAR_PATTERNS = ("full", "installments", "open", "overpaid", "paid_twice")
REGULAR_WEIGHTS = (60, 10, 30, 4, 3)


def _regular_history(world: World, s: Stream, n: int) -> None:
    """Ordinary sales. Besides full, split and open invoices, a few clean cases where money
    legitimately lands on UNAPPLIED although the payment names an invoice: an overpayment
    (the remainder is unapplied) and an invoice paid twice (the second payment finds no open
    balance)."""
    for _ in range(n):
        inv = new_invoice(world, s, tag="regular")
        pattern = s.weighted_choice(list(REGULAR_PATTERNS), list(REGULAR_WEIGHTS))
        if pattern == "full":
            pay_full(world, s, inv, tag="regular")
        elif pattern == "installments":
            pay_installments(world, s, inv, 2, tag="regular")
        elif pattern == "overpaid":
            ts = pay_time(world, s, inv.issued)
            if ts is not None:
                extra = draw_minor(s, inv.currency, 5, 200)
                add_payment(
                    world,
                    s,
                    payer=inv.customer,
                    ts=ts,
                    currency=inv.currency,
                    amount=inv.total + extra,
                    refs=[inv.number],
                    plan=[(inv.number, inv.total)],
                    tag="regular",
                )
        elif pattern == "paid_twice":
            first = pay_full(world, s, inv, tag="regular", max_days=12)
            ts = pay_time(world, s, first.ts, max_days=12) if first is not None else None
            if ts is not None:
                add_payment(
                    world,
                    s,
                    payer=inv.customer,
                    ts=ts,
                    currency=inv.currency,
                    amount=inv.total,
                    refs=[inv.number],
                    plan=[],
                    tag="regular",
                )


def _expenses(world: World, s: Stream, n: int) -> None:
    tl = world.tl
    for _ in range(n):
        ts = world.claim(lambda: tl.business_between(s, tl.range_start, tl.range_end))
        amount = draw_minor(s, world.base, 10, 3000)
        world.expenses.append(WExpense(world.mint("exp"), ts, amount))


# ---------------------------------------------------------------------------------- top-up
def _foreign_customer(world: World, s: Stream) -> Customer:
    pool = [c for c in world.customers if c.currency != world.base]
    if not pool:
        raise WorldError("no_foreign_customer")
    return s.choice(pool)


def _paid_invoice(world: World, s: Stream, cust: Customer) -> None:
    tl = world.tl
    issued = world.claim(lambda: tl.business_between(s, tl.range_start, tl.range_end - 10 * DAY))
    inv = new_invoice(world, s, tag="regular", customer=cust, issued=issued, lag=False)
    if pay_full(world, s, inv, tag="regular", max_days=8, lag=False) is None:
        raise WorldError("topup_payment")


def _topup_cutoff(world: World, s: Stream) -> None:
    tl = world.tl
    b = s.choice(tl.boundaries())
    if s.chance(1, 2):
        issued = world.claim(lambda: tl.business_between(s, b - BOUNDARY_WINDOW, b))
        inv = new_invoice(world, s, tag="regular", issued=issued, lag=False)
        if s.chance(1, 2):
            pay_full(world, s, inv, tag="regular")
        return
    lo = max(tl.range_start, b - 20 * DAY)
    issued = world.claim(lambda: tl.business_between(s, lo, b - BOUNDARY_WINDOW - DAY))
    inv = new_invoice(world, s, tag="regular", issued=issued, lag=False)
    ts = world.claim(lambda: tl.payment_between(s, b - BOUNDARY_WINDOW, b))
    _full_payment_at(world, s, inv, ts)


def _full_payment_at(world: World, s: Stream, inv: Any, ts: Any) -> None:
    add_payment(
        world,
        s,
        payer=inv.customer,
        ts=ts,
        currency=inv.currency,
        amount=inv.total,
        refs=[inv.number],
        plan=[(inv.number, inv.total)],
        tag="regular",
        lag=False,
    )


def _topup_fx(world: World, s: Stream) -> None:
    _paid_invoice(world, s, _foreign_customer(world, s))


def _topup_alloc(world: World, s: Stream) -> None:
    _paid_invoice(world, s, s.choice(world.customers))


def _topup_dup(world: World, s: Stream) -> None:
    tl = world.tl
    issued = world.claim(lambda: tl.business_between(s, tl.period_start, tl.period_end - 5 * DAY))
    inv = new_invoice(world, s, tag="regular", issued=issued)
    if s.chance(1, 2):
        pay_full(world, s, inv, tag="regular")


def _topup_dm(world: World, s: Stream) -> None:
    tl = world.tl
    issued = world.claim(lambda: tl.business_between(s, tl.period_start, tl.period_end - 8 * DAY))
    inv = new_invoice(world, s, tag="regular", issued=issued, lag=False)
    ts = world.claim(lambda: tl.payment_between(s, issued + DAY, issued + 6 * DAY))
    _full_payment_at(world, s, inv, ts)


def _topup_fxb(world: World, s: Stream) -> None:
    tl = world.tl
    cust = _foreign_customer(world, s)
    b = s.choice(tl.boundaries())
    lo = max(tl.range_start, b - 25 * DAY)
    issued = world.claim(lambda: tl.business_between(s, lo, b - BOUNDARY_WINDOW - DAY))
    inv = new_invoice(world, s, tag="regular", customer=cust, issued=issued, lag=False)
    ts = world.claim(lambda: tl.payment_between(s, b - BOUNDARY_WINDOW, b - HOUR))
    _full_payment_at(world, s, inv, ts)


TOPUP = {
    "fxb": _topup_fxb,
    "dm": _topup_dm,
    "cutoff": _topup_cutoff,
    "fx": _topup_fx,
    "dup": _topup_dup,
    "alloc": _topup_alloc,
}


def _topup(world: World, s: Stream, needs: dict[str, int]) -> None:
    for pool in candidates.POOLS:
        need = needs.get(pool, 0)
        added = 0
        while need > 0 and candidates.count(world, pool) < need:
            TOPUP[pool](world, s)
            added += 1
            if added > MAX_TOPUP:
                raise WorldError(f"topup_{pool}")
        if added:
            world.topups[pool] = added


# ---------------------------------------------------------------------------------- entry point
def build_world(root: Stream, cfg: dict[str, Any], needs: dict[str, int]) -> World:
    ws = root.child("world")
    cs = ws.child("company")
    base = cs.choice(cfg["currencies"]["base_pool"])
    pool = [c for c in cfg["currencies"]["foreign_pool"] if c != base]
    n_foreign = min(cs.in_range(cfg["currencies"]["foreign_count"]), len(pool))
    foreign = sorted(cs.sample(pool, n_foreign))
    tz_text = cs.choice(cfg["timezone_pool"])
    year = cs.choice(cfg["period"]["year_pool"])
    month = cs.randint(1, 12)
    tl = Timeline.build(tz_text, year, month, cfg["period"]["margin_days"])
    world = World(
        company=company_name(cs.child("name")),
        base=base,
        foreign=foreign,
        tl=tl,
        accounts=_accounts(cs.child("accounts")),
        fx_rule=cs.choice(FX_RATE_DATE_RULES),
        rounding=cs.choice(ROUNDING_RULES),
    )

    size = ws.child("size")
    n_customers = size.in_range(cfg["size"]["customers"])
    target_invoices = size.in_range(cfg["size"]["invoices"])
    n_expenses = size.in_range(cfg["size"]["background_entries"])
    _customers(world, ws.child("customers"), n_customers)

    dstreams = {kind: root.child(f"distractor/{kind}") for kind in DISTRACTOR_KINDS}
    dcounts = {kind: dstreams[kind].in_range(cfg["distractors"][kind]) for kind in DISTRACTOR_KINDS}
    spikes = plan_spikes(world, dstreams["fx_spike"], dcounts["fx_spike"])

    world.fx_first = tl.local_date(tl.range_start) - timedelta(days=FX_MARGIN_DAYS)
    world.fx_last = tl.local_date(tl.range_end) + timedelta(days=FX_MARGIN_DAYS)
    world.fx = build_fx(ws, base, foreign, world.fx_first, world.fx_last, spikes)

    distractor_invoices = sum(INVOICES_PER_INSTANCE[k] * dcounts[k] for k in DISTRACTOR_KINDS)
    regular = max(target_invoices - distractor_invoices, (target_invoices + 1) // 2, 4)
    _regular_history(world, ws.child("regular"), regular)

    for kind in DISTRACTOR_KINDS:
        BUILDERS[kind](world, dstreams[kind], dcounts[kind])

    _topup(world, ws.child("topup"), needs)
    _expenses(world, ws.child("background"), n_expenses)
    return world


def world_summary(world: World) -> dict[str, Any]:
    tl = world.tl
    return {
        "company": world.company,
        "base_currency": world.base,
        "foreign_currencies": list(world.foreign),
        "timezone": tl.tz_text,
        "period_start": tl.period_start.isoformat(),
        "period_end": tl.period_end.isoformat(),
        "range_start": tl.range_start.isoformat(),
        "range_end": tl.range_end.isoformat(),
        "fx_rate_date": world.fx_rule,
        "rounding": world.rounding,
        "accounts": dict(world.accounts),
        "customers": [
            {"key": c.key, "name": c.name, "currency": c.currency} for c in world.customers
        ],
        "fx_spikes": list(world.spikes),
        "topups": dict(sorted(world.topups.items())),
        "counts": {
            "invoices": len(world.invoices),
            "payments": len(world.payments),
            "expenses": len(world.expenses),
            "regular_invoices": sum(1 for i in world.invoices if i.tag == "regular"),
            "regular_payments": sum(1 for p in world.payments if p.tag == "regular"),
        },
    }
