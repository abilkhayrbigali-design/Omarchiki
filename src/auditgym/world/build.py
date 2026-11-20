"""Build the hidden, consistent company history of one episode (pipeline step 1).

Streams: "world" (company, period, customers, sizing, FX, carriers, regular history, background)
and one stream per distractor kind ("distractor/<kind>").

Order: company and period -> customers -> distractor counts -> sizing (fit to size.invoices) ->
FX table -> distractors -> carriers for the planned errors -> ordinary sales up to the target
number of invoice documents -> background expenses.

The world never learns which records will carry errors; it only receives how many carriers each
injector pool needs (`demand`), see auditgym.world.carriers.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from typing import Any

from auditgym.contracts.constants import DISTRACTOR_KINDS, FX_RATE_DATE_RULES, ROUNDING_RULES
from auditgym.util.rng import Stream
from auditgym.world.amounts import draw_minor, split_part
from auditgym.world.carriers import build_carriers, fxb_options
from auditgym.world.distractors import BUILDERS
from auditgym.world.docs import add_payment, new_invoice, pay_full, pay_installments, pay_time
from auditgym.world.fx import build_fx
from auditgym.world.model import Customer, WExpense, World, WorldError
from auditgym.world.names import company_name, customer_name
from auditgym.world.sizing import INVOICES_PER_INSTANCE, demand_invoices, fit
from auditgym.world.timeline import DAY, Timeline, draw_period

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
MAX_FX_REDRAWS = 20


# Ordinary sales. Besides full, split and open invoices there are clean cases where money
# legitimately lands on UNAPPLIED although the payment names an invoice, and where an instalment
# is applied only up to the remaining open balance; and payments naming several invoices.
REGULAR_PATTERNS = (
    "full",
    "installments",
    "installment_over",
    "open",
    "overpaid",
    "paid_twice",
    "multi_ref",
)
REGULAR_WEIGHTS = (50, 8, 5, 26, 3, 3, 6)


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
    for _ in range(n):
        cur = s.choice(payable)
        day = first + timedelta(days=s.randint(3, tl.period_days - 4))
        factor = s.decimal("1.03", "1.08", 3) if s.chance(1, 2) else s.decimal("0.92", "0.97", 3)
        if (cur, day) in spikes:
            continue
        spikes[(cur, day)] = factor
        world.spikes.append({"currency": cur, "day": day.isoformat(), "factor": str(factor)})
    return spikes


def _distractor_plan(cfg: dict[str, Any], streams: dict[str, Stream]) -> dict[str, list[int]]:
    """kind -> invoice documents per instance (the list length is the number of instances)."""
    out = {}
    for kind in DISTRACTOR_KINDS:
        s = streams[kind]
        n = s.in_range(cfg["distractors"][kind])
        if kind == "multi_invoice_payment":
            out[kind] = [s.randint(2, 3) for _ in range(n)]
        else:
            out[kind] = [INVOICES_PER_INSTANCE[kind]] * n
    return out


# ---------------------------------------------------------------------------------- ordinary sales
def _pay(
    world: World, s: Stream, inv: Any, ts: Any, amount: int, plan: list[tuple[str, int]]
) -> None:
    add_payment(
        world,
        s,
        payer=inv.customer,
        ts=ts,
        currency=inv.currency,
        amount=amount,
        refs=[inv.number],
        plan=plan,
        tag="regular",
    )


def _multi_ref(world: World, s: Stream) -> None:
    """Two invoices of one customer settled by one payment naming both, in the payer's order."""
    tl = world.tl
    cust = s.choice(world.customers)
    t0 = tl.business_between(s, tl.range_start, tl.range_end - 12 * DAY)
    invs = [
        new_invoice(
            world,
            s,
            tag="regular",
            customer=cust,
            issued=world.claim(lambda: tl.business_between(s, t0, t0 + 3 * DAY)),
        )
        for _ in range(2)
    ]
    order = s.shuffled(invs)
    plan = [(inv.number, inv.total) for inv in order]
    if s.chance(1, 3):
        part = split_part(s, order[-1].total)
        if part is not None:
            plan[-1] = (order[-1].number, part)
    ts = pay_time(world, s, max(inv.issued for inv in invs), max_days=8)
    if ts is not None:
        add_payment(
            world,
            s,
            payer=cust.key,
            ts=ts,
            currency=cust.currency,
            amount=sum(p for _, p in plan),
            refs=[inv.number for inv in order],
            plan=plan,
            tag="regular",
        )


def _regular_history(world: World, s: Stream, n: int) -> None:
    created = 0
    while created < n:
        pattern = s.weighted_choice(list(REGULAR_PATTERNS), list(REGULAR_WEIGHTS))
        if pattern == "multi_ref":
            if n - created < 2:
                continue
            _multi_ref(world, s)
            created += 2
            continue
        inv = new_invoice(world, s, tag="regular")
        created += 1
        if pattern == "full":
            pay_full(world, s, inv, tag="regular")
        elif pattern == "installments":
            pay_installments(world, s, inv, 2, tag="regular")
        elif pattern == "installment_over":
            # The second instalment pays more than is still open: it is applied up to the open
            # balance and the rest is unapplied.
            first = split_part(s, inv.total, 30, 70)
            t1 = pay_time(world, s, inv.issued, max_days=12) if first is not None else None
            t2 = pay_time(world, s, t1, max_days=12) if t1 is not None else None
            if first is not None and t1 is not None and t2 is not None:
                rest = inv.total - first
                _pay(world, s, inv, t1, first, [(inv.number, first)])
                extra = draw_minor(s, inv.currency, 5, 300)
                _pay(world, s, inv, t2, rest + extra, [(inv.number, rest)])
        elif pattern == "overpaid":
            ts = pay_time(world, s, inv.issued)
            if ts is not None:
                extra = draw_minor(s, inv.currency, 5, 200)
                _pay(world, s, inv, ts, inv.total + extra, [(inv.number, inv.total)])
        elif pattern == "paid_twice":
            first_pay = pay_full(world, s, inv, tag="regular", max_days=12)
            ts = pay_time(world, s, first_pay.ts, max_days=12) if first_pay is not None else None
            if ts is not None:
                _pay(world, s, inv, ts, inv.total, [])


def _expenses(world: World, s: Stream, n: int) -> None:
    tl = world.tl
    for _ in range(n):
        ts = world.claim(lambda: tl.business_between(s, tl.range_start, tl.range_end))
        amount = draw_minor(s, world.base, 10, 3000)
        world.expenses.append(WExpense(world.mint("exp"), ts, amount))


# ---------------------------------------------------------------------------------- entry point
def build_world(root: Stream, cfg: dict[str, Any], demand: dict[str, Any]) -> World:
    """`demand`: {"carriers": {pool: n}, "dup_copies": [copies per duplicate], "dm_copies": n}.

    `demand["dup_copies"]` may be reduced to fit size.invoices; the final values are in
    `world.sizing["dup_copies"]`.
    """
    ws = root.child("world")
    cs = ws.child("company")
    base = cs.choice(cfg["currencies"]["base_pool"])
    pool = [c for c in cfg["currencies"]["foreign_pool"] if c != base]
    n_foreign = min(cs.in_range(cfg["currencies"]["foreign_count"]), len(pool))
    foreign = sorted(cs.sample(pool, n_foreign))
    tz_text = cs.choice(cfg["timezone_pool"])
    year = cs.choice(cfg["period"]["year_pool"])
    kind, first, end = draw_period(cs.child("period"), year)
    tl = Timeline.build(tz_text, first, end, cfg["period"]["margin_days"], kind)
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
    n_expenses = size.in_range(cfg["size"]["background_entries"])
    _customers(world, ws.child("customers"), n_customers)

    dstreams = {k: root.child(f"distractor/{k}") for k in DISTRACTOR_KINDS}
    dplan = _distractor_plan(cfg, dstreams)
    demand = {
        "carriers": dict(demand["carriers"]),
        "dup_copies": list(demand["dup_copies"]),
        "dm_copies": demand["dm_copies"],
    }
    target = fit(size, cfg, dplan, demand)

    spikes = plan_spikes(world, dstreams["fx_spike"], len(dplan["fx_spike"]))
    world.fx_first = tl.local_date(tl.range_start) - timedelta(days=FX_MARGIN_DAYS)
    world.fx_last = tl.local_date(tl.range_end) + timedelta(days=FX_MARGIN_DAYS)
    # An fx_partial_at_boundary carrier needs a material rate move between the payment date and
    # a posting date just after a boundary. If the walk is calm around both boundaries for every
    # currency, redraw it from a numbered sub-stream (rare; recorded in sizing).
    redraws = 0
    world.fx = build_fx(ws, base, foreign, world.fx_first, world.fx_last, spikes)
    while demand["carriers"].get("fxb") and not fxb_options(world):
        redraws += 1
        if redraws > MAX_FX_REDRAWS:
            raise WorldError("fx_partial_infeasible")
        world.fx = build_fx(
            ws.child(f"fx-redraw-{redraws}"), base, foreign, world.fx_first, world.fx_last, spikes
        )

    for k in DISTRACTOR_KINDS:
        BUILDERS[k](world, dstreams[k], dplan[k])
    build_carriers(world, ws.child("carriers"), demand["carriers"])
    copies = sum(demand["dup_copies"]) + demand["dm_copies"]
    _regular_history(world, ws.child("regular"), target - copies - len(world.invoices))
    _expenses(world, ws.child("background"), n_expenses)
    world.sizing = {
        "target_invoices": target,
        "copies": copies,
        "dup_copies": demand["dup_copies"],
        "carriers": dict(sorted(demand["carriers"].items())),
        "carrier_invoices": sum(demand["carriers"].values()),
        "demand_invoices": demand_invoices(demand),
        "distractor_instances": {k: len(v) for k, v in dplan.items()},
        "fx_redraws": redraws,
    }
    return world


def world_summary(world: World) -> dict[str, Any]:
    tl = world.tl
    return {
        "company": world.company,
        "base_currency": world.base,
        "foreign_currencies": list(world.foreign),
        "timezone": tl.tz_text,
        "period_kind": tl.period_kind,
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
        "sizing": dict(world.sizing),
        "reserved_carriers": dict(sorted(world.reserved.items())),
        "counts": {
            "invoices": len(world.invoices),
            "payments": len(world.payments),
            "expenses": len(world.expenses),
            "regular_invoices": sum(1 for i in world.invoices if i.tag == "regular"),
            "regular_payments": sum(1 for p in world.payments if p.tag == "regular"),
        },
    }
