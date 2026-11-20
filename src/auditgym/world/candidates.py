"""Which world records can carry which injected error.

Shared by the world builder (to construct and verify carriers) and by the injectors (to pick one).

Rules for every pool:
- only "regular" records qualify: distractor records stay correct;
- a record already touched by an injection (`used`) is never touched again outside a scenario;
- a record reserved as a carrier for another pool is left to that pool (`World.allowed`);
- scope: the erroneous record lies in the reporting period by its document date, or its posting
  is moved into or out of the period by the error itself (cut-off). Records wholly in the margins
  before or after the period are context only.

Lists are returned in world order; the injectors pick from them with their own stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from auditgym.world.fx import convert_on, material, wrong_rate_dates
from auditgym.world.model import WInvoice, World, WPayment
from auditgym.world.timeline import DAY, HOUR

MIN_ALLOC_PART = 300  # a partial allocation leaves at least 100 applied and 100 short
MAX_EARLIER_PAYMENTS = 3
MAX_EVIDENCE_IDS = 10
POSTING_OFFSETS = (0, 1, 2, 3)  # days after the boundary a late posting lands on

POOLS = ("fxb", "dm", "cutoff", "fx", "dup", "alloc")


@dataclass(frozen=True)
class AllocCandidate:
    pay: WPayment
    inv: WInvoice
    shape: str  # "first" payment on the invoice, "later" instalment, or "multi_ref" payment


def plain_invoice(
    world: World, inv: WInvoice, used: set[str], counts: dict[str, int], pool: str
) -> bool:
    return (
        inv.tag == "regular"
        and inv.key not in used
        and inv.total >= 100
        and counts.get(inv.number, 0) == 1
        and world.allowed(inv.key, pool)
    )


def regular_payment(world: World, pay: WPayment, used: set[str], pool: str) -> bool:
    """A positive regular payment that names at least one invoice."""
    return (
        pay.tag == "regular"
        and pay.key not in used
        and pay.amount >= 100
        and bool(pay.refs)
        and world.allowed(pay.key, pool)
    )


def unlagged(world: World, doc_time: datetime, post_ts: datetime) -> bool:
    return world.tl.local_date(doc_time) == world.tl.local_date(post_ts)


def unapplied(pay: WPayment) -> int:
    return pay.amount - sum(part for _, part in pay.plan)


def earlier_payments(world: World, number: str, pay: WPayment) -> list[WPayment]:
    return [
        q
        for q in world.payments
        if q.key != pay.key and q.refs is not None and number in q.refs and q.ts < pay.ts
    ]


# ---------------------------------------------------------------------------------- pools
def cutoff_docs(world: World, used: set[str], pool: str = "cutoff") -> list[WInvoice | WPayment]:
    """Documents just before a period boundary whose posting can be moved across it."""
    tl, counts = world.tl, world.number_counts()
    out: list[WInvoice | WPayment] = []
    for inv in world.invoices:
        if (
            plain_invoice(world, inv, used, counts, pool)
            and tl.boundary_before(inv.issued) is not None
            and unlagged(world, inv.issued, inv.post_ts)
        ):
            out.append(inv)
    for pay in world.payments:
        if (
            regular_payment(world, pay, used, pool)
            and tl.boundary_before(pay.ts) is not None
            and unlagged(world, pay.ts, pay.post_ts)
        ):
            out.append(pay)
    return out


def fx_payments(world: World, used: set[str], pool: str = "fx") -> list[WPayment]:
    """Foreign payments in the period for which some rate date within a week is materially off."""
    return [
        p
        for p in world.payments
        if regular_payment(world, p, used, pool)
        and p.currency != world.base
        and world.tl.in_period(p.ts)
        and wrong_rate_dates(world, p)
    ]


def alloc_candidates(
    world: World, used: set[str], pool: str = "alloc", *, in_scope: bool = True
) -> list[AllocCandidate]:
    """(payment, invoice) pairs whose expected allocation follows from a small evidence set:
    the invoice is listed first by this payment and by every earlier payment that names it."""
    counts = world.number_counts()
    out: list[AllocCandidate] = []
    for pay in world.payments:
        if not regular_payment(world, pay, used, pool) or not pay.plan:
            continue
        if in_scope and not world.tl.in_period(pay.ts):
            continue
        assert pay.refs is not None
        number, part = pay.plan[0]
        if number != pay.refs[0] or part < MIN_ALLOC_PART:
            continue
        docs = world.invoice_docs(number)
        if len(docs) != 1 or not plain_invoice(world, docs[0], used, counts, pool):
            continue
        inv = docs[0]
        if inv.currency != pay.currency:
            continue
        earlier = earlier_payments(world, number, pay)
        if len(earlier) > MAX_EARLIER_PAYMENTS:
            continue
        if any(q.refs is None or q.refs[0] != number for q in earlier):
            continue
        # payment doc + invoice doc + CASH/FX_CLEARING + AR rows + UNAPPLIED (existing or added)
        evidence = 2 + 1 + len(pay.plan) + 1 + len(earlier)
        if evidence > MAX_EVIDENCE_IDS:
            continue
        shape = "multi_ref" if len(pay.refs) > 1 else ("later" if earlier else "first")
        out.append(AllocCandidate(pay, inv, shape))
    return out


def dup_invoices(world: World, used: set[str], pool: str = "dup") -> list[WInvoice]:
    """In-period invoices with room for two later copies inside the period."""
    tl, counts = world.tl, world.number_counts()
    return [
        inv
        for inv in world.invoices
        if plain_invoice(world, inv, used, counts, pool)
        and tl.in_period(inv.issued)
        and inv.issued + DAY <= tl.period_end
    ]


def dm_copy_window(world: World, pay: WPayment, inv: WInvoice) -> tuple[datetime, datetime]:
    """Interval for the copy in duplicate_misallocated: after the original, before the payment."""
    return inv.issued + HOUR, min(pay.ts, world.tl.period_end) - HOUR


def dm_pairs(world: World, used: set[str], pool: str = "dm") -> list[AllocCandidate]:
    out = []
    for c in alloc_candidates(world, used, pool):
        if not world.tl.in_period(c.inv.issued):
            continue
        lo, hi = dm_copy_window(world, c.pay, c.inv)
        if hi - lo >= HOUR:
            out.append(c)
    return out


def boundary_posting_offsets(world: World, pay: WPayment) -> list[int]:
    """Day offsets after the next boundary whose rate converts the payment materially wrong."""
    b = world.tl.boundary_before(pay.ts)
    if b is None:
        return []
    first = world.tl.local_date(b)
    return [
        off
        for off in POSTING_OFFSETS
        if material(
            world, pay, convert_on(world, pay.amount, pay.currency, first + timedelta(days=off))
        )
    ]


def fxb_pairs(world: World, used: set[str], pool: str = "fxb") -> list[AllocCandidate]:
    """Foreign payments just before a period boundary, for fx_partial_at_boundary. In scope: the
    payment is in the period (end boundary) or its posting moves into it (start boundary)."""
    return [
        c
        for c in alloc_candidates(world, used, pool, in_scope=False)
        if c.pay.currency != world.base
        and len(c.pay.refs or ()) == 1
        and unlagged(world, c.pay.ts, c.pay.post_ts)
        and boundary_posting_offsets(world, c.pay)
    ]


def pool_keys(world: World, pool: str, used: set[str] | None = None) -> set[str]:
    """Keys of the records that make up candidates of `pool` (for carrier verification)."""
    u = used if used is not None else set()
    if pool == "cutoff":
        return {d.key for d in cutoff_docs(world, u)}
    if pool == "fx":
        return {p.key for p in fx_payments(world, u)}
    if pool == "dup":
        return {i.key for i in dup_invoices(world, u)}
    if pool == "alloc":
        return {c.pay.key for c in alloc_candidates(world, u)}
    if pool == "dm":
        return {c.pay.key for c in dm_pairs(world, u)}
    if pool == "fxb":
        return {c.pay.key for c in fxb_pairs(world, u)}
    raise ValueError(f"unknown candidate pool {pool!r}")
