"""Which world records can carry which injected error.

Shared by the world builder (to guarantee enough candidates) and by the injectors (to pick one).
Only "regular" records qualify: distractor records stay correct, and a record already touched by
an injection (`used`) is never touched again outside an interacting scenario (DOMAIN 7).
Lists are returned in world order; the injectors pick from them with their own stream.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from auditgym.world.fx import convert_on, material, wrong_rate_dates
from auditgym.world.model import WInvoice, World, WPayment
from auditgym.world.timeline import HOUR

MIN_ALLOC_PART = 300  # a partial allocation leaves at least 100 applied and 100 short
MAX_EARLIER_PAYMENTS = 3  # keeps the allocation evidence set within 10 IDs
POSTING_OFFSETS = (0, 1, 2, 3)  # days after the boundary a late posting lands on


def plain_invoice(inv: WInvoice, used: set[str], counts: dict[str, int]) -> bool:
    return (
        inv.tag == "regular"
        and inv.key not in used
        and inv.total >= 100
        and counts.get(inv.number, 0) == 1
    )


def plain_payment(pay: WPayment, used: set[str]) -> bool:
    """A positive payment with one reference, fully applied to it."""
    return (
        pay.tag == "regular"
        and pay.key not in used
        and pay.amount >= 100
        and pay.refs is not None
        and len(pay.refs) == 1
        and len(pay.plan) == 1
        and pay.plan[0][0] == pay.refs[0]
        and pay.plan[0][1] == pay.amount
    )


def unlagged(world: World, doc_time: datetime, post_ts: datetime) -> bool:
    return world.tl.local_date(doc_time) == world.tl.local_date(post_ts)


def earlier_payments(world: World, number: str, pay: WPayment) -> list[WPayment]:
    return [
        q
        for q in world.payments
        if q.key != pay.key and q.refs is not None and number in q.refs and q.ts < pay.ts
    ]


def cutoff_docs(world: World, used: set[str]) -> list[WInvoice | WPayment]:
    """Documents just before a period boundary whose posting can be moved across it."""
    tl, counts = world.tl, world.number_counts()
    out: list[WInvoice | WPayment] = []
    for inv in world.invoices:
        if (
            plain_invoice(inv, used, counts)
            and tl.boundary_before(inv.issued) is not None
            and unlagged(world, inv.issued, inv.post_ts)
        ):
            out.append(inv)
    for pay in world.payments:
        if (
            plain_payment(pay, used)
            and tl.boundary_before(pay.ts) is not None
            and unlagged(world, pay.ts, pay.post_ts)
        ):
            out.append(pay)
    return out


def fx_payments(world: World, used: set[str]) -> list[WPayment]:
    """Foreign payments for which some rate date within a week gives a material difference."""
    return [
        p
        for p in world.payments
        if plain_payment(p, used) and p.currency != world.base and wrong_rate_dates(world, p)
    ]


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


def alloc_pairs(world: World, used: set[str]) -> list[tuple[WPayment, WInvoice]]:
    """(payment, invoice) pairs whose expected allocation is recomputable from a small evidence set."""
    counts = world.number_counts()
    out: list[tuple[WPayment, WInvoice]] = []
    for pay in world.payments:
        if not plain_payment(pay, used) or pay.amount < MIN_ALLOC_PART:
            continue
        assert pay.refs is not None
        docs = world.invoice_docs(pay.refs[0])
        if len(docs) != 1 or not plain_invoice(docs[0], used, counts):
            continue
        inv = docs[0]
        if inv.currency != pay.currency:
            continue
        earlier = earlier_payments(world, inv.number, pay)
        if len(earlier) > MAX_EARLIER_PAYMENTS:
            continue
        if any(q.refs is None or q.refs[0] != inv.number for q in earlier):
            continue
        out.append((pay, inv))
    return out


def dup_invoices(world: World, used: set[str]) -> list[WInvoice]:
    """In-period invoices with room for a later copy inside the same period."""
    tl, counts = world.tl, world.number_counts()
    return [
        inv
        for inv in world.invoices
        if plain_invoice(inv, used, counts)
        and tl.period_class(inv.issued) == "in"
        and inv.issued + 3 * HOUR < tl.period_end
    ]


def dm_copy_window(world: World, pay: WPayment, inv: WInvoice) -> tuple[datetime, datetime]:
    """Interval for the copy in duplicate_misallocated: after the original, before the payment."""
    return inv.issued + HOUR, min(pay.ts, world.tl.period_end) - HOUR


def dm_pairs(world: World, used: set[str]) -> list[tuple[WPayment, WInvoice]]:
    out = []
    for pay, inv in alloc_pairs(world, used):
        if world.tl.period_class(inv.issued) != "in":
            continue
        lo, hi = dm_copy_window(world, pay, inv)
        if hi - lo >= HOUR:
            out.append((pay, inv))
    return out


def fxb_pairs(world: World, used: set[str]) -> list[tuple[WPayment, WInvoice]]:
    """Foreign payments just before a period boundary, for fx_partial_at_boundary."""
    return [
        (pay, inv)
        for pay, inv in alloc_pairs(world, used)
        if pay.currency != world.base
        and unlagged(world, pay.ts, pay.post_ts)
        and boundary_posting_offsets(world, pay)
    ]


def count(world: World, name: str) -> int:
    used: set[str] = set()
    if name == "cutoff":
        return len(cutoff_docs(world, used))
    if name == "fx":
        return len(fx_payments(world, used))
    if name == "alloc":
        return len(alloc_pairs(world, used))
    if name == "dup":
        return len(dup_invoices(world, used))
    if name == "dm":
        return len(dm_pairs(world, used))
    if name == "fxb":
        return len(fxb_pairs(world, used))
    raise ValueError(f"unknown candidate pool {name!r}")


POOLS = ("fxb", "dm", "cutoff", "fx", "dup", "alloc")
