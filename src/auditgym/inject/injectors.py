"""The four error injectors (docs/DOMAIN.md, section 6). One call injects one error.

Each injector picks a candidate with its own stream, validates every amount before editing, edits
the working books, and records the truth (amount from documents only, evidence per the DOMAIN
table) together with record-level before/after values.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from auditgym.contracts.constants import FX_RATE_DATE_RULES
from auditgym.inject.common import (
    FX_TOLERANCE_MINOR,
    add_copy,
    alloc_evidence,
    apply_partial,
    ar_row,
    cutoff_evidence,
    fx_evidence,
    move_posting,
    posting_after,
    rebook_conversion,
)
from auditgym.inject.state import CandidateError, InjectState
from auditgym.util.money import convert_minor
from auditgym.util.rng import Stream
from auditgym.world import candidates
from auditgym.world.amounts import split_part
from auditgym.world.fx import convert_on, material, rate_date, rate_date_by, wrong_rate_dates
from auditgym.world.model import WInvoice, World, WPayment
from auditgym.world.timeline import DAY, HOUR


def _iso(state: InjectState, t: datetime) -> str:
    return state.world.tl.local(t).isoformat()


# ---------------------------------------------------------------------------------- duplicate_invoice
def copy_times(state: InjectState, s: Stream, inv: WInvoice, k: int) -> list[datetime]:
    """Issue (and posting) instants for k-1 copies: later than the original, same period,
    distinct from every document instant and from the original's posting instant."""
    tl, world = state.world.tl, state.world
    out: list[datetime] = []
    t = inv.issued
    for _ in range(k - 1):
        lo = t + HOUR
        hi = min(tl.period_end - HOUR, lo + 10 * DAY)
        if hi - lo < HOUR:
            break

        def draw(lo: datetime = lo, hi: datetime = hi) -> datetime:
            for _ in range(16):
                c = tl.business_between(s, lo, hi)
                if c != inv.post_ts:
                    return c
            raise CandidateError("duplicate_invoice_time")

        t = world.claim(draw)
        out.append(t)
    return out


def inject_duplicate(state: InjectState, s: Stream) -> None:
    pool = candidates.dup_invoices(state.world, state.used)
    if not pool:
        raise CandidateError("duplicate_invoice")
    inv = s.choice(pool)
    variant = s.choice(["same_invoice_id", "new_invoice_id"])
    k = 3 if s.chance(1, 5) else 2
    times = copy_times(state, s, inv, k)
    if not times:
        raise CandidateError("duplicate_invoice_time")
    state.begin("duplicate_invoice")
    copies = [add_copy(state, inv, t, new_number=(variant == "new_invoice_id")) for t in times]
    group = [inv.key] + [c.key for c in copies]
    amount = len(copies) * abs(inv.total)
    state.issue("duplicate_invoice", group, amount, inv.currency, [group])
    state.end(
        variant,
        group,
        copies=len(group),
        original=inv.key,
        copy_keys=[c.key for c in copies],
        copy_issued_at=[_iso(state, c.issued) for c in copies],
    )


# ---------------------------------------------------------------------------------- payment_allocation
def inject_allocation(state: InjectState, s: Stream) -> None:
    world = state.world
    pool = candidates.alloc_pairs(world, state.used)
    if not pool:
        raise CandidateError("payment_allocation")
    pay, inv = s.choice(pool)
    part = pay.plan[0][1]
    variant = s.weighted_choice(["none", "partial", "other_invoice"], [2, 3, 2])
    counts = world.number_counts()
    others = [
        b
        for b in world.invoices
        if b.key != inv.key
        and b.customer == inv.customer
        and b.currency == inv.currency
        and candidates.plain_invoice(b, state.used, counts)
    ]
    if variant == "other_invoice" and not others:
        variant = "partial"
    short = split_part(s, part) if variant == "partial" else None
    if variant == "partial" and short is None:
        variant = "none"

    state.begin("payment_allocation")
    row = ar_row(state, pay, inv.number)
    before = row.signed
    affected = [pay.key, inv.key]
    details: dict[str, Any] = {"payment": pay.key, "invoice": inv.key, "expected_applied": part}
    if variant == "none":
        state.touch_row(row)
        row.role = "UNAPPLIED"
        row.invoice = None
        amount = part
    elif variant == "partial":
        assert short is not None
        apply_partial(state, pay, inv.number, short)
        amount = short
    else:
        other = s.choice(others)
        state.touch_row(row).invoice = other.number
        details["applied_to"] = other.key
        state.used.add(other.key)
        amount = part
    details["ar_row_before"] = before
    details["ar_row_after"] = row.signed if row.role == "AR" else 0
    state.issue(
        "payment_allocation", affected, amount, pay.currency, alloc_evidence(state, pay, inv)
    )
    state.end(variant, affected, **details)


# ---------------------------------------------------------------------------------- fx_conversion
def _wrong_dates(s: Stream, world: World, pay: WPayment) -> list[date]:
    """Material wrong rate dates: first the mistakes of another date rule, then nearby days."""
    right = rate_date(world, pay.ts)
    material_dates = wrong_rate_dates(world, pay)
    out: list[date] = []
    for rule in FX_RATE_DATE_RULES:
        d = rate_date_by(world, rule, pay.ts)
        if d != right and d in material_dates and d not in out:
            out.append(d)
    out += [d for d in s.shuffled(material_dates) if d not in out]
    return out


def _wrong_conversion(
    state: InjectState, s: Stream, pay: WPayment, variant: str
) -> tuple[int, dict[str, str]] | None:
    world = state.world
    day = rate_date(world, pay.ts)
    if variant == "wrong_date":
        dates = _wrong_dates(s, world, pay)
        if not dates:
            return None
        wrong = convert_on(world, pay.amount, pay.currency, dates[0])
        assert wrong is not None
        return wrong, {"rate_date_used": dates[0].isoformat(), "rate_date_correct": day.isoformat()}
    others = s.shuffled([c for c in world.foreign if c != pay.currency])
    for other in others:
        rate = world.fx[(day.isoformat(), other)]
        wrong = convert_minor(pay.amount, pay.currency, world.base, rate, world.rounding)
        if material(world, pay, wrong):
            return wrong, {
                "pair_used": f"{other}/{world.base}",
                "pair_correct": f"{pay.currency}/{world.base}",
                "rate_date": day.isoformat(),
            }
    return None


def inject_fx(state: InjectState, s: Stream) -> None:
    world = state.world
    pool = candidates.fx_payments(world, state.used)
    for pay in s.shuffled(pool):
        variants = ["wrong_date"]
        if len(world.foreign) > 1:
            variants = s.shuffled(["wrong_date", "wrong_date", "wrong_pair"])
        for variant in dict.fromkeys(variants):
            found = _wrong_conversion(state, s, pay, variant)
            if found is None:
                continue
            wrong, details = found
            state.begin("fx_conversion")
            right = rebook_conversion(state, pay, wrong)
            state.issue(
                "fx_conversion",
                [pay.key],
                abs(wrong - right),
                world.base,
                fx_evidence(state, pay),
                tolerance=FX_TOLERANCE_MINOR,
            )
            state.end(variant, [pay.key], cash_before=right, cash_after=wrong, **details)
            return
    raise CandidateError("fx_conversion")


# ---------------------------------------------------------------------------------- period_cutoff
def inject_cutoff(state: InjectState, s: Stream) -> None:
    world = state.world
    pool = candidates.cutoff_docs(world, state.used)
    for doc in s.shuffled(pool):
        doc_time = doc.issued if isinstance(doc, WInvoice) else doc.ts
        boundary = world.tl.boundary_before(doc_time)
        assert boundary is not None
        new_ts = None
        for offset in s.shuffled([0, 1, 2, 3]):
            new_ts = posting_after(state, s, boundary, doc_time, offset)
            if new_ts is not None:
                break
        if new_ts is None:
            continue
        amount = abs(doc.total if isinstance(doc, WInvoice) else doc.amount)
        state.begin("period_cutoff")
        old = move_posting(state, doc, new_ts)
        state.issue(
            "period_cutoff", [doc.key], amount, doc.currency, cutoff_evidence(state, doc.key)
        )
        variant = "invoice_posted_late" if isinstance(doc, WInvoice) else "payment_posted_late"
        state.end(
            variant,
            [doc.key],
            boundary="period_start" if boundary == world.tl.period_start else "period_end",
            posted_before=_iso(state, old),
            posted_after=_iso(state, new_ts),
        )
        return
    raise CandidateError("period_cutoff")
