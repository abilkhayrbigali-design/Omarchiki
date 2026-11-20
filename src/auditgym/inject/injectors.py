"""The four error injectors (docs/DOMAIN.md, section 6). One call injects one error.

Each injector picks a candidate with its own stream, validates every amount before editing, edits
the working books, and records the truth (amount from documents only, evidence per the DOMAIN
table) together with record-level before/after values.

Variants (config `variants`, World.allowed_variants): an injector produces only allowed variants.
It draws among them in the canonical order of ISSUE_VARIANTS with the same weights, so with every
variant allowed the draws are exactly those of a config without the section. A fallback for a
variant the candidate cannot carry only goes to another allowed variant; candidates that can
carry none are skipped, and with no candidate left the attempt fails (CandidateError).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from auditgym.contracts.constants import FX_RATE_DATE_RULES
from auditgym.inject.common import (
    FX_TOLERANCE_MINOR,
    add_copy,
    alloc_evidence,
    apply_none,
    apply_partial,
    ar_row,
    cutoff_evidence,
    fx_evidence,
    move_posting,
    posting_after,
    rebook_conversion,
)
from auditgym.inject.state import CandidateError, InjectState
from auditgym.util.rng import Stream
from auditgym.world import candidates
from auditgym.world.amounts import can_split, split_part
from auditgym.world.fx import (
    convert_on,
    material,
    pair_conversion,
    rate_date,
    rate_date_by,
    wrong_rate_dates,
)
from auditgym.world.model import WInvoice, World, WPayment
from auditgym.world.timeline import DAY, HOUR


def _iso(state: InjectState, t: datetime) -> str:
    return state.world.tl.local(t).isoformat()


# ---------------------------------------------------------------------------------- duplicate_invoice
def copy_times(state: InjectState, s: Stream, inv: WInvoice, copies: int) -> list[datetime]:
    """Issue (and posting) instants for the copies: later than the original, inside the period,
    distinct from every document instant and from the original's posting instant."""
    tl, world = state.world.tl, state.world
    lo = inv.issued + HOUR
    hi = min(tl.period_end - HOUR, lo + 10 * DAY)
    step = int((hi - lo).total_seconds()) // copies
    if step < 60:
        raise CandidateError("duplicate_invoice_time")
    out: list[datetime] = []
    for i in range(copies):
        # Copy i lies in the i-th slice of the window, so later copies always have room.
        a = lo + timedelta(seconds=step * i)
        b = lo + timedelta(seconds=step * (i + 1))

        def draw(a: datetime = a, b: datetime = b) -> datetime:
            for _ in range(16):
                c = tl.business_between(s, a, b)
                if c != inv.post_ts:
                    return c
            raise CandidateError("duplicate_invoice_time")

        out.append(world.claim(draw))
    return out


def inject_duplicate(state: InjectState, s: Stream) -> None:
    pool = candidates.dup_invoices(state.world, state.used)
    if not pool:
        raise CandidateError("duplicate_invoice")
    inv = s.choice(pool)
    variant = s.choice(state.world.allowed_variants("duplicate_invoice"))
    planned = state.world.sizing.get("dup_copies", [])
    copies = planned[state.duplicates_done] if state.duplicates_done < len(planned) else 1
    state.duplicates_done += 1
    times = copy_times(state, s, inv, copies)
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
ALLOC_SHAPES = ("first", "later", "multi_ref")
ALLOC_SHAPE_WEIGHTS = (2, 4, 2)
ALLOC_VARIANT_WEIGHTS = {"none": 2, "partial": 3, "other_invoice": 2}
# A drawn variant the candidate cannot carry falls back along this chain, to allowed ones only.
ALLOC_FALLBACK = ("other_invoice", "partial", "none")


def _other_invoices(state: InjectState, inv: WInvoice, counts: dict[str, int]) -> list[WInvoice]:
    """Invoices the payment can be applied to by mistake: same customer and currency, plain,
    not reserved as a carrier."""
    world = state.world
    return [
        b
        for b in world.invoices
        if b.key != inv.key
        and b.customer == inv.customer
        and b.currency == inv.currency
        and b.key not in world.reserved
        and candidates.plain_invoice(world, b, state.used, counts, "alloc")
    ]


def _alloc_can_carry(variant: str, part: int, others: list[WInvoice]) -> bool:
    if variant == "other_invoice":
        return bool(others)
    if variant == "partial":
        return can_split(part)
    return True


def _alloc_fallback(
    variant: str, allowed: tuple[str, ...], part: int, others: list[WInvoice]
) -> str:
    """The next allowed variant down the chain the candidate can carry, else any allowed one."""
    chain = ALLOC_FALLBACK[ALLOC_FALLBACK.index(variant) + 1 :] + allowed
    for v in chain:
        if v in allowed and _alloc_can_carry(v, part, others):
            return v
    raise CandidateError("payment_allocation")


def inject_allocation(state: InjectState, s: Stream) -> None:
    """Shapes: the first payment on an invoice; a later instalment (expected = what is still
    open, which may be less than the payment); a payment naming several invoices, the target
    first. Variants: nothing applied, partly applied, applied to another invoice."""
    world = state.world
    allowed = world.allowed_variants("payment_allocation")
    counts = world.number_counts()
    pool = candidates.alloc_candidates(world, state.used)
    if "none" not in allowed:  # "none" fits every candidate; otherwise keep those that can carry
        pool = [
            c
            for c in pool
            if any(
                _alloc_can_carry(v, c.pay.plan[0][1], _other_invoices(state, c.inv, counts))
                for v in allowed
            )
        ]
    if not pool:
        raise CandidateError("payment_allocation")
    shapes = [sh for sh in ALLOC_SHAPES if any(c.shape == sh for c in pool)]
    shape = s.weighted_choice(
        shapes, [ALLOC_SHAPE_WEIGHTS[ALLOC_SHAPES.index(sh)] for sh in shapes]
    )
    cand = s.choice([c for c in pool if c.shape == shape])
    pay, inv = cand.pay, cand.inv
    part = pay.plan[0][1]
    variant = s.weighted_choice(allowed, [ALLOC_VARIANT_WEIGHTS[v] for v in allowed])
    others = _other_invoices(state, inv, counts)
    if not _alloc_can_carry(variant, part, others):
        variant = _alloc_fallback(variant, allowed, part, others)
    short = split_part(s, part) if variant == "partial" else None

    state.begin("payment_allocation")
    row = ar_row(state, pay, inv.number)
    before = row.signed
    affected = [pay.key, inv.key]
    details: dict[str, Any] = {
        "payment": pay.key,
        "invoice": inv.key,
        "shape": shape,
        "expected_applied": part,
        "payment_amount": pay.amount,
        "invoice_total": inv.total,
    }
    if variant == "none":
        apply_none(state, pay, inv.number)
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
    details["ar_row_after"] = row.signed if variant == "partial" else 0
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
        wrong = pair_conversion(world, pay, other)
        if material(world, pay, wrong):
            return wrong, {
                "pair_used": f"{other}/{world.base}",
                "pair_correct": f"{pay.currency}/{world.base}",
                "rate_date": day.isoformat(),
            }
    return None


def inject_fx(state: InjectState, s: Stream) -> None:
    world = state.world
    allowed = world.allowed_variants("fx_conversion")
    pool = candidates.fx_payments(world, state.used, variants=allowed)
    for pay in s.shuffled(pool):
        variants = ["wrong_date"]
        if len(world.foreign) > 1:
            variants = s.shuffled(["wrong_date", "wrong_date", "wrong_pair"])
        for variant in dict.fromkeys(variants):
            if variant not in allowed:
                continue
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
    pool = candidates.cutoff_docs(
        world, state.used, variants=world.allowed_variants("period_cutoff")
    )
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
