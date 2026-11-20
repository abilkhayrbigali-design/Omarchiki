"""Interacting scenarios (docs/DOMAIN.md, section 7). One call injects one scenario.

Each issue of a scenario is computed from the documents independently of the others, so the
reference answer is unambiguous even though several errors sit on the same records.
"""

from __future__ import annotations

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
from auditgym.util.rng import Stream
from auditgym.world import candidates
from auditgym.world.amounts import split_part
from auditgym.world.fx import conversion, convert_on, material


def fx_partial_at_boundary(state: InjectState, s: Stream) -> None:
    """A foreign payment just before a boundary is posted across it, converted at the rate of the
    posting date instead of the document date, and applied only partly."""
    world = state.world
    tl = world.tl
    for pay, inv in s.shuffled(candidates.fxb_pairs(world, state.used)):
        boundary = tl.boundary_before(pay.ts)
        assert boundary is not None
        part = pay.plan[0][1]
        short = split_part(s, part)
        offsets = candidates.boundary_posting_offsets(world, pay)
        if short is None or not offsets:
            continue
        right = conversion(world, pay)
        new_ts = posting_after(state, s, boundary, pay.ts, s.choice(offsets))
        if new_ts is None:
            continue
        wrong = convert_on(world, pay.amount, pay.currency, tl.local_date(new_ts))
        assert wrong is not None and material(world, pay, wrong)
        state.begin("fx_partial_at_boundary", scenario="fx_partial_at_boundary")
        old_ts = move_posting(state, pay, new_ts)
        rebook_conversion(state, pay, wrong)
        apply_partial(state, pay, inv.number, short)
        state.issue(
            "period_cutoff",
            [pay.key],
            abs(pay.amount),
            pay.currency,
            cutoff_evidence(state, pay.key),
        )
        state.issue(
            "fx_conversion",
            [pay.key],
            abs(wrong - right),
            world.base,
            fx_evidence(state, pay),
            tolerance=FX_TOLERANCE_MINOR,
        )
        state.issue(
            "payment_allocation",
            [pay.key, inv.key],
            short,
            pay.currency,
            alloc_evidence(state, pay, inv),
        )
        state.end(
            "posted_across_boundary_converted_at_posting_date_partly_applied",
            [pay.key, inv.key],
            boundary="period_start" if boundary == tl.period_start else "period_end",
            posted_before=tl.local(old_ts).isoformat(),
            posted_after=tl.local(new_ts).isoformat(),
            rate_date_used=tl.local_date(new_ts).isoformat(),
            cash_before=right,
            cash_after=wrong,
            expected_applied=part,
            applied=part - short,
        )
        return
    raise CandidateError("fx_partial_at_boundary")


def duplicate_misallocated(state: InjectState, s: Stream) -> None:
    """An invoice is entered again under a new invoice_id; a payment that references the original
    is applied to the copy."""
    world = state.world
    for pay, inv in s.shuffled(candidates.dm_pairs(world, state.used)):
        lo, hi = candidates.dm_copy_window(world, pay, inv)
        if hi - lo < candidates.HOUR:
            continue
        issued = world.claim(lambda lo=lo, hi=hi: world.tl.business_between(s, lo, hi))
        state.begin("duplicate_misallocated", scenario="duplicate_misallocated")
        dup = add_copy(state, inv, issued, new_number=True)
        row = state.touch_row(ar_row(state, pay, inv.number))
        row.invoice = dup.number
        part = pay.plan[0][1]
        group = [inv.key, dup.key]
        state.issue("duplicate_invoice", group, abs(inv.total), inv.currency, [group])
        state.issue(
            "payment_allocation",
            [pay.key, inv.key],
            part,
            pay.currency,
            alloc_evidence(state, pay, inv),
        )
        state.end(
            "copy_under_new_invoice_id_payment_applied_to_copy",
            [pay.key, inv.key, dup.key],
            original=inv.key,
            copy=dup.key,
            copy_issued_at=world.tl.local(issued).isoformat(),
            payment=pay.key,
            expected_applied=part,
        )
        return
    raise CandidateError("duplicate_misallocated")
