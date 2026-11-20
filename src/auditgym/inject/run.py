"""Run every planned injection on a working copy of the clean books (pipeline step 3)."""

from __future__ import annotations

from collections.abc import Callable

from auditgym.books.posting import WRow
from auditgym.inject.injectors import inject_allocation, inject_cutoff, inject_duplicate, inject_fx
from auditgym.inject.plan import ORDER, Plan
from auditgym.inject.scenarios import duplicate_misallocated, fx_partial_at_boundary
from auditgym.inject.state import InjectState
from auditgym.util.rng import Stream
from auditgym.world.model import World

INJECTORS: dict[str, Callable[[InjectState, Stream], None]] = {
    "fx_partial_at_boundary": fx_partial_at_boundary,
    "duplicate_misallocated": duplicate_misallocated,
    "period_cutoff": inject_cutoff,
    "fx_conversion": inject_fx,
    "duplicate_invoice": inject_duplicate,
    "payment_allocation": inject_allocation,
}


def run_injections(root: Stream, world: World, clean_rows: list[WRow], plan: Plan) -> InjectState:
    """Inject the plan into copies of `world` and `clean_rows`; the originals stay clean."""
    state = InjectState(world, clean_rows)
    for name in ORDER:
        stream = root.child(f"inject/{name}")
        for _ in range(plan.count(name)):
            INJECTORS[name](state, stream)
    return state
