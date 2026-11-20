"""How many invoice documents an episode has (config `size.invoices`).

Every invoice document of the published episode counts: distractor invoices, carriers of planned
errors, duplicate copies added by injectors, and ordinary invoices. The generator draws the total
inside [min, max] and builds exactly that many.

Costs in invoice documents:
- distractors: per instance X1 1, X2 2-3, X3 1, X4 2 (sale + credit note), X5 2, X6 2, X7 0,
  X8 1, X9 1, X10 1;
- every planned injection (scenarios included) gets one carrier invoice;
- duplicate_invoice adds 1 copy (sometimes 2), duplicate_misallocated adds 1 copy;
- at least MIN_FREE_REGULAR ordinary invoices besides carriers.

`check_capacity(cfg)` fails fast when even the largest possible plan of errors with the smallest
distractor counts cannot fit under size.invoices max. Per seed, `fit` first trims optional extras
(third copies, third X2 invoices) and then random distractor instances above their configured
minimum, so the configured range is never exceeded.
"""

from __future__ import annotations

from typing import Any

from auditgym.contracts.config import ConfigError
from auditgym.contracts.constants import (
    DISTRACTOR_KINDS,
    INTERACTION_ISSUE_COUNTS,
    INTERACTION_SCENARIOS,
    ISSUE_TYPES,
)
from auditgym.util.rng import Stream

INVOICES_PER_INSTANCE = {
    "split_payment": 1,
    "multi_invoice_payment": 2,  # 3 when drawn so and it fits
    "partial_open": 1,
    "credit_note_refund": 2,
    "subscription": 2,
    "same_amount_orders": 2,
    "unapplied_payment": 0,
    "boundary_offset": 1,
    "fx_spike": 1,
    "posting_lag": 1,
}
MIN_FREE_REGULAR = 2
# Invoice documents one planned injection needs: carrier + copies (smallest variant).
ISSUE_COST = {
    "duplicate_invoice": 2,
    "payment_allocation": 1,
    "fx_conversion": 1,
    "period_cutoff": 1,
}
SCENARIO_COST = {"fx_partial_at_boundary": 1, "duplicate_misallocated": 2}


class CapacityError(ConfigError):
    """size.invoices is too small for the configured errors and distractors."""


def _worst_issue_invoices(cfg: dict[str, Any]) -> int:
    """Largest number of invoice documents any plan allowed by the config can need."""
    cap = cfg["max_issues"]
    cost = sum(cfg["interactions"][s][1] * SCENARIO_COST[s] for s in INTERACTION_SCENARIOS)
    room = cap - sum(
        cfg["interactions"][s][1] * INTERACTION_ISSUE_COUNTS[s] for s in INTERACTION_SCENARIOS
    )
    counts = {t: cfg["errors"][t][0] for t in ISSUE_TYPES}
    room -= sum(counts.values())
    for t in sorted(ISSUE_TYPES, key=lambda t: (-ISSUE_COST[t], t)):
        extra = max(0, min(cfg["errors"][t][1] - counts[t], room))
        counts[t] += extra
        room -= extra
    return cost + sum(counts[t] * ISSUE_COST[t] for t in ISSUE_TYPES)


def required_invoices(cfg: dict[str, Any]) -> int:
    """Smallest size.invoices max that fits every plan the config allows."""
    distractors = sum(cfg["distractors"][k][0] * INVOICES_PER_INSTANCE[k] for k in DISTRACTOR_KINDS)
    return distractors + _worst_issue_invoices(cfg) + MIN_FREE_REGULAR


def check_capacity(cfg: dict[str, Any]) -> None:
    need = required_invoices(cfg)
    have = cfg["size"]["invoices"][1]
    if need > have:
        raise CapacityError(
            f"config.size.invoices: max {have} is too small; the configured errors and minimum "
            f"distractors need at least {need} invoice documents"
        )


def demand_invoices(demand: dict[str, Any]) -> int:
    return sum(demand["carriers"].values()) + sum(demand["dup_copies"]) + demand["dm_copies"]


def fit(
    s: Stream,
    cfg: dict[str, Any],
    distractors: dict[str, list[int]],
    demand: dict[str, Any],
) -> int:
    """Trim optional extras until everything fits; return the target number of invoices.

    `distractors` (kind -> invoices per instance) and `demand["dup_copies"]` are edited in place.
    """
    lo, hi = cfg["size"]["invoices"]

    def needed() -> int:
        total = sum(sum(sizes) for sizes in distractors.values())
        return total + demand_invoices(demand) + MIN_FREE_REGULAR

    copies = demand["dup_copies"]
    for i in range(len(copies)):
        if needed() > hi and copies[i] > 1:
            copies[i] = 1
    sizes = distractors["multi_invoice_payment"]
    for i in range(len(sizes)):
        if needed() > hi and sizes[i] > 2:
            sizes[i] = 2
    while needed() > hi:
        removable = [
            k
            for k in DISTRACTOR_KINDS
            if len(distractors[k]) > cfg["distractors"][k][0] and any(distractors[k])
        ]
        if not removable:
            raise CapacityError(
                f"config.size.invoices: max {hi} cannot hold this episode ({needed()} needed)"
            )
        kind = s.choice(removable)
        distractors[kind].pop()
    return s.randint(max(lo, needed()), hi)
