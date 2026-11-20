"""How many errors of each kind an episode gets (config["errors"], config["interactions"]).

Order fixed by auditgym.contracts.config: interaction counts first, then per-type counts; if the
total exceeds max_issues, randomly chosen per-type errors (never interactions) are removed until it
fits. Removal first keeps every per-type minimum; only if that is not enough does it go below the
minimums (possible when interactions are drawn high).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from auditgym.contracts.constants import (
    INTERACTION_ISSUE_COUNTS,
    INTERACTION_SCENARIOS,
    ISSUE_TYPES,
)
from auditgym.util.rng import Stream

# Injection order: the most constrained candidate pools first.
ORDER = (
    "fx_partial_at_boundary",
    "duplicate_misallocated",
    "period_cutoff",
    "fx_conversion",
    "duplicate_invoice",
    "payment_allocation",
)
POOL_OF = {
    "fx_partial_at_boundary": "fxb",
    "duplicate_misallocated": "dm",
    "period_cutoff": "cutoff",
    "fx_conversion": "fx",
    "duplicate_invoice": "dup",
    "payment_allocation": "alloc",
}


@dataclass
class Plan:
    interactions: dict[str, int]
    errors: dict[str, int]
    removed: list[str] = field(default_factory=list)
    dup_copies: list[int] = field(default_factory=list)  # copies per duplicate_invoice (1 or 2)

    def count(self, name: str) -> int:
        if name in self.interactions:
            return self.interactions[name]
        return self.errors[name]

    @property
    def total_issues(self) -> int:
        return sum(self.errors.values()) + sum(
            n * INTERACTION_ISSUE_COUNTS[s] for s, n in self.interactions.items()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "interactions": dict(self.interactions),
            "errors": dict(self.errors),
            "removed_by_max_issues": list(self.removed),
            "total_issues": self.total_issues,
            "duplicate_copies_drawn": list(self.dup_copies),
        }


def sample_plan(s: Stream, cfg: dict[str, Any]) -> Plan:
    interactions = {sc: s.in_range(cfg["interactions"][sc]) for sc in INTERACTION_SCENARIOS}
    errors = {t: s.in_range(cfg["errors"][t]) for t in ISSUE_TYPES}
    plan = Plan(interactions, errors)
    cap = cfg["max_issues"]
    for keep_minimum in (True, False):
        while plan.total_issues > cap:
            removable = []
            for t in ISSUE_TYPES:
                floor = cfg["errors"][t][0] if keep_minimum else 0
                removable += [t] * max(0, errors[t] - floor)
            if not removable:
                break
            t = s.choice(removable)
            errors[t] -= 1
            plan.removed.append(t)
    if plan.total_issues > cap:
        raise ValueError("max_issues cannot be met")  # excluded by normalize_config
    plan.dup_copies = [2 if s.chance(1, 5) else 1 for _ in range(errors["duplicate_invoice"])]
    return plan


def world_demand(plan: Plan) -> dict[str, Any]:
    """What the world must provide: one reserved carrier per planned injection, and room for
    the invoice copies the duplicate injectors will add (see auditgym.world.sizing)."""
    return {
        "carriers": {POOL_OF[name]: plan.count(name) for name in ORDER if plan.count(name)},
        "dup_copies": list(plan.dup_copies),
        "dm_copies": plan.interactions["duplicate_misallocated"],
    }
