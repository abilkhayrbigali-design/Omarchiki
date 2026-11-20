"""Volume check of the generator (K2): many seeds, every episode verified independently.

For each seed:
- clean books (before injection) must satisfy all invariants;
- the independent solver must return exactly the truth;
- every evidence set must, on its own, re-establish the issue and its amount;
- generation attempts > 1 are counted (each one is a generator inconsistency caught by the
  self-check).
Failures are kept with their seed so they can be replayed (measured failure analysis, K5).
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any

from auditgym.contracts.config import normalize_config
from auditgym.harness.transform import issue_key


def verify_episode(episode: Any, clean: Any) -> dict[str, Any]:
    """All K2 checks for one generated episode. Returns a dict with 'ok' and details."""
    from auditgym.books.invariants import check as check_invariants
    from auditgym.verify import solve
    from auditgym.verify.evidence import check_evidence

    problems: list[str] = []
    clean_violations = check_invariants(clean)
    if clean_violations:
        problems.append(f"clean books violate invariants: {[str(v) for v in clean_violations[:3]]}")

    truth = episode.private.truth
    truth_keys = sorted(issue_key(i) for i in truth.issues)
    solved_keys = sorted(issue_key(i) for i in solve(episode.public))
    if solved_keys != truth_keys:
        missing = [k for k in truth_keys if k not in solved_keys]
        extra = [k for k in solved_keys if k not in truth_keys]
        problems.append(f"solver != truth; missing {missing[:3]} extra {extra[:3]}")

    evidence_sets = 0
    for issue in truth.issues:
        if not issue.evidence_sets:
            problems.append(f"{issue.type} has no evidence set")
        for ev in issue.evidence_sets:
            evidence_sets += 1
            res = check_evidence(episode.public, issue.type, issue.affected_ids, ev)
            if (
                not res.ok
                or res.amount_minor != issue.expected_amount_minor
                or res.currency != issue.currency
            ):
                problems.append(
                    f"evidence set for {issue.type} {list(issue.affected_ids)} insufficient: {res.reason}"
                )
    return {
        "ok": not problems,
        "problems": problems,
        "issues": len(truth.issues),
        "evidence_sets": evidence_sets,
        "attempts": int(episode.private.provenance.get("attempts", 1)),
    }


def sweep(
    config: dict[str, Any] | None, seeds: list[int], max_failures: int = 20
) -> dict[str, Any]:
    from auditgym.episode.generate import generate_with_clean

    cfg = normalize_config(config)
    totals: Counter[str] = Counter()
    types: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    sizes = {"ledger": [], "documents": [], "issues": []}
    gen_seconds: list[float] = []

    for seed in seeds:
        started = time.perf_counter()
        try:
            episode, clean = generate_with_clean(seed, cfg)
        except Exception as exc:  # a crash is a failure to report, not to hide
            totals["generation_errors"] += 1
            if len(failures) < max_failures:
                failures.append(
                    {"seed": seed, "problems": [f"generation raised {type(exc).__name__}: {exc}"]}
                )
            continue
        gen_seconds.append(time.perf_counter() - started)
        result = verify_episode(episode, clean)
        totals["episodes"] += 1
        totals["ok"] += int(result["ok"])
        totals["retries"] += result["attempts"] - 1
        totals["evidence_sets"] += result["evidence_sets"]
        for issue in episode.private.truth.issues:
            types[issue.type] += 1
        if not episode.private.truth.issues:
            totals["clean_episodes"] += 1
        sizes["ledger"].append(len(episode.public.ledger))
        sizes["documents"].append(len(episode.public.invoices) + len(episode.public.payments))
        sizes["issues"].append(len(episode.private.truth.issues))
        if not result["ok"] and len(failures) < max_failures:
            failures.append({"seed": seed, "problems": result["problems"]})

    def span(values: list[int | float]) -> dict[str, Any]:
        if not values:
            return {}
        return {"min": min(values), "max": max(values), "mean": round(sum(values) / len(values), 3)}

    return {
        "config_id": cfg["config_id"],
        "seeds": [seeds[0], seeds[-1]] if seeds else [],
        "counts": dict(totals),
        "issue_types": dict(sorted(types.items())),
        "sizes": {k: span(v) for k, v in sizes.items()},
        "generation_seconds": span([round(s, 4) for s in gen_seconds]),
        "failures": failures,
    }
