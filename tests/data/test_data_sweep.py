"""Volume check of the generator over every shipped config (K2). Slow: `pytest -m slow`.

For each of SEEDS seeds per config:
- the first generation attempt succeeds (zero retries);
- the clean books satisfy every invariant and the injected books violate exactly the truth;
- the independent solver (auditgym.verify, written without reading the generator) returns exactly
  the truth on (type, affected_ids, amount, currency);
- every evidence set, on its own, re-establishes its issue and amount;
- undoing the provenance changes reproduces the clean books.
"""

from __future__ import annotations

import pytest
from genkit import SHIPPED, shipped_config, tables, undo_changes

from auditgym.episode.generate import generate_with_clean
from auditgym.episode.selfcheck import issue_signatures, problems

SEEDS = 300


@pytest.mark.slow
@pytest.mark.parametrize("name", SHIPPED)
def test_K2_sweep_zero_retries_zero_mismatches(name):
    from auditgym.verify import solve
    from auditgym.verify.evidence import check_evidence

    cfg = shipped_config(name)
    failures = []
    for seed in range(SEEDS):
        episode, clean = generate_with_clean(seed, cfg)
        prov = episode.private.provenance
        truth = episode.private.truth
        if prov["attempts"] != 1:
            failures.append((seed, "retry", prov["attempt_failures"]))
        found = problems(episode.public, truth, clean)
        if found:
            failures.append((seed, "self-check", found))
        if issue_signatures(solve(episode.public)) != issue_signatures(truth.issues):
            failures.append((seed, "solver != truth", None))
        for issue in truth.issues:
            for ev in issue.evidence_sets:
                res = check_evidence(episode.public, issue.type, issue.affected_ids, ev)
                if not (
                    res.ok
                    and res.amount_minor == issue.expected_amount_minor
                    and res.currency == issue.currency
                ):
                    failures.append((seed, "evidence", (issue.type, res.reason)))
        if undo_changes(episode.public, prov["changes"]) != tables(clean):
            failures.append((seed, "replay", None))
    assert failures == []
