"""J06, J07: the jury's published matcher conventions (docs/REQUIREMENTS.md, section 8)."""

from __future__ import annotations

import itertools

import pytest
from reward_helpers import EID, TINY, TRUTH, finding, oracle_finding, submission

from auditgym.contracts.types import Finding, Issue, Truth
from auditgym.grading import grade_submission, match_findings

KNOWN = TINY.public.open_ids()
CUT = next(i for i in TRUTH.issues if i.type == "period_cutoff")


def _grade(findings, truth=TRUTH):
    return grade_submission(
        submission(findings), truth, episode_id=EID, inspection_steps=0, known_ids=KNOWN
    )


def test_J06_evidence_id_outside_the_open_data_unmatches_the_finding():
    good = oracle_finding(CUT)
    padded = {**good, "evidence_ids": [*good["evidence_ids"], "U-NOT-THERE"]}
    extra_known = sorted(KNOWN - set(good["evidence_ids"]))[0]
    padded_known = {**good, "evidence_ids": [*good["evidence_ids"], extra_known]}
    assert (_grade([good]).tp, _grade([good]).fp) == (1, 0)
    assert (_grade([padded]).tp, _grade([padded]).fp) == (0, 1)  # M04
    assert (_grade([padded_known]).tp, _grade([padded_known]).fp) == (1, 0)  # M03


def test_J06_open_ids_cover_records_and_the_ids_they_carry():
    pub = TINY.public
    assert {r.row_id for r in pub.ledger} <= KNOWN
    assert {d.doc_id for d in (*pub.invoices, *pub.payments)} <= KNOWN
    assert {d.invoice_id for d in pub.invoices} <= KNOWN
    assert {f.rate_id for f in pub.fx} <= KNOWN
    assert "U-NOT-THERE" not in KNOWN


@pytest.mark.parametrize("currency", ["kzt", "KZT ", " KZT"])
def test_J06_currency_is_compared_exactly_and_misses_are_false_positives(currency):
    result = _grade([{**oracle_finding(CUT), "currency": currency}])
    assert result.valid and (result.tp, result.fp, result.fn) == (0, 1, len(TRUTH.issues))


def test_J06_limits_are_counted_before_removing_repeats():
    f = finding("period_cutoff", ["D-I5"], 1, "KZT", ["D-I5"] * 11)
    assert _grade([f]).valid is False  # M05
    g = finding("period_cutoff", ["D-I5", "D-I5", "D-I5"], 1, "KZT", ["D-I5"] * 10)
    assert _grade([g]).valid is True  # M01: repeats inside one list are one id


def _brute_force(adjacency: list[set[int]], issue_ids: list[str]) -> tuple[tuple[int, str], ...]:
    """Lexicographically smallest sorted (issue_id, finding) sequence among maximum matchings."""
    n_f, n_i = len(adjacency), len(issue_ids)
    best_size, best = 0, [()]
    for assign in itertools.product([None, *range(n_i)], repeat=n_f):
        used = [j for j in assign if j is not None]
        if len(used) != len(set(used)):
            continue
        if any(j is not None and j not in adjacency[i] for i, j in enumerate(assign)):
            continue
        seq = tuple(sorted((issue_ids[j], i) for i, j in enumerate(assign) if j is not None))
        if len(seq) > best_size:
            best_size, best = len(seq), [seq]
        elif len(seq) == best_size:
            best.append(seq)
    chosen = min(best)
    return tuple(sorted((i, iid) for iid, i in chosen))


@pytest.mark.parametrize("seed", range(60))
def test_J07_reported_pairs_follow_the_published_rule(seed):
    # Deterministic pseudo-random graphs without the random module: findings cite evidence ids,
    # issue j is satisfied by evidence "E{j}". Issue ids are not in truth order on purpose.
    n_i = 1 + seed % 4
    n_f = 1 + (seed * 7) % 5
    issue_ids = [f"iss_{(seed * 13 + 5 * j) % 17:02d}_{j}" for j in range(n_i)]
    issues = tuple(
        Issue(issue_ids[j], "period_cutoff", ("R1",), 100, "KZT", 0, ((f"E{j}",),))
        for j in range(n_i)
    )
    adjacency = []
    findings = []
    for i in range(n_f):
        cited = {j for j in range(n_i) if (seed * 31 + i * 11 + j * 5) % 3 != 0}
        adjacency.append(cited)
        findings.append(
            Finding("period_cutoff", ("R1",), 100, "KZT", tuple(sorted(f"E{j}" for j in cited)), "")
        )
    result = match_findings(findings, Truth(EID, issues).issues)
    expected = _brute_force(adjacency, issue_ids)
    assert result.tp == len(expected)
    assert result.pairs == expected


def test_J07_open_examples_m10_m11():
    a = Issue("A", "period_cutoff", ("R1",), 100, "KZT", 0, (("D1",),))
    b = Issue("B", "period_cutoff", ("R1",), 100, "KZT", 0, (("D2",),))
    both = Finding("period_cutoff", ("R1",), 100, "KZT", ("D1", "D2"), "")
    only_a = Finding("period_cutoff", ("R1",), 100, "KZT", ("D1",), "")
    m10 = match_findings([both, only_a], (a, b))
    assert m10.tp == 2 and m10.pairs == ((0, "B"), (1, "A"))  # not the greedy F0 -> A
    m11 = match_findings([both, both], (b, a))
    assert m11.tp == 2 and m11.pairs == ((0, "A"), (1, "B"))
