"""K2 on the hand-built tiny fixture: the solver finds exactly the reference answer, and every
reference evidence set is independently sufficient and minimal."""

import pytest
import verify_builders as vb

from auditgym.verify import check_evidence, solve


def key4(issues):
    return {(i.type, i.affected_ids, i.expected_amount_minor, i.currency) for i in issues}


def test_K2_solve_equals_truth_on_tiny(tiny_episode):
    got = solve(tiny_episode.public)
    assert key4(got) == key4(tiny_episode.private.truth.issues)
    assert len(got) == len(tiny_episode.private.truth.issues) == 4


def test_K2_solver_tolerances_and_evidence_match_truth_on_tiny(tiny_episode):
    got = {i.key(): i for i in solve(tiny_episode.public)}
    for truth in tiny_episode.private.truth.issues:
        mine = got[truth.key()]
        assert mine.amount_tolerance_minor == truth.amount_tolerance_minor
        assert set(mine.evidence_sets) == set(truth.evidence_sets)


def test_K2_every_truth_evidence_set_is_sufficient(tiny_episode):
    for issue in tiny_episode.private.truth.issues:
        for ev in issue.evidence_sets:
            check = check_evidence(tiny_episode.public, issue.type, issue.affected_ids, ev)
            assert check.ok, (issue.type, ev, check.reason)
            assert check.amount_minor == issue.expected_amount_minor
            assert check.currency == issue.currency


def _sets(tiny):
    for issue in tiny.private.truth.issues:
        for ev in issue.evidence_sets:
            for dropped in ev:
                yield issue, ev, dropped


@pytest.mark.parametrize("case", list(_sets(vb.build_tiny_episode())), ids=lambda c: c[2])
def test_K2_removing_any_evidence_id_fails(case):
    issue, ev, dropped = case
    public = vb.tiny_public()
    rest = [x for x in ev if x != dropped]
    check = check_evidence(public, issue.type, issue.affected_ids, rest)
    assert not check.ok or check.amount_minor != issue.expected_amount_minor


def test_T5_corrected_tiny_is_clean():
    assert solve(vb.clean_tiny()) == ()


def test_G01_each_tiny_error_alone():
    """Re-inject the tiny fixture's errors one at a time into the corrected copy."""
    tiny = vb.tiny_public()
    clean = vb.clean_tiny()
    by_id = {r.row_id: r for r in tiny.ledger}

    def ledger_with(**rows):
        out = [rows.get(r.row_id, r) for r in clean.ledger]
        return out + [r for k, r in rows.items() if k not in {x.row_id for x in clean.ledger}]

    dup = vb.rebuild(
        clean,
        invoices=list(clean.invoices) + [i for i in tiny.invoices if i.doc_id == "D-I4"],
        ledger=ledger_with(**{"R-007": by_id["R-007"], "R-008": by_id["R-008"]}),
    )
    fx = vb.rebuild(clean, ledger=ledger_with(**{"R-017": by_id["R-017"], "R-018": by_id["R-018"]}))
    alloc = vb.rebuild(
        clean, ledger=ledger_with(**{"R-020": by_id["R-020"], "R-021": by_id["R-021"]})
    )
    cutoff = vb.rebuild(
        clean, ledger=ledger_with(**{"R-011": by_id["R-011"], "R-012": by_id["R-012"]})
    )
    truth = {i.type: i for i in vb.build_tiny_episode().private.truth.issues}
    for episode, issue_type in (
        (dup, "duplicate_invoice"),
        (fx, "fx_conversion"),
        (alloc, "payment_allocation"),
        (cutoff, "period_cutoff"),
    ):
        assert key4(solve(episode)) == key4([truth[issue_type]]), issue_type
