"""K2, independent sufficiency of evidence: check_evidence uses only the named records."""

from dataclasses import replace

import pytest
import verify_builders as vb

from auditgym.verify import check_evidence, solve
from auditgym.verify.evidence import restrict


def _issues():
    public, _ = vb.kitchen_sink()
    return public, solve(public)


def test_K2_solver_evidence_sets_are_sufficient():
    """Every evidence set the solver emits re-derives its issue, except the documented gap."""
    public, issues = _issues()
    checked = 0
    for issue in issues:
        if issue.key() in vb.EVIDENCE_GAP_KEYS:
            continue
        for ev in issue.evidence_sets:
            check = check_evidence(public, issue.type, issue.affected_ids, ev)
            assert check.ok, (issue.key(), ev, check.reason)
            assert (check.amount_minor, check.currency) == (
                issue.expected_amount_minor,
                issue.currency,
            )
            checked += 1
    assert checked >= 20


def test_K2_evidence_ignores_records_outside_the_set():
    """Same verdict on an episode that holds only the evidence records."""
    public, issues = _issues()
    for issue in issues:
        for ev in issue.evidence_sets:
            view = restrict(public, frozenset(ev))
            only = replace(
                public,
                invoices=view.invoices,
                payments=view.payments,
                ledger=view.rows,
                fx=view.rates,
            )
            assert check_evidence(only, issue.type, issue.affected_ids, ev) == check_evidence(
                public, issue.type, issue.affected_ids, ev
            )


def test_K2_extra_ids_are_allowed():
    public, issues = _issues()
    issue = next(i for i in issues if i.type == "fx_conversion")
    ev = list(issue.evidence_sets[0]) + ["D-I1", "R-001", "no-such-id"]
    check = check_evidence(public, issue.type, issue.affected_ids, ev)
    assert check.ok and check.amount_minor == issue.expected_amount_minor


def test_K2_documented_gap_second_invoice_of_multi_invoice_payment():
    """DOMAIN's set for payment_allocation lists the payment, the invoice, the payment rows and
    earlier payments of that invoice. When the invoice is not first in invoice_refs, the
    expected amount also depends on the invoices listed before it, which the set lacks."""
    public = vb.apply(vb.clean_tiny(), vb.alloc_second_ref())
    (issue,) = solve(public)
    (ev,) = issue.evidence_sets
    check = check_evidence(public, issue.type, issue.affected_ids, ev)
    assert not check.ok and check.reason.startswith("insufficient")
    check = check_evidence(public, issue.type, issue.affected_ids, list(ev) + ["D-AG1"])
    assert check.ok and check.amount_minor == 150_000


def test_K2_earlier_payment_changes_the_amount():
    """Without the earlier payment the open balance looks larger: 700000 - 500000 = 200000
    instead of 600000 - 500000 = 100000."""
    public = vb.apply(vb.clean_tiny(), vb.alloc_partial_after_earlier_payment())
    (issue,) = solve(public)
    (ev,) = issue.evidence_sets
    assert "D-AP2" in ev
    assert check_evidence(public, issue.type, issue.affected_ids, ev).amount_minor == 100_000
    rest = [x for x in ev if x != "D-AP2"]
    assert check_evidence(public, issue.type, issue.affected_ids, rest).amount_minor == 200_000


def test_K2_each_allocation_row_is_needed():
    public = vb.apply(vb.clean_tiny(), vb.alloc_partial_after_earlier_payment())
    (issue,) = solve(public)
    (ev,) = issue.evidence_sets
    for row_id in [x for x in ev if x.startswith("RAP-")]:
        rest = [x for x in ev if x != row_id]
        assert not check_evidence(public, issue.type, issue.affected_ids, rest).ok


@pytest.mark.parametrize(
    "issue_type, affected, evidence, reason_part",
    [
        ("fx_conversion", ["D-P1"], ["D-P1", "R-013"], "base currency"),
        ("period_cutoff", ["D-I1"], ["D-I1", "R-001"], "same period"),
        ("duplicate_invoice", ["D-I1", "D-I6"], ["D-I1", "D-I6"], "neither"),
        ("payment_allocation", ["D-P1", "D-I1"], ["D-P1", "D-I1", "R-013", "R-014"], "at least"),
        ("payment_allocation", ["D-P1", "D-I6"], ["D-P1", "D-I6", "R-013", "R-014"], "reference"),
        ("fx_conversion", ["D-P2"], ["D-P2", "R-017"], "rate"),
        ("period_cutoff", ["D-I5"], ["D-I5"], "no row"),
        ("period_cutoff", ["D-I5"], ["R-011"], "not in the evidence"),
        ("bogus_type", ["D-I5"], ["D-I5", "R-011"], "unknown issue type"),
        ("period_cutoff", ["D-I5"], ["D-I5", "R-011"] + [f"x{i}" for i in range(9)], "more than"),
        ("period_cutoff", [], ["D-I5", "R-011"], "no affected"),
        ("fx_conversion", ["D-P2", "D-I2"], ["D-P2", "D-I2", "R-017", "2026-03-20/USD/KZT"], "one"),
    ],
)
def test_K2_wrong_claims_are_rejected(issue_type, affected, evidence, reason_part):
    check = check_evidence(vb.tiny_public(), issue_type, affected, evidence)
    assert not check.ok
    assert check.amount_minor is None and check.currency is None
    assert reason_part in check.reason


def test_K2_duplicate_must_name_every_copy_in_evidence():
    public = vb.apply(vb.clean_tiny(), vb.dup_triple())
    (issue,) = solve(public)
    assert issue.affected_ids == ("D-DT1", "D-DT2", "D-DT3")
    two = ["D-DT1", "D-DT2"]
    assert check_evidence(public, issue.type, two, two).ok  # the set alone shows a pair
    check = check_evidence(public, issue.type, two, two + ["D-DT3"])
    assert not check.ok and "exactly one group" in check.reason


def test_K2_cutoff_on_shared_invoice_id_is_not_attributable():
    public = vb.tiny_public()
    check = check_evidence(public, "period_cutoff", ["D-I3"], ["D-I3", "D-I4", "R-005"])
    assert not check.ok and "attributed" in check.reason


def test_K2_fx_evidence_rows_must_agree():
    """CASH and base FX_CLEARING rows of one payment that contradict each other."""
    public = vb.tiny_public()
    ledger = [
        replace(r, credit_minor=60_000_000) if r.row_id == "R-018" else r for r in public.ledger
    ]
    broken = replace(public, ledger=tuple(ledger))
    ev = ["D-P2", "R-017", "R-018", "2026-03-20/USD/KZT"]
    check = check_evidence(broken, "fx_conversion", ["D-P2"], ev)
    assert not check.ok and "disagree" in check.reason
