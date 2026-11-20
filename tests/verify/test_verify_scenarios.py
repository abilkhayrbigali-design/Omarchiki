"""Hand-built scenarios: distractors X1-X10 give nothing (G03), each issue type and variant is
found alone (G01), both interacting scenarios give all their issues (G02), and policy options
(rate date rule, rounding, timezone, base currency, account codes) are read from the episode."""

import pytest
import verify_builders as vb

from auditgym.verify import solve


def key4(issues):
    return {(i.type, i.affected_ids, i.expected_amount_minor, i.currency) for i in issues}


@pytest.mark.parametrize("make", vb.DISTRACTORS, ids=lambda f: f.__name__)
def test_G03_distractor_alone_gives_nothing(make):
    assert solve(vb.apply(vb.clean_tiny(), make())) == ()


def test_G03_all_distractors_together_give_nothing():
    assert solve(vb.apply(vb.clean_tiny(), *(f() for f in vb.DISTRACTORS))) == ()


@pytest.mark.parametrize("make", vb.ISSUES, ids=lambda f: f.__name__)
def test_G01_issue_alone(make):
    patch = make()
    assert key4(solve(vb.apply(vb.clean_tiny(), patch))) == set(patch.expected)


@pytest.mark.parametrize("make", vb.INTERACTIONS, ids=lambda f: f.__name__)
def test_G02_interacting_scenario(make):
    patch = make()
    got = solve(vb.apply(vb.clean_tiny(), patch))
    assert key4(got) == set(patch.expected)
    assert len(got) == len(patch.expected) >= 2


def test_G02_interacting_issues_among_distractors():
    public, expected = vb.kitchen_sink()
    assert key4(solve(public)) == expected


def test_tolerances_come_from_policy():
    public, _ = vb.kitchen_sink()
    for issue in solve(public):
        want = 1 if issue.type == "fx_conversion" else 0
        assert issue.amount_tolerance_minor == want


def test_issues_sorted_and_unique():
    public, _ = vb.kitchen_sink()
    got = solve(public)
    keys = [i.key() for i in got]
    assert keys == sorted(keys) and len(set(keys)) == len(keys)
    assert solve(public) == got  # deterministic, issue_id included


def _sets(issue):
    return {frozenset(s) for s in issue.evidence_sets}


def test_evidence_sets_fx_partial_at_boundary():
    """Rows RIB-03..07: FX_CLEARING USD, AR, UNAPPLIED, CASH KZT, FX_CLEARING KZT."""
    got = {i.type: i for i in solve(vb.apply(vb.clean_tiny(), vb.fx_partial_at_boundary()))}
    rate = "2026-03-31/USD/KZT"
    assert _sets(got["payment_allocation"]) == {
        frozenset({"D-IB2", "D-IB1", "RIB-03", "RIB-04", "RIB-05"})
    }
    assert _sets(got["fx_conversion"]) == {
        frozenset({"D-IB2", rate, "RIB-06"}),
        frozenset({"D-IB2", rate, "RIB-07"}),
    }
    assert _sets(got["period_cutoff"]) == {frozenset({"D-IB2", f"RIB-0{k}"}) for k in range(3, 8)}


def test_allocation_evidence_lists_only_earlier_payments():
    (issue,) = solve(vb.apply(vb.clean_tiny(), vb.alloc_partial_after_earlier_payment()))
    assert _sets(issue) == {frozenset({"D-AP3", "D-AP1", "RAP-05", "RAP-06", "RAP-07", "D-AP2"})}


def test_allocation_on_invoice_id_with_two_documents_names_the_earliest():
    """DOMAIN 6: the invoice document of a repeated invoice_id is the earliest issued_at.
    The later copy is written first so that position cannot pick the right one."""
    r = vb.Rows("RAD-")
    first = vb.invoice("D-AD9", "INV-AD", vb.ts("03-02"), "KZT", 300_000, "ORD-AD")
    copy = vb.invoice("D-AD1", "INV-AD", vb.ts("03-05"), "KZT", 300_000, "ORD-AD")
    p = vb.payment("D-AD5", "PAY-AD", vb.ts("03-09"), "KZT", 300_000, ("INV-AD",))
    rows = r.invoice(copy) + r.invoice(first) + r.payment(p, unapplied=300_000)
    public = vb.apply(vb.clean_tiny(), vb.Patch("ad", [copy, first], [p], rows))
    assert key4(solve(public)) == {
        ("duplicate_invoice", ("D-AD1", "D-AD9"), 300_000, "KZT"),
        ("payment_allocation", ("D-AD5", "D-AD9"), 300_000, "KZT"),
    }


# ---------------------------------------------------------------------------------- policy
RATE_DATE_CASH = {
    "payment_local_date": 10_050_000,
    "payment_local_date_minus_1": 10_045_000,
    "payment_utc_date": 10_045_000,
}


@pytest.mark.parametrize("rule", sorted(RATE_DATE_CASH))
def test_D08_rate_date_rule_correct_booking_is_clean(rule):
    assert solve(vb.rate_date_case(RATE_DATE_CASH[rule], rule)) == ()


def test_D08_rate_date_rule_other_date_is_fx_issue():
    got = solve(vb.rate_date_case(10_050_000, "payment_utc_date"))
    assert key4(got) == {("fx_conversion", ("D-RD2",), 5_000, "KZT")}
    got = solve(vb.rate_date_case(10_045_000, "payment_local_date"))
    assert key4(got) == {("fx_conversion", ("D-RD2",), 5_000, "KZT")}


WEST_CASH = {
    "payment_local_date": 21_780,
    "payment_local_date_minus_1": 21_760,
    "payment_utc_date": 21_800,
}


@pytest.mark.parametrize("rule", sorted(WEST_CASH))
def test_G06_west_timezone_other_base_and_codes(rule):
    """-04:00, base USD, EUR payment, account codes unlike the tiny fixture."""
    assert solve(vb.west_case(WEST_CASH[rule], rule)) == ()
    for other in sorted(set(WEST_CASH) - {rule}):
        diff = abs(WEST_CASH[other] - WEST_CASH[rule])
        assert key4(solve(vb.west_case(WEST_CASH[other], rule))) == {
            ("fx_conversion", ("WD-2",), diff, "USD")
        }


@pytest.mark.parametrize(
    "rounding, cash, refund_cash",
    [("half_even", 5_045_504, -5_045_504), ("half_up", 5_045_505, -5_045_505)],
)
def test_D08_rounding_mode(rounding, cash, refund_cash):
    assert solve(vb.rounding_case(cash, refund_cash, rounding)) == ()


def test_D08_rounding_off_by_one_is_reported():
    """No grey zone: any non-zero difference is reported, here one minor unit."""
    got = solve(vb.rounding_case(5_045_505, -5_045_504, "half_even"))
    assert key4(got) == {("fx_conversion", ("D-RN2",), 1, "KZT")}
    got = solve(vb.rounding_case(5_045_505, -5_045_504, "half_up"))
    assert key4(got) == {("fx_conversion", ("D-RN4",), 1, "KZT")}


def test_X8_rate_date_utc_at_boundary():
    """The X8 foreign payment (local 04-01, UTC 03-31) under the UTC rule needs the 03-31
    rate: 10000 * 507.50 = 5075000, not the booked 5077500."""
    public = vb.apply(
        vb.clean_tiny_base_only(), vb.x8_boundary_offset(), rate_date="payment_utc_date"
    )
    assert key4(solve(public)) == {("fx_conversion", ("D-B5",), 2_500, "KZT")}


def test_I4_period_boundary_is_local_midnight_end_exclusive():
    """2026-03-31T19:00Z is exactly 2026-04-01 00:00 local: after the period. A posting one
    second earlier (local) is in March, so the document and its posting straddle the end."""
    inv = vb.invoice("D-BX1", "INV-BX1", "2026-03-31T19:00:00+00:00", "KZT", 120_000, "ORD-BX1")
    rows = vb.Rows("RBX-").invoice(inv, "2026-04-01T00:00:00+05:00")
    assert solve(vb.apply(vb.clean_tiny(), vb.Patch("bx", [inv], [], rows))) == ()
    rows = vb.Rows("RBX-").invoice(inv, "2026-03-31T23:59:59+05:00")
    assert key4(solve(vb.apply(vb.clean_tiny(), vb.Patch("bx", [inv], [], rows)))) == {
        ("period_cutoff", ("D-BX1",), 120_000, "KZT")
    }
