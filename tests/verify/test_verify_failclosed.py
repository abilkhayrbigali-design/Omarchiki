"""Fail closed: malformed or contradictory public data raises VerifyError with a fixed code
instead of being skipped (T5: no discrepancy may exist outside the four issue types)."""

from dataclasses import replace

import pytest
import verify_builders as vb

from auditgym.verify import VerifyError, solve
from auditgym.verify import errors as E


def _rows(public, **changes):
    """Replace fields of ledger rows by row_id: _rows(p, R_001={"debit_minor": 5})."""
    by = {k.replace("_", "-"): v for k, v in changes.items()}
    return [replace(r, **by[r.row_id]) if r.row_id in by else r for r in public.ledger]


def _raises(public, code):
    with pytest.raises(VerifyError) as info:
        solve(public)
    assert info.value.code == code, str(info.value)


def clean():
    return vb.clean_tiny()


def test_policy_and_manifest_must_agree():
    p = clean()
    _raises(replace(p, policy={**p.policy, "base_currency": "USD"}), E.POLICY_INVALID)
    _raises(replace(p, manifest={**p.manifest, "timezone": "+06:00"}), E.POLICY_INVALID)
    _raises(
        replace(p, manifest={**p.manifest, "period_end": "2026-05-01T00:00:00+05:00"}),
        E.POLICY_INVALID,
    )
    fx = {**p.policy["fx"], "rate_date": "payment_tomorrow"}
    _raises(replace(p, policy={**p.policy, "fx": fx}), E.POLICY_INVALID)
    accounts = {**p.policy["accounts"], "AR": p.policy["accounts"]["CASH"]}
    _raises(replace(p, policy={**p.policy, "accounts": accounts}), E.POLICY_INVALID)


def test_period_boundary_must_be_local_midnight():
    p = clean()
    period = {**p.policy["period"], "start": "2026-03-01T00:00:00+00:00"}
    manifest = {**p.manifest, "period_start": "2026-03-01T00:00:00+00:00"}
    _raises(replace(p, policy={**p.policy, "period": period}, manifest=manifest), E.POLICY_INVALID)


def test_manifest_counts_must_match():
    p = clean()
    _raises(replace(p, ledger=p.ledger[:-1]), E.COUNTS_MISMATCH)


def test_ids_must_be_unique_across_tables():
    p = clean()
    ledger = list(p.ledger)
    ledger[1] = replace(ledger[1], row_id=ledger[0].row_id)
    _raises(vb.rebuild(p, ledger=ledger), E.ID_NOT_UNIQUE)
    ledger = list(p.ledger)
    ledger[0] = replace(ledger[0], row_id="D-I1")
    _raises(vb.rebuild(p, ledger=ledger), E.ID_NOT_UNIQUE)


def test_row_with_two_sides_or_unknown_account():
    p = clean()
    _raises(vb.rebuild(p, ledger=_rows(p, R_001={"credit_minor": 1})), E.RECORD_INVALID)
    _raises(vb.rebuild(p, ledger=_rows(p, R_001={"account": "9999"})), E.RECORD_INVALID)
    _raises(
        vb.rebuild(p, ledger=_rows(p, R_001={"timestamp": "2026-03-03 10:00"})), E.RECORD_INVALID
    )


def test_invoice_lines_must_add_up():
    p = clean()
    inv = p.invoices[0]
    bad = replace(inv, total_minor=inv.total_minor + 1)
    _raises(vb.rebuild(p, invoices=[bad, *p.invoices[1:]]), E.RECORD_INVALID)


def test_payment_reference_to_unknown_invoice():
    p = clean()
    pays = [replace(x, invoice_refs=("INV-404",)) if x.doc_id == "D-P4" else x for x in p.payments]
    _raises(vb.rebuild(p, payments=pays), E.UNKNOWN_REFERENCE)


def test_payment_reference_in_other_currency():
    p = clean()
    pays = [replace(x, invoice_refs=("INV-1002",)) if x.doc_id == "D-P4" else x for x in p.payments]
    _raises(vb.rebuild(p, payments=pays), E.UNKNOWN_REFERENCE)


def test_missing_fx_rate():
    p = clean()
    _raises(vb.rebuild(p, fx=[f for f in p.fx if f.date != "2026-03-20"]), E.FX_RATE_MISSING)


def test_document_without_posting():
    p = clean()
    drop = {"R-001", "R-002"}
    _raises(
        vb.rebuild(p, ledger=[r for r in p.ledger if r.row_id not in drop]), E.UNPOSTED_DOCUMENT
    )


def test_rows_for_unknown_document():
    p = clean()
    _raises(vb.rebuild(p, ledger=_rows(p, R_013={"payment_id": "PAY-404"})), E.UNKNOWN_REFERENCE)


def test_unbalanced_payment_posting():
    p = clean()
    _raises(vb.rebuild(p, ledger=_rows(p, R_014={"credit_minor": 9_000_000})), E.POSTING_INVALID)


def test_invoice_posting_with_wrong_amount():
    p = clean()
    ledger = _rows(p, R_001={"debit_minor": 9_000_000}, R_002={"credit_minor": 9_000_000})
    _raises(vb.rebuild(p, ledger=ledger), E.UNEXPLAINED_DISCREPANCY)


def test_base_payment_cash_differs_from_document():
    p = clean()
    ledger = _rows(p, R_013={"debit_minor": 9_000_000}, R_014={"credit_minor": 9_000_000})
    _raises(vb.rebuild(p, ledger=ledger), E.UNEXPLAINED_DISCREPANCY)


def test_unapplied_short_without_invoice_shortfall():
    """Overpayment booked fully to the invoice: no referenced invoice is short, UNAPPLIED is."""
    r = vb.Rows("RZ-")
    inv = vb.invoice("D-Z1", "INV-Z1", vb.ts("03-02"), "KZT", 500_000, "ORD-Z1")
    pay = vb.payment("D-Z2", "PAY-Z2", vb.ts("03-03"), "KZT", 700_000, ("INV-Z1",))
    rows = r.invoice(inv) + r.payment(pay, ar={"INV-Z1": 700_000})
    _raises(vb.apply(clean(), vb.Patch("z", [inv], [pay], rows)), E.UNEXPLAINED_DISCREPANCY)


def test_equal_instants_on_one_invoice():
    r = vb.Rows("RZ-")
    inv = vb.invoice("D-Z1", "INV-Z1", vb.ts("03-02"), "KZT", 500_000, "ORD-Z1")
    p1 = vb.payment("D-Z2", "PAY-Z2", vb.ts("03-03"), "KZT", 300_000, ("INV-Z1",))
    # Same instant written with another offset.
    p2 = vb.payment("D-Z3", "PAY-Z3", "2026-03-03T05:00:00+00:00", "KZT", 300_000, ("INV-Z1",))
    rows = r.invoice(inv) + r.payment(p1, ar={"INV-Z1": 300_000})
    rows += r.payment(p2, ar={"INV-Z1": 200_000}, unapplied=100_000)
    _raises(vb.apply(clean(), vb.Patch("z", [inv], [p1, p2], rows)), E.ORDER_AMBIGUOUS)


def test_credit_note_sharing_order_ref_is_not_silently_a_duplicate():
    """I7 says one invoice per order_ref. A credit note on the original order_ref breaks it, and
    the duplicate amount (k-1)*|total| is undefined for different totals."""
    r = vb.Rows("RZ-")
    inv = vb.invoice("D-Z1", "INV-Z1", vb.ts("03-02"), "KZT", 500_000, "ORD-Z1")
    cn = vb.invoice("D-Z2", "INV-Z2", vb.ts("03-04"), "KZT", -100_000, "ORD-Z1", desc="Return")
    rows = r.invoice(inv) + r.invoice(cn)
    _raises(vb.apply(clean(), vb.Patch("z", [inv, cn], [], rows)), E.DUPLICATE_GROUP_INVALID)


def test_cutoff_on_shared_invoice_id_is_ambiguous():
    """Both copies of INV-1003 in March, one posting moved to April: which document?"""
    tiny = vb.tiny_public()
    ledger = _rows(tiny, R_007={"timestamp": vb.ts("04-02")}, R_008={"timestamp": vb.ts("04-02")})
    _raises(vb.rebuild(tiny, ledger=ledger), E.CUTOFF_AMBIGUOUS)


def test_shared_invoice_id_with_equal_issued_at():
    tiny = vb.tiny_public()
    invs = [
        replace(i, issued_at="2026-03-10T09:30:00+05:00") if i.doc_id == "D-I4" else i
        for i in tiny.invoices
    ]
    _raises(vb.rebuild(tiny, invoices=invs), E.DUPLICATE_GROUP_INVALID)


def test_background_row_on_document_account():
    p = clean()
    _raises(
        vb.rebuild(p, ledger=_rows(p, R_026={"account": p.policy["accounts"]["AR"]})),
        E.POSTING_INVALID,
    )


def test_payment_rows_with_two_timestamps():
    p = clean()
    _raises(vb.rebuild(p, ledger=_rows(p, R_013={"timestamp": vb.ts("03-16")})), E.POSTING_INVALID)


def test_evidence_too_large():
    """An allocation shortfall after nine earlier payments needs more than 10 evidence ids."""
    r = vb.Rows("RZ-")
    inv = vb.invoice("D-Z0", "INV-Z0", vb.ts("03-01"), "KZT", 2_000_000, "ORD-Z0")
    pays, rows = [], r.invoice(inv)
    for k in range(1, 10):
        p = vb.payment(
            f"D-Z{k}", f"PAY-Z{k}", vb.ts(f"03-{k + 1:02d}"), "KZT", 100_000, ("INV-Z0",)
        )
        pays.append(p)
        rows += r.payment(p, ar={"INV-Z0": 100_000})
    last = vb.payment("D-Z10", "PAY-Z10", vb.ts("03-20"), "KZT", 100_000, ("INV-Z0",))
    pays.append(last)
    rows += r.payment(last, unapplied=100_000)
    _raises(vb.apply(clean(), vb.Patch("z", [inv], pays, rows)), E.EVIDENCE_TOO_LARGE)
