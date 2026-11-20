"""The hand-built fixture is internally consistent and its truth matches hand computation."""

from collections import defaultdict

from auditgym.contracts.io import fingerprint, read_episode
from auditgym.util.money import convert_minor


def test_every_row_has_exactly_one_side(tiny_episode):
    for row in tiny_episode.public.ledger:
        assert (row.debit_minor > 0) != (row.credit_minor > 0)
        assert row.debit_minor >= 0 and row.credit_minor >= 0


def test_ledger_balances_per_currency(tiny_episode):
    total = defaultdict(int)
    for row in tiny_episode.public.ledger:
        total[row.currency] += row.debit_minor - row.credit_minor
    assert all(v == 0 for v in total.values())


def test_invoice_totals_match_lines(tiny_episode):
    for inv in tiny_episode.public.invoices:
        assert inv.total_minor == sum(li.amount_minor for li in inv.line_items)
        for li in inv.line_items:
            assert li.amount_minor == li.quantity * li.unit_price_minor


def test_fx_issue_amount(tiny_episode):
    rates = {f.rate_id: f.rate for f in tiny_episode.public.fx}
    right = convert_minor(120_000, "USD", "KZT", rates["2026-03-20/USD/KZT"], "half_even")
    wrong = convert_minor(120_000, "USD", "KZT", rates["2026-03-19/USD/KZT"], "half_even")
    issue = next(i for i in tiny_episode.private.truth.issues if i.type == "fx_conversion")
    assert right - wrong == issue.expected_amount_minor == 30_000


def test_evidence_ids_exist(tiny_episode):
    pub = tiny_episode.public
    ids = (
        {r.row_id for r in pub.ledger}
        | {d.doc_id for d in pub.invoices}
        | {d.doc_id for d in pub.payments}
        | {f.rate_id for f in pub.fx}
    )
    for issue in tiny_episode.private.truth.issues:
        assert set(issue.affected_ids) <= ids
        for ev in issue.evidence_sets:
            assert set(ev) <= ids
            assert len(ev) <= 10


def test_manifest_counts(tiny_episode):
    m = tiny_episode.public.manifest
    assert m["counts"]["ledger"] == len(tiny_episode.public.ledger) == 27
    assert m["counts"]["documents"] == 11


def test_disk_roundtrip_is_exact(tiny_episode, tiny_episode_dir):
    again = read_episode(tiny_episode_dir)
    assert again == tiny_episode
    assert fingerprint(again) == fingerprint(tiny_episode)


def test_public_files_hold_no_truth(tiny_episode_dir):
    public_bytes = b"".join(
        p.read_bytes() for p in sorted(tiny_episode_dir.iterdir()) if p.is_file()
    )
    for marker in (b"iss_", b"expected_amount", b"evidence_sets", b"provenance"):
        assert marker not in public_bytes
