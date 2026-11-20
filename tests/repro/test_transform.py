"""Paired-check transform: renaming and permutation keep the episode's meaning (K4, V02)."""

from collections import Counter

from auditgym.contracts.io import fingerprint
from auditgym.harness.transform import issue_key, map_issue_key, transform_episode


def _ids(public):
    return (
        {r.row_id for r in public.ledger}
        | {d.doc_id for d in public.invoices}
        | {d.doc_id for d in public.payments}
    )


def test_K4_transform_is_deterministic(tiny_episode):
    a = transform_episode(tiny_episode, "salt-1").episode
    b = transform_episode(tiny_episode, "salt-1").episode
    assert fingerprint(a) == fingerprint(b)
    c = transform_episode(tiny_episode, "salt-2").episode
    assert fingerprint(a) != fingerprint(c)


def test_K4_every_opaque_id_renamed(tiny_episode):
    t = transform_episode(tiny_episode, "s")
    old, new = _ids(tiny_episode.public), _ids(t.episode.public)
    assert old.isdisjoint(new)
    assert len(new) == len(old)
    assert t.episode.public.episode_id != tiny_episode.public.episode_id
    assert t.episode.private.truth.episode_id == t.episode.public.episode_id


def test_K4_truth_maps_through_renaming(tiny_episode):
    t = transform_episode(tiny_episode, "s")
    expected = sorted(map_issue_key(i, t) for i in tiny_episode.private.truth.issues)
    got = sorted(issue_key(i) for i in t.episode.private.truth.issues)
    assert expected == got
    ids = _ids(t.episode.public) | {f.rate_id for f in t.episode.public.fx}
    for issue in t.episode.private.truth.issues:
        for ev in issue.evidence_sets:
            assert set(ev) <= ids


def test_K4_references_stay_consistent(tiny_episode):
    pub = transform_episode(tiny_episode, "s").episode.public
    invoice_ids = {i.invoice_id for i in pub.invoices}
    payment_ids = {p.payment_id for p in pub.payments}
    for pay in pub.payments:
        assert set(pay.invoice_refs or ()) <= invoice_ids
    for row in pub.ledger:
        assert row.invoice_id is None or row.invoice_id in invoice_ids
        assert row.payment_id is None or row.payment_id in payment_ids


def test_K4_multisets_of_values_unchanged(tiny_episode):
    pub0, pub1 = tiny_episode.public, transform_episode(tiny_episode, "s").episode.public

    def money(pub):
        return Counter(
            (r.account, r.debit_minor, r.credit_minor, r.currency, r.timestamp) for r in pub.ledger
        )

    assert money(pub0) == money(pub1)
    assert Counter(f.to_dict()["rate"] for f in pub0.fx) == Counter(
        f.to_dict()["rate"] for f in pub1.fx
    )


def test_K4_permutation_only_keeps_ids(tiny_episode):
    t = transform_episode(tiny_episode, "s", rename=False)
    assert _ids(t.episode.public) == _ids(tiny_episode.public)
    assert t.episode.public.ledger != tiny_episode.public.ledger
