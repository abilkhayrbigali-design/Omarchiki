"""Truth, invariants and evidence of generated episodes (K2: G01-G05, D01, D10, T1-T5)."""

from __future__ import annotations

from collections import Counter

import pytest
from genkit import all_distractors, all_public_ids, gen, only, shipped_config

from auditgym.books.invariants import check, id_problems
from auditgym.contracts.config import normalize_config
from auditgym.contracts.constants import DISTRACTOR_KINDS, ISSUE_TYPES, MAX_EVIDENCE_IDS
from auditgym.episode.selfcheck import expected_violations, found_violations

EXPECTED_CODES = {
    "duplicate_invoice": {"I7"},  # plus I3 when the copies share the invoice_id
    "payment_allocation": {"I5"},
    "fx_conversion": {"I6"},
    "period_cutoff": {"I4"},
}


@pytest.mark.parametrize("name", ["train", "dev", "smoke", "interacting", "clean"])
def test_K2_clean_books_satisfy_all_invariants(name):
    cfg = shipped_config(name)
    for seed in range(25):
        episode, clean = gen(seed, cfg)
        assert check(clean) == [], (name, seed)
        assert id_problems(clean) == []
        assert id_problems(episode.public) == []


@pytest.mark.parametrize("issue_type", ISSUE_TYPES)
def test_G01_each_issue_type_alone(issue_type):
    cfg = only(issue_type)
    for seed in range(12):
        episode, clean = gen(seed, cfg)
        issues = episode.private.truth.issues
        assert [i.type for i in issues] == [issue_type]
        assert check(clean) == []
        violations = check(episode.public)
        codes = {v.code for v in violations}
        if issue_type == "duplicate_invoice":
            assert codes in ({"I7"}, {"I3", "I7"})
        else:
            assert codes == EXPECTED_CODES[issue_type]
        assert found_violations(episode.public) == expected_violations(
            episode.public, episode.private.truth
        )
        for v in violations:
            assert v.ids == issues[0].affected_ids
            assert v.amount_minor == issues[0].expected_amount_minor
            assert v.currency == issues[0].currency


@pytest.mark.parametrize("issue_type", ISSUE_TYPES)
def test_G01_issue_types_are_independently_configurable(issue_type):
    raw = {
        "config_id": f"two-{issue_type}",
        "errors": {t: (2 if t == issue_type else 0) for t in ISSUE_TYPES},
        "interactions": {"fx_partial_at_boundary": 0, "duplicate_misallocated": 0},
    }
    cfg = normalize_config(raw)
    for seed in range(6):
        types = Counter(i.type for i in gen(seed, cfg)[0].private.truth.issues)
        assert types == Counter({issue_type: 2})


def test_G01_all_variants_occur():
    cfg = shipped_config("train")
    variants = Counter()
    for seed in range(80):
        for inj in gen(seed, cfg)[0].private.provenance["injections"]:
            variants[(inj["injector"], inj["variant"])] += 1
            if inj["injector"] == "period_cutoff":
                variants[("period_cutoff", inj["details"]["boundary"])] += 1
    expected = {
        ("duplicate_invoice", "same_invoice_id"),
        ("duplicate_invoice", "new_invoice_id"),
        ("payment_allocation", "none"),
        ("payment_allocation", "partial"),
        ("payment_allocation", "other_invoice"),
        ("fx_conversion", "wrong_date"),
        ("fx_conversion", "wrong_pair"),
        ("period_cutoff", "invoice_posted_late"),
        ("period_cutoff", "payment_posted_late"),
        ("period_cutoff", "period_start"),
        ("period_cutoff", "period_end"),
    }
    assert expected <= set(variants), sorted(set(expected) - set(variants))


def test_G01_duplicate_with_three_copies_occurs():
    cfg = only("duplicate_invoice")
    sizes = {len(gen(seed, cfg)[0].private.truth.issues[0].affected_ids) for seed in range(40)}
    assert sizes == {2, 3}


def test_G03_clean_config_gives_empty_truth_and_no_violations():
    cfg = shipped_config("clean")
    for seed in range(30):
        episode, clean = gen(seed, cfg)
        assert episode.private.truth.issues == ()
        assert check(episode.public) == []
        assert episode.private.provenance["changes"] == []
        assert episode.public.ledger == clean.ledger


def test_G03_all_distractors_present_and_clean():
    cfg = all_distractors()
    for seed in range(10):
        episode, clean = gen(seed, cfg)
        prov = episode.private.provenance
        kinds = Counter(d["kind"] for d in prov["distractors"])
        assert set(kinds) == set(DISTRACTOR_KINDS), (seed, kinds)
        distractor_ids = {i for d in prov["distractors"] for i in d["ids"]}
        for issue in episode.private.truth.issues:
            assert not set(issue.affected_ids) & distractor_ids
        assert check(clean) == []


def test_G03_distractor_shapes():
    cfg = all_distractors("all-distractors-shapes")
    for seed in range(10):
        pub = gen(seed, cfg)[0].public
        tz = pub.manifest["timezone"]
        assert any(i.total_minor < 0 for i in pub.invoices)  # X4 credit note
        assert any(p.amount_minor < 0 for p in pub.payments)  # X4 refund
        assert any(p.invoice_refs is None for p in pub.payments)  # X7
        assert any(p.invoice_refs and len(p.invoice_refs) > 1 for p in pub.payments)  # X2
        assert any(li.service_period for i in pub.invoices for li in i.line_items)  # X5
        offsets = {
            ts[-6:]
            for ts in [i.issued_at for i in pub.invoices] + [p.timestamp for p in pub.payments]
        }
        assert offsets - {tz}  # X8 writes some documents in another offset
        assert all(r.timestamp.endswith(tz) for r in pub.ledger)


def test_G02_interacting_config_produces_both_scenarios():
    cfg = shipped_config("interacting")
    for seed in range(20):
        episode, _ = gen(seed, cfg)
        prov = episode.private.provenance
        assert sorted(s["scenario"] for s in prov["scenarios"]) == [
            "duplicate_misallocated",
            "fx_partial_at_boundary",
        ]
        by_id = {i.issue_id: i for i in episode.private.truth.issues}
        for sc in prov["scenarios"]:
            issues = [by_id[i] for i in sc["issue_ids"]]
            if sc["scenario"] == "fx_partial_at_boundary":
                assert sorted(i.type for i in issues) == [
                    "fx_conversion",
                    "payment_allocation",
                    "period_cutoff",
                ]
                payment = {i.affected_ids for i in issues if i.type != "payment_allocation"}
                assert len(payment) == 1
                (pay_doc,) = payment.pop()
                alloc = next(i for i in issues if i.type == "payment_allocation")
                assert pay_doc in alloc.affected_ids
            else:
                assert sorted(i.type for i in issues) == ["duplicate_invoice", "payment_allocation"]
                dup = next(i for i in issues if i.type == "duplicate_invoice")
                alloc = next(i for i in issues if i.type == "payment_allocation")
                assert len(set(dup.affected_ids) & set(alloc.affected_ids)) == 1


@pytest.mark.parametrize("scenario", ["fx_partial_at_boundary", "duplicate_misallocated"])
def test_G02_each_scenario_alone(scenario):
    cfg = only(scenario=scenario)
    for seed in range(10):
        episode, clean = gen(seed, cfg)
        assert check(clean) == []
        assert found_violations(episode.public) == expected_violations(
            episode.public, episode.private.truth
        )
        assert [s["scenario"] for s in episode.private.provenance["scenarios"]] == [scenario]


@pytest.mark.parametrize("name", ["train", "dev", "interacting", "smoke"])
def test_K2_violations_match_truth_exactly(name):
    cfg = shipped_config(name)
    for seed in range(30):
        episode, _ = gen(seed, cfg)
        assert found_violations(episode.public) == expected_violations(
            episode.public, episode.private.truth
        ), (name, seed)


@pytest.mark.parametrize("name", ["train", "dev", "interacting"])
def test_D10_evidence_ids_exist_and_sets_are_small(name):
    cfg = shipped_config(name)
    for seed in range(30):
        episode, _ = gen(seed, cfg)
        ids = all_public_ids(episode.public)
        for issue in episode.private.truth.issues:
            assert set(issue.affected_ids) <= ids
            assert issue.evidence_sets
            for ev in issue.evidence_sets:
                assert 1 <= len(ev) <= MAX_EVIDENCE_IDS
                assert set(ev) <= ids
            # Each evidence set names the affected documents (DOMAIN evidence table).
            for ev in issue.evidence_sets:
                assert set(issue.affected_ids) <= set(ev)


def test_D10_evidence_sets_follow_domain_table():
    cfg = shipped_config("dev")
    for seed in range(20):
        episode, _ = gen(seed, cfg)
        pub = episode.public
        rows = {r.row_id: r for r in pub.ledger}
        pays = {p.doc_id: p for p in pub.payments}
        role = {code: r for r, code in pub.policy["accounts"].items()}
        base = pub.manifest["base_currency"]
        for issue in episode.private.truth.issues:
            sets = [set(ev) for ev in issue.evidence_sets]
            if issue.type == "duplicate_invoice":
                assert sets == [set(issue.affected_ids)]
            elif issue.type == "fx_conversion":
                (pay_doc,) = issue.affected_ids
                pay = pays[pay_doc]
                assert len(sets) == 2
                roles = sorted(role[rows[x].account] for ev in sets for x in ev if x in rows)
                assert roles == ["CASH", "FX_CLEARING"]
                for ev in sets:
                    fx_ids = [x for x in ev if "/" in x]
                    assert len(fx_ids) == 1 and fx_ids[0].split("/")[1:] == [pay.currency, base]
            elif issue.type == "period_cutoff":
                (doc,) = issue.affected_ids
                assert all(len(ev) == 2 and doc in ev for ev in sets)
            else:
                pay_doc = next(x for x in issue.affected_ids if x in pays)
                pay = pays[pay_doc]
                (ev,) = sets
                entry = {
                    r.row_id
                    for r in pub.ledger
                    if r.payment_id == pay.payment_id and r.currency == pay.currency
                }
                assert entry <= ev


def test_K2_every_discrepancy_at_least_100_minor_units():
    for name in ("train", "dev", "interacting", "smoke"):
        cfg = shipped_config(name)
        for seed in range(25):
            for issue in gen(seed, cfg)[0].private.truth.issues:
                assert issue.expected_amount_minor >= 100


def test_T2_type_and_affected_ids_unique():
    for name in ("train", "dev", "interacting"):
        cfg = shipped_config(name)
        for seed in range(25):
            issues = gen(seed, cfg)[0].private.truth.issues
            assert len({i.key() for i in issues}) == len(issues)


def test_D01_max_issues_caps_the_total():
    raw = {
        "config_id": "capped",
        "max_issues": 5,
        "errors": {t: [0, 3] for t in ISSUE_TYPES},
        "interactions": {"fx_partial_at_boundary": [0, 1], "duplicate_misallocated": [0, 0]},
    }
    cfg = normalize_config(raw)
    totals = Counter()
    for seed in range(30):
        episode, _ = gen(seed, cfg)
        n = len(episode.private.truth.issues)
        assert n <= 5
        totals[n] += 1
        plan = episode.private.provenance["plan"]
        assert plan["total_issues"] == n
    assert totals[5] > 0  # the cap actually binds


def test_T1_amounts_recomputed_from_documents():
    """Truth amounts follow the DOMAIN definitions, recomputed here from the public documents."""
    cfg = shipped_config("train")
    for seed in range(20):
        episode, clean = gen(seed, cfg)
        pub = episode.public
        invs = {i.doc_id: i for i in pub.invoices}
        pays = {p.doc_id: p for p in pub.payments}
        for issue in episode.private.truth.issues:
            if issue.type == "duplicate_invoice":
                docs = [invs[d] for d in issue.affected_ids]
                assert issue.expected_amount_minor == (len(docs) - 1) * abs(docs[0].total_minor)
                assert issue.currency == docs[0].currency
            elif issue.type == "period_cutoff":
                (d,) = issue.affected_ids
                doc = invs.get(d) or pays[d]
                amount = doc.total_minor if d in invs else doc.amount_minor
                assert issue.expected_amount_minor == abs(amount)
                assert issue.currency == doc.currency
            elif issue.type == "fx_conversion":
                assert issue.currency == pub.manifest["base_currency"]
                assert (
                    issue.amount_tolerance_minor == pub.policy["tolerances"]["fx_conversion_minor"]
                )
            else:
                pay = next(pays[d] for d in issue.affected_ids if d in pays)
                assert issue.currency == pay.currency
