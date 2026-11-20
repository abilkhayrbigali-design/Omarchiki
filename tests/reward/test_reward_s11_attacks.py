"""S11: guessed ids, unconfirmed evidence, currencies, resubmission, bad input, timeout.

Also the experimental reward (viewed evidence only), which is reported separately (S08).
"""

from __future__ import annotations

import copy
from fractions import Fraction

import pytest
from reward_helpers import (
    CLEAN_TRUTH,
    EID,
    ISSUES,
    TRUTH,
    expected_reward,
    finding,
    grade,
    oracle_finding,
    oracle_findings,
    public_ids,
    submission,
)

from auditgym.contracts.constants import MAX_STEPS
from auditgym.grading import empty_submission, grade_submission

DUP = ISSUES["duplicate_invoice"]
FX = ISSUES["fx_conversion"]
ALLOC = ISSUES["payment_allocation"]
CUT = ISSUES["period_cutoff"]
ALL_EVIDENCE = frozenset(e for issue in TRUTH.issues for s in issue.evidence_sets for e in s)


# ------------------------------------------------------------------------------ guessed ids
@pytest.mark.parametrize(
    "ids",
    [
        ["D-I99", "D-I98"],  # ids that do not exist
        ["ep_TINY000001"],  # the episode id
        ["iss_tiny_dup"],  # a reference issue id
        ["INV-1003"],  # business key instead of doc_id
        ["*"],
        ["D-I%"],
    ],
)
def test_S11_guessed_affected_ids_do_not_match(ids):
    f = oracle_finding(DUP)
    f["affected_ids"] = ids
    result = grade([f])
    assert (result.tp, result.fp) == (0, 1)
    assert result.reward == -1.0


def test_S11_guessed_evidence_ids_do_not_match():
    for issue in TRUTH.issues:
        f = oracle_finding(issue)
        f["evidence_ids"] = ["R-999", "D-X1", "2026-03-19/USD/KZT"] + list(issue.affected_ids)
        f["evidence_ids"] = f["evidence_ids"][:10]
        result = grade([f], type(TRUTH)(EID, (issue,)))
        if set(issue.affected_ids) in [set(s) for s in issue.evidence_sets]:
            # duplicate_invoice: the copies themselves are the whole evidence set (DOMAIN 6).
            assert result.tp == 1
        else:
            assert result.tp == 0, issue.type


def test_S11_wrong_fx_rate_row_is_not_evidence():
    # The rate actually used (03-19) instead of the correct one (03-20).
    f = finding("fx_conversion", ["D-P2"], 30_000, "KZT", ["D-P2", "R-017", "2026-03-19/USD/KZT"])
    assert grade([f]).tp == 0


def test_S11_guessing_evidence_by_listing_ten_random_public_ids_fails():
    ids = public_ids()
    for start in range(0, len(ids) - 10, 7):
        window = ids[start : start + 10]
        for issue in (FX, ALLOC, CUT):
            if any(set(s) <= set(window) for s in issue.evidence_sets):
                continue
            f = oracle_finding(issue)
            f["evidence_ids"] = window
            assert grade([f]).tp == 0


# ------------------------------------------------------------------------------ evidence
def test_S11_unconfirmed_evidence_cannot_earn_a_true_positive():
    no_evidence = [{**f, "evidence_ids": []} for f in oracle_findings()]
    result = grade(no_evidence, steps=5)
    assert (result.tp, result.fp, result.fn) == (0, 4, 4)
    assert result.reward == -1.0
    # Right fields without evidence are no better than silence.
    assert result.reward <= grade([], steps=5).reward


def test_S11_partial_evidence_for_every_issue_scores_like_nothing():
    partial = []
    for f in oracle_findings():
        partial.append({**f, "evidence_ids": f["evidence_ids"][:-1]})
    result = grade(partial)
    assert result.tp == 0 and result.reward == -1.0


# ------------------------------------------------------------------------------ currencies
def test_S11_payment_currency_instead_of_base_currency_fails_for_fx():
    # The fx_conversion discrepancy is in the base currency (KZT), not the payment's USD.
    f = oracle_finding(FX)
    f["currency"] = "USD"
    f["claimed_amount_minor"] = 60  # 30000 KZT is about 60 US cents at 504.75
    assert grade([f]).tp == 0


@pytest.mark.parametrize("currency", ["USD", "EUR", "JPY", "XXX"])
def test_S11_wrong_valid_currency_is_a_false_positive(currency):
    f = oracle_finding(CUT)
    f["currency"] = currency
    result = grade([f])
    assert result.valid and (result.tp, result.fp) == (0, 1)


@pytest.mark.parametrize("currency", ["kzt", "Kzt", "KZT ", " KZT", "KZT\n", "KZ", "₸", "398"])
def test_S11_malformed_currency_is_invalid(currency):
    f = oracle_finding(CUT)
    f["currency"] = currency
    result = grade([f] + oracle_findings())
    assert result.valid is False and result.reward == -1.0


def test_S11_same_amount_in_other_currency_is_not_converted():
    # 3_000_000 minor units of USD are not 3_000_000 minor units of KZT.
    f = oracle_finding(CUT)
    f["currency"] = "USD"
    assert grade([f]).tp == 0


# ------------------------------------------------------------------------------ amounts
@pytest.mark.parametrize(
    ("issue_type", "delta", "tp"),
    [
        ("fx_conversion", 1, 1),
        ("fx_conversion", -1, 1),
        ("fx_conversion", 2, 0),
        ("fx_conversion", -2, 0),
        ("period_cutoff", 1, 0),
        ("period_cutoff", -1, 0),
        ("duplicate_invoice", 1, 0),
        ("payment_allocation", -1, 0),
    ],
)
def test_S11_amount_just_inside_or_outside_tolerance(issue_type, delta, tp):
    issue = ISSUES[issue_type]
    f = oracle_finding(issue)
    f["claimed_amount_minor"] += delta
    assert grade([f]).tp == tp


def test_S11_document_total_instead_of_discrepancy_fails_for_duplicates_and_allocation():
    # duplicate_invoice expects (k-1) x total = 7_500_000 here; with k = 2 that equals the
    # total, so use payment_allocation: the payment amount is not the under-applied amount.
    f = oracle_finding(ALLOC)
    f["claimed_amount_minor"] = 2_500_000
    assert grade([f]).tp == 0


# ------------------------------------------------------------------------------ resubmission
def test_S11_grader_is_pure_same_input_same_result():
    payload = submission(oracle_findings()[:3] + [finding("period_cutoff", ["D-I1"], 1, "KZT", [])])
    snapshot = copy.deepcopy(payload)
    results = [
        grade_submission(
            payload, TRUTH, episode_id=EID, inspection_steps=12, viewed_ids=ALL_EVIDENCE
        )
        for _ in range(5)
    ]
    assert all(r == results[0] for r in results)
    assert payload == snapshot  # the payload is not mutated
    assert TRUTH == type(TRUTH).from_dict(TRUTH.to_dict())  # truth unchanged


def test_S11_resubmitting_the_same_answer_inside_one_payload_brings_nothing():
    first = grade(oracle_findings()[:2], steps=10)
    twice = grade(oracle_findings()[:2] * 2, steps=10)
    assert twice.tp == first.tp == 2
    assert twice.reward < first.reward


def test_S11_order_of_resubmitted_findings_does_not_matter():
    a = grade(oracle_findings() + oracle_findings()[:1], steps=3)
    b = grade(oracle_findings()[:1] + list(reversed(oracle_findings())), steps=3)
    assert (a.tp, a.fp, a.fn, a.reward) == (b.tp, b.fp, b.fn, b.reward)


# ------------------------------------------------------------------------------ bad input
@pytest.mark.parametrize(
    "payload",
    [
        None,
        "",
        b"{}",
        0,
        1.5,
        [],
        {},
        {"episode_id": EID},
        {"findings": []},
        {"episode_id": EID, "findings": None},
        {"episode_id": EID, "findings": [None]},
        {"episode_id": EID, "findings": [{}]},
        {"episode_id": EID, "findings": [[]]},
        {"episode_id": EID, "findings": "[]"},
        {"episode_id": EID, "findings": [], "extra": None},
        {"episode_id": None, "findings": []},
        {"episode_id": 12345, "findings": []},
        {"episode_id": EID, "findings": [oracle_findings()[0]] * 31},
        {"episode_id": EID, "findings": [{**oracle_findings()[0], "type": None}]},
    ],
    ids=range(19),
)
def test_S11_malformed_input_ends_with_minus_one_and_never_raises(payload):
    for steps in (0, 50):
        result = grade_submission(payload, TRUTH, episode_id=EID, inspection_steps=steps)
        assert result.valid is False and result.reward == -1.0
        result = grade_submission(payload, CLEAN_TRUTH, episode_id=EID, inspection_steps=steps)
        assert result.valid is False and result.reward == -1.0


def test_S11_hostile_python_objects_are_rejected_without_being_called():
    class Evil(str):
        def __eq__(self, other):
            raise AssertionError("parser must not compare agent objects")

        __hash__ = str.__hash__

    class EvilDict(dict):
        def __getitem__(self, key):
            raise AssertionError("parser must not index agent subclasses")

    payloads = [
        EvilDict(submission(oracle_findings())),
        {"episode_id": Evil(EID), "findings": []},
        {"episode_id": EID, "findings": [{**oracle_findings()[0], "type": Evil("period_cutoff")}]},
    ]
    for payload in payloads:
        result = grade_submission(payload, TRUTH, episode_id=EID, inspection_steps=0)
        assert result.valid is False and result.reward == -1.0


def test_S11_huge_values_are_handled():
    f = oracle_finding(CUT)
    f["claimed_amount_minor"] = 10**4000
    result = grade([f])
    assert result.valid and result.tp == 0 and result.reward == -1.0


# ------------------------------------------------------------------------------ timeout
def test_S11_timeout_is_graded_as_an_empty_submission():
    # E10: steps or time run out -> truncated, the final answer is empty, all steps counted.
    on_errors = grade_submission(
        empty_submission(EID), TRUTH, episode_id=EID, inspection_steps=MAX_STEPS
    )
    assert on_errors.valid and (on_errors.tp, on_errors.fp, on_errors.fn) == (0, 0, 4)
    assert on_errors.reward == -1.0
    on_clean = grade_submission(
        empty_submission(EID), CLEAN_TRUTH, episode_id=EID, inspection_steps=MAX_STEPS
    )
    assert on_clean.valid and on_clean.f1 == 1.0
    assert on_clean.reward == float(1 - Fraction(MAX_STEPS, 500))  # 0.8
    # Timing out is never better than submitting the same empty answer earlier.
    early = grade_submission(
        empty_submission(EID), CLEAN_TRUTH, episode_id=EID, inspection_steps=10
    )
    assert early.reward > on_clean.reward


def test_S11_timeout_never_beats_a_correct_answer():
    for steps in range(MAX_STEPS):
        assert grade(oracle_findings(), steps=steps).reward > grade([], steps=MAX_STEPS).reward


# ------------------------------------------------------------------------------ experimental
def test_S08_experimental_reward_is_none_without_viewed_ids():
    assert grade(oracle_findings()).experimental_reward is None


def test_S08_experimental_reward_equals_shared_when_all_evidence_was_viewed():
    result = grade(oracle_findings(), steps=20, viewed=ALL_EVIDENCE)
    assert result.experimental_reward == result.reward == 0.96


def test_S11_experimental_reward_rejects_evidence_the_agent_never_requested():
    shared_only = grade(oracle_findings(), steps=0)
    blind = grade(oracle_findings(), steps=0, viewed=frozenset())
    assert blind.reward == shared_only.reward == 1.0  # the shared reward never changes
    assert blind.experimental_reward == -1.0
    assert (blind.tp, blind.fp, blind.fn) == (4, 0, 0)


def test_S11_experimental_reward_counts_only_issues_with_a_viewed_evidence_set():
    viewed = frozenset(CUT.evidence_sets[1]) | frozenset(FX.evidence_sets[0])
    result = grade(oracle_findings(), steps=10, viewed=viewed)
    # cut-off: its finding lists set 0 only, set 1 was viewed -> not counted.
    # fx: set 0 listed and viewed -> counted. TP_exp = 1, FP_exp = 3, FN_exp = 3.
    assert result.experimental_reward == expected_reward(1, 3, 3, 10)
    assert result.reward == expected_reward(4, 0, 0, 10)


def test_S11_experimental_reward_accepts_any_viewed_complete_set():
    f = oracle_finding(CUT)
    f["evidence_ids"] = sorted({e for s in CUT.evidence_sets for e in s})  # both sets listed
    viewed = frozenset(CUT.evidence_sets[1])  # only the second one viewed
    result = grade([f], type(TRUTH)(EID, (CUT,)), viewed=viewed)
    assert result.experimental_reward == 1.0


def test_S11_experimental_reward_needs_the_whole_set_viewed():
    for issue in TRUTH.issues:
        for evidence_set in issue.evidence_sets:
            for missing in evidence_set:
                viewed = frozenset(evidence_set) - {missing}
                result = grade(
                    [oracle_finding(issue, issue.evidence_sets.index(evidence_set))],
                    type(TRUTH)(EID, (issue,)),
                    viewed=viewed,
                )
                assert result.reward == 1.0 and result.experimental_reward == -1.0


def test_S08_experimental_reward_accepts_any_iterable_of_ids():
    for viewed in (set(ALL_EVIDENCE), sorted(ALL_EVIDENCE), tuple(ALL_EVIDENCE)):
        assert grade(oracle_findings(), viewed=viewed).experimental_reward == 1.0
