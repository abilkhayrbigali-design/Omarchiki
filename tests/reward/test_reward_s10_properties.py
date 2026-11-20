"""S10: the five properties the case asks the reward tests to show, with numbers.

All episodes are the tiny fixture (4 reference issues) or its clean counterpart (empty truth).
`steps` is the number of inspection steps before submit.
"""

from __future__ import annotations

from fractions import Fraction

import pytest
from reward_helpers import (
    CLEAN_TRUTH,
    ISSUES,
    TINY,
    expected_reward,
    finding,
    grade,
    oracle_findings,
    public_ids,
    rows_of_document,
)

from auditgym.contracts.constants import ISSUE_TYPES, MAX_FINDINGS, MAX_STEPS

TYPICAL_STEPS = (0, 10, 30, 60, 99)

# False accusations aimed at the fixture's correct distractors (docs/DOMAIN.md, section 8).
FALSE_ACCUSATIONS = [
    # X7: payment without references, correctly booked to UNAPPLIED.
    finding("payment_allocation", ["D-P5"], 500_000, "KZT", ["D-P5", "R-024", "R-025"]),
    # X3: legitimate partial payment, INV-1003 stays open.
    finding("payment_allocation", ["D-P4", "D-I3"], 3_500_000, "KZT", ["D-P4", "D-I3", "R-023"]),
    # X10: lagged posting inside the same period.
    finding("period_cutoff", ["D-I6"], 2_500_000, "KZT", ["D-I6", "R-009"]),
    # X8: payment timestamp written in UTC, converted correctly.
    finding("fx_conversion", ["D-P1"], 1, "KZT", ["D-P1", "R-013"]),
    # X6-like: same entity, recurring amount.
    finding("duplicate_invoice", ["D-I1", "D-I5"], 3_000_000, "KZT", ["D-I1", "D-I5"]),
]


# ------------------------------------------------------------------------------ 1
@pytest.mark.parametrize("steps", TYPICAL_STEPS)
def test_S10_full_correct_answer_beats_silence_on_books_with_errors(steps):
    full = grade(oracle_findings(), steps=steps)
    silence = grade([], steps=steps)
    assert (full.tp, full.fp, full.fn, full.f1) == (4, 0, 0, 1.0)
    assert (silence.tp, silence.fp, silence.fn, silence.f1) == (0, 0, 4, 0.0)
    assert full.reward == float(1 - Fraction(steps, 500))  # 1.0 ... 0.802
    assert silence.reward == -1.0
    assert full.reward - silence.reward >= 1.8


def test_S10_full_answer_with_minimum_steps_scores_the_maximum():
    assert grade(oracle_findings(), steps=0).reward == 1.0


def test_S10_any_correct_part_beats_silence_on_books_with_errors():
    found = oracle_findings()
    rewards = [grade(found[:k], steps=20).reward for k in range(5)]
    # 0, 1, 2, 3, 4 correct findings: -1, -0.24, 0.293.., 0.674.., 0.96
    assert rewards == sorted(rewards) and len(set(rewards)) == 5
    assert rewards[0] == -1.0 and rewards[4] == 0.96


# ------------------------------------------------------------------------------ 2
@pytest.mark.parametrize("steps", TYPICAL_STEPS)
def test_S10_silence_beats_false_accusations_on_clean_books(steps):
    silence = grade([], CLEAN_TRUTH, steps=steps)
    assert silence.f1 == 1.0
    assert silence.reward == float(1 - Fraction(steps, 500))
    for k in range(1, len(FALSE_ACCUSATIONS) + 1):
        accusing = grade(FALSE_ACCUSATIONS[:k], CLEAN_TRUTH, steps=steps)
        assert (accusing.tp, accusing.fp, accusing.f1) == (0, k, 0.0)
        assert accusing.reward == -1.0
        assert silence.reward - accusing.reward >= 1.8


def test_S10_even_true_findings_of_other_books_are_false_on_clean_books():
    accusing = grade(oracle_findings(), CLEAN_TRUTH)
    assert accusing.reward == -1.0 < grade([], CLEAN_TRUTH).reward == 1.0


def test_S10_false_accusations_also_cost_on_books_with_errors():
    honest = grade(oracle_findings(), steps=20)
    padded = grade(oracle_findings() + FALSE_ACCUSATIONS, steps=20)
    assert (padded.tp, padded.fp) == (4, 5)
    assert padded.reward == expected_reward(4, 5, 0, 20) < honest.reward


# ------------------------------------------------------------------------------ 3
@pytest.mark.parametrize("copies", [2, 3, 7])
def test_S10_duplicated_findings_bring_no_gain(copies):
    once = grade(oracle_findings(), steps=10)
    repeated = grade(oracle_findings() * copies, steps=10)
    assert repeated.tp == once.tp == 4
    assert repeated.fp == 4 * (copies - 1)
    assert repeated.reward == expected_reward(4, 4 * (copies - 1), 0, 10) < once.reward


def test_S10_repeating_one_correct_finding_is_worse_than_stating_it_once():
    one = ISSUES["period_cutoff"]
    f = finding(
        one.type, one.affected_ids, one.expected_amount_minor, one.currency, one.evidence_sets[0]
    )
    rewards = [grade([f] * n, steps=10).reward for n in (1, 2, 5, 30)]
    assert [grade([f] * n).tp for n in (1, 2, 5, 30)] == [1, 1, 1, 1]
    assert rewards == sorted(rewards, reverse=True) and len(set(rewards)) == 4
    assert rewards[0] == expected_reward(1, 0, 3, 10)  # F1 = 2/5 -> -0.22


def test_S10_restating_a_finding_with_other_evidence_or_amount_is_still_a_duplicate():
    fx = ISSUES["fx_conversion"]
    variants = [
        finding(fx.type, fx.affected_ids, 30_000, "KZT", fx.evidence_sets[0]),
        finding(fx.type, fx.affected_ids, 30_000, "KZT", fx.evidence_sets[1]),
        finding(fx.type, fx.affected_ids, 30_001, "KZT", fx.evidence_sets[0]),
        finding(fx.type, fx.affected_ids, 29_999, "KZT", fx.evidence_sets[1]),
    ]
    result = grade(variants)
    assert (result.tp, result.fp, result.fn) == (1, 3, 3)


# ------------------------------------------------------------------------------ 4
def _flag_all_rows_packed() -> list[dict]:
    """Every ledger row and document of the fixture under every issue type, 10 ids a finding."""
    pub = TINY.public
    ids = [r.row_id for r in pub.ledger] + [d.doc_id for d in pub.invoices + pub.payments]
    chunks = [ids[k : k + 10] for k in range(0, len(ids), 10)]
    return [finding(t, chunk, 0, "KZT", chunk) for t in ISSUE_TYPES for chunk in chunks]


def _flag_all_ids_packed() -> list[dict]:
    """Same with FX rows included: more chunks than findings allowed."""
    ids = public_ids()
    chunks = [ids[k : k + 10] for k in range(0, len(ids), 10)]
    return [finding(t, chunk, 0, "KZT", chunk) for t in ISSUE_TYPES for chunk in chunks]


def _flag_every_document_every_type() -> list[dict]:
    """One finding per (type, document) with the document's own amount and rows as evidence.

    Strongest guesses first (period_cutoff, where the amount is the document total), so the
    30-finding cut keeps them. 44 combinations exist; all of them at once is over the limit.
    """
    pub = TINY.public
    docs = [(d.doc_id, abs(d.total_minor), d.currency) for d in pub.invoices] + [
        (p.doc_id, abs(p.amount_minor), p.currency) for p in pub.payments
    ]
    order = ("period_cutoff", "payment_allocation", "fx_conversion", "duplicate_invoice")
    return [
        finding(t, [doc], amount, cur, ([doc] + rows_of_document(doc))[:10])
        for t in order
        for doc, amount, cur in docs
    ]


def test_S10_flag_all_rows_and_types_scores_minus_one():
    attack = _flag_all_rows_packed()
    assert len(attack) == 16  # 27 rows + 11 documents = 38 ids -> 4 chunks x 4 types
    result = grade(attack, steps=0)
    assert (result.tp, result.fp, result.fn) == (0, 16, 4)
    assert result.reward == -1.0
    assert grade(attack, CLEAN_TRUTH).reward == -1.0
    # Including the FX rows needs more findings than allowed: structurally invalid, also -1.
    everything = _flag_all_ids_packed()
    assert len(everything) > MAX_FINDINGS
    assert grade(everything).valid is False and grade(everything).reward == -1.0


def test_S10_flag_every_document_for_every_type_over_the_limit_is_invalid():
    attack = _flag_every_document_every_type()
    assert len(attack) == 44 > MAX_FINDINGS
    result = grade(attack)
    assert result.valid is False and result.reward == -1.0


def test_S10_flag_every_document_within_the_limit_scores_badly():
    attack = _flag_every_document_every_type()[:MAX_FINDINGS]
    result = grade(attack, steps=0)
    # Only the cut-off guess on D-I5 (amount = invoice total, evidence = its rows) hits.
    assert (result.tp, result.fp, result.fn) == (1, 29, 3)
    assert result.f1 == float(Fraction(1, 17))
    assert result.reward == float(Fraction(-15, 17))  # -0.882
    honest_partial = grade(oracle_findings()[:1], steps=60)  # one correct finding, 60 steps
    assert result.reward < honest_partial.reward == float(Fraction(-1, 5) - Fraction(60, 500))
    assert result.reward < -0.85
    assert grade(attack, CLEAN_TRUTH).reward == -1.0


def test_S10_all_types_on_the_true_subject_get_at_most_one_hit():
    cut = ISSUES["period_cutoff"]
    attack = [
        finding(t, cut.affected_ids, cut.expected_amount_minor, cut.currency, cut.evidence_sets[0])
        for t in ISSUE_TYPES
    ]
    result = grade(attack)
    assert (result.tp, result.fp) == (1, 3)
    assert result.reward == expected_reward(1, 3, 3, 0) < 0


def test_S10_brute_forcing_amounts_gets_one_hit_per_issue():
    fx = ISSUES["fx_conversion"]
    attack = [
        finding(
            fx.type, fx.affected_ids, fx.expected_amount_minor + d, fx.currency, fx.evidence_sets[0]
        )
        for d in range(-15, 15)
    ]
    result = grade(attack)
    assert len(attack) == 30
    assert (result.tp, result.fp, result.fn) == (1, 29, 3)
    assert result.reward == float(Fraction(-15, 17))


# ------------------------------------------------------------------------------ 5
def test_S10_step_savings_only_slightly_separate_equal_answers():
    rewards = {s: grade(oracle_findings(), steps=s).reward for s in range(MAX_STEPS)}
    # Every step costs 0.002 (exactly 1/500 before the final float conversion).
    for s in range(1, MAX_STEPS):
        assert rewards[s] == float(1 - Fraction(s, 500))
        assert rewards[s - 1] - rewards[s] == pytest.approx(0.002, abs=1e-12)
    widest_gap = rewards[0] - rewards[MAX_STEPS - 1]  # 0 vs 99 inspection steps
    assert widest_gap == pytest.approx(0.198)
    # The whole budget is worth less than one missed issue or one false positive.
    one_missed = rewards[0] - grade(oracle_findings()[:3], steps=0).reward  # 2/7 = 0.2857
    one_false = rewards[0] - grade(oracle_findings() + FALSE_ACCUSATIONS[:1], steps=0).reward
    assert widest_gap < one_false < one_missed
    assert one_false == pytest.approx(2 / 9) and one_missed == pytest.approx(2 / 7)


def test_S10_step_savings_never_reorder_answers_of_different_quality_here():
    # Full answer with 99 steps still beats a 3-of-4 answer with 0 steps.
    assert grade(oracle_findings(), steps=99).reward > grade(oracle_findings()[:3], steps=0).reward


def test_S10_step_cost_applies_on_clean_books_too():
    assert grade([], CLEAN_TRUTH, steps=0).reward == 1.0
    assert grade([], CLEAN_TRUTH, steps=50).reward == 0.9
