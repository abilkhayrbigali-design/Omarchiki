"""S07, S08: metrics with their edge cases and the shared reward, in exact arithmetic."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import pytest
from reward_helpers import CLEAN_TRUTH, finding, grade, oracle_findings

from auditgym.grading import STEP_COST_FRACTION, metrics, reward_exact, shared_reward

F = Fraction


# ------------------------------------------------------------------------------ S07
@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        # (tp, fp, fn, n_truth, n_findings) -> (P, R, F1)
        ((4, 0, 0, 4, 4), (F(1), F(1), F(1))),
        ((3, 1, 1, 4, 4), (F(3, 4), F(3, 4), F(3, 4))),
        ((1, 4, 3, 4, 5), (F(1, 5), F(1, 4), F(2, 9))),
        ((2, 0, 2, 4, 2), (F(1), F(1, 2), F(2, 3))),
        ((0, 3, 4, 4, 3), (F(0), F(0), F(0))),
        # zero denominators
        ((0, 0, 4, 4, 0), (F(0), F(0), F(0))),  # silence on books with errors
        ((0, 0, 0, 0, 0), (F(0), F(0), F(1))),  # truth and findings empty -> F1 = 1
        ((0, 2, 0, 0, 2), (F(0), F(0), F(0))),  # truth empty, findings present -> F1 = 0
        ((0, 30, 0, 0, 30), (F(0), F(0), F(0))),
    ],
)
def test_S07_metrics_exact_values(counts, expected):
    assert metrics(*counts) == expected
    assert all(isinstance(x, Fraction) for x in metrics(*counts))


@pytest.mark.parametrize(
    "counts",
    [(1, 0, 0, 2, 1), (1, 0, 0, 1, 2), (-1, 0, 0, -1, -1), (True, 0, 0, 1, 1), (1.0, 0, 0, 1, 1)],
)
def test_S07_metrics_reject_inconsistent_counts(counts):
    with pytest.raises(ValueError):
        metrics(*counts)


def test_S07_grade_reports_metrics_of_the_matching():
    junk = finding("period_cutoff", ["D-I1"], 10_000_000, "KZT", ["D-I1", "R-001"])
    result = grade(oracle_findings()[:3] + [junk])  # TP 3, FP 1, FN 1
    assert (result.tp, result.fp, result.fn) == (3, 1, 1)
    assert (result.precision, result.recall, result.f1) == (0.75, 0.75, 0.75)


def test_S07_clean_books_edge_cases_through_the_grader():
    silent = grade([], CLEAN_TRUTH)
    assert (silent.tp, silent.fp, silent.fn, silent.f1) == (0, 0, 0, 1.0)
    assert (silent.precision, silent.recall) == (0.0, 0.0)
    accusing = grade(oracle_findings(), CLEAN_TRUTH)
    assert (accusing.tp, accusing.fp, accusing.fn, accusing.f1) == (0, 4, 0, 0.0)


# ------------------------------------------------------------------------------ S08
def test_S08_step_cost_is_exactly_one_five_hundredth():
    assert STEP_COST_FRACTION == F(1, 500)


@pytest.mark.parametrize(
    ("f1", "steps", "expected"),
    [
        (F(1), 0, F(1)),
        (F(1), 1, F(499, 500)),
        (F(1), 50, F(9, 10)),
        (F(1), 99, F(401, 500)),
        (F(1), 100, F(4, 5)),
        (F(1), 1000, F(-1)),  # 1 - 2 = -1, exactly the lower bound
        (F(1), 5000, F(-1)),  # clipped
        (F(0), 0, F(-1)),
        (F(0), 37, F(-1)),  # clipped
        (F(1, 2), 0, F(0)),
        (F(1, 2), 10, F(-1, 50)),
        (F(2, 3), 3, F(1, 3) - F(3, 500)),
        (F(6, 7), 0, F(5, 7)),
    ],
)
def test_S08_reward_formula_exact(f1, steps, expected):
    assert reward_exact(f1, steps) == expected
    assert shared_reward(f1, steps) == float(expected)


def test_S08_exact_arithmetic_avoids_float_drift():
    # In floats 2*(2/3) - 1 - 0.002*3 is not the correctly rounded value of 1/3 - 3/500.
    exact = F(1, 3) - F(3, 500)
    assert shared_reward(F(2, 3), 3) == float(exact)
    for steps in range(0, 101):
        for f1 in (F(1, 3), F(2, 3), F(2, 9), F(8, 11), F(1, 17)):
            value = shared_reward(f1, steps)
            assert value == float(max(F(-1), min(F(1), 2 * f1 - 1 - F(steps, 500))))


def test_S08_reward_accepts_other_exact_number_types():
    assert shared_reward(1, 10) == 0.98
    assert shared_reward(Decimal("0.5"), 0) == 0.0
    assert shared_reward(0.75, 0) == 0.5


@pytest.mark.parametrize("f1", [F(-1, 10), F(11, 10), True, "1", None])
def test_S08_reward_rejects_f1_outside_unit_interval(f1):
    with pytest.raises(ValueError):
        reward_exact(f1, 0)


@pytest.mark.parametrize("steps", [-1, True, 1.5, None])
def test_S08_reward_rejects_bad_step_counts(steps):
    with pytest.raises(ValueError):
        reward_exact(F(1), steps)


def test_S08_grade_reward_equals_formula_from_counts():
    for steps in (0, 1, 13, 57, 99, 100):
        result = grade(oracle_findings()[:2], steps=steps)  # TP 2, FP 0, FN 2 -> F1 = 2/3
        assert result.f1 == float(F(2, 3))
        assert result.reward == float(max(F(-1), F(1, 3) - F(steps, 500)))
        assert result.inspection_steps == steps
