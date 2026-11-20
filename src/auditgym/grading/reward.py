"""Metrics (S07) and the shared terminal reward (S08), in exact rational arithmetic.

    P  = TP / (TP + FP)            0 when TP + FP = 0
    R  = TP / (TP + FN)            0 when TP + FN = 0
    F1 = 2TP / (2TP + FP + FN)     1 when truth and findings are both empty,
                                   0 when truth is empty and findings are not
    reward = clip(2*F1 - 1 - 0.002*inspection_steps, -1, 1)

Everything is a `fractions.Fraction`; `float` appears only in `shared_reward`'s return value.
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from numbers import Rational

from auditgym.contracts.constants import STEP_COST

STEP_COST_FRACTION = Fraction(STEP_COST)  # Decimal("0.002") -> exactly 1/500
_ZERO = Fraction(0)
_ONE = Fraction(1)


def _count(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative int")
    return value


def metrics(
    tp: int, fp: int, fn: int, n_truth: int, n_findings: int
) -> tuple[Fraction, Fraction, Fraction]:
    """Exact (precision, recall, F1) per S07. Raises ValueError on inconsistent counts."""
    tp, fp, fn = _count(tp, "tp"), _count(fp, "fp"), _count(fn, "fn")
    n_truth, n_findings = _count(n_truth, "n_truth"), _count(n_findings, "n_findings")
    if tp + fn != n_truth or tp + fp != n_findings:
        raise ValueError("counts must satisfy tp + fn = n_truth and tp + fp = n_findings")

    precision = Fraction(tp, tp + fp) if tp + fp else _ZERO
    recall = Fraction(tp, tp + fn) if tp + fn else _ZERO
    if n_truth == 0:
        f1 = _ONE if n_findings == 0 else _ZERO
    else:
        f1 = Fraction(2 * tp, 2 * tp + fp + fn)
    return precision, recall, f1


def _as_fraction(f1: object) -> Fraction:
    if isinstance(f1, bool) or not isinstance(f1, Rational | float | Decimal):
        raise ValueError("f1 must be a number")
    value = Fraction(f1)  # exact for int, Fraction, Decimal and the binary value of a float
    if not _ZERO <= value <= _ONE:
        raise ValueError("f1 must be within [0, 1]")
    return value


def reward_exact(f1: object, inspection_steps: int) -> Fraction:
    """clip(2*F1 - 1 - 0.002*inspection_steps, -1, 1) as an exact Fraction."""
    steps = _count(inspection_steps, "inspection_steps")
    raw = 2 * _as_fraction(f1) - 1 - STEP_COST_FRACTION * steps
    return min(_ONE, max(-_ONE, raw))


def shared_reward(f1: object, inspection_steps: int) -> float:
    """The shared terminal reward of the case (S08). Pass F1 as a Fraction to stay exact."""
    return float(reward_exact(f1, inspection_steps))
