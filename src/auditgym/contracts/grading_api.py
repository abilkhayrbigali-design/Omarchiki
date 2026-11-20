"""Interface between the environment (WP3) and the grader (WP4).

The environment calls a grader function with this signature:

    grade_submission(
        payload: Any,                       # the raw `findings` argument of submit
        truth: Truth,
        *,
        episode_id: str,                    # the current episode id
        inspection_steps: int,              # steps used before submit (errors included)
        viewed_ids: frozenset[str] | None,  # ids the agent requested; for the experimental reward
    ) -> GradeResult

The real implementation is auditgym.grading.grade_submission. Tests of the environment may pass
any function with the same signature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from auditgym.contracts.types import Truth


@dataclass(frozen=True, slots=True)
class GradeResult:
    valid: bool  # False: structurally invalid submission -> reward -1 (S03)
    error: str | None  # why invalid; grader log only, never shown to the agent
    reward: float  # shared terminal reward (S08)
    experimental_reward: float | None  # optional extra reward, reported separately
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    inspection_steps: int
    matches: tuple[tuple[int, str], ...] = field(default=())  # (finding index, issue_id)

    def to_log(self) -> dict[str, Any]:
        """Full record for the grader log."""
        return {
            "valid": self.valid,
            "error": self.error,
            "reward": self.reward,
            "experimental_reward": self.experimental_reward,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "inspection_steps": self.inspection_steps,
            "matches": [list(m) for m in self.matches],
        }


class GraderFn(Protocol):
    def __call__(
        self,
        payload: Any,
        truth: Truth,
        *,
        episode_id: str,
        inspection_steps: int,
        viewed_ids: frozenset[str] | None = None,
    ) -> GradeResult: ...
