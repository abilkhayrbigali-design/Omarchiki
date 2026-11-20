"""grade_submission: structure -> matching -> metrics -> rewards. Pure: no I/O, no state."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from auditgym.contracts.grading_api import GradeResult
from auditgym.contracts.types import Truth
from auditgym.grading.match import match_findings
from auditgym.grading.parse import SubmissionError, SubmissionErrorCode, parse_submission
from auditgym.grading.reward import metrics, reward_exact

INVALID_REWARD = -1.0  # S03


def _id_set(ids: Iterable[str] | None, name: str) -> frozenset[str] | None:
    if ids is None:
        return None
    if isinstance(ids, str | bytes):
        raise TypeError(f"{name} must be a collection of ids, not a single string")
    out = frozenset(ids)
    if not all(type(x) is str for x in out):
        raise TypeError(f"{name} must contain only strings")
    return out


def grade_submission(
    payload: Any,
    truth: Truth,
    *,
    episode_id: str,
    inspection_steps: int,
    viewed_ids: Iterable[str] | None = None,
    known_ids: Iterable[str] | None = None,
) -> GradeResult:
    """Grade the `findings` argument of submit against `truth`.

    - Structurally invalid payload (schemas/submission.schema.json, or a different episode_id):
      valid=False, reward=-1, TP=FP=0, FN=len(truth.issues), P=R=F1=0, no matches.
    - Otherwise: one-to-one matching, S07 metrics, S08 shared reward.
    - experimental_reward: None without `viewed_ids`; else the same formula where a TP also needs
      its satisfied evidence set inside `viewed_ids` (-1 for an invalid payload). It never
      changes `reward`.
    - known_ids: every id of the open data (PublicEpisode.open_ids). A finding that cites an
      evidence id outside it matches nothing (M04). None skips this check (unit tests only).

    Never raises because of `payload`: any agent-controlled value yields a GradeResult (the
    environment would otherwise score an internal failure). Raises only on caller errors:
    negative or non-int `inspection_steps`, `viewed_ids` or `known_ids` that is not a collection
    of strings.
    """
    if type(inspection_steps) is not int or inspection_steps < 0:
        raise ValueError("inspection_steps must be a non-negative int")
    viewed = _id_set(viewed_ids, "viewed_ids")
    known = _id_set(known_ids, "known_ids")
    issues = truth.issues

    error: str | None = None
    try:
        findings = parse_submission(payload, episode_id)
    except SubmissionError as exc:
        error = str(exc)
    except Exception:  # noqa: BLE001 - trust boundary: no agent payload may crash the grader
        # parse_submission only raises SubmissionError by design (tests/reward cover it); this
        # is defence in depth. The exception text is not logged: it could echo agent values.
        error = f"{SubmissionErrorCode.UNPARSEABLE.value} at $"
    if error is not None:
        return GradeResult(
            valid=False,
            error=error,
            reward=INVALID_REWARD,
            experimental_reward=None if viewed is None else INVALID_REWARD,
            tp=0,
            fp=0,
            fn=len(issues),
            precision=0.0,
            recall=0.0,
            f1=0.0,
            inspection_steps=inspection_steps,
            matches=(),
        )

    shared = match_findings(findings, issues, known_ids=known)
    precision, recall, f1 = metrics(shared.tp, shared.fp, shared.fn, len(issues), len(findings))
    reward = reward_exact(f1, inspection_steps)

    experimental: float | None = None
    if viewed is not None:
        exp = match_findings(findings, issues, viewed_ids=viewed, known_ids=known)
        _, _, exp_f1 = metrics(exp.tp, exp.fp, exp.fn, len(issues), len(findings))
        experimental = float(reward_exact(exp_f1, inspection_steps))

    return GradeResult(
        valid=True,
        error=None,
        reward=float(reward),
        experimental_reward=experimental,
        tp=shared.tp,
        fp=shared.fp,
        fn=shared.fn,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        inspection_steps=inspection_steps,
        matches=shared.pairs,
    )
