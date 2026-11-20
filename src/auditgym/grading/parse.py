"""Structural validation of the `submit` argument (S01, S02, S03).

The accepted structure is exactly `schemas/submission.schema.json`, checked by hand with the
standard library only:

    {"episode_id": str, "findings": [finding, ...]}          # no other keys, at most 30 findings
    finding = {
        "type": str,
        "affected_ids": [str, ...],                          # at most 10 ids
        "claimed_amount_minor": int,
        "currency": str,
        "evidence_ids": [str, ...],                          # at most 10 ids
        "explanation": str,                                  # at most 1000 characters
    }                                                        # exactly these six keys

`episode_id` must also equal the id of the current episode.

What is structural and what is only a wrong answer follows the jury's published matcher
conventions (docs/REQUIREMENTS.md, section 8): a wrong JSON type, a missing or extra field, or a
list or text over its limit makes the whole submission invalid (reward -1). A structurally valid
finding with a wrong value - an unknown `type` string, a currency spelled in another case or with
a space, a negative amount, an id that does not exist - is parsed as is and simply matches no
issue, so it is a false positive.

Limits are checked on the raw lists before de-duplication (11 evidence ids that repeat one id are
over the limit), and the explanation length is counted in Unicode code points after JSON decoding.

Python types are checked exactly (`type(x) is int`, not `isinstance`): `True` is not an integer
and a string is not an amount; subclasses of dict, list and str are rejected. Integers follow
JSON Schema's "integer": a number with a zero fraction (`5.0`) is accepted and converted to the
exact int; any other float is rejected. Money stays `int` after parsing (AGENTS.md rule 1).

Errors carry a code from `SubmissionErrorCode` and a JSON path such as `$.findings[3].currency`.
They never echo submitted values, so the grader log stays bounded and free of agent text.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from auditgym.contracts.constants import (
    MAX_EVIDENCE_IDS,
    MAX_EXPLANATION_CHARS,
    MAX_FINDINGS,
)
from auditgym.contracts.types import Finding

# Limits of schemas/submission.schema.json that have no constant in contracts/.
# tests/reward/test_reward_parse.py checks them against the schema file.
MAX_AFFECTED_IDS = 10
MIN_AFFECTED_IDS = 0
MAX_EPISODE_ID_CHARS = 64

SUBMISSION_KEYS = ("episode_id", "findings")
FINDING_KEYS = (
    "type",
    "affected_ids",
    "claimed_amount_minor",
    "currency",
    "evidence_ids",
    "explanation",
)


class SubmissionErrorCode(StrEnum):
    NOT_AN_OBJECT = "not_an_object"
    MISSING_FIELD = "missing_field"
    UNKNOWN_FIELD = "unknown_field"
    WRONG_TYPE = "wrong_type"
    WRONG_EPISODE_ID = "wrong_episode_id"
    TOO_MANY_FINDINGS = "too_many_findings"
    BAD_ID_COUNT = "bad_id_count"
    EXPLANATION_TOO_LONG = "explanation_too_long"
    UNPARSEABLE = "unparseable"  # fallback in grade_submission; parse_submission never uses it


class SubmissionError(ValueError):
    """The submission does not match schemas/submission.schema.json (S03: reward -1)."""

    def __init__(self, code: SubmissionErrorCode, path: str) -> None:
        super().__init__(f"{code.value} at {path}")
        self.code = code
        self.path = path


def _fail(code: SubmissionErrorCode, path: str) -> SubmissionError:
    return SubmissionError(code, path)


def _check_keys(obj: dict[Any, Any], keys: tuple[str, ...], path: str) -> None:
    for key in obj:
        if type(key) is not str or key not in keys:
            raise _fail(SubmissionErrorCode.UNKNOWN_FIELD, path)
    for key in keys:
        if key not in obj:
            raise _fail(SubmissionErrorCode.MISSING_FIELD, f"{path}.{key}")


def _id_list(value: Any, path: str, min_items: int, max_items: int) -> tuple[str, ...]:
    if type(value) is not list:
        raise _fail(SubmissionErrorCode.WRONG_TYPE, path)
    if not min_items <= len(value) <= max_items:  # raw length: before de-duplication
        raise _fail(SubmissionErrorCode.BAD_ID_COUNT, path)
    for k, item in enumerate(value):
        if type(item) is not str:
            raise _fail(SubmissionErrorCode.WRONG_TYPE, f"{path}[{k}]")
    # Normalization: ids are a set. Sorting only gives the tuple a canonical spelling.
    return tuple(sorted(set(value)))


def _string(value: Any, path: str) -> str:
    if type(value) is not str:
        raise _fail(SubmissionErrorCode.WRONG_TYPE, path)
    return value


def _finding(value: Any, path: str) -> Finding:
    if type(value) is not dict:
        raise _fail(SubmissionErrorCode.NOT_AN_OBJECT, path)
    _check_keys(value, FINDING_KEYS, path)

    issue_type = _string(value["type"], f"{path}.type")
    affected = _id_list(
        value["affected_ids"], f"{path}.affected_ids", MIN_AFFECTED_IDS, MAX_AFFECTED_IDS
    )

    amount = value["claimed_amount_minor"]
    if type(amount) is float and amount.is_integer() and abs(amount) <= 2**53:
        amount = int(amount)  # JSON Schema "integer" admits 5.0; the value is exact below 2**53
    if type(amount) is not int:
        raise _fail(SubmissionErrorCode.WRONG_TYPE, f"{path}.claimed_amount_minor")

    currency = _string(value["currency"], f"{path}.currency")
    evidence = _id_list(value["evidence_ids"], f"{path}.evidence_ids", 0, MAX_EVIDENCE_IDS)

    explanation = _string(value["explanation"], f"{path}.explanation")
    if len(explanation) > MAX_EXPLANATION_CHARS:  # code points, after JSON decoding
        raise _fail(SubmissionErrorCode.EXPLANATION_TOO_LONG, f"{path}.explanation")

    return Finding(
        type=issue_type,
        affected_ids=affected,
        claimed_amount_minor=amount,
        currency=currency,
        evidence_ids=evidence,
        explanation=explanation,
    )


def parse_submission(payload: Any, episode_id: str) -> tuple[Finding, ...]:
    """Validate `payload` (the `findings` argument of submit) and return normalized findings.

    Raises SubmissionError on any deviation from schemas/submission.schema.json or when the
    payload's `episode_id` is not `episode_id`. Findings keep their submitted order; inside each
    finding `affected_ids` and `evidence_ids` are de-duplicated and sorted.
    """
    if type(payload) is not dict:
        raise _fail(SubmissionErrorCode.NOT_AN_OBJECT, "$")
    _check_keys(payload, SUBMISSION_KEYS, "$")

    submitted_id = payload["episode_id"]
    if type(submitted_id) is not str:
        raise _fail(SubmissionErrorCode.WRONG_TYPE, "$.episode_id")
    if not 1 <= len(submitted_id) <= MAX_EPISODE_ID_CHARS or submitted_id != episode_id:
        raise _fail(SubmissionErrorCode.WRONG_EPISODE_ID, "$.episode_id")

    findings = payload["findings"]
    if type(findings) is not list:
        raise _fail(SubmissionErrorCode.WRONG_TYPE, "$.findings")
    if len(findings) > MAX_FINDINGS:
        raise _fail(SubmissionErrorCode.TOO_MANY_FINDINGS, "$.findings")
    return tuple(_finding(item, f"$.findings[{k}]") for k, item in enumerate(findings))


def empty_submission(episode_id: str) -> dict[str, Any]:
    """The answer graded when an episode is truncated (E10): no findings."""
    return {"episode_id": episode_id, "findings": []}
