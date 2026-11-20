"""Builders shared by tests/reward. Plain functions so hypothesis tests can use them too."""

from __future__ import annotations

import copy
from fractions import Fraction
from typing import Any

from auditgym.contracts.types import Issue, Truth
from auditgym.fixtures.tiny import EPISODE_ID, build_tiny_episode
from auditgym.grading import grade_submission

EID = EPISODE_ID
TINY = build_tiny_episode()
TRUTH = TINY.private.truth
CLEAN_TRUTH = Truth(EID, ())
ISSUES = {issue.type: issue for issue in TRUTH.issues}  # one issue of each type in the fixture


def finding(
    type_: str,
    affected: list[str] | tuple[str, ...],
    amount: int,
    currency: str,
    evidence: list[str] | tuple[str, ...],
    explanation: str = "",
) -> dict[str, Any]:
    return {
        "type": type_,
        "affected_ids": list(affected),
        "claimed_amount_minor": amount,
        "currency": currency,
        "evidence_ids": list(evidence),
        "explanation": explanation,
    }


def oracle_finding(issue: Issue, evidence_set: int = 0) -> dict[str, Any]:
    return finding(
        issue.type,
        issue.affected_ids,
        issue.expected_amount_minor,
        issue.currency,
        issue.evidence_sets[evidence_set],
    )


def oracle_findings(truth: Truth = TRUTH) -> list[dict[str, Any]]:
    """Each issue with its first evidence set: the organizers' oracle."""
    return [oracle_finding(issue) for issue in truth.issues]


def submission(findings: list[dict[str, Any]], episode_id: str = EID) -> dict[str, Any]:
    return {"episode_id": episode_id, "findings": copy.deepcopy(findings)}


def grade(
    findings: list[dict[str, Any]],
    truth: Truth = TRUTH,
    *,
    steps: int = 0,
    viewed: frozenset[str] | None = None,
    episode_id: str = EID,
):
    return grade_submission(
        submission(findings, episode_id),
        truth,
        episode_id=truth.episode_id,
        inspection_steps=steps,
        viewed_ids=viewed,
    )


def expected_reward(tp: int, fp: int, fn: int, steps: int) -> float:
    """S07 + S08 written out independently of the grader, for numeric assertions."""
    if tp + fp + fn == 0:
        f1 = Fraction(1)
    else:
        f1 = Fraction(2 * tp, 2 * tp + fp + fn)
    raw = 2 * f1 - 1 - Fraction(2, 1000) * steps
    return float(min(Fraction(1), max(Fraction(-1), raw)))


def public_ids() -> list[str]:
    """Every id the agent can see in the tiny fixture: ledger rows, documents, FX rows."""
    pub = TINY.public
    return (
        [row.row_id for row in pub.ledger]
        + [inv.doc_id for inv in pub.invoices]
        + [pay.doc_id for pay in pub.payments]
        + [fx.rate_id for fx in pub.fx]
    )


def rows_of_document(doc_id: str) -> list[str]:
    """Ledger row ids linked to a document of the tiny fixture (by invoice_id or payment_id)."""
    pub = TINY.public
    inv = {i.doc_id: i.invoice_id for i in pub.invoices}
    pay = {p.doc_id: p.payment_id for p in pub.payments}
    if doc_id in inv:
        return [
            r.row_id for r in pub.ledger if r.invoice_id == inv[doc_id] and r.payment_id is None
        ]
    return [r.row_id for r in pub.ledger if r.payment_id == pay[doc_id]]
