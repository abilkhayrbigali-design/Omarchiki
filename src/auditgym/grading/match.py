"""Compatibility of (finding, issue) pairs and the one-to-one matcher (S04, S05, S06).

The rules are published in the package docstring (`auditgym.grading`). In short:

- a finding and an issue are compatible when type, the set of affected ids, currency and the
  amount (within the issue's tolerance) agree and the finding's evidence contains a whole
  evidence set of the issue;
- TP is the size of a maximum matching in the bipartite compatibility graph; every finding left
  out is FP, every issue left out is FN;
- the matching is built by offering findings one by one, in a canonical order, to Kuhn's
  augmenting-path algorithm. The order only decides which of several equally valid findings is
  reported as the TP; the counts are the same for every order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from auditgym.contracts.types import Finding, Issue


@dataclass(frozen=True, slots=True)
class MatchResult:
    tp: int
    fp: int
    fn: int
    pairs: tuple[tuple[int, str], ...]  # (finding index, issue_id), sorted by finding index


def normalize(finding: Finding) -> Finding:
    """Canonical form: id lists become sorted tuples of distinct ids. Other fields unchanged."""
    return Finding(
        type=finding.type,
        affected_ids=tuple(sorted(set(finding.affected_ids))),
        claimed_amount_minor=finding.claimed_amount_minor,
        currency=finding.currency,
        evidence_ids=tuple(sorted(set(finding.evidence_ids))),
        explanation=finding.explanation,
    )


def amount_within_tolerance(claimed_minor: int, issue: Issue) -> bool:
    """|claimed - expected| <= tolerance, inclusive on both sides."""
    return abs(claimed_minor - issue.expected_amount_minor) <= issue.amount_tolerance_minor


def evidence_satisfied(
    evidence_ids: frozenset[str], issue: Issue, viewed_ids: frozenset[str] | None = None
) -> bool:
    """True when some evidence set of `issue` is wholly inside `evidence_ids`.

    With `viewed_ids` (experimental reward only) that set must also be wholly inside
    `viewed_ids`, i.e. the agent actually requested every record of it.
    """
    for evidence_set in issue.evidence_sets:
        required = frozenset(evidence_set)
        if required <= evidence_ids and (viewed_ids is None or required <= viewed_ids):
            return True
    return False


def compatible(finding: Finding, issue: Issue, viewed_ids: frozenset[str] | None = None) -> bool:
    """Whether `finding` may be matched to `issue` (S04). `explanation` is never read (S09)."""
    return (
        finding.type == issue.type
        and frozenset(finding.affected_ids) == frozenset(issue.affected_ids)
        and finding.currency == issue.currency
        and amount_within_tolerance(finding.claimed_amount_minor, issue)
        and evidence_satisfied(frozenset(finding.evidence_ids), issue, viewed_ids)
    )


def _content(finding: Finding) -> tuple[str, tuple[str, ...], int, str, tuple[str, ...]]:
    return (
        finding.type,
        finding.affected_ids,
        finding.claimed_amount_minor,
        finding.currency,
        finding.evidence_ids,
    )


def match_findings(
    findings: Sequence[Finding],
    issues: Sequence[Issue],
    viewed_ids: frozenset[str] | None = None,
) -> MatchResult:
    """One-to-one matching of findings to reference issues.

    `issues` are tried in the given order (Truth keeps them sorted by (type, affected_ids)).
    With `viewed_ids`, compatibility additionally requires the satisfied evidence set to be
    viewed (experimental reward); the shared reward always calls this with `viewed_ids=None`.
    """
    normalized = [normalize(f) for f in findings]
    adjacency: list[list[int]] = [
        [j for j, issue in enumerate(issues) if compatible(f, issue, viewed_ids)]
        for f in normalized
    ]

    # Canonical offer order (published rule, see the package docstring). Position is the last
    # key and only separates findings whose normalized content is identical.
    def offer_key(i: int) -> tuple[object, ...]:
        f = normalized[i]
        gap = min(
            abs(f.claimed_amount_minor - issues[j].expected_amount_minor) for j in adjacency[i]
        )
        return (gap, len(f.evidence_ids), f.claimed_amount_minor, _content(f), i)

    order = sorted((i for i in range(len(normalized)) if adjacency[i]), key=offer_key)

    owner: list[int | None] = [None] * len(issues)  # issue index -> finding index

    def augment(i: int, seen: list[bool]) -> bool:
        for j in adjacency[i]:
            if seen[j]:
                continue
            seen[j] = True
            current = owner[j]
            if current is None or augment(current, seen):
                owner[j] = i
                return True
        return False

    for i in order:
        augment(i, [False] * len(issues))

    pairs = tuple(sorted((i, issues[j].issue_id) for j, i in enumerate(owner) if i is not None))
    tp = len(pairs)
    return MatchResult(tp=tp, fp=len(normalized) - tp, fn=len(issues) - tp, pairs=pairs)
