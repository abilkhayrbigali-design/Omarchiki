"""Compatibility of (finding, issue) pairs and the one-to-one matcher (S04, S05, S06).

The rules are published in the package docstring (`auditgym.grading`) and follow the jury's
matcher conventions (docs/REQUIREMENTS.md, section 8). In short:

- a finding and an issue are compatible when type, the set of affected ids, currency and the
  amount (within the issue's tolerance) agree, the finding's evidence contains a whole evidence
  set of the issue, and every evidence id the finding cites exists in the open data;
- TP is the size of a maximum matching in the bipartite compatibility graph; every finding left
  out is FP, every issue left out is FN;
- among the maximum matchings the reported one is fixed by the published rule: issues in
  ascending `issue_id` order (Unicode code points), the lexicographically smallest sorted
  sequence of (issue_id, finding index) pairs. The choice never changes TP, FP, FN or a reward.
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


def evidence_exists(evidence_ids: frozenset[str], known_ids: frozenset[str] | None) -> bool:
    """Every cited evidence id is an id of the open data (M04). No check without `known_ids`."""
    return known_ids is None or evidence_ids <= known_ids


def compatible(
    finding: Finding,
    issue: Issue,
    viewed_ids: frozenset[str] | None = None,
    known_ids: frozenset[str] | None = None,
) -> bool:
    """Whether `finding` may be matched to `issue` (S04). `explanation` is never read (S09)."""
    evidence = frozenset(finding.evidence_ids)
    return (
        finding.type == issue.type
        and frozenset(finding.affected_ids) == frozenset(issue.affected_ids)
        and finding.currency == issue.currency
        and amount_within_tolerance(finding.claimed_amount_minor, issue)
        and evidence_satisfied(evidence, issue, viewed_ids)
        and evidence_exists(evidence, known_ids)
    )


def _max_matching(adjacency: dict[int, list[int]], findings: list[int], issues: set[int]) -> int:
    """Size of a maximum matching between `findings` and `issues` (Kuhn's algorithm)."""
    owner: dict[int, int] = {}

    def augment(i: int, seen: set[int]) -> bool:
        for j in adjacency[i]:
            if j not in issues or j in seen:
                continue
            seen.add(j)
            if j not in owner or augment(owner[j], seen):
                owner[j] = i
                return True
        return False

    return sum(1 for i in findings if augment(i, set()))


def match_findings(
    findings: Sequence[Finding],
    issues: Sequence[Issue],
    viewed_ids: frozenset[str] | None = None,
    known_ids: frozenset[str] | None = None,
) -> MatchResult:
    """One-to-one matching of findings to reference issues.

    With `viewed_ids`, compatibility additionally requires the satisfied evidence set to be
    viewed (experimental reward); the shared reward always calls this with `viewed_ids=None`.
    With `known_ids`, a finding that cites an evidence id outside it is compatible with nothing.
    """
    normalized = [normalize(f) for f in findings]
    adjacency: dict[int, list[int]] = {
        i: [j for j, issue in enumerate(issues) if compatible(f, issue, viewed_ids, known_ids)]
        for i, f in enumerate(normalized)
    }
    every_finding = list(range(len(normalized)))
    tp = _max_matching(adjacency, every_finding, set(range(len(issues))))

    # Published choice among maximum matchings: walk the issues by issue_id; give each the
    # smallest finding index that still leaves a maximum matching for the issues after it; an
    # issue that no maximum matching can cover (given the choices so far) stays unmatched.
    order = sorted(range(len(issues)), key=lambda j: issues[j].issue_id)
    used: set[int] = set()
    chosen: list[tuple[int, str]] = []
    for pos, j in enumerate(order):
        later = set(order[pos + 1 :])
        for i in sorted(f for f in every_finding if f not in used and j in adjacency[f]):
            rest = [f for f in every_finding if f not in used and f != i]
            if len(chosen) + 1 + _max_matching(adjacency, rest, later) == tp:
                used.add(i)
                chosen.append((i, issues[j].issue_id))
                break

    pairs = tuple(sorted(chosen))
    return MatchResult(tp=tp, fp=len(normalized) - tp, fn=len(issues) - tp, pairs=pairs)
