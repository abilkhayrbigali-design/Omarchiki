"""Property tests over random truths and submissions (S04-S08, S10, K4).

- permuting findings leaves TP/FP/FN, rewards and the matched contents unchanged;
- consistently renaming every id (truth, findings, viewed ids) leaves TP/FP/FN and rewards
  unchanged;
- rewards are always within [-1, 1], for any payload;
- the oracle (each issue with its first evidence set) always gets F1 = 1;
- adding a copy of a finding never adds a TP and never raises the reward;
- explanation text never changes the result.
"""

from __future__ import annotations

import copy
from fractions import Fraction
from typing import Any

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from auditgym.contracts.constants import ISSUE_TYPES, MAX_FINDINGS
from auditgym.contracts.types import Issue, Truth
from auditgym.grading import grade_submission, normalize, parse_submission

EID = "ep_PROP0000001"
ID_POOL = tuple(f"id{k:02d}" for k in range(24))
CURRENCIES = ("KZT", "USD", "EUR")
SETTINGS = settings(
    max_examples=300,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

ids = st.sampled_from(ID_POOL)


@st.composite
def truths(draw, max_issues: int = 12) -> Truth:
    issues: list[Issue] = []
    keys: set[tuple[str, tuple[str, ...]]] = set()
    for k in range(draw(st.integers(min_value=0, max_value=max_issues))):
        issue_type = draw(st.sampled_from(ISSUE_TYPES))
        affected = tuple(sorted(draw(st.sets(ids, min_size=1, max_size=3))))
        if (issue_type, affected) in keys:  # T2: (type, affected_ids) unique
            continue
        keys.add((issue_type, affected))
        evidence_sets = tuple(
            tuple(sorted(draw(st.sets(ids, min_size=1, max_size=5))))
            for _ in range(draw(st.integers(min_value=1, max_value=3)))
        )
        issues.append(
            Issue(
                issue_id=f"iss{k:02d}",
                type=issue_type,
                affected_ids=affected,
                expected_amount_minor=draw(st.integers(min_value=100, max_value=10**9)),
                currency=draw(st.sampled_from(CURRENCIES)),
                amount_tolerance_minor=draw(st.integers(min_value=0, max_value=2)),
                evidence_sets=evidence_sets,
            )
        )
    return Truth(EID, tuple(issues))


def _oracle(issue: Issue, evidence_set: int = 0) -> dict[str, Any]:
    return {
        "type": issue.type,
        "affected_ids": list(issue.affected_ids),
        "claimed_amount_minor": issue.expected_amount_minor,
        "currency": issue.currency,
        "evidence_ids": list(issue.evidence_sets[evidence_set]),
        "explanation": "",
    }


@st.composite
def finding_near(draw, issue: Issue) -> dict[str, Any]:
    """A finding derived from `issue`: exact, or with one field perturbed."""
    f = _oracle(issue, draw(st.integers(min_value=0, max_value=len(issue.evidence_sets) - 1)))
    kind = draw(st.integers(min_value=0, max_value=7))
    if kind == 1:
        f["claimed_amount_minor"] = max(0, f["claimed_amount_minor"] + draw(st.integers(-3, 3)))
    elif kind == 2:
        f["type"] = draw(st.sampled_from(ISSUE_TYPES))
    elif kind == 3:
        f["currency"] = draw(st.sampled_from(CURRENCIES))
    elif kind == 4:
        f["evidence_ids"] = f["evidence_ids"][: draw(st.integers(0, len(f["evidence_ids"])))]
    elif kind == 5:
        f["affected_ids"] = draw(st.lists(ids, min_size=1, max_size=4))
    elif kind == 6:
        extra = draw(st.lists(ids, max_size=10 - len(f["evidence_ids"])))
        f["evidence_ids"] = f["evidence_ids"] + extra
    f["explanation"] = draw(st.text(max_size=20))
    return f


@st.composite
def random_finding(draw) -> dict[str, Any]:
    return {
        "type": draw(st.sampled_from(ISSUE_TYPES)),
        "affected_ids": draw(st.lists(ids, min_size=1, max_size=4)),
        "claimed_amount_minor": draw(st.integers(min_value=0, max_value=10**9)),
        "currency": draw(st.sampled_from(CURRENCIES)),
        "evidence_ids": draw(st.lists(ids, max_size=10)),
        "explanation": draw(st.text(max_size=20)),
    }


@st.composite
def cases(draw) -> tuple[Truth, list[dict[str, Any]], frozenset[str] | None, int]:
    truth = draw(truths())
    findings: list[dict[str, Any]] = []
    for issue in truth.issues:
        for _ in range(draw(st.integers(min_value=0, max_value=2))):
            findings.append(draw(finding_near(issue)))
    findings += draw(st.lists(random_finding(), max_size=4))
    findings = findings[:MAX_FINDINGS]
    order = draw(st.permutations(range(len(findings))))
    findings = [findings[k] for k in order]
    viewed = draw(st.none() | st.frozensets(ids, max_size=len(ID_POOL)))
    steps = draw(st.integers(min_value=0, max_value=120))
    return truth, findings, viewed, steps


def _grade(truth: Truth, findings: list[dict[str, Any]], viewed, steps: int):
    payload = {"episode_id": truth.episode_id, "findings": copy.deepcopy(findings)}
    return grade_submission(
        payload, truth, episode_id=truth.episode_id, inspection_steps=steps, viewed_ids=viewed
    )


def _summary(result) -> tuple:
    return (
        result.valid,
        result.tp,
        result.fp,
        result.fn,
        result.precision,
        result.recall,
        result.f1,
        result.reward,
        result.experimental_reward,
    )


def _matched_contents(truth: Truth, findings: list[dict[str, Any]], result) -> list[tuple]:
    parsed = parse_submission({"episode_id": truth.episode_id, "findings": findings}, EID)
    out = []
    for index, issue_id in result.matches:
        f = normalize(parsed[index])
        out.append(
            (issue_id, f.type, f.affected_ids, f.claimed_amount_minor, f.currency, f.evidence_ids)
        )
    return sorted(out)


# ------------------------------------------------------------------------------ properties
@SETTINGS
@given(cases(), st.data())
def test_S05_permuting_findings_changes_nothing(case, data):
    truth, findings, viewed, steps = case
    base = _grade(truth, findings, viewed, steps)
    order = data.draw(st.permutations(range(len(findings))))
    shuffled = [findings[k] for k in order]
    other = _grade(truth, shuffled, viewed, steps)
    assert _summary(other) == _summary(base)
    # Canonical matches: the same finding contents are matched to the same issues.
    assert _matched_contents(truth, shuffled, other) == _matched_contents(truth, findings, base)


def _rename(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list | tuple):
        return type(value)(_rename(v, mapping) for v in value)
    return value


@SETTINGS
@given(cases(), st.data())
def test_S04_consistent_id_renaming_changes_nothing(case, data):
    truth, findings, viewed, steps = case
    base = _grade(truth, findings, viewed, steps)

    # A bijection onto new spellings whose sort order differs from the original one.
    targets = data.draw(
        st.permutations([f"Z{k}-{s}" for k, s in enumerate("qwertyuiopasdfghjklzxcvb")])
    )
    mapping = dict(zip(ID_POOL, targets, strict=True))
    new_issue_ids = data.draw(st.permutations([f"ISS-{k}" for k in range(len(truth.issues))]))
    renamed_truth = Truth(
        truth.episode_id,
        tuple(
            Issue(
                issue_id=new_issue_ids[k],
                type=i.type,
                affected_ids=_rename(i.affected_ids, mapping),
                expected_amount_minor=i.expected_amount_minor,
                currency=i.currency,
                amount_tolerance_minor=i.amount_tolerance_minor,
                evidence_sets=tuple(_rename(s, mapping) for s in i.evidence_sets),
            )
            for k, i in enumerate(truth.issues)
        ),
    )
    renamed_findings = [
        {
            **f,
            "affected_ids": _rename(f["affected_ids"], mapping),
            "evidence_ids": _rename(f["evidence_ids"], mapping),
        }
        for f in findings
    ]
    renamed_viewed = None if viewed is None else frozenset(mapping[v] for v in viewed)
    other = _grade(renamed_truth, renamed_findings, renamed_viewed, steps)
    assert _summary(other) == _summary(base)
    assert len(other.matches) == len(base.matches)


json_values = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(max_size=5),
    lambda c: st.lists(c, max_size=3) | st.dictionaries(st.text(max_size=3), c, max_size=3),
    max_leaves=10,
)


@SETTINGS
@given(
    truths(),
    st.one_of(
        json_values,
        st.builds(
            lambda fs: {"episode_id": EID, "findings": fs}, st.lists(random_finding(), max_size=35)
        ),
        st.builds(
            lambda fs: {"episode_id": EID, "findings": fs}, st.lists(json_values, max_size=3)
        ),
    ),
    st.none() | st.frozensets(ids),
    st.integers(min_value=0, max_value=2000),
)
def test_S08_reward_is_always_within_bounds(truth, payload, viewed, steps):
    result = grade_submission(
        payload, truth, episode_id=EID, inspection_steps=steps, viewed_ids=viewed
    )
    assert isinstance(result.reward, float)
    assert -1.0 <= result.reward <= 1.0
    assert 0.0 <= result.f1 <= 1.0
    if viewed is None:
        assert result.experimental_reward is None
    else:
        assert -1.0 <= result.experimental_reward <= 1.0
    if not result.valid:
        assert result.reward == -1.0


@SETTINGS
@given(cases())
def test_S08_bounds_and_count_identities_on_structured_cases(case):
    truth, findings, viewed, steps = case
    result = _grade(truth, findings, viewed, steps)
    assert result.valid
    assert result.tp + result.fn == len(truth.issues)
    assert result.tp + result.fp == len(findings)
    assert len(result.matches) == result.tp
    assert len({i for i, _ in result.matches}) == result.tp  # one-to-one on findings
    assert len({j for _, j in result.matches}) == result.tp  # one-to-one on issues
    assert -1.0 <= result.reward <= 1.0
    if viewed is not None:
        # Experimental compatibility is a subset of shared compatibility.
        assert result.experimental_reward <= result.reward


@SETTINGS
@given(truths(), st.integers(min_value=0, max_value=120), st.data())
def test_S04_oracle_always_gets_f1_one(truth, steps, data):
    findings = [_oracle(issue) for issue in truth.issues]
    order = data.draw(st.permutations(range(len(findings))))
    findings = [findings[k] for k in order]
    every_evidence = frozenset(e for i in truth.issues for s in i.evidence_sets for e in s)
    result = _grade(truth, findings, every_evidence, steps)
    assert (result.tp, result.fp, result.fn) == (len(truth.issues), 0, 0)
    assert result.f1 == 1.0
    expected = float(max(Fraction(-1), 1 - Fraction(steps, 500)))
    assert result.reward == expected
    assert result.experimental_reward == expected


@SETTINGS
@given(cases(), st.data())
def test_S05_copies_of_a_finding_never_help(case, data):
    truth, findings, viewed, steps = case
    assume(0 < len(findings) < MAX_FINDINGS)
    base = _grade(truth, findings, viewed, steps)
    k = data.draw(st.integers(min_value=0, max_value=len(findings) - 1))
    copy_of_k = {**findings[k], "explanation": data.draw(st.text(max_size=10))}
    more = _grade(truth, findings + [copy_of_k], viewed, steps)
    assert more.tp == base.tp
    assert more.fp == base.fp + 1
    assert more.reward <= base.reward
    if viewed is not None:
        assert more.experimental_reward <= base.experimental_reward


@SETTINGS
@given(cases(), st.data())
def test_S09_explanations_never_matter(case, data):
    truth, findings, viewed, steps = case
    base = _grade(truth, findings, viewed, steps)
    texts = data.draw(
        st.lists(
            st.text(max_size=60) | st.just("x" * 1000),
            min_size=len(findings),
            max_size=len(findings),
        )
    )
    rewritten = [{**f, "explanation": t} for f, t in zip(findings, texts, strict=True)]
    assert _grade(truth, rewritten, viewed, steps) == base
