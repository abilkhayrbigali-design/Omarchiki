"""S03, S11: no payload can make the grader raise.

The environment treats an exception from the grader as an internal failure, so every
agent-controlled payload must come back as a GradeResult; anything off-schema as valid=False,
reward=-1.0. `parse_submission` itself raises nothing but SubmissionError, so the fallback in
`grade_submission` is never needed for these inputs; it is tested separately.
"""

from __future__ import annotations

import decimal
import json
import math

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from reward_helpers import CLEAN_TRUTH, EID, TRUTH, oracle_findings

import auditgym.grading.grade as grade_module
from auditgym.contracts.grading_api import GradeResult
from auditgym.grading import SubmissionError, grade_submission, parse_submission

HUGE = "x" * 10_000_000


def _deep_list(depth: int) -> list:
    root: list = []
    node = root
    for _ in range(depth):
        child: list = []
        node.append(child)
        node = child
    return root


def _deep_dict(depth: int) -> dict:
    root: dict = {}
    node = root
    for _ in range(depth):
        child: dict = {}
        node["findings"] = child
        node = child
    return root


def _self_list() -> list:
    a: list = []
    a.append(a)
    return a


def _self_dict() -> dict:
    d: dict = {"episode_id": EID}
    d["findings"] = [d]
    return d


def _f(**changes) -> dict:
    return {"episode_id": EID, "findings": [{**oracle_findings()[0], **changes}]}


class FloatSub(float):
    pass


class IntSub(int):
    pass


EDGE_PAYLOADS = {
    "none": None,
    "true": True,
    "nan": math.nan,
    "inf": math.inf,
    "neg_inf": -math.inf,
    "complex": 1j,
    "decimal_nan": decimal.Decimal("NaN"),
    "ellipsis": ...,
    "object": object(),
    "function": len,
    "exception": ValueError("boom"),
    "type": dict,
    "bytes": b'{"episode_id": "ep_TINY000001", "findings": []}',
    "bytearray": bytearray(b"{}"),
    "set": {1, 2},
    "frozenset": frozenset({"episode_id"}),
    "tuple": (EID, []),
    "range": range(10),
    "json_text": json.dumps({"episode_id": EID, "findings": oracle_findings()}),
    "huge_string": HUGE,
    "deep_list": _deep_list(100_000),
    "deep_dict": _deep_dict(100_000),
    "self_list": _self_list(),
    "self_dict": _self_dict(),
    "findings_deep": {"episode_id": EID, "findings": _deep_list(100_000)},
    "findings_million": {"episode_id": EID, "findings": [None] * 1_000_000},
    "findings_generator": {"episode_id": EID, "findings": (x for x in oracle_findings())},
    "many_keys": {**{f"k{n}": n for n in range(100_000)}, "episode_id": EID, "findings": []},
    "huge_episode_id": {"episode_id": HUGE, "findings": []},
    "nan_episode_id": {"episode_id": math.nan, "findings": []},
    "nan_amount": _f(claimed_amount_minor=math.nan),
    "inf_amount": _f(claimed_amount_minor=math.inf),
    "float_sub_amount": _f(claimed_amount_minor=FloatSub(7_500_000)),
    "int_sub_amount": _f(claimed_amount_minor=IntSub(7_500_000)),
    "decimal_amount": _f(claimed_amount_minor=decimal.Decimal(7_500_000)),
    "complex_amount": _f(claimed_amount_minor=7_500_000 + 0j),
    "huge_explanation": _f(explanation=HUGE),
    "million_evidence": _f(evidence_ids=["D-I3"] * 1_000_000),
    "deep_affected": _f(affected_ids=_deep_list(100_000)),
    "self_affected": _f(affected_ids=_self_list()),
    "nan_currency": _f(currency=math.nan),
    "bytes_currency": _f(currency=b"KZT"),
    "bytes_type": _f(type=b"duplicate_invoice"),
    "set_affected": _f(affected_ids={"D-I3", "D-I4"}),
    "tuple_affected": _f(affected_ids=("D-I3", "D-I4")),
    "nonstring_finding_key": {"episode_id": EID, "findings": [{**oracle_findings()[0], 7: 7}]},
    "tuple_key": {("episode_id",): EID, "findings": []},
}


# Odd but structurally valid values: graded without raising, the finding matches nothing.
ODD_VALUES = {
    "huge_id": _f(affected_ids=[HUGE, "D-I4"]),
    "huge_evidence": _f(evidence_ids=[HUGE]),
    "nul_currency": _f(currency="KZ\x00"),
}


@pytest.mark.parametrize("name", sorted(ODD_VALUES))
def test_S04_odd_values_are_graded_as_false_positives(name):
    result = grade_submission(ODD_VALUES[name], TRUTH, episode_id=EID, inspection_steps=7)
    assert result.valid is True and result.tp == 0 and result.fp == 1
    json.dumps(result.to_log())


@pytest.mark.parametrize("name", sorted(EDGE_PAYLOADS))
def test_S03_edge_payload_never_raises_and_scores_minus_one(name):
    payload = EDGE_PAYLOADS[name]
    for truth in (TRUTH, CLEAN_TRUTH):
        result = grade_submission(
            payload, truth, episode_id=EID, inspection_steps=7, viewed_ids=frozenset()
        )
        assert isinstance(result, GradeResult)
        assert result.valid is False
        assert result.reward == -1.0
        assert result.experimental_reward == -1.0
        assert result.error is not None and len(result.error) < 100
        assert not result.error.startswith("unparseable")  # the parser itself handled it
        json.dumps(result.to_log())  # loggable


@pytest.mark.parametrize("name", sorted(EDGE_PAYLOADS))
def test_S03_parser_raises_only_submission_error(name):
    with pytest.raises(SubmissionError):
        parse_submission(EDGE_PAYLOADS[name], EID)


# Arbitrary Python data, including NaN/inf, bytes, tuples, sets and nested containers.
anything = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats()  # NaN and infinities included
    | st.text(max_size=20)
    | st.binary(max_size=8)
    | st.complex_numbers()
    | st.decimals(),
    lambda c: (
        st.lists(c, max_size=4)
        | st.tuples(c, c)
        | st.frozensets(st.integers() | st.text(max_size=3), max_size=3)
        | st.dictionaries(st.text(max_size=12) | st.integers(), c, max_size=4)
    ),
    max_leaves=20,
)
finding_fields = st.fixed_dictionaries(
    {},
    optional={
        "type": anything | st.sampled_from(["duplicate_invoice", "period_cutoff"]),
        "affected_ids": anything | st.lists(st.text(max_size=70), max_size=12),
        "claimed_amount_minor": anything,
        "currency": anything | st.text(max_size=4),
        "evidence_ids": anything | st.lists(st.text(max_size=70), max_size=12),
        "explanation": anything,
        "extra": anything,
    },
)
garbage_payloads = st.one_of(
    anything,
    st.fixed_dictionaries(
        {},
        optional={
            "episode_id": anything | st.just(EID),
            "findings": anything | st.lists(finding_fields | anything, max_size=33),
            "other": anything,
        },
    ),
)


@settings(
    max_examples=1500,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@given(garbage_payloads, st.booleans())
def test_S03_fuzzed_payloads_never_raise(payload, clean):
    try:
        parse_submission(payload, EID)
        structurally_valid = True
    except SubmissionError:
        structurally_valid = False
    result = grade_submission(
        payload,
        CLEAN_TRUTH if clean else TRUTH,
        episode_id=EID,
        inspection_steps=3,
        viewed_ids=frozenset(),
    )
    assert isinstance(result, GradeResult)
    assert result.valid is structurally_valid
    assert -1.0 <= result.reward <= 1.0
    if not result.valid:
        assert result.reward == -1.0 and result.experimental_reward == -1.0


def test_S03_odd_but_well_formed_strings_are_graded_not_rejected():
    # A lone surrogate or control character is still a string of 1-64 characters: structurally
    # valid, simply a wrong id, so the finding is a false positive.
    for odd in ("\ud800", "\x00", "‮", "😀" * 64):
        result = grade_submission(
            _f(affected_ids=[odd, "D-I4"]), TRUTH, episode_id=EID, inspection_steps=0
        )
        assert result.valid is True and (result.tp, result.fp) == (0, 1)
        assert result.reward == -1.0


def test_S03_unexpected_parser_failure_still_scores_minus_one(monkeypatch):
    secret = "AGENT_SECRET_TEXT"

    def broken(payload, episode_id):
        raise RuntimeError(secret)

    monkeypatch.setattr(grade_module, "parse_submission", broken)
    for exc_payload in (None, {"episode_id": EID, "findings": []}):
        result = grade_submission(exc_payload, TRUTH, episode_id=EID, inspection_steps=0)
        assert result.valid is False and result.reward == -1.0
        assert result.error == "unparseable at $"
        assert secret not in json.dumps(result.to_log())
