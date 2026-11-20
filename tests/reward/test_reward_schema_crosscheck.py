"""S01/S03: the hand-written parser agrees with schemas/submission.schema.json.

Thousands of generated payloads (valid ones and valid ones with 1-3 mutations) are checked by
both `parse_submission` and the `jsonschema` package. The parser accepts exactly when the schema
accepts and the episode id is right, except for two documented cases where the parser is
stricter on purpose (see auditgym/grading/parse.py).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from jsonschema import Draft202012Validator
from reward_helpers import EID

from auditgym.contracts.constants import ISSUE_TYPES
from auditgym.grading import SubmissionError, parse_submission

SCHEMA = json.loads(
    (Path(__file__).resolve().parents[2] / "schemas" / "submission.schema.json").read_text("utf-8")
)
VALIDATOR = Draft202012Validator(SCHEMA)
FINDING_KEYS = (
    "type",
    "affected_ids",
    "claimed_amount_minor",
    "currency",
    "evidence_ids",
    "explanation",
)

# ------------------------------------------------------------------------------ strategies
json_values = st.recursive(
    st.none()
    | st.booleans()
    | st.integers(min_value=-(10**20), max_value=10**20)
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(max_size=6),
    lambda children: (
        st.lists(children, max_size=3) | st.dictionaries(st.text(max_size=4), children, max_size=3)
    ),
    max_leaves=8,
)
valid_ids = st.one_of(
    st.sampled_from(["D-I3", "D-I4", "R-011", "2026-03-20/USD/KZT", "x" * 64, "я"]),
    st.text(min_size=1, max_size=64),
)
bad_ids = st.one_of(
    st.sampled_from(["", "x" * 65]),
    st.integers(),
    st.none(),
    st.booleans(),
    st.lists(st.text(max_size=2), max_size=2),
)
valid_currencies = st.from_regex(r"\A[A-Z]{3}\Z")
bad_currencies = st.one_of(
    st.sampled_from(["kzt", "Kzt", "KZ", "KZTT", "", "KZ1", "K Z", "\nKZT", "KZT\n", "ÄBC", "₸"]),
    st.text(max_size=5),
    json_values,
)
valid_amounts = st.one_of(st.sampled_from([0, 1, 30_000, 7_500_000]), st.integers(min_value=0))
bad_amounts = st.one_of(
    st.sampled_from([-1, 1.0, 0.0, 30_000.0, 0.5, True, False, "1", None]),
    st.floats(allow_nan=False, allow_infinity=False),
    st.integers(max_value=-1),
    json_values,
)
valid_explanations = st.one_of(st.text(max_size=40), st.sampled_from(["x" * 1000, "😀" * 1000]))
bad_explanations = st.one_of(st.sampled_from(["x" * 1001, "😀" * 1001]), json_values)
bad_types = st.one_of(
    st.sampled_from(["", "fraud", "Duplicate_invoice", "period_cutoff ", "PERIOD_CUTOFF"]),
    json_values,
)


@st.composite
def valid_finding(draw) -> dict[str, Any]:
    return {
        "type": draw(st.sampled_from(ISSUE_TYPES)),
        "affected_ids": draw(st.lists(valid_ids, min_size=1, max_size=10)),
        "claimed_amount_minor": draw(valid_amounts),
        "currency": draw(valid_currencies),
        "evidence_ids": draw(st.lists(valid_ids, min_size=0, max_size=10)),
        "explanation": draw(valid_explanations),
    }


BAD_FIELD_VALUES = {
    "type": bad_types,
    "affected_ids": st.one_of(
        st.just([]),
        st.lists(valid_ids, min_size=11, max_size=12),
        st.lists(st.one_of(valid_ids, bad_ids), min_size=1, max_size=4),
        json_values,
    ),
    "claimed_amount_minor": bad_amounts,
    "currency": bad_currencies,
    "evidence_ids": st.one_of(
        st.lists(valid_ids, min_size=11, max_size=12),
        st.lists(st.one_of(valid_ids, bad_ids), min_size=1, max_size=4),
        json_values,
    ),
    "explanation": bad_explanations,
}


def _mutate(draw, payload: dict[str, Any]) -> Any:
    kind = draw(st.integers(min_value=0, max_value=9))
    findings = payload["findings"]
    if kind == 0:
        return draw(json_values)  # replace everything
    if kind == 1:
        key = draw(st.sampled_from(["episode_id", "findings"]))
        payload.pop(key, None)
    elif kind == 2:
        payload[draw(st.text(min_size=1, max_size=5))] = draw(json_values)
    elif kind == 3:
        payload["episode_id"] = draw(
            st.one_of(
                st.sampled_from(["", "ep_OTHER", EID.lower(), EID + " ", "e" * 65]), json_values
            )
        )
    elif kind == 4:
        payload["findings"] = draw(st.one_of(json_values, st.tuples(valid_finding())))
    elif kind == 5:
        one = draw(valid_finding())  # pad to 31 findings: one over the limit
        findings.extend(copy.deepcopy(one) for _ in range(31 - len(findings)))
    elif findings:
        k = draw(st.integers(min_value=0, max_value=len(findings) - 1))
        if kind == 6:
            findings[k] = draw(json_values)
        elif kind == 7 and isinstance(findings[k], dict):
            findings[k].pop(draw(st.sampled_from(FINDING_KEYS)), None)
        elif kind == 8 and isinstance(findings[k], dict):
            findings[k][draw(st.text(min_size=1, max_size=5))] = draw(json_values)
        elif isinstance(findings[k], dict):
            field = draw(st.sampled_from(FINDING_KEYS))
            findings[k][field] = draw(BAD_FIELD_VALUES[field])
    return payload


@st.composite
def payloads(draw) -> Any:
    payload = {
        "episode_id": EID,
        "findings": draw(st.lists(valid_finding(), min_size=0, max_size=31)),
    }
    for _ in range(draw(st.integers(min_value=0, max_value=3))):
        if not isinstance(payload, dict) or not isinstance(payload.get("findings"), list):
            break
        payload = _mutate(draw, payload)
    return payload


# ------------------------------------------------------------------------------ the check
def _known_stricter(payload: Any) -> bool:
    """Payloads the schema validator accepts but the parser rejects on purpose.

    1. `claimed_amount_minor` given as an integral float beyond 2**53: JSON Schema "integer"
       accepts it, but it cannot be an exact amount. (Integral floats up to 2**53, like 5.0, are
       accepted by both and converted to int.)
    2. `currency` with a trailing newline ("USD\\n"): Python's re `$` matches before a final
       newline, ECMA-262 `$` (the JSON Schema regex dialect) does not.
    """
    if type(payload) is not dict or type(payload.get("findings")) is not list:
        return False
    for item in payload["findings"]:
        if type(item) is not dict:
            continue
        amount = item.get("claimed_amount_minor")
        if isinstance(amount, float) and abs(amount) > 2**53:
            return True
        currency = item.get("currency")
        if isinstance(currency, str) and currency.endswith("\n"):
            return True
    return False


def _parser_accepts(payload: Any) -> bool:
    try:
        parse_submission(copy.deepcopy(payload), EID)
    except SubmissionError:
        return False
    return True


def _check_agreement(payload: Any) -> None:
    schema_ok = VALIDATOR.is_valid(payload)
    parser_ok = _parser_accepts(payload)
    # The parser is never more lenient than the schema.
    assert not parser_ok or schema_ok
    if schema_ok:
        expected = payload["episode_id"] == EID and not _known_stricter(payload)
        assert parser_ok == expected


_HEALTH = [HealthCheck.too_slow, HealthCheck.data_too_large]


@settings(max_examples=1000, deadline=None, suppress_health_check=_HEALTH)
@given(payloads())
def test_S01_parser_agrees_with_json_schema(payload):
    _check_agreement(payload)


@pytest.mark.slow
@settings(max_examples=10_000, deadline=None, suppress_health_check=_HEALTH, database=None)
@given(payloads())
def test_S01_parser_agrees_with_json_schema_10k(payload):
    _check_agreement(payload)


def test_S01_crosscheck_generator_produces_both_outcomes():
    """Guard against a vacuous cross-check: the generator yields valid and invalid payloads."""
    seen: set[bool] = set()

    @settings(max_examples=300, deadline=None, database=None)
    @given(payloads())
    def collect(payload):
        seen.add(_parser_accepts(payload))

    collect()
    assert seen == {True, False}


def test_S01_documented_divergences_from_python_jsonschema():
    base = {
        "type": "period_cutoff",
        "affected_ids": ["D-I5"],
        "claimed_amount_minor": 3_000_000,
        "currency": "KZT",
        "evidence_ids": ["D-I5", "R-011"],
        "explanation": "",
    }
    integral_float = {"episode_id": EID, "findings": [{**base, "claimed_amount_minor": 3e6}]}
    newline_currency = {"episode_id": EID, "findings": [{**base, "currency": "KZT\n"}]}
    huge_float = {"episode_id": EID, "findings": [{**base, "claimed_amount_minor": 1e300}]}
    assert VALIDATOR.is_valid(integral_float) and _parser_accepts(integral_float)
    parsed = parse_submission(copy.deepcopy(integral_float), EID)
    assert (
        type(parsed[0].claimed_amount_minor) is int and parsed[0].claimed_amount_minor == 3_000_000
    )
    assert VALIDATOR.is_valid(newline_currency) and _parser_accepts(newline_currency)
    for payload in (huge_float,):
        assert VALIDATOR.is_valid(payload)  # generic Python validator lets these through
        assert not _parser_accepts(payload)  # the grader does not
