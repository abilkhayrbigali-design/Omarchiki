"""S01-S03: structure of the submission, limits, and -1 for any structural deviation."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from reward_helpers import EID, TRUTH, finding, grade, oracle_findings, submission

from auditgym.contracts.constants import (
    ISSUE_TYPES,
    MAX_EVIDENCE_IDS,
    MAX_EXPLANATION_CHARS,
    MAX_FINDINGS,
)
from auditgym.contracts.types import Finding
from auditgym.grading import (
    SubmissionError,
    SubmissionErrorCode,
    empty_submission,
    grade_submission,
    parse_submission,
)
from auditgym.grading import parse as parse_module

SCHEMA = json.loads(
    (Path(__file__).resolve().parents[2] / "schemas" / "submission.schema.json").read_text("utf-8")
)
C = SubmissionErrorCode


def _ok() -> dict:
    return submission(oracle_findings())


def _with(path: tuple, value) -> dict:
    """Copy of the oracle submission with one value replaced (path into the payload)."""
    payload = _ok()
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return payload


def _without(path: tuple) -> dict:
    payload = _ok()
    target = payload
    for key in path[:-1]:
        target = target[key]
    del target[path[-1]]
    return payload


# ------------------------------------------------------------------------------ valid input
def test_S01_oracle_submission_parses_to_findings():
    findings = parse_submission(_ok(), EID)
    assert len(findings) == len(TRUTH.issues)
    assert all(isinstance(f, Finding) for f in findings)
    assert [f.type for f in findings] == [i.type for i in TRUTH.issues]


def test_S01_parse_normalizes_id_lists_to_sorted_distinct_tuples():
    payload = submission([finding("period_cutoff", ["D-I5", "D-I5"], 1, "KZT", ["R-2", "R-1"])])
    (f,) = parse_submission(payload, EID)
    assert f.affected_ids == ("D-I5",)
    assert f.evidence_ids == ("R-1", "R-2")


def test_S01_parse_does_not_mutate_payload():
    payload = submission([finding("period_cutoff", ["b", "a"], 1, "KZT", ["d", "c"])])
    before = copy.deepcopy(payload)
    parse_submission(payload, EID)
    assert payload == before


def test_S01_empty_findings_are_valid():
    assert parse_submission({"episode_id": EID, "findings": []}, EID) == ()
    assert empty_submission(EID) == {"episode_id": EID, "findings": []}


def test_S02_limits_match_the_schema_file():
    item = SCHEMA["properties"]["findings"]["items"]["properties"]
    assert SCHEMA["properties"]["findings"]["maxItems"] == MAX_FINDINGS == 30
    assert item["evidence_ids"]["maxItems"] == MAX_EVIDENCE_IDS == 10
    assert item["explanation"]["maxLength"] == MAX_EXPLANATION_CHARS == 1000
    assert item["affected_ids"]["maxItems"] == parse_module.MAX_AFFECTED_IDS
    assert item["affected_ids"].get("minItems", 0) == parse_module.MIN_AFFECTED_IDS
    assert item["affected_ids"]["items"] == {"type": "string"}
    assert item["evidence_ids"]["items"] == {"type": "string"}
    assert SCHEMA["properties"]["episode_id"]["maxLength"] == parse_module.MAX_EPISODE_ID_CHARS
    # Values are not constrained by the schema: a wrong value is a false positive, not -1.
    assert item["type"]["type"] == item["currency"]["type"] == "string"
    assert "minimum" not in item["claimed_amount_minor"] and "enum" not in item["type"]
    for issue_type in ISSUE_TYPES:
        assert issue_type in item["type"]["description"]
    assert tuple(SCHEMA["required"]) == parse_module.SUBMISSION_KEYS
    assert tuple(SCHEMA["properties"]["findings"]["items"]["required"]) == parse_module.FINDING_KEYS


# ------------------------------------------------------------------------------ limits (S02)
def _many(n: int) -> dict:
    one = finding("period_cutoff", ["D-I5"], 3_000_000, "KZT", ["D-I5", "R-011"])
    return submission([one] * n)


def test_S02_thirty_findings_accepted_thirty_one_rejected():
    assert len(parse_submission(_many(30), EID)) == 30
    with pytest.raises(SubmissionError) as err:
        parse_submission(_many(31), EID)
    assert err.value.code is C.TOO_MANY_FINDINGS


@pytest.mark.parametrize(
    ("n", "ok"), [(0, True), (1, True), (10, True), (11, False)], ids=lambda x: str(x)
)
def test_S02_evidence_ids_limit_is_ten(n, ok):
    payload = submission(
        [finding("period_cutoff", ["D-I5"], 1, "KZT", [f"E{k}" for k in range(n)])]
    )
    if ok:
        assert len(parse_submission(payload, EID)[0].evidence_ids) == n
    else:
        with pytest.raises(SubmissionError) as err:
            parse_submission(payload, EID)
        assert err.value.code is C.BAD_ID_COUNT


def test_S02_evidence_limit_counts_items_before_deduplication():
    payload = submission([finding("period_cutoff", ["D-I5"], 1, "KZT", ["D-I5"] * 11)])
    with pytest.raises(SubmissionError):
        parse_submission(payload, EID)


@pytest.mark.parametrize(("n", "ok"), [(0, True), (1, True), (10, True), (11, False)])
def test_S02_affected_ids_at_most_ten(n, ok):
    payload = submission([finding("period_cutoff", [f"A{k}" for k in range(n)], 1, "KZT", [])])
    if ok:
        parse_submission(payload, EID)
    else:
        with pytest.raises(SubmissionError) as err:
            parse_submission(payload, EID)
        assert err.value.code is C.BAD_ID_COUNT


@pytest.mark.parametrize(("length", "ok"), [(0, True), (1000, True), (1001, False)])
def test_S02_explanation_at_most_1000_characters(length, ok):
    # Non-ASCII characters: the limit counts characters (code points), not bytes.
    payload = submission([finding("period_cutoff", ["D-I5"], 1, "KZT", [], "я" * length)])
    if ok:
        parse_submission(payload, EID)
    else:
        with pytest.raises(SubmissionError) as err:
            parse_submission(payload, EID)
        assert err.value.code is C.EXPLANATION_TOO_LONG


@pytest.mark.parametrize("length", [0, 1, 64, 65, 500])
def test_S02_any_id_string_is_structurally_valid(length):
    # Ids are compared exactly; an id that does not exist only makes the finding unmatched.
    for field in ("affected_ids", "evidence_ids"):
        parse_submission(_with(("findings", 0, field), ["x" * length]), EID)


# ------------------------------------------------------------------------------ malformed (S03)
MALFORMED = {
    # top level
    "payload_none": (None, C.NOT_AN_OBJECT),
    "payload_list": ([], C.NOT_AN_OBJECT),
    "payload_string": ('{"episode_id": "ep_TINY000001", "findings": []}', C.NOT_AN_OBJECT),
    "payload_int": (0, C.NOT_AN_OBJECT),
    "payload_findings_list_only": (oracle_findings(), C.NOT_AN_OBJECT),
    "missing_episode_id": (_without(("episode_id",)), C.MISSING_FIELD),
    "missing_findings": (_without(("findings",)), C.MISSING_FIELD),
    "extra_top_key": ({**_ok(), "reward": 1.0}, C.UNKNOWN_FIELD),
    "non_string_top_key": ({**_ok(), 1: "x"}, C.UNKNOWN_FIELD),
    "episode_id_none": (_with(("episode_id",), None), C.WRONG_TYPE),
    "episode_id_int": (_with(("episode_id",), 1), C.WRONG_TYPE),
    "episode_id_wrong": (_with(("episode_id",), "ep_OTHER000001"), C.WRONG_EPISODE_ID),
    "episode_id_case": (_with(("episode_id",), EID.lower()), C.WRONG_EPISODE_ID),
    "episode_id_padded": (_with(("episode_id",), EID + " "), C.WRONG_EPISODE_ID),
    "episode_id_empty": (_with(("episode_id",), ""), C.WRONG_EPISODE_ID),
    "episode_id_too_long": (_with(("episode_id",), "e" * 65), C.WRONG_EPISODE_ID),
    "findings_none": (_with(("findings",), None), C.WRONG_TYPE),
    "findings_dict": (_with(("findings",), {"0": oracle_findings()[0]}), C.WRONG_TYPE),
    "findings_tuple": (_with(("findings",), tuple(oracle_findings())), C.WRONG_TYPE),
    "findings_string": (_with(("findings",), "[]"), C.WRONG_TYPE),
    # one finding
    "finding_none": (_with(("findings", 0), None), C.NOT_AN_OBJECT),
    "finding_list": (_with(("findings", 0), list(oracle_findings()[0].values())), C.NOT_AN_OBJECT),
    "finding_missing_type": (_without(("findings", 0, "type")), C.MISSING_FIELD),
    "finding_missing_affected": (_without(("findings", 0, "affected_ids")), C.MISSING_FIELD),
    "finding_missing_amount": (_without(("findings", 0, "claimed_amount_minor")), C.MISSING_FIELD),
    "finding_missing_currency": (_without(("findings", 0, "currency")), C.MISSING_FIELD),
    "finding_missing_evidence": (_without(("findings", 0, "evidence_ids")), C.MISSING_FIELD),
    "finding_missing_explanation": (_without(("findings", 0, "explanation")), C.MISSING_FIELD),
    "finding_extra_key": (_with(("findings", 0, "confidence"), 1.0), C.UNKNOWN_FIELD),
    "finding_extra_issue_id": (_with(("findings", 0, "issue_id"), "iss_tiny_dup"), C.UNKNOWN_FIELD),
    "type_none": (_with(("findings", 0, "type"), None), C.WRONG_TYPE),
    "type_list": (_with(("findings", 0, "type"), list(ISSUE_TYPES)), C.WRONG_TYPE),
    "affected_string": (_with(("findings", 0, "affected_ids"), "D-I3"), C.WRONG_TYPE),
    "affected_none": (_with(("findings", 0, "affected_ids"), None), C.WRONG_TYPE),
    "affected_int_item": (_with(("findings", 0, "affected_ids"), [3]), C.WRONG_TYPE),
    "affected_none_item": (_with(("findings", 0, "affected_ids"), [None]), C.WRONG_TYPE),
    "affected_nested": (_with(("findings", 0, "affected_ids"), [["D-I3"]]), C.WRONG_TYPE),
    "amount_float": (_with(("findings", 0, "claimed_amount_minor"), 7_500_000.5), C.WRONG_TYPE),
    "amount_fraction": (_with(("findings", 0, "claimed_amount_minor"), 0.5), C.WRONG_TYPE),
    "amount_bool": (_with(("findings", 0, "claimed_amount_minor"), True), C.WRONG_TYPE),
    "amount_string": (_with(("findings", 0, "claimed_amount_minor"), "7500000"), C.WRONG_TYPE),
    "amount_none": (_with(("findings", 0, "claimed_amount_minor"), None), C.WRONG_TYPE),
    "currency_none": (_with(("findings", 0, "currency"), None), C.WRONG_TYPE),
    "evidence_string": (_with(("findings", 0, "evidence_ids"), "D-I3"), C.WRONG_TYPE),
    "evidence_none": (_with(("findings", 0, "evidence_ids"), None), C.WRONG_TYPE),
    "evidence_int_item": (_with(("findings", 0, "evidence_ids"), ["D-I3", 4]), C.WRONG_TYPE),
    "explanation_none": (_with(("findings", 0, "explanation"), None), C.WRONG_TYPE),
    "explanation_list": (_with(("findings", 0, "explanation"), ["ok"]), C.WRONG_TYPE),
    "explanation_too_long": (
        _with(("findings", 0, "explanation"), "x" * 1001),
        C.EXPLANATION_TOO_LONG,
    ),
    "affected_eleven": (_with(("findings", 0, "affected_ids"), ["D-I3"] * 11), C.BAD_ID_COUNT),
    "evidence_eleven": (_with(("findings", 0, "evidence_ids"), ["D-I3"] * 11), C.BAD_ID_COUNT),
    "bad_last_finding": (_with(("findings", 3, "currency"), None), C.WRONG_TYPE),
}

# Structurally valid findings with a wrong value: the submission stays valid, the finding is a
# false positive and its issue a false negative (jury conventions M06, "unknown type").
WRONG_VALUES = {
    "type_unknown": (("findings", 0, "type"), "fraud"),
    "type_case": (("findings", 0, "type"), "Duplicate_Invoice"),
    "type_padded": (("findings", 0, "type"), "duplicate_invoice "),
    "affected_empty": (("findings", 0, "affected_ids"), []),
    "affected_empty_item": (("findings", 0, "affected_ids"), [""]),
    "amount_negative": (("findings", 0, "claimed_amount_minor"), -7_500_000),
    "currency_lower": (("findings", 0, "currency"), "kzt"),
    "currency_mixed": (("findings", 0, "currency"), "Kzt"),
    "currency_short": (("findings", 0, "currency"), "KZ"),
    "currency_long": (("findings", 0, "currency"), "KZTT"),
    "currency_newline": (("findings", 0, "currency"), "KZT\n"),
    "currency_padded": (("findings", 0, "currency"), " KZT"),
    "currency_symbol": (("findings", 0, "currency"), "₸"),
    "currency_empty": (("findings", 0, "currency"), ""),
    "evidence_empty_item": (("findings", 0, "evidence_ids"), [""]),
    "last_finding_currency": (("findings", 3, "currency"), "usd"),
}


@pytest.mark.parametrize("name", sorted(WRONG_VALUES))
def test_S04_wrong_value_is_a_false_positive_not_a_structural_error(name):
    path, value = WRONG_VALUES[name]
    payload = _with(path, value)
    parse_submission(copy.deepcopy(payload), EID)
    result = grade_submission(payload, TRUTH, episode_id=EID, inspection_steps=0)
    n = len(TRUTH.issues)
    assert result.valid is True
    assert (result.tp, result.fp, result.fn) == (n - 1, 1, 1)


@pytest.mark.parametrize("name", sorted(MALFORMED))
def test_S03_malformed_submission_rejected_with_fixed_code(name):
    payload, code = MALFORMED[name]
    with pytest.raises(SubmissionError) as err:
        parse_submission(copy.deepcopy(payload), EID)
    assert err.value.code is code


@pytest.mark.parametrize("name", sorted(MALFORMED))
def test_S03_malformed_submission_gets_reward_minus_one(name):
    payload, _ = MALFORMED[name]
    viewed = frozenset()
    result = grade_submission(
        copy.deepcopy(payload), TRUTH, episode_id=EID, inspection_steps=5, viewed_ids=viewed
    )
    assert result.valid is False
    assert result.reward == -1.0
    assert result.experimental_reward == -1.0
    assert (result.tp, result.fp, result.fn) == (0, 0, len(TRUTH.issues))
    assert (result.precision, result.recall, result.f1) == (0.0, 0.0, 0.0)
    assert result.matches == ()
    assert result.error is not None and result.error.startswith(MALFORMED[name][1].value)


def test_S03_invalid_reward_is_minus_one_even_with_zero_steps_and_empty_truth():
    clean = type(TRUTH)(EID, ())
    result = grade_submission(None, clean, episode_id=EID, inspection_steps=0)
    assert result.valid is False and result.reward == -1.0 and result.f1 == 0.0
    assert result.experimental_reward is None


def test_S03_error_never_echoes_submitted_values():
    secret = "AGENT_TEXT_" + "z" * 40
    payloads = [
        _with(("findings", 0, "explanation"), secret * 30),
        _with(("findings", 0, "affected_ids"), [secret] * 11),
        _with(("findings", 0, secret), 1),
        _with(("episode_id",), secret),
        {**_ok(), secret: 1},
    ]
    for payload in payloads:
        result = grade_submission(payload, TRUTH, episode_id=EID, inspection_steps=0)
        assert result.valid is False
        assert "AGENT_TEXT_" not in result.error
        assert "AGENT_TEXT_" not in json.dumps(result.to_log())


def test_S03_wrong_episode_id_is_invalid_even_for_a_perfect_answer():
    result = grade(oracle_findings(), episode_id="ep_OTHER000001")
    assert result.valid is False
    assert result.reward == -1.0
    assert result.error == "wrong_episode_id at $.episode_id"


def test_S03_grade_log_is_json_serializable():
    for payload in (_ok(), None):
        log = grade_submission(payload, TRUTH, episode_id=EID, inspection_steps=3).to_log()
        assert json.loads(json.dumps(log)) == log


@pytest.mark.parametrize("steps", [-1, True, 1.0, "3", None])
def test_S03_bad_inspection_steps_is_a_caller_error(steps):
    with pytest.raises(ValueError):
        grade_submission(_ok(), TRUTH, episode_id=EID, inspection_steps=steps)


def test_S03_viewed_ids_as_a_single_string_is_a_caller_error():
    with pytest.raises(TypeError):
        grade_submission(_ok(), TRUTH, episode_id=EID, inspection_steps=0, viewed_ids="D-I3")
