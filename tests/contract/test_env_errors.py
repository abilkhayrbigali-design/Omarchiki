"""Invalid input: every malformed action gets a fixed error code and costs a step (E07, K1)."""

from __future__ import annotations

import pytest
from envkit import OBS_SCHEMA, act, list_page

from auditgym.contracts.constants import MAX_STEPS
from auditgym.contracts.errors import ERROR_MESSAGES, ErrorCode
from auditgym.env import observe

CASES: list[tuple[str, object, str]] = [
    # (case id, action, expected code)
    ("not_a_dict_list", ["list_records"], "invalid_arguments"),
    ("not_a_dict_str", "list_records", "invalid_arguments"),
    ("not_a_dict_none", None, "invalid_arguments"),
    ("not_a_dict_int", 42, "invalid_arguments"),
    ("empty_dict", {}, "invalid_arguments"),
    ("action_not_str", {"action": 5, "args": {}}, "invalid_arguments"),
    ("unknown_action", act("drop_table"), "unknown_action"),
    ("unknown_action_shell", act("shell", cmd="ls"), "unknown_action"),
    ("unknown_action_case", act("LIST_RECORDS", table="ledger"), "unknown_action"),
    ("missing_args", {"action": "list_records"}, "invalid_arguments"),
    ("args_not_dict", {"action": "list_records", "args": ["ledger"]}, "invalid_arguments"),
    ("args_null", {"action": "calculator", "args": None}, "invalid_arguments"),
    (
        "extra_top_level_key",
        {"action": "list_records", "args": {"table": "ledger"}, "debug": True},
        "invalid_arguments",
    ),
    ("extra_arg", act("list_records", table="ledger", sql="select"), "invalid_arguments"),
    ("missing_table", act("list_records", limit=5), "invalid_arguments"),
    ("table_not_str", act("list_records", table=1), "invalid_arguments"),
    ("unknown_table", act("list_records", table="truth"), "unknown_table"),
    ("unknown_table_get", act("get_record", table="users", id="R-001"), "unknown_table"),
    ("unknown_table_path", act("list_records", table="../private"), "unknown_table"),
    ("cursor_not_str", act("list_records", table="ledger", cursor=3), "invalid_arguments"),
    ("limit_str", act("list_records", table="ledger", limit="10"), "invalid_arguments"),
    ("limit_bool", act("list_records", table="ledger", limit=True), "invalid_arguments"),
    ("limit_fraction", act("list_records", table="ledger", limit=2.5), "invalid_arguments"),
    ("limit_zero", act("list_records", table="ledger", limit=0), "limit_out_of_range"),
    ("limit_negative", act("list_records", table="ledger", limit=-1), "limit_out_of_range"),
    ("limit_51", act("list_records", table="ledger", limit=51), "limit_out_of_range"),
    ("limit_huge", act("list_records", table="ledger", limit=10**30), "limit_out_of_range"),
    ("bad_cursor", list_page("ledger", "c_deadbeef", 5), "invalid_cursor"),
    ("get_record_missing_id", act("get_record", table="ledger"), "invalid_arguments"),
    ("get_record_id_int", act("get_record", table="ledger", id=1), "invalid_arguments"),
    ("get_record_id_empty", act("get_record", table="ledger", id=""), "invalid_arguments"),
    ("get_record_id_long", act("get_record", table="ledger", id="R" * 65), "invalid_arguments"),
    ("get_record_unknown", act("get_record", table="ledger", id="R-999"), "unknown_id"),
    ("get_record_path", act("get_record", table="ledger", id="../truth.json"), "unknown_id"),
    ("get_document_unknown", act("get_document", doc_id="D-X9"), "unknown_id"),
    ("get_document_extra", act("get_document", doc_id="D-I1", full=True), "invalid_arguments"),
    ("get_document_list", act("get_document", doc_id=["D-I1"]), "invalid_arguments"),
    ("calculator_not_str", act("calculator", expression=12), "invalid_arguments"),
    ("calculator_missing", act("calculator"), "invalid_arguments"),
    ("calculator_code", act("calculator", expression="open('/etc/passwd')"), "calculator_error"),
    ("calculator_div0", act("calculator", expression="1/0"), "calculator_error"),
    ("non_json_value", act("calculator", expression={1, 2}), "invalid_arguments"),
    ("non_str_key", {"action": "list_records", "args": {1: "ledger"}}, "invalid_arguments"),
    ("nan", act("list_records", table="ledger", limit=float("nan")), "invalid_arguments"),
]


@pytest.mark.parametrize("action,code", [(a, c) for _, a, c in CASES], ids=[i for i, _, _ in CASES])
def test_E07_invalid_action_gets_fixed_error_and_costs_a_step(make_env, grader, action, code):
    env = make_env()
    env.reset(0)
    obs, reward, terminated, truncated, info = env.step(action)
    OBS_SCHEMA.validate(obs)
    assert obs["result"] is None
    assert obs["error"] == {"code": code, "message": ERROR_MESSAGES[ErrorCode(code)]}
    assert obs["steps_remaining"] == MAX_STEPS - 1
    assert (reward, terminated, truncated, info) == (0.0, False, False, {})
    assert grader.calls == []
    # The episode goes on.
    assert env.step(list_page("ledger", None, 1))[0]["error"] is None


def _deep(depth: int) -> object:
    value: object = []
    for _ in range(depth):
        value = [value]
    return value


EPISODE_ID = "ep_TINY000001"
FLAT = {"episode_id": EPISODE_ID, "findings": []}

# (case id, malformed submit action, what the grader must receive)
MALFORMED_SUBMITS: list[tuple[str, dict, object]] = [
    ("no_args", {"action": "submit"}, None),
    ("args_empty", {"action": "submit", "args": {}}, {}),
    ("args_null", {"action": "submit", "args": None}, None),
    ("args_list", {"action": "submit", "args": [FLAT]}, [FLAT]),
    ("args_str", {"action": "submit", "args": "findings"}, "findings"),
    (
        "extra_arg",
        act("submit", findings=FLAT, note="x"),
        {"findings": FLAT, "note": "x"},
    ),
    (
        "extra_top_level_key",
        {"action": "submit", "args": {"findings": FLAT}, "debug": True},
        {"findings": FLAT},
    ),
    ("too_deep", act("submit", findings=_deep(80)), {"findings": _deep(80)}),
    # The findings object sent without its wrapper: still a malformed action, never graded valid.
    ("flattened", {"action": "submit", "args": FLAT}, None),
    ("flattened_extra", {"action": "submit", "args": dict(FLAT, x=1)}, dict(FLAT, x=1)),
]


@pytest.mark.parametrize(
    "action,payload",
    [(a, p) for _, a, p in MALFORMED_SUBMITS],
    ids=[i for i, _, _ in MALFORMED_SUBMITS],
)
@pytest.mark.parametrize("mode", ["open", "closed"])
def test_S03_E07_malformed_submit_action_ends_episode_as_invalid(make_env, mode, action, payload):
    """Any {"action": "submit", ...} ends the episode; a malformed one is graded invalid (-1)."""
    from auditgym.grading import grade_submission

    records: list[dict] = []
    calls: list[object] = []

    def grader(p, truth, **kw):
        calls.append(p)
        return grade_submission(p, truth, **kw)

    env = make_env(mode, grader_fn=grader, grader_log=records.append)
    env.reset(0)
    env.step(list_page("ledger", None, 1))
    obs, reward, terminated, truncated, info = env.step(action)
    OBS_SCHEMA.validate(obs)
    assert terminated is True and truncated is False
    assert obs == {
        "result": None,
        "error": {
            "code": "invalid_arguments",
            "message": ERROR_MESSAGES[ErrorCode.INVALID_ARGUMENTS],
        },
        "steps_remaining": MAX_STEPS - 2,
    }
    if mode == "open":
        assert reward == -1.0 and info == {"reward_withheld": False}
    else:
        assert reward == 0.0 and info == {"reward_withheld": True}
    assert calls == [payload]
    (record,) = records
    assert record["end"] == "submit" and record["action_error"] == "invalid_arguments"
    assert record["valid"] is False and record["reward"] == -1.0 and record["error"]
    assert record["inspection_steps"] == 1 and record["step_count"] == 2
    # Ended: nothing more happens.
    assert env.step(list_page("ledger"))[0]["error"]["code"] == "episode_finished"
    assert len(records) == 1


def test_S03_malformed_submit_on_last_step_terminates(make_env, grader):
    env = make_env()
    env.reset(0)
    for _ in range(MAX_STEPS - 1):
        env.step(act("teleport"))
    obs, reward, terminated, truncated, info = env.step({"action": "submit"})
    assert terminated is True and truncated is False
    assert obs["steps_remaining"] == 0
    (call,) = grader.calls
    assert call["payload"] is None and call["inspection_steps"] == MAX_STEPS - 1


def test_S03_non_submit_actions_named_like_submit_do_not_end_episode(make_env, grader):
    env = make_env()
    env.reset(0)
    for action in (act("SUBMIT", findings=FLAT), act("submit_now"), ["submit"], "submit"):
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs["error"] is not None and not terminated and not truncated
    assert grader.calls == []


def test_E07_every_env_error_code_is_reachable(make_env):
    """unknown_action ... internal_error: all codes the environment emits during a run."""
    seen = {c for _, _, c in CASES}
    assert seen >= {
        "unknown_action",
        "invalid_arguments",
        "unknown_table",
        "unknown_id",
        "invalid_cursor",
        "limit_out_of_range",
        "calculator_error",
    }


def test_E07_episode_finished_error(make_env):
    env = make_env()
    env.reset(0)
    env.step(act("submit", findings={"episode_id": "ep_TINY000001", "findings": []}))
    obs, reward, terminated, truncated, info = env.step(list_page("ledger"))
    assert obs["error"] == {
        "code": "episode_finished",
        "message": ERROR_MESSAGES[ErrorCode.EPISODE_FINISHED],
    }
    assert reward == 0.0


def test_E07_internal_error_is_fixed_and_costs_a_step(make_env, monkeypatch):
    env = make_env()
    env.reset(0)

    def boom(store, action):
        raise RuntimeError("/secret/path/truth.json line 7")

    monkeypatch.setattr(observe, "run_tool", boom)
    obs, reward, terminated, truncated, info = env.step(list_page("ledger"))
    assert obs == {
        "result": None,
        "error": {"code": "internal_error", "message": ERROR_MESSAGES[ErrorCode.INTERNAL_ERROR]},
        "steps_remaining": MAX_STEPS - 1,
    }
    assert (reward, terminated, truncated, info) == (0.0, False, False, {})


def test_E07_messages_are_constant_per_code(make_env):
    env = make_env()
    env.reset(0)
    messages: dict[str, set[str]] = {}
    for _, action, _code in CASES:
        obs = env.step(action)[0]
        messages.setdefault(obs["error"]["code"], set()).add(obs["error"]["message"])
    assert all(len(m) == 1 for m in messages.values())


def test_E07_step_before_reset_is_bounded(make_env, grader):
    env = make_env()
    obs, reward, terminated, truncated, info = env.step(list_page("ledger"))
    assert obs["error"]["code"] == "episode_finished"
    assert obs["result"] is None and obs["steps_remaining"] == 0
    assert reward == 0.0 and terminated is True and truncated is False
    assert grader.calls == []
