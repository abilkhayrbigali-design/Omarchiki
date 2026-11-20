"""reset/step shapes, schemas and JSON compatibility (E01, E02, E04, E08)."""

from __future__ import annotations

import json

import pytest
from envkit import (
    ACTION_SCHEMA,
    OBS_SCHEMA,
    act,
    assert_valid_observation,
    list_page,
    oracle_submission,
    schema_validator,
    scripted_run,
)

from auditgym.contracts.constants import MAX_STEPS
from auditgym.env import AuditEnv
from auditgym.env.observe import TASK
from auditgym.fixtures.tiny import build_tiny_episode


def test_E01_reset_returns_observation_and_info(make_env):
    env = make_env()
    obs, info = env.reset(seed=3)
    assert set(obs) == {"task", "manifest", "policy", "steps_remaining"}
    assert obs["steps_remaining"] == MAX_STEPS
    assert obs["task"] == TASK
    episode = build_tiny_episode()
    assert obs["manifest"] == episode.public.manifest
    assert obs["policy"] == episode.public.policy
    assert_valid_observation(obs)
    schema_validator("manifest").validate(obs["manifest"])
    schema_validator("policy").validate(obs["policy"])
    assert isinstance(info, dict)
    assert json.loads(json.dumps(info)) == info


def test_E01_reset_passes_seed_and_config_to_generator(grader):
    calls = []

    def generate(seed, config=None):
        calls.append((seed, config))
        return build_tiny_episode()

    env = AuditEnv(generate_fn=generate, grader=grader)
    env.reset(17, {"config_id": "x"})
    env.reset(18)
    assert calls == [(17, {"config_id": "x"}), (18, None)]


@pytest.mark.parametrize("seed", [True, 1.5, "7", None])
def test_E01_reset_rejects_non_integer_seed(make_env, seed):
    with pytest.raises(TypeError):
        make_env().reset(seed)


def test_E01_task_mentions_tools_and_submit_format():
    for word in (
        "list_records",
        "get_record",
        "get_document",
        "calculator",
        "submit",
        "episode_id",
        "claimed_amount_minor",
        "evidence_ids",
        "100 steps",
    ):
        assert word in TASK


def test_E02_step_returns_five_json_values(make_env):
    env = make_env()
    env.reset(0)
    res = env.step(list_page("ledger", None, 5))
    assert len(res) == 5
    obs, reward, terminated, truncated, info = res
    assert isinstance(obs, dict) and isinstance(info, dict)
    assert type(reward) is float and reward == 0.0
    assert terminated is False and truncated is False
    assert json.loads(json.dumps([obs, reward, terminated, truncated, info])) == [
        obs,
        reward,
        terminated,
        truncated,
        info,
    ]


@pytest.mark.parametrize("mode", ["open", "closed"])
def test_E02_whole_scripted_run_is_json_and_schema_valid(make_env, mode):
    env = make_env(mode)
    obs, info = env.reset(0)
    assert_valid_observation(obs)
    for obs, reward, terminated, truncated, info in scripted_run(
        env, submit=oracle_submission(build_tiny_episode())
    ):
        assert_valid_observation(obs)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool) and isinstance(truncated, bool)
        assert json.loads(json.dumps(info)) == info


def test_E04_actions_used_in_tests_match_action_schema():
    episode = build_tiny_episode()
    for action in (
        list_page("ledger"),
        act("list_records", table="fx"),
        act("get_record", table="invoices", id="D-I1"),
        act("get_document", doc_id="D-P1"),
        act("calculator", expression="1+2"),
        act("submit", findings=oracle_submission(episode)),
    ):
        ACTION_SCHEMA.validate(action)


def test_E04_list_records_shape(make_env):
    env = make_env()
    env.reset(0)
    obs = env.step(list_page("ledger", None, 3))[0]
    assert obs["error"] is None
    assert set(obs["result"]) == {"table", "records", "next_cursor"}
    assert obs["result"]["table"] == "ledger"
    assert [r["row_id"] for r in obs["result"]["records"]] == ["R-001", "R-002", "R-003"]
    assert obs["steps_remaining"] == MAX_STEPS - 1


def test_E04_invoices_listed_as_summary_documents_in_full(make_env):
    env = make_env()
    env.reset(0)
    episode = build_tiny_episode()
    inv = episode.public.invoices[0]
    listed = env.step(list_page("invoices", None, 50))[0]["result"]["records"]
    assert listed[0] == inv.summary()
    assert all("line_items" not in r for r in listed)
    rec = env.step(act("get_record", table="invoices", id=inv.doc_id))[0]["result"]
    assert rec == {"table": "invoices", "record": inv.summary()}
    doc = env.step(act("get_document", doc_id=inv.doc_id))[0]["result"]
    assert doc == {"document": inv.to_dict()}
    pay = episode.public.payments[1]
    doc = env.step(act("get_document", doc_id=pay.doc_id))[0]["result"]
    assert doc == {"document": pay.to_dict()}


def test_E04_record_ids_per_table(make_env):
    env = make_env()
    env.reset(0)
    episode = build_tiny_episode()
    cases = {
        "ledger": (episode.public.ledger[4].row_id, episode.public.ledger[4].to_dict()),
        "invoices": (episode.public.invoices[2].doc_id, episode.public.invoices[2].summary()),
        "payments": (episode.public.payments[3].doc_id, episode.public.payments[3].to_dict()),
        "fx": (episode.public.fx[5].rate_id, episode.public.fx[5].to_dict()),
    }
    for table, (rec_id, expected) in cases.items():
        obs = env.step(act("get_record", table=table, id=rec_id))[0]
        assert obs["error"] is None
        assert obs["result"] == {"table": table, "record": expected}


def test_E04_ids_do_not_cross_tables(make_env):
    env = make_env()
    env.reset(0)
    for table, rec_id in (("ledger", "D-I1"), ("invoices", "D-P1"), ("payments", "R-001")):
        obs = env.step(act("get_record", table=table, id=rec_id))[0]
        assert obs["error"]["code"] == "unknown_id"
    for doc_id in ("R-001", "2026-03-20/USD/KZT"):
        assert env.step(act("get_document", doc_id=doc_id))[0]["error"]["code"] == "unknown_id"


def test_E04_calculator_result_shape(make_env):
    env = make_env()
    env.reset(0)
    obs = env.step(act("calculator", expression="120000 * 504.75"))[0]
    assert obs == {"result": {"value": "60570000"}, "error": None, "steps_remaining": 99}


def test_S01_submit_result_says_nothing_about_correctness(make_env):
    env = make_env()
    env.reset(0)
    payload = oracle_submission(build_tiny_episode())
    obs = env.step(act("submit", findings=payload))[0]
    assert obs == {
        "result": {"submitted": True, "findings_count": 4},
        "error": None,
        "steps_remaining": 99,
    }


def test_S01_submit_payload_reaches_grader_unchanged(make_env, grader):
    env = make_env()
    env.reset(0)
    payload = oracle_submission(build_tiny_episode())
    env.step(act("submit", findings=payload))
    (call,) = grader.calls
    assert call["payload"] == payload
    assert call["episode_id"] == "ep_TINY000001"
    assert call["truth"] == build_tiny_episode().private.truth


def test_E08_observation_has_only_result_error_steps(make_env):
    env = make_env()
    env.reset(0)
    for action in (list_page("fx", None, 2), act("teleport"), act("calculator", expression="2")):
        obs = env.step(action)[0]
        assert set(obs) == {"result", "error", "steps_remaining"}
        OBS_SCHEMA.validate(obs)


def test_E08_error_observation_has_null_result(make_env):
    env = make_env()
    env.reset(0)
    obs = env.step(act("get_document", doc_id="D-NOPE"))[0]
    assert obs["result"] is None
    assert obs["error"]["code"] == "unknown_id"
