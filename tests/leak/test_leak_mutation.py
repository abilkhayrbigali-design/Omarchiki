"""Returned objects are copies: mutating them cannot change the environment (ARCHITECTURE §8)."""

from __future__ import annotations

import json

from envkit import act, list_page, oracle_submission

from auditgym.env import AuditEnv
from auditgym.fixtures.tiny import build_tiny_episode


def test_mutating_first_observation_changes_nothing(make_env, grader):
    env = make_env()
    obs, info = env.reset(0)
    pristine = json.loads(json.dumps(obs))
    obs["manifest"]["episode_id"] = "ep_HACKED"
    obs["manifest"]["counts"]["ledger"] = 0
    obs["policy"]["accounts"].clear()
    obs["policy"]["tolerances"]["default_minor"] = 10**9
    obs["steps_remaining"] = 10**6
    info["mode"] = "open"
    info["episode_id"] = "ep_HACKED"
    page = env.step(list_page("ledger", None, 50))[0]
    assert len(page["result"]["records"]) == 27
    assert page["steps_remaining"] == 99
    env.step(act("submit", findings={"episode_id": "ep_TINY000001", "findings": []}))
    assert grader.calls[0]["episode_id"] == "ep_TINY000001"
    obs2, info2 = env.reset(0)
    assert obs2 == pristine
    assert info2["episode_id"] == "ep_TINY000001"


def test_mutating_returned_records_changes_nothing(make_env):
    env = make_env()
    env.reset(0)
    page = env.step(list_page("ledger", None, 5))[0]
    page["result"]["records"][0]["debit_minor"] = 1
    page["result"]["records"].append({"row_id": "R-FAKE"})
    rec = env.step(act("get_record", table="ledger", id="R-001"))[0]
    assert rec["result"]["record"]["debit_minor"] == 10_000_000
    rec["result"]["record"]["account"] = "9999"
    doc = env.step(act("get_document", doc_id="D-I3"))[0]
    doc["result"]["document"]["line_items"][0]["amount_minor"] = 0
    doc["result"]["document"]["line_items"].clear()
    again = env.step(act("get_document", doc_id="D-I3"))[0]
    assert again["result"]["document"] == build_tiny_episode().public.invoices[2].to_dict()
    assert (
        env.step(act("get_record", table="ledger", id="R-001"))[0]["result"]["record"]["account"]
        == "1200"
    )
    assert env.step(act("get_record", table="ledger", id="R-FAKE"))[0]["error"]["code"] == (
        "unknown_id"
    )


def test_mutating_error_payload_changes_nothing(make_env):
    env = make_env()
    env.reset(0)
    err = env.step(act("teleport"))[0]
    err["error"]["message"] = "changed"
    err["error"]["code"] = "changed"
    again = env.step(act("teleport"))[0]
    assert again["error"] == {"code": "unknown_action", "message": "Unknown action."}


def test_mutating_source_episode_after_reset_changes_nothing(grader):
    episode = build_tiny_episode()
    env = AuditEnv(generate_fn=lambda s, c=None: episode, grader=grader)
    obs, _ = env.reset(0)
    episode.public.manifest["episode_id"] = "ep_CHANGED"
    episode.public.policy["accounts"]["AR"] = "0000"
    page = env.step(list_page("ledger", None, 1))[0]
    assert page["result"]["records"][0]["account"] == "1200"
    env.step(act("submit", findings={"episode_id": "ep_TINY000001", "findings": []}))
    assert grader.calls[0]["episode_id"] == "ep_TINY000001"


def test_mutating_action_after_step_changes_nothing(make_env, grader):
    env = make_env()
    env.reset(0)
    payload = oracle_submission(build_tiny_episode())
    action = act("submit", findings=payload)
    env.step(action)
    payload["findings"].clear()
    action["args"]["findings"] = "changed"
    assert len(grader.calls[0]["payload"]["findings"]) == 4


def test_mutating_action_args_cannot_reach_store(make_env):
    env = make_env()
    env.reset(0)
    args = {"table": "ledger", "cursor": None, "limit": 2}
    first = env.step({"action": "list_records", "args": args})[0]
    args["limit"] = 50
    args["table"] = "fx"
    second = env.step(list_page("ledger", None, 2))[0]
    assert first["result"] == second["result"]
