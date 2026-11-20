"""Step budget, termination, truncation and exploration reward (E09, E10, E11, S03)."""

from __future__ import annotations

import pytest
from envkit import RecordingGrader, act, list_page, oracle_submission

from auditgym.contracts.constants import MAX_STEPS
from auditgym.fixtures.tiny import build_tiny_episode

EMPTY = {"episode_id": "ep_TINY000001", "findings": []}


def _explore(env, n):
    results = []
    for i in range(n):
        action = list_page("ledger", None, 1) if i % 2 == 0 else act("teleport")
        results.append(env.step(action))
    return results


def test_E09_submit_terminates(make_env, grader):
    env = make_env()
    env.reset(0)
    obs, reward, terminated, truncated, info = env.step(act("submit", findings=EMPTY))
    assert terminated is True and truncated is False
    assert reward == grader.reward
    assert obs["steps_remaining"] == MAX_STEPS - 1
    assert grader.calls[0]["inspection_steps"] == 0


def test_E09_budget_counts_down_by_one_per_step(make_env):
    env = make_env()
    obs, _ = env.reset(0)
    assert obs["steps_remaining"] == MAX_STEPS
    for i, res in enumerate(_explore(env, 10), start=1):
        assert res[0]["steps_remaining"] == MAX_STEPS - i


def test_E09_submit_on_step_100_terminates_not_truncates(make_env, grader):
    env = make_env()
    env.reset(0)
    for _obs, reward, terminated, truncated, _info in _explore(env, MAX_STEPS - 1):
        assert (reward, terminated, truncated) == (0.0, False, False)
    obs, reward, terminated, truncated, info = env.step(act("submit", findings=EMPTY))
    assert terminated is True and truncated is False
    assert obs["steps_remaining"] == 0
    assert grader.calls[0]["inspection_steps"] == MAX_STEPS - 1
    assert grader.calls[0]["payload"] == EMPTY


def test_E10_step_100_without_submit_truncates_with_empty_submission(make_env, grader):
    env = make_env()
    env.reset(0)
    results = _explore(env, MAX_STEPS)
    for _obs, reward, terminated, truncated, info in results[:-1]:
        assert (reward, terminated, truncated, info) == (0.0, False, False, {})
    obs, reward, terminated, truncated, info = results[-1]
    assert truncated is True and terminated is False
    assert obs["steps_remaining"] == 0
    assert obs["error"]["code"] == "unknown_action"  # the last action still gets its answer
    assert reward == grader.reward
    assert info == {"reward_withheld": False}
    (call,) = grader.calls
    assert call["payload"] == EMPTY
    assert call["inspection_steps"] == MAX_STEPS


def test_E10_last_step_valid_tool_result_is_returned(make_env):
    env = make_env()
    env.reset(0)
    _explore(env, MAX_STEPS - 1)
    obs, reward, terminated, truncated, info = env.step(act("calculator", expression="2*3"))
    assert obs["result"] == {"value": "6"} and truncated is True


def test_E10_truncate_ends_episode_as_timeout(make_env, grader):
    env = make_env()
    env.reset(0)
    _explore(env, 7)
    obs, reward, terminated, truncated, info = env.truncate()
    assert truncated is True and terminated is False
    assert obs == {"result": None, "error": None, "steps_remaining": MAX_STEPS - 7}
    assert reward == grader.reward
    (call,) = grader.calls
    assert call["payload"] == EMPTY and call["inspection_steps"] == 7
    # Ended: further steps and truncates change nothing.
    obs2, reward2, terminated2, truncated2, _ = env.step(list_page("ledger"))
    assert obs2["error"]["code"] == "episode_finished" and reward2 == 0.0
    assert (terminated2, truncated2) == (False, True)
    assert env.truncate()[0]["error"]["code"] == "episode_finished"
    assert len(grader.calls) == 1


def test_E10_truncate_in_closed_mode_withholds_reward(make_env, grader):
    env = make_env("closed")
    env.reset(0)
    obs, reward, terminated, truncated, info = env.truncate()
    assert (reward, truncated, info) == (0.0, True, {"reward_withheld": True})
    assert len(grader.calls) == 1


def test_E09_after_termination_steps_change_nothing(make_env, grader):
    env = make_env()
    env.reset(0)
    _explore(env, 3)
    final = env.step(act("submit", findings=EMPTY))
    for action in (list_page("ledger"), act("teleport"), act("submit", findings=EMPTY), None):
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs == {
            "result": None,
            "error": {"code": "episode_finished", "message": "Episode has finished. Call reset."},
            "steps_remaining": final[0]["steps_remaining"],
        }
        assert (reward, terminated, truncated, info) == (0.0, True, False, {})
    assert len(grader.calls) == 1


def test_E10_after_truncation_steps_change_nothing(make_env, grader):
    env = make_env()
    env.reset(0)
    _explore(env, MAX_STEPS)
    obs, reward, terminated, truncated, info = env.step(act("submit", findings=EMPTY))
    assert obs["error"]["code"] == "episode_finished"
    assert (reward, terminated, truncated) == (0.0, False, True)
    assert len(grader.calls) == 1


def test_S05_second_submit_gets_no_reward(make_env, grader):
    env = make_env()
    env.reset(0)
    payload = oracle_submission(build_tiny_episode())
    first = env.step(act("submit", findings=payload))
    second = env.step(act("submit", findings=payload))
    assert first[1] == grader.reward
    assert second[1] == 0.0
    assert second[0]["error"]["code"] == "episode_finished"
    assert len(grader.calls) == 1


def test_E11_exploration_reward_is_zero_and_info_empty(make_env):
    env = make_env()
    env.reset(0)
    episode = build_tiny_episode()
    actions = [
        list_page("ledger"),
        act("get_record", table="invoices", id="D-I3"),
        act("get_record", table="invoices", id="D-I4"),
        act("get_document", doc_id="D-P2"),
        act("calculator", expression="60570000 - 60540000"),
        act("get_record", table="ledger", id="R-NOPE"),
    ] + [act("get_document", doc_id=i.affected_ids[0]) for i in episode.private.truth.issues]
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        assert (reward, terminated, truncated, info) == (0.0, False, False, {})


def test_S03_malformed_findings_still_terminate(make_env):
    grader = RecordingGrader(valid=False)
    env = make_env(grader_fn=grader)
    env.reset(0)
    for payload in ("garbage", None, [], {"findings": "x"}, {"episode_id": 1}):
        env.reset(0)
        obs, reward, terminated, truncated, info = env.step(act("submit", findings=payload))
        assert terminated is True and reward == -1.0
        assert obs["error"] is None and obs["result"]["submitted"] is True
    assert [c["payload"] for c in grader.calls] == [
        "garbage",
        None,
        [],
        {"findings": "x"},
        {"episode_id": 1},
    ]


def test_S03_findings_count_comes_from_payload_only(make_env):
    env = make_env()
    for payload, count in (
        ({"episode_id": "x", "findings": [{}, {}, {}]}, 3),
        ({"episode_id": "x", "findings": "nope"}, 0),
        ("garbage", 0),
    ):
        env.reset(0)
        obs = env.step(act("submit", findings=payload))[0]
        assert obs["result"] == {"submitted": True, "findings_count": count}


def test_E09_inspection_steps_include_errors(make_env, grader):
    env = make_env()
    env.reset(0)
    env.step(act("teleport"))
    env.step(list_page("nope"))
    env.step(list_page("ledger"))
    env.step(act("submit", findings=EMPTY))
    assert grader.calls[0]["inspection_steps"] == 3


def test_viewed_ids_passed_to_grader(make_env, grader):
    env = make_env()
    env.reset(0)
    env.step(list_page("payments", None, 2))
    env.step(act("get_record", table="fx", id="2026-03-20/USD/KZT"))
    env.step(act("get_document", doc_id="D-I3"))
    env.step(act("get_document", doc_id="D-NOPE"))
    env.step(act("submit", findings=EMPTY))
    assert grader.calls[0]["viewed_ids"] == frozenset(
        {"D-P1", "D-P2", "2026-03-20/USD/KZT", "D-I3"}
    )


@pytest.mark.parametrize("mode", ["open", "closed"])
def test_grader_failure_ends_episode_without_leaking(make_env, mode):
    logs = []

    def broken(*args, **kwargs):
        raise RuntimeError("SECRET-TRACE /tmp/truth.json")

    env = make_env(mode, grader_fn=broken, grader_log=logs.append)
    env.reset(0)
    obs, reward, terminated, truncated, info = env.step(act("submit", findings=EMPTY))
    assert terminated is True and reward == 0.0
    assert "SECRET" not in str(obs) and "SECRET" not in str(info)
    if mode == "open":
        assert obs["error"]["code"] == "internal_error"
    else:
        assert obs["error"] is None and info == {"reward_withheld": True}
    assert logs[0]["grader_failure"] is True
    assert "SECRET" not in str(logs)
