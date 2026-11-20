"""Gymnasium-style reset, failed resets and abandoned episodes in-process (E01, E03, E10)."""

from __future__ import annotations

import pytest
from envkit import act, list_page, oracle_submission, with_episode_id

from auditgym.contracts.config import ConfigError
from auditgym.contracts.constants import MAX_STEPS
from auditgym.env import AuditEnv
from auditgym.fixtures.tiny import build_tiny_episode


def _recording_env(grader, grader_log=None, fail_on=None):
    calls: list[tuple] = []

    def generate(seed, config=None):
        calls.append((seed, config))
        if fail_on is not None and fail_on(seed, config):
            raise ConfigError("unknown key 'nope'")
        return with_episode_id(build_tiny_episode(), f"ep_SEED{seed:08d}")

    env = AuditEnv(generate_fn=generate, grader=grader, grader_log=grader_log)
    return env, calls


# ------------------------------------------------------------------ Gymnasium signature
def test_E01_gymnasium_reset_signature(grader):
    env, calls = _recording_env(grader)
    obs, info = env.reset(seed=1, options=None)
    assert info["episode_id"] == "ep_SEED00000001"
    env.reset(seed=2, options={"config": {"config_id": "x"}})
    env.reset(3, {"config_id": "y"})
    env.reset(seed=4, config=None, options={})
    env.reset(seed=5, options={"config": None})
    assert calls == [
        (1, None),
        (2, {"config_id": "x"}),
        (3, {"config_id": "y"}),
        (4, None),
        (5, None),
    ]


def test_E01_reset_without_seed_is_a_deterministic_sequence(grader):
    env, calls = _recording_env(grader)
    env.reset()
    env.reset()
    env.reset(seed=41)
    env.reset(options={"config": {"config_id": "z"}})
    env.reset_with_episode(build_tiny_episode())  # not part of the seed sequence
    env.reset(seed=None)
    assert [c[0] for c in calls] == [0, 1, 41, 42, 43]
    other, other_calls = _recording_env(grader)
    other.reset()
    other.reset()
    assert [c[0] for c in other_calls] == [0, 1]


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"options": {"seed": 1}}, ValueError),
        ({"options": {"config": {}, "render": True}}, ValueError),
        ({"options": "config"}, TypeError),
        ({"config": {"config_id": "a"}, "options": {"config": {"config_id": "b"}}}, ValueError),
        ({"options": {"config": [1]}}, TypeError),
        ({"config": "configs/dev.v1.json"}, TypeError),
    ],
)
def test_E01_bad_reset_arguments_raise_and_keep_running_episode(grader, kwargs, error):
    env, calls = _recording_env(grader)
    env.reset(seed=7)
    env.step(list_page("ledger", None, 2))
    with pytest.raises(error):
        env.reset(seed=8, **kwargs)
    assert calls == [(7, None)]
    obs = env.step(list_page("ledger", None, 1))[0]
    assert obs["error"] is None and obs["steps_remaining"] == MAX_STEPS - 2


# ------------------------------------------------------------------ failed reset
def test_E03_failed_reset_leaves_running_episode_intact(grader):
    records: list[dict] = []
    env, calls = _recording_env(grader, records.append, fail_on=lambda s, c: c is not None)
    env.reset(seed=3)
    env.step(list_page("ledger", None, 2))
    env.step(act("get_document", doc_id="D-I3"))
    with pytest.raises(ConfigError):
        env.reset(seed=4, config={"nope": 1})
    with pytest.raises(ConfigError):
        env.reset(options={"config": {"nope": 1}})
    assert records == [] and grader.calls == []  # nothing ended, nothing graded
    obs, reward, terminated, truncated, info = env.step(list_page("ledger", None, 1))
    assert obs["error"] is None and obs["steps_remaining"] == MAX_STEPS - 3
    assert not terminated and not truncated
    env.step(act("submit", findings=oracle_submission(build_tiny_episode())))
    (call,) = grader.calls
    assert call["episode_id"] == "ep_SEED00000003" and call["inspection_steps"] == 3
    assert call["viewed_ids"] == frozenset({"R-001", "R-002", "D-I3"})
    # A failed seed=None reset does not advance the sequence.
    assert [c[0] for c in calls] == [3, 4, 4]


def test_E03_failed_reset_with_broken_episode_keeps_running_episode(grader):
    env = AuditEnv(generate_fn=lambda s, c=None: build_tiny_episode(), grader=grader)
    env.reset(0)
    env.step(list_page("ledger", None, 1))
    with pytest.raises(AttributeError):
        env.reset_with_episode(object())  # type: ignore[arg-type]
    assert env.active
    assert env.step(list_page("fx", None, 1))[0]["steps_remaining"] == MAX_STEPS - 2


# ------------------------------------------------------------------ abandoned episodes
@pytest.mark.parametrize("how", ["reset", "reset_with_episode"])
def test_E10_reset_mid_episode_logs_abandoned_empty_grade(grader, how):
    records: list[dict] = []
    env, _ = _recording_env(grader, records.append)
    env.reset(seed=1)
    for _ in range(99):
        env.step(act("teleport"))
    if how == "reset":
        obs, _info = env.reset(seed=1)
    else:
        obs, _info = env.reset_with_episode(
            with_episode_id(build_tiny_episode(), "ep_SEED00000001")
        )
    assert obs["steps_remaining"] == MAX_STEPS
    (record,) = records
    assert record["end"] == "abandoned" and record["episode_id"] == "ep_SEED00000001"
    assert record["step_count"] == 99 and record["inspection_steps"] == 99
    (call,) = grader.calls
    assert call["payload"] == {"episode_id": "ep_SEED00000001", "findings": []}
    assert call["inspection_steps"] == 99


def test_E10_five_abandoned_rounds_are_five_logged_grades(grader):
    """The reviewer's attack: explore, reset, explore, ... then submit. Every round is graded."""
    records: list[dict] = []
    env, _ = _recording_env(grader, records.append)
    payload = oracle_submission(with_episode_id(build_tiny_episode(), "ep_SEED00000001"))
    for _ in range(5):
        env.reset(seed=1)
        for _ in range(99):
            env.step(list_page("ledger", None, 1))
    env.reset(seed=1)
    env.step(act("submit", findings=payload))
    assert [r["end"] for r in records] == ["abandoned"] * 5 + ["submit"]
    assert [r["inspection_steps"] for r in records] == [99] * 5 + [0]
    assert [c["payload"]["findings"] == [] for c in grader.calls] == [True] * 5 + [False]


def test_E10_reset_after_end_logs_nothing_more(grader):
    records: list[dict] = []
    env, _ = _recording_env(grader, records.append)
    env.reset(seed=1)
    env.truncate()
    env.reset(seed=2)
    env.step(act("submit", findings={"episode_id": "ep_SEED00000002", "findings": []}))
    env.reset(seed=3)
    assert [r["end"] for r in records] == ["timeout", "submit"]


def test_E10_abandon_is_explicit_and_idempotent(grader):
    records: list[dict] = []
    env, _ = _recording_env(grader, records.append)
    env.reset(seed=1)
    env.step(list_page("ledger", None, 1))
    obs, reward, terminated, truncated, info = env.abandon()
    assert truncated is True and terminated is False
    assert obs == {"result": None, "error": None, "steps_remaining": MAX_STEPS - 1}
    assert env.abandon()[0]["error"]["code"] == "episode_finished"
    assert [r["end"] for r in records] == ["abandoned"]
    assert not env.active


def test_E12_abandon_in_closed_mode_withholds_reward(grader):
    records: list[dict] = []
    env = AuditEnv(
        mode="closed",
        generate_fn=lambda s, c=None: build_tiny_episode(),
        grader=grader,
        grader_log=records.append,
    )
    env.reset(0)
    assert env.abandon()[1:] == (0.0, False, True, {"reward_withheld": True})
    assert records[0]["reward"] == grader.reward
