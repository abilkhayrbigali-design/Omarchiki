"""Wall-clock limit per episode: truncated=true, empty answer, late action not executed (E10).

The limit is enforced by the server (EpisodeHost), by the harness and, when asked for with
`AuditEnv(time_limit_s=...)`, by the environment itself; all three take a fake clock.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from envkit import RecordingGrader, act, list_page, oracle_submission, with_episode_id
from serverkit import FakeClock, episode_ids, read_log, run_server, write_episodes

from auditgym.contracts.constants import MAX_STEPS
from auditgym.env import AuditEnv
from auditgym.fixtures.tiny import build_tiny_episode
from auditgym.harness.run import run_episode
from auditgym.server.host import EpisodeHost


@pytest.fixture
def log(tmp_path) -> Path:
    return tmp_path / "organizer" / "grader_log.jsonl"


def _closed(log: Path, dirs: list[Path], *extra: str) -> list[str]:
    argv = ["--mode", "closed", "--grader-log", str(log), *extra]
    for d in dirs:
        argv += ["--episode-dir", str(d)]
    return argv


def _oracle_step(eid: str) -> dict:
    payload = oracle_submission(with_episode_id(build_tiny_episode(), eid))
    return {"op": "step", "action": act("submit", findings=payload)}


# ------------------------------------------------------------------ server
def test_E10_server_late_submit_is_not_executed(tmp_path, log):
    ids = episode_ids(1)
    dirs = write_episodes(tmp_path / "eps", ids)
    clock = FakeClock()
    grader = RecordingGrader()
    _, responses, _ = run_server(
        _closed(log, dirs),  # closed mode: 60 s by default
        [
            (0.0, {"op": "reset"}),
            (30.0, {"op": "step", "action": list_page("ledger", None, 2)}),
            (60.0, {"op": "step", "action": list_page("ledger", None, 1)}),  # at the limit: ok
            (60.5, _oracle_step(ids[0])),  # after it: not executed
            (61.0, {"op": "step", "action": list_page("ledger", None, 1)}),
        ],
        grader=grader,
        clock=clock,
    )
    assert responses[1]["observation"]["error"] is None
    assert responses[2]["observation"]["steps_remaining"] == MAX_STEPS - 2
    late = responses[3]
    assert late["ok"] is True and late["op"] == "step"
    assert late["truncated"] is True and late["terminated"] is False
    assert late["observation"] == {"result": None, "error": None, "steps_remaining": MAX_STEPS - 2}
    assert late["reward"] == 0.0 and late["info"] == {"reward_withheld": True}
    assert responses[4]["observation"]["error"]["code"] == "episode_finished"
    # The grader saw the empty answer, not the late oracle submission.
    (call,) = grader.calls
    assert call["payload"] == {"episode_id": ids[0], "findings": []}
    assert call["inspection_steps"] == 2
    (record,) = read_log(log)
    assert record["end"] == "timeout" and record["step_count"] == 2


@pytest.mark.parametrize("ending", ["reset", "close", "truncate"])
def test_E10_server_walkaway_after_limit_is_logged_as_timeout(tmp_path, log, ending):
    ids = episode_ids(2)
    dirs = write_episodes(tmp_path / "eps", ids)
    clock = FakeClock()
    run_server(
        _closed(log, dirs),
        [
            (0.0, {"op": "reset"}),
            (1.0, {"op": "step", "action": act("teleport")}),
            (75.0, {"op": ending}),
        ],
        clock=clock,
    )
    records = read_log(log)
    assert records[0]["episode_id"] == ids[0]
    assert records[0]["end"] == "timeout" and records[0]["step_count"] == 1


def test_E10_server_eof_after_limit_is_timeout(tmp_path, log):
    ids = episode_ids(1)
    dirs = write_episodes(tmp_path / "eps", ids)
    clock = FakeClock()
    # The last line arrives after the limit and is malformed: the episode is still running when
    # EOF comes, and by then the limit has passed.
    run_server(
        _closed(log, dirs),
        [(0.0, {"op": "reset"}), (90.0, b"not json\n")],
        clock=clock,
    )
    (record,) = read_log(log)
    assert record["end"] == "timeout"


def test_E10_server_time_limit_option_and_open_default(tmp_path, log):
    ids = episode_ids(2)
    dirs = write_episodes(tmp_path / "eps", ids)
    # Open mode: no limit unless asked for.
    clock = FakeClock()
    _, responses, _ = run_server(
        ["--mode", "open", "--episode-dir", str(dirs[0])],
        [(0.0, {"op": "reset"}), (10_000.0, {"op": "step", "action": act("teleport")})],
        clock=clock,
    )
    assert responses[1]["truncated"] is False
    assert responses[1]["observation"]["error"]["code"] == "unknown_action"
    # Explicit limit, closed mode.
    clock = FakeClock()
    _, responses, _ = run_server(
        _closed(log, dirs, "--time-limit", "5"),
        [
            (0.0, {"op": "reset"}),
            (5.0, {"op": "step", "action": act("teleport")}),
            (5.001, {"op": "step", "action": act("teleport")}),
            (100.0, {"op": "reset"}),  # the clock restarts at every reset
            (104.0, {"op": "step", "action": act("teleport")}),
        ],
        clock=clock,
    )
    assert responses[1]["truncated"] is False
    assert responses[2]["truncated"] is True
    assert responses[3]["info"]["episode_id"] == ids[1]
    assert responses[4]["truncated"] is False


def test_E10_host_reads_no_clock_without_a_limit():
    clock = FakeClock()
    env = AuditEnv(generate_fn=lambda s, c=None: build_tiny_episode(), grader=RecordingGrader())
    host = EpisodeHost(env, clock=clock)
    host.handle({"op": "reset", "seed": 1})
    host.handle({"op": "step", "action": act("teleport")})
    host.handle({"op": "close"})
    assert clock.reads == 0


def test_E10_host_rejects_bad_limits():
    env = AuditEnv(mode="open")
    for bad in (0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            EpisodeHost(env, time_limit_s=bad)


# ------------------------------------------------------------------ harness
class SlowPolicy:
    """Submits the oracle answer, but takes `seconds` of (fake) time to think each step."""

    name = "slow"

    def __init__(self, clock: FakeClock, seconds: float, episode) -> None:
        self.clock = clock
        self.seconds = seconds
        self.episode = episode
        self.acts = 0

    def reset(self, observation, info) -> None:
        pass

    def act(self, observation):
        self.acts += 1
        self.clock.now += self.seconds
        if self.acts < 3:
            return list_page("ledger", None, 1)
        return act("submit", findings=oracle_submission(self.episode))


def test_E10_harness_discards_action_that_comes_back_late(monkeypatch):
    episode = build_tiny_episode()
    clock = FakeClock()
    policy = SlowPolicy(clock, 25.0, episode)  # 25, 50, 75 s: the submit comes back late
    run = run_episode(policy, episode, time_limit_s=60.0, clock=clock)
    assert run.truncated is True and run.terminated is False
    assert run.steps == 3 and policy.acts == 3
    assert run.grade["end"] == "timeout"
    assert run.grade["inspection_steps"] == 2 and run.grade["tp"] == 0
    assert run.trajectory[-1]["action"] is None  # the late action was never executed
    assert run.trajectory[-1]["truncated"] is True


def test_E10_harness_fast_policy_unaffected():
    episode = build_tiny_episode()
    clock = FakeClock()
    policy = SlowPolicy(clock, 1.0, episode)
    run = run_episode(policy, episode, time_limit_s=60.0, clock=clock)
    assert run.terminated is True and run.truncated is False
    assert run.grade["end"] == "submit" and run.grade["reward"] == pytest.approx(1.0 - 0.004)
    assert run.trajectory[-1]["action"]["action"] == "submit"


def test_E10_harness_no_limit():
    episode = build_tiny_episode()
    clock = FakeClock()
    run = run_episode(SlowPolicy(clock, 1000.0, episode), episode, time_limit_s=None, clock=clock)
    assert run.terminated is True and run.grade["end"] == "submit"


# ------------------------------------------------------------------ in-process environment
def _timed_env(clock, limit=60.0, *, mode="closed", grader=None, log=None) -> AuditEnv:
    return AuditEnv(
        mode=mode,
        generate_fn=lambda seed, config=None: build_tiny_episode(),
        grader=grader if grader is not None else RecordingGrader(),
        grader_log=log,
        time_limit_s=limit,
        clock=clock,
    )


def test_E10_env_late_submit_is_not_executed():
    clock = FakeClock()
    grader = RecordingGrader()
    records: list[dict] = []
    env = _timed_env(clock, grader=grader, log=records.append)
    obs, _ = env.reset(seed=1)
    eid = obs["manifest"]["episode_id"]
    start = clock.now
    clock.now = start + 30.0
    obs, *_ = env.step(list_page("ledger", None, 2))
    assert obs["error"] is None
    clock.now = start + 60.0  # at the limit: still executed
    obs, _, _, truncated, _ = env.step(list_page("ledger", None, 1))
    assert obs["steps_remaining"] == MAX_STEPS - 2 and truncated is False
    clock.now = start + 60.5  # after it: not executed, no step spent
    late = act("submit", findings=oracle_submission(build_tiny_episode()))
    obs, reward, terminated, truncated, info = env.step(late)
    assert truncated is True and terminated is False
    assert obs == {"result": None, "error": None, "steps_remaining": MAX_STEPS - 2}
    assert reward == 0.0 and info == {"reward_withheld": True}
    # The grader saw the empty answer, not the late oracle submission.
    (call,) = grader.calls
    assert call["payload"] == {"episode_id": eid, "findings": []}
    assert call["inspection_steps"] == 2
    (record,) = records
    assert record["end"] == "timeout" and record["step_count"] == 2
    obs, reward, terminated, truncated, _ = env.step(list_page("ledger", None, 1))
    assert obs["error"]["code"] == "episode_finished" and reward == 0.0
    assert truncated is True and terminated is False
    assert len(grader.calls) == 1 and len(records) == 1


def test_E10_env_open_mode_returns_the_reward_of_the_empty_answer():
    episode = build_tiny_episode()
    assert episode.private.truth.issues  # books with errors: the empty answer scores -1
    clock = FakeClock()
    env = AuditEnv(mode="open", time_limit_s=5, clock=clock)  # the real grader
    env.reset_with_episode(episode)
    clock.now += 5.001
    late = act("submit", findings=oracle_submission(episode))
    obs, reward, terminated, truncated, info = env.step(late)
    assert truncated is True and terminated is False
    assert reward == -1.0 and info == {"reward_withheld": False}
    assert obs == {"result": None, "error": None, "steps_remaining": MAX_STEPS}


def test_E10_env_clock_restarts_at_every_reset():
    clock = FakeClock()
    env = _timed_env(clock, 5.0)
    env.reset(seed=1)
    clock.now += 100.0
    env.reset(seed=2)
    clock.now += 5.0
    assert env.step(act("teleport"))[3] is False
    clock.now += 0.001
    assert env.step(act("teleport"))[3] is True
    env.reset_with_episode(build_tiny_episode())
    assert env.step(act("teleport"))[3] is False


@pytest.mark.parametrize(("wait", "end"), [(75.0, "timeout"), (10.0, "abandoned")])
def test_E10_env_reset_over_a_running_episode_is_logged(wait, end):
    clock = FakeClock()
    records: list[dict] = []
    env = _timed_env(clock, log=records.append)
    env.reset(seed=1)
    env.step(act("teleport"))
    clock.now += wait
    env.reset(seed=2)
    assert [r["end"] for r in records] == [end]
    assert records[0]["step_count"] == 1


def test_E10_env_reads_no_clock_without_a_limit():
    clock = FakeClock()
    records: list[dict] = []
    env = AuditEnv(
        generate_fn=lambda seed, config=None: build_tiny_episode(),
        grader=RecordingGrader(),
        grader_log=records.append,
        clock=clock,
    )
    env.reset(seed=1)
    env.step(act("teleport"))
    env.reset(seed=2)
    env.step(act("submit", findings=oracle_submission(build_tiny_episode())))
    env.truncate()
    assert clock.reads == 0 and [r["end"] for r in records] == ["abandoned", "submit"]


def test_E10_env_limit_not_reached_changes_nothing():
    """Before the limit the clock is invisible: same observations, rewards and flags (E03)."""

    def rollout(**kwargs) -> str:
        env = AuditEnv(mode="open", **kwargs)
        episode = build_tiny_episode()
        out = [env.reset_with_episode(episode)]
        out.append(env.step(list_page("ledger", None, 5)))
        out.append(env.step(act("teleport")))
        out.append(env.step(act("submit", findings=oracle_submission(episode))))
        out.append(env.step(act("teleport")))
        return json.dumps(out, sort_keys=True)

    clock = FakeClock()
    assert rollout() == rollout(time_limit_s=60.0, clock=clock)
    assert clock.reads > 0


def test_E10_env_default_clock_is_the_server_hosts(monkeypatch):
    from auditgym.server import host

    clock = FakeClock()
    monkeypatch.setattr(host, "MONOTONIC", clock)
    env = AuditEnv(
        generate_fn=lambda seed, config=None: build_tiny_episode(),
        grader=RecordingGrader(),
        time_limit_s=60,
    )
    env.reset(seed=1)
    assert env.step(act("teleport"))[3] is False
    clock.now += 61.0
    assert env.step(act("teleport"))[3] is True


def test_E10_env_real_clock_smoke():
    env = AuditEnv(mode="open", time_limit_s=3600)
    env.reset_with_episode(build_tiny_episode())
    obs, reward, terminated, truncated, _ = env.step(list_page("ledger", None, 1))
    assert obs["error"] is None and (reward, terminated, truncated) == (0.0, False, False)


def test_E10_env_rejects_bad_limits():
    for bad in (0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            AuditEnv(time_limit_s=bad)
    for bad in (True, "60", [60]):
        with pytest.raises(TypeError):
            AuditEnv(time_limit_s=bad)
    with pytest.raises(TypeError):
        AuditEnv(time_limit_s=60, clock=1000.0)
