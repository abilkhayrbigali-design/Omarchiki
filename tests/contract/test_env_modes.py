"""Open and closed modes; the grader log (E12, S03)."""

from __future__ import annotations

import json

import pytest
from envkit import RecordingGrader, act, list_page, oracle_submission, scripted_run

from auditgym.env import AuditEnv
from auditgym.fixtures.tiny import build_tiny_episode

PAYLOAD = oracle_submission(build_tiny_episode())


def test_E12_open_mode_returns_terminal_reward(make_env):
    env = make_env("open", grader_fn=RecordingGrader(reward=0.75))
    env.reset(0)
    obs, reward, terminated, truncated, info = env.step(act("submit", findings=PAYLOAD))
    assert reward == 0.75 and terminated is True
    assert info == {"reward_withheld": False}


def test_E12_closed_mode_withholds_reward(make_env):
    env = make_env("closed", grader_fn=RecordingGrader(reward=0.75))
    env.reset(0)
    obs, reward, terminated, truncated, info = env.step(act("submit", findings=PAYLOAD))
    assert reward == 0.0 and terminated is True
    assert info == {"reward_withheld": True}
    assert obs == {
        "result": {"submitted": True, "findings_count": 4},
        "error": None,
        "steps_remaining": 99,
    }


def test_E12_closed_mode_invalid_submission_looks_the_same(make_env):
    good = make_env("closed", grader_fn=RecordingGrader(reward=0.9))
    bad = make_env("closed", grader_fn=RecordingGrader(valid=False))
    good.reset(0)
    bad.reset(0)
    assert good.step(act("submit", findings=PAYLOAD)) == bad.step(act("submit", findings=PAYLOAD))


def test_E12_closed_mode_whole_stream_independent_of_grade(make_env):
    runs = []
    for g in (
        RecordingGrader(reward=1.0),
        RecordingGrader(reward=-0.5),
        RecordingGrader(valid=False),
    ):
        env = make_env("closed", grader_fn=g)
        runs.append((env.reset(0), scripted_run(env, submit=PAYLOAD)))
    assert runs[0] == runs[1] == runs[2]


@pytest.mark.parametrize("mode", ["open", "closed"])
def test_E12_grader_log_file_written_in_both_modes(make_env, tmp_path, mode):
    log = tmp_path / "grader" / "log.jsonl"
    env = make_env(mode, grader_fn=RecordingGrader(reward=0.5), grader_log=log)
    env.reset(0)
    env.step(list_page("ledger"))
    env.step(act("submit", findings=PAYLOAD))
    env.reset(0)
    env.truncate()
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first, second = (json.loads(x) for x in lines)
    assert first["episode_id"] == "ep_TINY000001"
    assert first["step_count"] == 2
    assert first["end"] == "submit"
    assert first["reward"] == 0.5 and first["inspection_steps"] == 1
    assert first["mode"] == mode
    for key in ("valid", "error", "tp", "fp", "fn", "precision", "recall", "f1", "matches"):
        assert key in first
    assert second["end"] == "timeout" and second["step_count"] == 0


def test_E12_grader_log_callable(make_env):
    records = []
    env = make_env("closed", grader_fn=RecordingGrader(reward=0.5), grader_log=records.append)
    env.reset(0)
    env.step(act("submit", findings=PAYLOAD))
    assert len(records) == 1
    assert records[0]["reward"] == 0.5 and records[0]["matches"]


def test_E12_truncation_record_in_log(make_env):
    records = []
    env = make_env("closed", grader_log=records.append)
    env.reset(0)
    for _ in range(100):
        env.step(act("teleport"))
    assert records[0]["end"] == "step_limit"
    assert records[0]["step_count"] == 100 and records[0]["inspection_steps"] == 100


def test_S03_invalid_submission_logged_with_reason(make_env):
    records = []
    env = make_env("open", grader_fn=RecordingGrader(valid=False), grader_log=records.append)
    env.reset(0)
    obs, reward, terminated, truncated, info = env.step(act("submit", findings="garbage"))
    assert reward == -1.0 and terminated is True
    assert records[0]["valid"] is False and records[0]["error"] == "structure"


def test_mode_is_validated():
    with pytest.raises(ValueError):
        AuditEnv(mode="debug")
