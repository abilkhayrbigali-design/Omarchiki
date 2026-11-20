"""reset isolates episodes; same seed, config and actions give the same stream (E03, K4)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from envkit import act, big_episode, list_page, oracle_submission, scripted_run, with_episode_id

from auditgym.contracts.constants import MAX_STEPS
from auditgym.env import AuditEnv
from auditgym.fixtures.tiny import build_tiny_episode
from auditgym.util import canon

PAYLOAD = oracle_submission(build_tiny_episode())


def _stream(env):
    first = env.reset(42, {"config_id": "smoke"})
    return canon.dumps([list(first), [list(r) for r in scripted_run(env, submit=PAYLOAD)]])


def test_E03_same_actions_give_byte_identical_stream(make_env):
    a = _stream(make_env("open"))
    b = _stream(make_env("open"))
    assert a == b


def test_E03_reset_on_same_env_gives_same_stream(make_env):
    env = make_env("open")
    assert _stream(env) == _stream(env)


def test_E03_reset_clears_state(make_env, grader):
    env = make_env()
    env.reset(0)
    for _ in range(30):
        env.step(list_page("ledger", None, 2))
    env.step(act("get_document", doc_id="D-I3"))
    env.step(act("submit", findings=PAYLOAD))
    obs, info = env.reset(0)
    assert obs["steps_remaining"] == MAX_STEPS
    obs, reward, terminated, truncated, info = env.step(list_page("ledger", None, 2))
    assert obs["steps_remaining"] == MAX_STEPS - 1 and not terminated and not truncated
    env.step(act("submit", findings=PAYLOAD))
    second = grader.calls[1]
    assert second["inspection_steps"] == 1
    assert second["viewed_ids"] == frozenset({"R-001", "R-002"})


def test_E03_reset_mid_episode_starts_fresh(make_env, grader):
    env = make_env()
    env.reset(0)
    for _ in range(99):
        env.step(act("teleport"))
    env.reset(0)
    obs, reward, terminated, truncated, info = env.step(act("teleport"))
    assert obs["steps_remaining"] == MAX_STEPS - 1 and not truncated
    assert grader.calls == []  # abandoned episode is never graded


def test_E03_reset_switches_episode_completely(grader):
    episodes = [big_episode(80), with_episode_id(build_tiny_episode(), "ep_SECOND00001")]
    it = iter(episodes)
    env = AuditEnv(generate_fn=lambda seed, config=None: next(it), grader=grader)
    env.reset(1)
    cursor = env.step(list_page("ledger", None, 50))[0]["result"]["next_cursor"]
    assert cursor is not None
    obs, info = env.reset(2)
    assert obs["manifest"]["episode_id"] == "ep_SECOND00001"
    assert info["episode_id"] == "ep_SECOND00001"
    assert env.step(list_page("ledger", cursor, 50))[0]["error"]["code"] == "invalid_cursor"
    page = env.step(list_page("ledger", None, 50))[0]["result"]
    assert len(page["records"]) == 27 and page["next_cursor"] is None
    assert env.step(act("get_record", table="ledger", id="R-X0001"))[0]["error"]["code"] == (
        "unknown_id"
    )
    env.step(act("submit", findings=PAYLOAD))
    assert grader.calls[0]["episode_id"] == "ep_SECOND00001"


def test_E03_stream_identical_across_hash_seeds(tmp_path):
    """Two interpreter processes with different PYTHONHASHSEED produce the same bytes."""
    script = tmp_path / "run_stream.py"
    contract_dir = Path(__file__).resolve().parent
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(contract_dir)!r})\n"
        "from envkit import RecordingGrader, env_factory, oracle_submission, scripted_run\n"
        "from auditgym.fixtures.tiny import build_tiny_episode\n"
        "from auditgym.util import canon\n"
        "env = env_factory(RecordingGrader())('open')\n"
        "first = env.reset(5)\n"
        "rest = scripted_run(env, submit=oracle_submission(build_tiny_episode()))\n"
        "sys.stdout.write(canon.hash_obj([list(first), [list(r) for r in rest]]))\n",
        encoding="utf-8",
    )
    outs = []
    for hash_seed in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        out = subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True, env=env, check=True
        )
        outs.append(out.stdout)
    assert outs[0] == outs[1] and len(outs[0]) == 64
