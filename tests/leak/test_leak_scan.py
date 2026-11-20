"""No truth, issue id, seed, config, path or traceback in any observation, info or error (D12)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from envkit import (
    FORBIDDEN_KEYS,
    TruthGrader,
    act,
    collect_keys,
    list_page,
    oracle_submission,
    scripted_run,
    truth_markers,
)

import auditgym
from auditgym.contracts.errors import ERROR_MESSAGES
from auditgym.env import AuditEnv
from auditgym.fixtures.tiny import build_tiny_episode
from auditgym.util import canon

SECRET_SEED = 918273645
SECRET_CONFIG = {"config_id": "cfg-secret-55331", "note": "/srv/private/episodes"}


def _everything_seen(mode: str, tmp_path: Path) -> list:
    episode = build_tiny_episode()
    log = tmp_path / "grader_log.jsonl"
    env = AuditEnv(
        mode=mode, generate_fn=lambda s, c=None: episode, grader=TruthGrader(), grader_log=log
    )
    seen: list = []
    seen.append(env.reset(SECRET_SEED, SECRET_CONFIG))
    seen += scripted_run(env, submit=oracle_submission(episode))
    # Truncation path, with errors and an internal failure on the way.
    seen.append(env.reset(SECRET_SEED, SECRET_CONFIG))
    for i in range(100):
        action = (
            list_page("ledger", None, 50)
            if i % 4 == 0
            else act("get_document", doc_id=episode.private.truth.issues[i % 4].affected_ids[0])
            if i % 4 == 1
            else act("get_record", table="ledger", id="../private/truth.json")
            if i % 4 == 2
            else {"action": "submit"}
        )
        seen.append(env.step(action))
    seen.append(env.step(act("submit", findings=oracle_submission(episode))))
    # Timeout path.
    seen.append(env.reset(SECRET_SEED, SECRET_CONFIG))
    seen.append(env.truncate())
    seen.append(env.truncate())
    assert log.exists()  # the grade went to the grader log, not to the agent
    return seen


@pytest.mark.parametrize("mode", ["open", "closed"])
def test_D12_no_truth_fields_in_observations_or_info(tmp_path, mode):
    seen = _everything_seen(mode, tmp_path)
    keys = collect_keys(seen)
    assert not keys & FORBIDDEN_KEYS, sorted(keys & FORBIDDEN_KEYS)


@pytest.mark.parametrize("mode", ["open", "closed"])
def test_D12_no_truth_values_seed_config_or_paths(tmp_path, mode):
    text = canon.dumps(_everything_seen(mode, tmp_path))
    markers = truth_markers(build_tiny_episode()) + [
        str(SECRET_SEED),
        "cfg-secret",
        "/srv/private",
        str(tmp_path),
        os.getcwd(),
        str(Path(auditgym.__file__).resolve().parent),
        "auditgym/",
        ".py",
        ".json",
        "Traceback",
        "site-packages",
        "Error(",
    ]
    found = [m for m in markers if m in text]
    assert not found, found


@pytest.mark.parametrize("mode", ["open", "closed"])
def test_D12_errors_use_only_fixed_messages(tmp_path, mode):
    fixed = {msg for msg in ERROR_MESSAGES.values()}
    for item in _everything_seen(mode, tmp_path):
        obs = item[0]
        if obs.get("error") is not None:
            assert set(obs["error"]) == {"code", "message"}
            assert obs["error"]["message"] in fixed


@pytest.mark.parametrize("mode", ["open", "closed"])
def test_D12_info_keys_are_whitelisted(tmp_path, mode):
    allowed = {"episode_id", "mode", "env_version", "max_steps", "reward_withheld"}
    for item in _everything_seen(mode, tmp_path):
        info = item[-1]
        assert set(info) <= allowed


def test_E12_closed_mode_reward_never_reaches_agent(tmp_path):
    for item in _everything_seen("closed", tmp_path):
        if len(item) == 5:
            assert item[1] == 0.0


def test_D12_grader_exception_text_never_reaches_agent(tmp_path):
    def broken(*args, **kwargs):
        raise ValueError(f"truth at {tmp_path}/private/truth.json: iss_tiny_dup")

    for mode in ("open", "closed"):
        env = AuditEnv(mode=mode, generate_fn=lambda s, c=None: build_tiny_episode(), grader=broken)
        seen = [env.reset(1)]
        seen.append(env.step(act("submit", findings={"episode_id": "x", "findings": []})))
        env.reset(1)
        seen.append(env.truncate())
        text = canon.dumps(seen)
        assert "iss_tiny" not in text and str(tmp_path) not in text and "truth" not in text
