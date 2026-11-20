"""The grader runs in its own process; the agent gets JSON lines and nothing else (D11, D12).

Uses `tests/leak/fake_grader_server.py` (the real server entry point with a stand-in grader)
for submit paths, and `python -m auditgym.server` itself for exploration.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from envkit import (
    FORBIDDEN_KEYS,
    SECRET_REASON,
    act,
    alternative_truths,
    collect_keys,
    list_page,
    oracle_submission,
    scripted_run,
    truth_markers,
    with_truth,
)

from auditgym.client import RemoteEnv, RemoteEnvError
from auditgym.contracts.io import read_episode, write_episode
from auditgym.env import AuditEnv
from auditgym.fixtures.tiny import build_tiny_episode
from auditgym.util import canon

FAKE_SERVER = Path(__file__).resolve().parent / "fake_grader_server.py"


def _fake_server_cmd(mode: str, episode_dir: Path, log: Path) -> list[str]:
    return [
        sys.executable,
        str(FAKE_SERVER),
        "--mode",
        mode,
        "--episode-dir",
        str(episode_dir),
        "--grader-log",
        str(log),
    ]


@pytest.fixture
def episode_dir(tmp_path):
    out = tmp_path / "episode"
    write_episode(build_tiny_episode(), out)
    return out


def _remote_stream(remote, payload):
    first = remote.reset(5)
    return [list(first), [list(s) for s in scripted_run(remote, submit=payload)]]


@pytest.mark.parametrize("mode", ["open", "closed"])
def test_D11_remote_env_matches_in_process_env(tmp_path, episode_dir, mode):
    import fake_grader_server

    payload = oracle_submission(build_tiny_episode())
    log = tmp_path / "log.jsonl"
    with RemoteEnv(command=_fake_server_cmd(mode, episode_dir, log)) as remote:
        remote_stream = _remote_stream(remote, payload)
    local = AuditEnv(
        mode=mode,
        generate_fn=lambda s, c=None: read_episode(episode_dir),
        grader=fake_grader_server.grader,
    )
    first = local.reset(5)
    local_stream = [list(first), [list(s) for s in scripted_run(local, submit=payload)]]
    assert canon.dumps(remote_stream) == canon.dumps(local_stream)
    submit_reward = remote_stream[1][-2][1]
    if mode == "open":
        assert submit_reward == pytest.approx(0.4 - 0.002 * (len(remote_stream[1]) - 2))
    else:
        assert submit_reward == 0.0
        assert remote_stream[1][-2][4] == {"reward_withheld": True}
    (record,) = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    assert record["episode_id"] == "ep_TINY000001"
    assert record["error"] == SECRET_REASON and len(record["matches"]) == 4
    assert record["reward"] == pytest.approx(0.4 - 0.002 * (len(remote_stream[1]) - 2))


def test_D12_remote_stream_carries_no_truth(tmp_path, episode_dir):
    payload = oracle_submission(build_tiny_episode())
    for mode in ("open", "closed"):
        with RemoteEnv(
            command=_fake_server_cmd(mode, episode_dir, tmp_path / f"{mode}.jsonl")
        ) as remote:
            stream = _remote_stream(remote, payload)
            remote.reset(5)
            stream.append(list(remote.truncate()))
        text = canon.dumps(stream)
        found = [m for m in truth_markers(build_tiny_episode()) if m in text]
        assert not found, found
        assert str(tmp_path) not in text and str(episode_dir) not in text
        assert not collect_keys(stream) & FORBIDDEN_KEYS


def test_D12_blind_swap_across_process_boundary(tmp_path):
    """Same public files, different private truth: the closed-mode transcripts are identical."""
    base = build_tiny_episode()
    payload = oracle_submission(base)
    transcripts = set()
    for i, truth in enumerate(alternative_truths(base)):
        directory = tmp_path / f"ep{i}"
        write_episode(with_truth(base, truth), directory)
        with RemoteEnv(
            command=_fake_server_cmd("closed", directory, tmp_path / f"log{i}.jsonl")
        ) as remote:
            transcripts.add(canon.dumps(_remote_stream(remote, payload)))
    assert len(transcripts) == 1
    rewards = {
        json.loads((tmp_path / f"log{i}.jsonl").read_text(encoding="utf-8"))["reward"]
        for i in range(3)
    }
    assert len(rewards) == 3  # the grader process did see three different truths


def test_D11_default_server_entry_point(tmp_path, episode_dir):
    """`python -m auditgym.server` serves exploration exactly like the in-process env."""
    actions = [
        list_page("ledger", None, 10),
        act("get_document", doc_id="D-I4"),
        act("calculator", expression="7500000 * 1"),
        act("teleport"),
        {"weird": [1, 2, 3]},
        act("calculator", expression={1, 2}),  # not JSON: sent as an invalid action
    ]
    with RemoteEnv(mode="open", episode_dir=episode_dir) as remote:
        remote_out = [list(remote.reset(0))] + [list(remote.step(a)) for a in actions]
    local = AuditEnv(generate_fn=lambda s, c=None: read_episode(episode_dir))
    local_out = [list(local.reset(0))] + [list(local.step(a)) for a in actions]
    assert remote_out == local_out
    assert remote_out[-1][0]["error"]["code"] == "invalid_arguments"


def test_D11_remote_errors_are_fixed_codes(episode_dir):
    with RemoteEnv(mode="open", episode_dir=episode_dir) as remote:
        with pytest.raises(RemoteEnvError) as exc:
            remote.reset("seven")  # type: ignore[arg-type]
        assert exc.value.code == "invalid_request"
        obs, reward, terminated, truncated, info = remote.step(list_page("ledger"))
        assert obs["error"]["code"] == "episode_finished"  # nothing reset yet
        remote.reset(1)
        assert remote.step(list_page("fx", None, 1))[0]["error"] is None


def test_D11_server_survives_garbage_lines(tmp_path, episode_dir):
    proc = subprocess.Popen(
        [sys.executable, "-m", "auditgym.server", "--episode-dir", str(episode_dir)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    lines = [
        b"hello\n",
        b'{"op": "rm -rf"}\n',
        b'{"op": "reset", "seed": 1, "config": "x"}\n',
        b"\x00\x01\x02\n",
        b'{"op": "reset", "seed": 1}\n',
        b'{"op": "step", "action": {"action": "calculator", "args": {"expression": "2*21"}}}\n',
        b'{"op": "close"}\n',
    ]
    out, err = proc.communicate(b"".join(lines), timeout=60)
    responses = [json.loads(x) for x in out.decode("utf-8").splitlines()]
    assert proc.returncode == 0
    assert [r["ok"] for r in responses] == [False, False, False, False, True, True, True]
    assert {r["error"]["code"] for r in responses[:4]} == {"invalid_request"}
    assert responses[5]["observation"]["result"] == {"value": "42"}
    assert b"Traceback" not in out


def test_D11_closed_server_writes_log_only_where_told(tmp_path, episode_dir):
    log = tmp_path / "closed" / "grader.jsonl"
    with RemoteEnv(command=_fake_server_cmd("closed", episode_dir, log)) as remote:
        remote.reset(1)
        remote.step(act("submit", findings={"episode_id": "ep_TINY000001", "findings": []}))
        o, r, t, tr, info = remote.step(list_page("ledger"))
        assert o["error"]["code"] == "episode_finished" and r == 0.0
    assert log.exists() and len(log.read_text(encoding="utf-8").splitlines()) == 1


def test_D11_client_close_is_idempotent(episode_dir):
    remote = RemoteEnv(mode="open", episode_dir=episode_dir)
    remote.reset(1)
    remote.close()
    remote.close()
    with pytest.raises(RemoteEnvError):
        remote.step(list_page("ledger"))
