"""The grader runs in its own process; the agent gets JSON lines and nothing else (D11, D12).

Uses `tests/leak/fake_grader_server.py` (the real server entry point with a stand-in grader)
for submit paths, and `python -m auditgym.server` itself for exploration.
"""

from __future__ import annotations

import json
import os
import signal
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
SRC_DIR = str(Path(__file__).resolve().parents[2] / "src")


def _fake_server_cmd(mode: str, episode_dirs: list[Path], log: Path) -> list[str]:
    cmd = [sys.executable, str(FAKE_SERVER), "--mode", mode, "--grader-log", str(log)]
    for d in episode_dirs:
        cmd += ["--episode-dir", str(d)]
    return cmd


def _read_log(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def episode_dir(tmp_path):
    out = tmp_path / "episode"
    write_episode(build_tiny_episode(), out)
    return out


def _remote_stream(remote, payload):
    first = remote.reset()
    return [list(first), [list(s) for s in scripted_run(remote, submit=payload)]]


@pytest.mark.parametrize("mode", ["open", "closed"])
def test_D11_remote_env_matches_in_process_env(tmp_path, episode_dir, mode):
    import fake_grader_server

    payload = oracle_submission(build_tiny_episode())
    log = tmp_path / "log.jsonl"
    with RemoteEnv(command=_fake_server_cmd(mode, [episode_dir], log)) as remote:
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
    (record,) = _read_log(log)
    assert record["episode_id"] == "ep_TINY000001"
    assert record["error"] == SECRET_REASON and len(record["matches"]) == 4
    assert record["reward"] == pytest.approx(0.4 - 0.002 * (len(remote_stream[1]) - 2))


def test_D12_remote_stream_carries_no_truth(tmp_path, episode_dir):
    payload = oracle_submission(build_tiny_episode())
    for mode in ("open", "closed"):
        log = tmp_path / f"{mode}.jsonl"
        with RemoteEnv(command=_fake_server_cmd(mode, [episode_dir, episode_dir], log)) as remote:
            stream = _remote_stream(remote, payload)
            remote.reset()
            stream.append(list(remote.truncate()))
            with pytest.raises(RemoteEnvError) as exc:
                remote.reset()
            stream.append(exc.value.code)
        text = canon.dumps(stream)
        found = [m for m in truth_markers(build_tiny_episode()) if m in text]
        assert not found, found
        assert str(tmp_path) not in text and str(episode_dir) not in text
        assert not collect_keys(stream) & FORBIDDEN_KEYS
        assert stream[-1] == "schedule_exhausted"


def test_D12_blind_swap_across_process_boundary(tmp_path):
    """Same public files, different private truth: the closed-mode transcripts are identical."""
    base = build_tiny_episode()
    payload = oracle_submission(base)
    transcripts = set()
    for i, truth in enumerate(alternative_truths(base)):
        directory = tmp_path / f"ep{i}"
        write_episode(with_truth(base, truth), directory)
        with RemoteEnv(
            command=_fake_server_cmd("closed", [directory], tmp_path / f"log{i}.jsonl")
        ) as remote:
            transcripts.add(canon.dumps(_remote_stream(remote, payload)))
    assert len(transcripts) == 1
    rewards = {_read_log(tmp_path / f"log{i}.jsonl")[0]["reward"] for i in range(3)}
    assert len(rewards) == 3  # the grader process did see three different truths


def test_D11_default_server_entry_point(tmp_path, episode_dir):
    """`python -m auditgym.server` serves exploration exactly like the in-process env."""
    actions = [
        list_page("ledger", None, 10),
        act("get_document", doc_id="D-I4"),
        act("calculator", expression="7500000 * 1"),
        act("calculator", expression="+.5 * 2"),
        act("teleport"),
        {"weird": [1, 2, 3]},
        act("calculator", expression={1, 2}),  # not JSON: sent as an invalid action
    ]
    with RemoteEnv(mode="open", episode_dir=episode_dir) as remote:
        remote_out = [list(remote.reset())] + [list(remote.step(a)) for a in actions]
    local = AuditEnv(generate_fn=lambda s, c=None: read_episode(episode_dir))
    local_out = [list(local.reset(0))] + [list(local.step(a)) for a in actions]
    assert remote_out == local_out
    assert remote_out[4][0]["result"] == {"value": "1"}
    assert remote_out[-1][0]["error"]["code"] == "invalid_arguments"


def test_D11_remote_errors_are_fixed_codes(episode_dir):
    with RemoteEnv(mode="open", episode_dir=episode_dir) as remote:
        with pytest.raises(RemoteEnvError) as exc:
            remote.reset("seven")  # type: ignore[arg-type]
        assert exc.value.code == "invalid_request"
        with pytest.raises(RemoteEnvError) as exc:
            remote.reset(7)  # the server serves a schedule: the client may not pick
        assert exc.value.code == "invalid_request"
        obs, reward, terminated, truncated, info = remote.step(list_page("ledger"))
        assert obs["error"]["code"] == "episode_finished"  # nothing reset yet
        remote.reset()
        assert remote.step(list_page("fx", None, 1))[0]["error"] is None
        with pytest.raises(RemoteEnvError) as exc:
            remote.reset()
        assert exc.value.code == "schedule_exhausted"


def test_D11_open_server_without_schedule_takes_client_seed(tmp_path):
    """Training: open mode, the client picks seed and config (stand-in generator)."""
    with RemoteEnv(mode="open") as remote:
        with pytest.raises(RemoteEnvError) as exc:
            remote.reset()  # no schedule: a seed is required
        assert exc.value.code == "invalid_request"
        with pytest.raises(RemoteEnvError) as exc:
            remote.reset(1, {"no_such_key": 1})
        assert exc.value.code == "invalid_config"


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
        b'{"op": "reset"}\n',
        b'{"op": "step", "action": {"action": "calculator", "args": {"expression": "2*21"}}}\n',
        b'{"op": "close"}\n',
    ]
    out, err = proc.communicate(b"".join(lines), timeout=60)
    responses = [json.loads(x) for x in out.decode("utf-8").splitlines()]
    assert proc.returncode == 0
    assert [r["ok"] for r in responses] == [False, False, False, False, False, True, True, True]
    assert {r["error"]["code"] for r in responses[:5]} == {"invalid_request"}
    assert responses[6]["observation"]["result"] == {"value": "42"}
    assert b"Traceback" not in out


def test_D11_closed_server_writes_log_only_where_told(tmp_path, episode_dir):
    log = tmp_path / "closed" / "grader.jsonl"
    with RemoteEnv(command=_fake_server_cmd("closed", [episode_dir], log)) as remote:
        remote.reset()
        remote.step(act("submit", findings={"episode_id": "ep_TINY000001", "findings": []}))
        o, r, t, tr, info = remote.step(list_page("ledger"))
        assert o["error"]["code"] == "episode_finished" and r == 0.0
    assert log.exists() and len(log.read_text(encoding="utf-8").splitlines()) == 1


def _server_proc(args: list[str], cwd: Path | None = None) -> subprocess.Popen:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (SRC_DIR, env.get("PYTHONPATH", "")) if p)
    return subprocess.Popen(
        [sys.executable, "-m", "auditgym.server", *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=env,
    )


@pytest.mark.parametrize(
    "args",
    [
        ["--mode", "closed", "--episode-dir", "{ep}"],
        ["--mode", "closed", "--episode-dir", "{ep}", "--grader-log", "runs/grader_log.jsonl"],
        ["--mode", "closed", "--grader-log", "{log}"],
    ],
)
def test_D11_closed_server_process_refuses_to_start(tmp_path, episode_dir, args):
    argv = [a.format(ep=episode_dir, log=tmp_path / "log.jsonl") for a in args]
    proc = _server_proc(argv, cwd=tmp_path)
    out, err = proc.communicate(b'{"op": "reset"}\n', timeout=60)
    assert proc.returncode == 2 and out == b""
    assert b"closed mode requires" in err
    assert not (tmp_path / "runs").exists()


def test_D11_cli_serve_passes_options_after_double_dash(tmp_path, episode_dir):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (SRC_DIR, env.get("PYTHONPATH", "")) if p)
    proc = subprocess.run(
        [sys.executable, "-m", "auditgym", "serve", "--", "--mode", "closed"],
        input=b'{"op": "reset"}\n',
        capture_output=True,
        cwd=tmp_path,
        env=env,
        timeout=60,
    )
    assert proc.returncode == 2 and proc.stdout == b""
    assert b"closed mode requires" in proc.stderr


def test_D11_organizer_started_server_agent_attaches_to_streams(tmp_path, episode_dir):
    """The real closed run: the organizer starts the server, the agent only gets the channel."""
    log = tmp_path / "organizer" / "grader_log.jsonl"
    proc = _server_proc(
        ["--mode", "closed", "--grader-log", str(log), "--episode-dir", str(episode_dir)]
    )
    env = RemoteEnv.from_streams(proc.stdout, proc.stdin)
    obs, info = env.reset()
    assert info["episode_id"] == "ep_TINY000001" and "seed" not in info
    env.step(list_page("ledger", None, 3))
    _, reward, terminated, _, info = env.step(
        act("submit", findings=oracle_submission(build_tiny_episode()))
    )
    assert terminated and reward == 0.0 and info == {"reward_withheld": True}
    with pytest.raises(RemoteEnvError) as exc:
        env.reset()
    assert exc.value.code == "schedule_exhausted"
    env.close()
    proc.stdin.close()
    assert proc.wait(timeout=30) == 0
    proc.stdout.close()
    proc.stderr.close()
    (record,) = _read_log(log)
    assert record["end"] == "submit" and record["valid"] is True and record["reward"] > 0.9


def test_D11_sigterm_mid_episode_is_logged(tmp_path, episode_dir):
    log = tmp_path / "grader_log.jsonl"
    proc = _server_proc(
        ["--mode", "closed", "--grader-log", str(log), "--episode-dir", str(episode_dir)]
    )
    env = RemoteEnv.from_streams(proc.stdout, proc.stdin)
    env.reset()
    env.step(list_page("ledger", None, 1))
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=30)
    for stream in (proc.stdin, proc.stdout, proc.stderr):
        stream.close()
    (record,) = _read_log(log)
    assert record["end"] == "abandoned" and record["step_count"] == 1


def test_D11_client_close_is_idempotent(episode_dir):
    remote = RemoteEnv(mode="open", episode_dir=episode_dir)
    remote.reset()
    remote.close()
    remote.close()
    with pytest.raises(RemoteEnvError):
        remote.step(list_page("ledger"))
