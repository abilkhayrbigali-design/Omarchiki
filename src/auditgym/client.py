"""RemoteEnv: the reset/step API over a grader-side server process (D11).

The client starts `python -m auditgym.server` as a subprocess and talks JSON lines to it. The
reference answer stays in the server process; this module only ever holds what the server
sends back. It must not import grading, inject, world, verify or episode.

    with RemoteEnv(mode="closed", grader_log="runs/grader_log.jsonl") as env:
        obs, info = env.reset(seed=7)
        obs, reward, terminated, truncated, info = env.step({"action": ..., "args": {...}})

The server's stderr is discarded by default, so no traceback or path can reach the agent.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from auditgym.util import canon

_SRC_DIR = str(Path(__file__).resolve().parents[1])


class RemoteEnvError(RuntimeError):
    """The server refused a request or went away. `code` is a fixed protocol code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class RemoteEnv:
    def __init__(
        self,
        mode: str = "open",
        grader_log: str | os.PathLike | None = None,
        episode_dir: str | os.PathLike | None = None,
        *,
        command: Sequence[str] | None = None,
        stderr: Any = subprocess.DEVNULL,
    ) -> None:
        """Start the server. `command` replaces the default `python -m auditgym.server ...`."""
        if command is None:
            command = [sys.executable, "-m", "auditgym.server", "--mode", mode]
            if grader_log is not None:
                command += ["--grader-log", os.fspath(grader_log)]
            if episode_dir is not None:
                command += ["--episode-dir", os.fspath(episode_dir)]
        child_env = dict(os.environ)
        child_env["PYTHONPATH"] = os.pathsep.join(
            p for p in (_SRC_DIR, child_env.get("PYTHONPATH", "")) if p
        )
        self._proc: subprocess.Popen[bytes] | None = subprocess.Popen(
            list(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            env=child_env,
        )

    # ------------------------------------------------------------------ transport
    def _request(self, message: dict[str, Any]) -> dict[str, Any]:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise RemoteEnvError("closed")
        try:
            proc.stdin.write((canon.dumps(message) + "\n").encode("utf-8"))
            proc.stdin.flush()
            line = proc.stdout.readline()
        except (BrokenPipeError, OSError) as exc:
            raise RemoteEnvError("server_gone") from exc
        if not line:
            raise RemoteEnvError("server_gone")
        response = canon.loads(line.decode("utf-8"))
        if not response.get("ok"):
            raise RemoteEnvError(str(response.get("error", {}).get("code", "internal_error")))
        return response

    @staticmethod
    def _step_tuple(r: dict[str, Any]) -> tuple[dict, float, bool, bool, dict]:
        return r["observation"], r["reward"], r["terminated"], r["truncated"], r["info"]

    # ------------------------------------------------------------------ API
    def reset(self, seed: int, config: dict[str, Any] | None = None) -> tuple[dict, dict]:
        r = self._request({"op": "reset", "seed": seed, "config": config})
        return r["observation"], r["info"]

    def step(self, action: Any) -> tuple[dict, float, bool, bool, dict]:
        try:
            canon.dumps(action)
        except (TypeError, ValueError):
            # Not JSON: the server sees an invalid action, exactly like the in-process env.
            action = None
        return self._step_tuple(self._request({"op": "step", "action": action}))

    def truncate(self) -> tuple[dict, float, bool, bool, dict]:
        return self._step_tuple(self._request({"op": "truncate"}))

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin is not None:
                proc.stdin.write(b'{"op":"close"}\n')
                proc.stdin.flush()
                if proc.stdout is not None:
                    proc.stdout.readline()
        except OSError:
            pass
        for stream in (proc.stdin, proc.stdout):
            if stream is not None:
                stream.close()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    def __enter__(self) -> RemoteEnv:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
