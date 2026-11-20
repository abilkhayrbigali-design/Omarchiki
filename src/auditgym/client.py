"""RemoteEnv: the reset/step API over a grader-side server process (D11).

The reference answer stays in the server process; this module only ever holds what the server
sends back. It must not import grading, inject, world, verify or episode.

Real closed evaluation: the organizer starts the server (`python -m auditgym.server --mode closed
--schedule ... --grader-log /abs/path`, as a different OS user) and wires its stdin/stdout to the
agent. The agent then attaches to the channel it was given and never chooses episodes:

    env = RemoteEnv.from_streams(sys.stdin.buffer, sys.stdout.buffer)
    obs, info = env.reset()                   # next scheduled episode
    obs, reward, terminated, truncated, info = env.step({"action": ..., "args": {...}})

For training and tests RemoteEnv can also start the server itself (the agent then owns that
process, so this is not a closed evaluation):

    with RemoteEnv(mode="open") as env:
        obs, info = env.reset(seed=7)

The server's stderr is discarded by default, so no traceback or path can reach the agent.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import IO, Any

from auditgym.util import canon

_SRC_DIR = str(Path(__file__).resolve().parents[1])

PathArg = str | os.PathLike


class RemoteEnvError(RuntimeError):
    """The server refused a request or went away. `code` is a fixed protocol code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class RemoteEnv:
    def __init__(
        self,
        mode: str = "open",
        grader_log: PathArg | None = None,
        episode_dir: PathArg | Sequence[PathArg] | None = None,
        *,
        schedule: PathArg | None = None,
        time_limit: float | None = None,
        command: Sequence[str] | None = None,
        stderr: Any = subprocess.DEVNULL,
    ) -> None:
        """Start the server. `command` replaces the default `python -m auditgym.server ...`."""
        if command is None:
            command = [sys.executable, "-m", "auditgym.server", "--mode", mode]
            if grader_log is not None:
                command += ["--grader-log", os.fspath(grader_log)]
            if schedule is not None:
                command += ["--schedule", os.fspath(schedule)]
            if episode_dir is not None:
                dirs = (
                    [episode_dir]
                    if isinstance(episode_dir, (str, os.PathLike))
                    else list(episode_dir)
                )
                for d in dirs:
                    command += ["--episode-dir", os.fspath(d)]
            if time_limit is not None:
                command += ["--time-limit", repr(float(time_limit))]
        child_env = dict(os.environ)
        child_env["PYTHONPATH"] = os.pathsep.join(
            p for p in (_SRC_DIR, child_env.get("PYTHONPATH", "")) if p
        )
        proc = subprocess.Popen(
            list(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            env=child_env,
        )
        self._proc: subprocess.Popen[bytes] | None = proc
        self._reader: IO[bytes] | None = proc.stdout
        self._writer: IO[bytes] | None = proc.stdin

    @classmethod
    def from_streams(cls, reader: IO[bytes], writer: IO[bytes]) -> RemoteEnv:
        """Attach to a server channel someone else opened (e.g. the organizer's harness).

        `reader` carries the server's responses, `writer` takes requests. The streams stay open
        after close(); their owner closes them.
        """
        env = cls.__new__(cls)
        env._proc = None
        env._reader = reader
        env._writer = writer
        return env

    # ------------------------------------------------------------------ transport
    def _request(self, message: dict[str, Any]) -> dict[str, Any]:
        reader, writer = self._reader, self._writer
        if reader is None or writer is None:
            raise RemoteEnvError("closed")
        try:
            writer.write((canon.dumps(message) + "\n").encode("utf-8"))
            writer.flush()
            line = reader.readline()
        except (BrokenPipeError, OSError, ValueError) as exc:
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
    def reset(
        self,
        seed: int | None = None,
        config: dict[str, Any] | None = None,
        *,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict, dict]:
        """Next episode. With a scheduled server (closed mode) pass no seed and no config."""
        if options and options.get("config") is not None:
            if config is not None:
                raise ValueError("config given both as an argument and in options")
            config = options["config"]
        message: dict[str, Any] = {"op": "reset"}
        if seed is not None:
            message["seed"] = seed
        if config is not None:
            message["config"] = config
        r = self._request(message)
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
        reader, writer, proc = self._reader, self._writer, self._proc
        self._reader = self._writer = self._proc = None
        if writer is None:
            return
        try:
            if proc is None or proc.poll() is None:
                writer.write(b'{"op":"close"}\n')
                writer.flush()
                if reader is not None:
                    reader.readline()
        except (OSError, ValueError):
            pass
        if proc is None:
            return
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
