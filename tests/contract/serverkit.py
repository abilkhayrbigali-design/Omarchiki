"""Helpers for in-process server tests: scripted stdin with arrival times, a fake clock.

Imported by the contract tests as `serverkit`. Owned by WP3.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from envkit import RecordingGrader, with_episode_id

from auditgym.contracts.io import write_episode
from auditgym.fixtures.tiny import build_tiny_episode
from auditgym.server.app import main


class FakeClock:
    """A monotonic clock the test moves by hand."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now
        self.reads = 0

    def __call__(self) -> float:
        self.reads += 1
        return self.now


class TimedInput:
    """stdin whose request lines arrive at given clock times.

    `requests` is a list of (seconds after the start or None, request); a request is a dict
    (sent as JSON) or raw bytes. Reading a line first moves the clock to its arrival time.
    """

    def __init__(self, requests: list[tuple[float | None, Any]], clock: FakeClock | None) -> None:
        self._items = list(requests)
        self._clock = clock
        self._start = clock.now if clock is not None else 0.0

    def readline(self, size: int = -1) -> bytes:
        if not self._items:
            return b""
        at, request = self._items.pop(0)
        if at is not None and self._clock is not None:
            self._clock.now = self._start + at
        if isinstance(request, bytes):
            return request
        return (json.dumps(request) + "\n").encode("utf-8")


def line(obj: Any) -> tuple[None, Any]:
    return (None, obj)


def episode_ids(n: int, prefix: str = "ep_SCHED") -> list[str]:
    return [f"{prefix}{i:06d}" for i in range(n)]


def write_episode_at(directory: Path, episode_id: str) -> Path:
    """The tiny fixture with `episode_id`, written to `directory` (private files included)."""
    write_episode(with_episode_id(build_tiny_episode(), episode_id), directory)
    return directory


def write_episodes(root: Path, ids: list[str]) -> list[Path]:
    """Tiny-fixture episodes with the given ids under `root`, one directory each, in id order."""
    return [write_episode_at(root / f"e{i:03d}", eid) for i, eid in enumerate(ids)]


def run_server(
    argv: list[str],
    requests: list[tuple[float | None, Any]],
    *,
    grader: Any = None,
    generate_fn: Any = None,
    clock: FakeClock | None = None,
) -> tuple[int, list[dict[str, Any]], bytes]:
    out = io.BytesIO()
    code = main(
        argv,
        grader=grader if grader is not None else RecordingGrader(),
        generate_fn=generate_fn,
        stdin=TimedInput(requests, clock),
        stdout=out,
        clock=clock,
    )
    raw = out.getvalue()
    return code, [json.loads(x) for x in raw.decode("utf-8").splitlines()], raw


def read_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
