"""Server-side episode lifecycle: schedule, walk-aways and the wall-clock limit.

EpisodeHost sits between the JSON-lines protocol and one AuditEnv:

- **Who picks the episode.** With a schedule (always in closed mode) the server serves the next
  scheduled episode on every `reset`; a reset that carries `seed` or `config` is rejected with
  `invalid_request` and changes nothing. When the schedule is used up, reset answers
  `schedule_exhausted`. Without a schedule (open mode only) the client picks seed and config.
- **No free retries.** A valid reset, `close` or EOF while an episode is running ends that
  episode first: truncated, graded as an empty submission, logged with end="abandoned" (or
  end="timeout" once the time limit has passed). A scheduled episode is never served twice.
- **Time limit.** Measured with a monotonic clock from the moment the reset is answered. A step
  that arrives after the limit is not executed: the episode is ended with env.truncate() and that
  result is the answer. The clock lives only here; the environment and the generator never read
  it, so their determinism is untouched. The clock is injectable for tests.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from typing import Any

from auditgym.server.protocol import ProtocolCode, error_response, step_response
from auditgym.server.schedule import ScheduleEntry

Clock = Callable[[], float]

# The package's single wall-clock source (tests/leak/test_leak_import_boundary.py). EpisodeHost
# uses it by default; AuditEnv takes it from here when it is given a time limit and no clock.
MONOTONIC: Clock = time.monotonic


class EpisodeHost:
    def __init__(
        self,
        env: Any,
        *,
        schedule: Sequence[ScheduleEntry] | None = None,
        generate_fn: Callable[..., Any] | None = None,
        time_limit_s: float | None = None,
        clock: Clock | None = None,
    ) -> None:
        if env.mode == "closed" and schedule is None:
            raise ValueError("closed mode needs a schedule: the client may not pick episodes")
        if time_limit_s is not None and not (math.isfinite(time_limit_s) and time_limit_s > 0):
            raise ValueError("time limit must be a positive number of seconds")
        self._env = env
        self._schedule = list(schedule) if schedule is not None else None
        self._next = 0
        self._generate_fn = generate_fn
        self._limit = time_limit_s
        self._clock: Clock = clock if clock is not None else MONOTONIC
        self._started: float | None = None

    @property
    def env(self) -> Any:
        return self._env

    @property
    def remaining_episodes(self) -> int | None:
        """Scheduled episodes not yet served (None without a schedule)."""
        return None if self._schedule is None else len(self._schedule) - self._next

    # ------------------------------------------------------------------ requests
    def handle(self, request: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Answer one structurally valid request. Returns (response, close the server)."""
        op = request["op"]
        if op == "close":
            self.finish()
            return {"ok": True, "op": "close"}, True
        if op == "reset":
            return self._reset(request), False
        if op == "step":
            if self._expired():
                # Late: the action is not executed.
                return step_response("step", self._env.truncate()), False
            return step_response("step", self._env.step(request["action"])), False
        return step_response("truncate", self._env.truncate()), False

    def finish(self) -> None:
        """Close or EOF: end the running episode, if any. Idempotent."""
        if self._env.active:
            if self._expired():
                self._env.truncate()
            else:
                self._env.abandon()
        self._started = None

    # ------------------------------------------------------------------ internals
    def _expired(self) -> bool:
        if self._limit is None or self._started is None or not self._env.active:
            return False
        return self._clock() - self._started > self._limit

    def _reset(self, request: dict[str, Any]) -> dict[str, Any]:
        chooses = "seed" in request or "config" in request
        if self._schedule is not None and chooses:
            return error_response(ProtocolCode.INVALID_REQUEST)  # nothing changes
        if self._schedule is None and "seed" not in request:
            return error_response(ProtocolCode.INVALID_REQUEST)
        self.finish()
        if self._schedule is None:
            obs, info = self._env.reset(request["seed"], request.get("config"))
        else:
            if self._next >= len(self._schedule):
                return error_response(ProtocolCode.SCHEDULE_EXHAUSTED)
            entry = self._schedule[self._next]
            self._next += 1  # consumed even if it fails to load: never served twice
            obs, info = self._env.reset_with_episode(entry.load(self._generate_fn))
        self._started = self._clock() if self._limit is not None else None
        return {"ok": True, "op": "reset", "observation": obs, "info": info}
