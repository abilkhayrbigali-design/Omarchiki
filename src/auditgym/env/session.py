"""Episode session: the reset/step controller (E01-E03, E07, E09-E12, S03).

This is the only module of the environment that holds the reference answer and talks to the
grader. Observations are built by auditgym.env.observe from the public RecordStore alone.

Step accounting: every step() call on a running episode costs one step, errors included.
`submit` grades the findings with inspection_steps = steps before it and ends the episode with
terminated=True. If step MAX_STEPS is not a submit, the episode ends with truncated=True and is
graded as an empty submission (inspection_steps = MAX_STEPS). `truncate()` (wall-clock limit
reached, decided by the harness) ends it the same way without costing a step. After the end,
step() returns `episode_finished`, reward 0.0, and changes nothing.

Modes: `open` returns the terminal reward to the caller; `closed` returns 0.0 and
info["reward_withheld"] = True. If `grader_log` is set (a path or a callable), the full grade
record is written there in both modes; it never reaches an observation or info.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from auditgym.contracts.constants import MAX_STEPS
from auditgym.contracts.errors import ErrorCode
from auditgym.contracts.grading_api import GraderFn
from auditgym.contracts.types import Episode, Truth
from auditgym.contracts.versions import ENV_VERSION
from auditgym.env import observe
from auditgym.env.actions import ActionError, parse_action
from auditgym.env.store import RecordStore
from auditgym.util import canon

MODES = ("open", "closed")

GenerateFn = Callable[..., Episode]
GraderLog = str | os.PathLike | Callable[[dict[str, Any]], None]
StepResult = tuple[dict[str, Any], float, bool, bool, dict[str, Any]]


def _default_generate() -> GenerateFn:
    from auditgym.episode.generate import generate_episode

    return generate_episode


def _default_grader() -> GraderFn:
    from auditgym.grading import grade_submission

    return grade_submission


class AuditEnv:
    """reset/step environment over one generated episode at a time."""

    def __init__(
        self,
        mode: str = "open",
        generate_fn: GenerateFn | None = None,
        grader: GraderFn | None = None,
        grader_log: GraderLog | None = None,
    ) -> None:
        if mode not in MODES:
            raise ValueError("mode must be 'open' or 'closed'")
        self._mode = mode
        self._generate_fn = generate_fn
        self._grader = grader
        self._grader_log = grader_log
        self._unlogged: list[dict[str, Any]] = []  # grade records when no grader_log is set
        self._clear()

    @property
    def mode(self) -> str:
        return self._mode

    # ------------------------------------------------------------------ state
    def _clear(self) -> None:
        self._store: RecordStore | None = None
        self._truth: Truth | None = None
        self._active = False
        self._steps_used = 0
        self._viewed: set[str] = set()
        self._terminated = False
        self._truncated = False

    def _remaining(self) -> int:
        return MAX_STEPS - self._steps_used

    # ------------------------------------------------------------------ reset
    def reset(self, seed: int, config: dict[str, Any] | None = None) -> tuple[dict, dict]:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if config is not None and not isinstance(config, dict):
            raise TypeError("config must be a dict or None")
        self._clear()
        generate = self._generate_fn or _default_generate()
        episode = generate(seed, config)
        return self.reset_with_episode(episode)

    def reset_with_episode(self, episode: Episode) -> tuple[dict, dict]:
        self._clear()
        store = RecordStore(episode.public)
        self._store = store
        self._truth = episode.private.truth
        self._active = True
        obs = observe.first_observation(store, self._remaining())
        info = {
            "episode_id": store.episode_id,
            "mode": self._mode,
            "env_version": ENV_VERSION,
            "max_steps": MAX_STEPS,
        }
        return obs, info

    # ------------------------------------------------------------------ step
    def step(self, action: Any) -> StepResult:
        if not self._active:
            return self._finished()
        self._steps_used += 1
        try:
            return self._step(action)
        except Exception:
            # Fail closed: a fixed code, no traceback. The step is spent; the episode goes on
            # unless it was the last step.
            if not self._active:
                obs = observe.step_observation(None, ErrorCode.INTERNAL_ERROR, self._remaining())
                return obs, 0.0, self._terminated, self._truncated, {}
            return self._after_exploration(None, ErrorCode.INTERNAL_ERROR, [])

    def _step(self, raw: Any) -> StepResult:
        try:
            action = parse_action(raw)
        except ActionError as err:
            return self._after_exploration(None, err.code, [])
        if action.name == "submit":
            payload = action.args["findings"]
            result = observe.submit_result(payload)
            reward, info, failed = self._end(
                payload, inspection_steps=self._steps_used - 1, end="submit", truncated=False
            )
            obs = self._terminal_observation(result, None, failed)
            return obs, reward, True, False, info
        result, error, ids = observe.run_tool(self._store, action)
        return self._after_exploration(result, error, ids)

    def _after_exploration(
        self, result: dict[str, Any] | None, error: ErrorCode | None, ids: list[str]
    ) -> StepResult:
        if error is None:
            self._viewed.update(ids)
        if self._steps_used >= MAX_STEPS:
            reward, info, failed = self._end(
                self._empty_submission(),
                inspection_steps=self._steps_used,
                end="step_limit",
                truncated=True,
            )
            obs = self._terminal_observation(result, error, failed)
            return obs, reward, False, True, info
        obs = observe.step_observation(result, error, self._remaining())
        return obs, 0.0, False, False, {}

    def truncate(self) -> StepResult:
        """End the running episode as a timeout: truncated=True, graded as an empty submission.

        Called by the harness when the wall-clock limit is reached; the environment itself never
        reads the clock. Costs no step.
        """
        if not self._active:
            return self._finished()
        try:
            reward, info, failed = self._end(
                self._empty_submission(),
                inspection_steps=self._steps_used,
                end="timeout",
                truncated=True,
            )
        except Exception:
            obs = observe.step_observation(None, ErrorCode.INTERNAL_ERROR, self._remaining())
            return obs, 0.0, False, True, {}
        obs = self._terminal_observation(None, None, failed)
        return obs, reward, False, True, info

    def _terminal_observation(
        self, result: dict[str, Any] | None, error: ErrorCode | None, grader_failed: bool
    ) -> dict[str, Any]:
        # A grader failure is reported only in open mode; in closed mode the observation must not
        # depend on anything that happens inside the grader.
        if grader_failed and self._mode == "open":
            return observe.step_observation(None, ErrorCode.INTERNAL_ERROR, self._remaining())
        return observe.step_observation(result, error, self._remaining())

    def _finished(self) -> StepResult:
        remaining = self._remaining() if self._store is not None else 0
        obs = observe.step_observation(None, ErrorCode.EPISODE_FINISHED, remaining)
        never_started = self._store is None
        return obs, 0.0, self._terminated or never_started, self._truncated, {}

    # ------------------------------------------------------------------ grading
    def _empty_submission(self) -> dict[str, Any]:
        return {"episode_id": self._store.episode_id, "findings": []}

    def _end(
        self, payload: Any, *, inspection_steps: int, end: str, truncated: bool
    ) -> tuple[float, dict[str, Any], bool]:
        """Close the episode, grade it, log the grade.

        Returns (reward for the caller, info, grader failed).
        """
        self._active = False
        self._terminated = not truncated
        self._truncated = truncated
        record: dict[str, Any] = {
            "episode_id": self._store.episode_id,
            "step_count": self._steps_used,
            "end": end,
            "env_version": ENV_VERSION,
            "mode": self._mode,
        }
        reward = 0.0
        failed = False
        try:
            grader = self._grader or _default_grader()
            result = grader(
                payload,
                self._truth,
                episode_id=self._store.episode_id,
                inspection_steps=inspection_steps,
                viewed_ids=frozenset(self._viewed),
            )
            record.update(result.to_log())
            reward = float(result.reward)
            if not math.isfinite(reward):
                reward = 0.0
            reward = min(1.0, max(-1.0, reward))
        except Exception:
            failed = True
            reward = 0.0
            record["grader_failure"] = True
        self._write_log(record)
        if self._mode == "closed":
            return 0.0, {"reward_withheld": True}, failed
        return reward, {"reward_withheld": False}, failed

    def _write_log(self, record: dict[str, Any]) -> None:
        target = self._grader_log
        if target is None:
            self._unlogged.append(record)
            return
        if callable(target):
            target(record)
            return
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(canon.dumps(record) + "\n")
