"""Episode session: the reset/step controller (E01-E03, E07, E09-E12, S03).

This is the only module of the environment that holds the reference answer and talks to the
grader. Observations are built by auditgym.env.observe from the public RecordStore alone.

Step accounting: every step() call on a running episode costs one step, errors included.
`submit` grades the findings with inspection_steps = steps before it and ends the episode with
terminated=True. Any action that is a JSON object with "action": "submit" ends the episode, even
when its arguments are malformed (missing `args`, extra keys, too deep): it is then graded as a
structurally invalid submission and the observation carries `invalid_arguments`. If step
MAX_STEPS is not a submit, the episode ends with truncated=True and is graded as an empty
submission (inspection_steps = MAX_STEPS). `truncate()` (wall-clock limit reached, decided by the
caller) and `abandon()` (the caller walked away: reset, close, disconnect) end it the same way
without costing a step. After the end, step() returns `episode_finished`, reward 0.0, and changes
nothing.

reset: `reset(seed=None, config=None, *, options=None)` in the Gymnasium style. The new episode
is generated and checked before the running one is touched, so a failed reset raises and leaves
the running episode as it was. A successful reset during a running episode ends that episode as
abandoned; it is graded and logged when a grader log is configured.

Modes: `open` returns the terminal reward to the caller; `closed` returns 0.0 and
info["reward_withheld"] = True. If `grader_log` is set (a path or a callable), the full grade
record is written there in both modes; it never reaches an observation or info.

Time: by default the environment never reads a clock; the caller (server or harness) enforces
the wall-clock limit through truncate(). For in-process use without a server,
`AuditEnv(time_limit_s=60)` makes the environment enforce it itself, with the same rules as the
server: the clock starts when reset returns; a step() that arrives after the limit is not
executed and costs no step, the episode ends with truncated=True, is graded as an empty
submission and logged with end="timeout". The limit is checked only when step() or reset() is
called, never in between. Before the limit the clock changes nothing, so transitions and rewards
stay a function of seed, config and actions. `clock` is injectable for tests; the default is the
server host's monotonic clock, the only place of the package that imports `time`.
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
from auditgym.env.actions import ActionError, parse_action, plain_json
from auditgym.env.store import RecordStore
from auditgym.util import canon

MODES = ("open", "closed")
RESET_OPTIONS = frozenset({"config"})

GenerateFn = Callable[..., Episode]
GraderLog = str | os.PathLike | Callable[[dict[str, Any]], None]
Clock = Callable[[], float]
StepResult = tuple[dict[str, Any], float, bool, bool, dict[str, Any]]


def _default_generate() -> GenerateFn:
    from auditgym.episode.generate import generate_episode

    return generate_episode


def _default_grader() -> GraderFn:
    from auditgym.grading import grade_submission

    return grade_submission


def _default_clock() -> Clock:
    # Looked up at first use: this module never imports `time` itself.
    from auditgym.server import host

    return host.MONOTONIC


def _is_submit(raw: Any) -> bool:
    """A JSON object whose "action" is "submit", whatever else is wrong with it."""
    if not isinstance(raw, dict):
        return False
    name = dict.get(raw, "action")
    return isinstance(name, str) and str(name) == "submit"


def _malformed_submit_payload(raw: dict[Any, Any]) -> Any:
    """What the grader gets for a submit action that does not match the action schema.

    The raw `args` value (None when it is missing), so the grader records why it is invalid. One
    exception: `args` that are themselves a findings object (`{"episode_id", "findings"}`, sent
    without the `findings` wrapper) are replaced by None, so that a malformed action can never be
    graded as a valid submission.
    """
    if not dict.__contains__(raw, "args"):
        return None
    args = dict.__getitem__(raw, "args")
    try:
        args = plain_json(args)
    except (ValueError, RecursionError):
        return args  # too deep or not JSON: the grader rejects it on its own
    if isinstance(args, dict) and set(args) == {"episode_id", "findings"}:
        return None
    return args


class AuditEnv:
    """reset/step environment over one generated episode at a time."""

    def __init__(
        self,
        mode: str = "open",
        generate_fn: GenerateFn | None = None,
        grader: GraderFn | None = None,
        grader_log: GraderLog | None = None,
        *,
        time_limit_s: float | None = None,
        clock: Clock | None = None,
    ) -> None:
        if mode not in MODES:
            raise ValueError("mode must be 'open' or 'closed'")
        if time_limit_s is not None:
            if isinstance(time_limit_s, bool) or not isinstance(time_limit_s, int | float):
                raise TypeError("time_limit_s must be a number of seconds or None")
            if not (math.isfinite(time_limit_s) and time_limit_s > 0):
                raise ValueError("time_limit_s must be a positive number of seconds")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable or None")
        self._mode = mode
        self._generate_fn = generate_fn
        self._grader = grader
        self._grader_log = grader_log
        self._time_limit_s = time_limit_s
        self._clock = clock
        self._started: float | None = None  # clock reading when the running episode began
        self._unlogged: list[dict[str, Any]] = []  # grade records when no grader_log is set
        self._last_seed: int | None = None
        self._clear()

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def active(self) -> bool:
        """True while an episode is running: reset done, not yet ended."""
        return self._active

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

    def _now(self) -> float:
        if self._clock is None:
            self._clock = _default_clock()
        return self._clock()

    def _expired(self) -> bool:
        """The running episode is past its time limit. Reads no clock when no limit is set."""
        if self._time_limit_s is None or self._started is None or not self._active:
            return False
        return self._now() - self._started > self._time_limit_s

    # ------------------------------------------------------------------ reset
    def reset(
        self,
        seed: int | None = None,
        config: dict[str, Any] | None = None,
        *,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict, dict]:
        """Start a new episode from `seed` and `config` (or `options["config"]`).

        `seed=None` continues a deterministic sequence: 0 on the first reset, then the previous
        seed + 1. Raises TypeError / ValueError on bad arguments and whatever the generator raises
        (ConfigError for a bad config); in every failure case the running episode is untouched.
        """
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise TypeError("seed must be an integer or None")
        if options is not None:
            if not isinstance(options, dict):
                raise TypeError("options must be a dict or None")
            if not set(options) <= RESET_OPTIONS:
                raise ValueError("reset options may only contain 'config'")
            if options.get("config") is not None:
                if config is not None:
                    raise ValueError("config given both as an argument and in options")
                config = options["config"]
        if config is not None and not isinstance(config, dict):
            raise TypeError("config must be a dict or None")
        if seed is None:
            seed = 0 if self._last_seed is None else self._last_seed + 1
        generate = self._generate_fn or _default_generate()
        episode = generate(seed, config)
        result = self._start(episode)
        self._last_seed = seed
        return result

    def reset_with_episode(self, episode: Episode) -> tuple[dict, dict]:
        """Start the given episode (tests, fixtures, episodes read from disk)."""
        return self._start(episode)

    def _start(self, episode: Episode) -> tuple[dict, dict]:
        # Build everything the new episode needs first: if anything fails, the running episode
        # is left exactly as it was.
        store = RecordStore(episode.public)
        truth = episode.private.truth
        if not isinstance(truth, Truth):
            raise TypeError("episode.private.truth must be a Truth")
        obs = observe.first_observation(store, MAX_STEPS)
        if self._active and self._grader_log is not None:
            self._end_early("timeout" if self._expired() else "abandoned")
        self._clear()
        self._store = store
        self._truth = truth
        self._active = True
        # The limit runs from the moment reset returns: generation time is not counted.
        self._started = self._now() if self._time_limit_s is not None else None
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
        if self._expired():
            return self.truncate()  # late: the action is not executed and costs no step
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
            if _is_submit(raw):
                return self._submit(
                    _malformed_submit_payload(raw), None, ErrorCode.INVALID_ARGUMENTS
                )
            return self._after_exploration(None, err.code, [])
        if action.name == "submit":
            payload = action.args["findings"]
            return self._submit(payload, observe.submit_result(payload), None)
        result, error, ids = observe.run_tool(self._store, action)
        return self._after_exploration(result, error, ids)

    def _submit(
        self, payload: Any, result: dict[str, Any] | None, action_error: ErrorCode | None
    ) -> StepResult:
        reward, info, failed = self._end(
            payload,
            inspection_steps=self._steps_used - 1,
            end="submit",
            truncated=False,
            action_error=action_error,
        )
        obs = self._terminal_observation(result, action_error, failed)
        return obs, reward, True, False, info

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

        Called by the server or the harness when the wall-clock limit is reached, and by step()
        itself when the environment was given `time_limit_s`. Costs no step.
        """
        return self._end_early("timeout")

    def abandon(self) -> StepResult:
        """End the running episode because the caller left it (reset, close, disconnect).

        Same as truncate() (truncated=True, empty submission, no step spent), logged with
        end="abandoned". The episode is always graded, so a walk-away is never a free retry.
        """
        return self._end_early("abandoned")

    def _end_early(self, end: str) -> StepResult:
        if not self._active:
            return self._finished()
        try:
            reward, info, failed = self._end(
                self._empty_submission(),
                inspection_steps=self._steps_used,
                end=end,
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
        self,
        payload: Any,
        *,
        inspection_steps: int,
        end: str,
        truncated: bool,
        action_error: ErrorCode | None = None,
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
        if action_error is not None:
            record["action_error"] = action_error.value
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
