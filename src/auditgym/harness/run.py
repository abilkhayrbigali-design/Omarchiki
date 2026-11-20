"""Run policies on environments, record trajectories and grader diagnostics.

The harness plays the organizers' side: it creates the environment, owns the grader log, and
for the oracle it reads the truth of the generated episode. Wall-clock timing is measured here
(diagnostics only); the environment and the generator never read the clock.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from auditgym.contracts.constants import MAX_STEPS
from auditgym.contracts.types import Episode
from auditgym.policies.base import Policy

EPISODE_TIME_LIMIT_S = 60.0


@dataclass
class EpisodeRun:
    episode_id: str
    policy: str
    seed: int | None
    config_id: str | None
    category: str  # "clean", "errors" or "interacting"
    steps: int
    terminated: bool
    truncated: bool
    reward_returned: float
    grade: dict[str, Any] | None
    seconds: float
    trajectory: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        g = self.grade or {}
        return {
            "episode_id": self.episode_id,
            "policy": self.policy,
            "seed": self.seed,
            "config_id": self.config_id,
            "category": self.category,
            "steps": self.steps,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "reward": g.get("reward", self.reward_returned),
            "experimental_reward": g.get("experimental_reward"),
            "precision": g.get("precision"),
            "recall": g.get("recall"),
            "f1": g.get("f1"),
            "tp": g.get("tp"),
            "fp": g.get("fp"),
            "fn": g.get("fn"),
            "valid": g.get("valid"),
            "seconds": round(self.seconds, 4),
        }


def episode_category(episode: Episode) -> str:
    issues = episode.private.truth.issues
    if not issues:
        return "clean"
    scenarios = episode.private.provenance.get("scenarios") or []
    return "interacting" if scenarios else "errors"


def make_env(mode: str, grader_log: Callable[[dict[str, Any]], None]) -> Any:
    from auditgym.env import AuditEnv

    return AuditEnv(mode=mode, grader_log=grader_log)


def run_episode(
    policy: Policy,
    episode: Episode,
    *,
    mode: str = "open",
    seed: int | None = None,
    config_id: str | None = None,
    time_limit_s: float = EPISODE_TIME_LIMIT_S,
    record: bool = True,
) -> EpisodeRun:
    """Play one episode. The environment is created fresh for every episode (E03)."""
    grades: list[dict[str, Any]] = []
    env = make_env(mode, grades.append)
    episode_id = episode.public.episode_id
    trajectory: list[dict[str, Any]] = []

    started = time.perf_counter()
    obs, info = env.reset_with_episode(episode)
    policy.reset(obs, info)
    if record:
        trajectory.append(_step_line(episode_id, 0, None, obs, 0.0, False, False))

    reward, terminated, truncated, steps = 0.0, False, False, 0
    while not (terminated or truncated):
        if time.perf_counter() - started > time_limit_s:
            obs, reward, terminated, truncated, info = env.truncate()
            action = None
        else:
            action = policy.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        if record:
            trajectory.append(
                _step_line(episode_id, steps, action, obs, reward, terminated, truncated)
            )
        if steps > MAX_STEPS + 1:  # defensive: the env must end episodes by itself
            raise RuntimeError("environment did not end the episode after the step limit")
    seconds = time.perf_counter() - started

    return EpisodeRun(
        episode_id=episode_id,
        policy=getattr(policy, "name", type(policy).__name__),
        seed=seed,
        config_id=config_id,
        category=episode_category(episode),
        steps=steps,
        terminated=terminated,
        truncated=truncated,
        reward_returned=float(reward),
        grade=grades[-1] if grades else None,
        seconds=seconds,
        trajectory=trajectory,
    )


def _step_line(
    episode_id: str,
    step: int,
    action: dict[str, Any] | None,
    observation: dict[str, Any],
    reward: float,
    terminated: bool,
    truncated: bool,
) -> dict[str, Any]:
    return {
        "episode_id": episode_id,
        "step": step,
        "action": action,
        "observation": observation,
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def aggregate(runs: Iterable[EpisodeRun]) -> dict[str, Any]:
    """Mean diagnostics per policy and per (policy, category)."""
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for run in runs:
        s = run.summary()
        buckets.setdefault((s["policy"], "all"), []).append(s)
        buckets.setdefault((s["policy"], s["category"]), []).append(s)

    def mean(rows: list[dict[str, Any]], key: str) -> float | None:
        values = [r[key] for r in rows if r.get(key) is not None]
        return round(sum(values) / len(values), 4) if values else None

    out: dict[str, Any] = {}
    for (policy, category), rows in sorted(buckets.items()):
        # On clean books S07 sets P = R = 0 even for a perfect (empty) answer, so the means of
        # P and R are taken over episodes that have errors only; F1 and reward over all.
        with_errors = [r for r in rows if r["category"] != "clean"]
        out.setdefault(policy, {})[category] = {
            "episodes": len(rows),
            "reward": mean(rows, "reward"),
            "experimental_reward": mean(rows, "experimental_reward"),
            "precision": mean(with_errors, "precision"),
            "recall": mean(with_errors, "recall"),
            "f1": mean(rows, "f1"),
            "steps": mean(rows, "steps"),
            "seconds": mean(rows, "seconds"),
            "truncated": sum(1 for r in rows if r["truncated"]),
            "invalid": sum(1 for r in rows if r.get("valid") is False),
        }
    return out
