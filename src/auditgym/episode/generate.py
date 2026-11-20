"""Episode generator: world -> clean books -> injections -> presentation -> bundle -> self-check.

    episode = generate_episode(seed, config)
    episode, clean = generate_with_clean(seed, config)

Every random choice comes from a named child of
`root_stream(seed, stream_config_hash(cfg), GENERATOR_VERSION, attempt)`:
"plan", "world" (and its children), "distractor/<kind>", "inject/<name>", "presentation".

`stream_config_hash` is the hash of the normalized config without its `presentation` section, so
`presentation.id_salt` and `presentation.order` change IDs and row order but never the semantic
content of the episode. `secret_salt` is part of it.

Size: the published episode has a number of invoice documents inside `size.invoices`. A config
whose errors and minimum distractors cannot fit raises CapacityError (a ConfigError) before any
attempt (auditgym.world.sizing).

Self-check: every attempt runs the invariant check against the truth (auditgym.episode.selfcheck)
and, if given, `verify_fn(public)` against the truth on (type, affected_ids, amount, currency). A
mismatch retries with the next attempt number; `provenance["attempts"]` records how many attempts
were made. A retry is a generator bug; the target is zero.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from auditgym.books.posting import post_all
from auditgym.contracts.config import config_hash, normalize_config
from auditgym.contracts.types import Episode, Issue, PrivateEpisode, PublicEpisode
from auditgym.contracts.versions import GENERATOR_VERSION
from auditgym.episode.bundle import Built, bundle
from auditgym.episode.selfcheck import issue_signatures, problems
from auditgym.inject.plan import sample_plan, world_demand
from auditgym.inject.run import run_injections
from auditgym.inject.state import CandidateError
from auditgym.present.presenter import Presenter
from auditgym.util.rng import root_stream
from auditgym.world.build import build_world
from auditgym.world.model import WorldError
from auditgym.world.sizing import CapacityError, check_capacity

MAX_ATTEMPTS = 8

VerifyFn = Callable[[PublicEpisode], Iterable[Issue]]

__all__ = [
    "MAX_ATTEMPTS",
    "CapacityError",
    "GenerationError",
    "build_attempt",
    "generate_episode",
    "generate_with_clean",
    "stream_config_hash",
]


class GenerationError(RuntimeError):
    """No consistent episode within MAX_ATTEMPTS attempts."""


def stream_config_hash(cfg: dict[str, Any]) -> str:
    return config_hash({k: v for k, v in cfg.items() if k != "presentation"})


def build_attempt(seed: int, cfg: dict[str, Any], attempt: int) -> Built:
    """One deterministic generation attempt. Raises WorldError / CandidateError on failure."""
    sem = stream_config_hash(cfg)
    root = root_stream(seed, sem, GENERATOR_VERSION, attempt)
    plan = sample_plan(root.child("plan"), cfg)
    world = build_world(root, cfg, world_demand(plan))
    clean_rows = post_all(world)
    state = run_injections(root, world, clean_rows, plan)
    presenter = Presenter(root.child("presentation"), cfg["presentation"])
    return bundle(
        seed=seed,
        cfg=cfg,
        stream_config_hash=sem,
        plan=plan,
        world=world,
        clean_rows=clean_rows,
        state=state,
        presenter=presenter,
    )


def _generate(
    seed: int, config: dict[str, Any] | None, verify_fn: VerifyFn | None
) -> tuple[Episode, PublicEpisode]:
    cfg = normalize_config(config)
    check_capacity(cfg)  # fails fast, before any attempt, when size.invoices is too small
    failures: list[dict[str, Any]] = []
    for attempt in range(MAX_ATTEMPTS):
        try:
            built = build_attempt(seed, cfg, attempt)
        except (WorldError, CandidateError) as exc:
            failures.append({"attempt": attempt, "reason": f"construction: {exc.code}"})
            continue
        found = problems(built.public, built.truth, built.clean)
        lo, hi = cfg["size"]["invoices"]
        if not lo <= len(built.public.invoices) <= hi:
            found.append(
                f"{len(built.public.invoices)} invoices outside size.invoices [{lo}, {hi}]"
            )
        if found:
            failures.append({"attempt": attempt, "reason": "self-check", "problems": found[:10]})
            continue
        if verify_fn is not None:
            solved = issue_signatures(verify_fn(built.public))
            if solved != issue_signatures(built.truth.issues):
                failures.append({"attempt": attempt, "reason": "verify_fn disagrees with truth"})
                continue
        built.provenance["attempts"] = attempt + 1
        built.provenance["attempt_failures"] = failures
        episode = Episode(
            public=built.public,
            private=PrivateEpisode(truth=built.truth, provenance=built.provenance),
        )
        return episode, built.clean
    raise GenerationError(f"no consistent episode after {MAX_ATTEMPTS} attempts")


def generate_episode(
    seed: int, config: dict[str, Any] | None = None, *, verify_fn: VerifyFn | None = None
) -> Episode:
    """The episode for (seed, config). Same inputs and GENERATOR_VERSION give the same bytes."""
    episode, _ = _generate(seed, config, verify_fn)
    return episode


def generate_with_clean(
    seed: int, config: dict[str, Any] | None = None, *, verify_fn: VerifyFn | None = None
) -> tuple[Episode, PublicEpisode]:
    """The episode and the same books before error injection, with the same public IDs."""
    return _generate(seed, config, verify_fn)
