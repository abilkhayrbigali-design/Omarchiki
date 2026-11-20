"""Helpers for the generator tests: configs, cached generation, replay of provenance changes."""

from __future__ import annotations

import copy
import functools
import json
from pathlib import Path
from typing import Any

from auditgym.contracts.config import normalize_config
from auditgym.contracts.constants import DISTRACTOR_KINDS, INTERACTION_SCENARIOS, ISSUE_TYPES
from auditgym.contracts.types import Episode, PublicEpisode
from auditgym.episode.generate import generate_with_clean

REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "configs"
SHIPPED = ("train", "dev", "smoke", "clean", "interacting")


def shipped_config(name: str) -> dict[str, Any]:
    raw = json.loads((CONFIG_DIR / f"{name}.v1.json").read_text(encoding="utf-8"))
    return normalize_config(raw)


def only(
    issue_type: str | None = None, scenario: str | None = None, **extra: Any
) -> dict[str, Any]:
    """A config with exactly one error of `issue_type` or one `scenario`, nothing else."""
    raw: dict[str, Any] = {
        "config_id": f"only-{issue_type or scenario}",
        "errors": {t: (1 if t == issue_type else 0) for t in ISSUE_TYPES},
        "interactions": {s: (1 if s == scenario else 0) for s in INTERACTION_SCENARIOS},
    }
    raw.update(extra)
    return normalize_config(raw)


def all_distractors(config_id: str = "all-distractors") -> dict[str, Any]:
    return normalize_config(
        {"config_id": config_id, "distractors": {k: [1, 2] for k in DISTRACTOR_KINDS}}
    )


@functools.lru_cache(maxsize=256)
def _cached(seed: int, cfg_json: str) -> tuple[Episode, PublicEpisode]:
    return generate_with_clean(seed, json.loads(cfg_json))


def gen(seed: int, cfg: dict[str, Any]) -> tuple[Episode, PublicEpisode]:
    """Cached generate_with_clean. Records are immutable, so sharing them is safe."""
    return _cached(seed, json.dumps(cfg, sort_keys=True))


def tables(public: PublicEpisode) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        "ledger": {r.row_id: r.to_dict() for r in public.ledger},
        "invoices": {d.doc_id: d.to_dict() for d in public.invoices},
        "payments": {d.doc_id: d.to_dict() for d in public.payments},
    }


def undo_changes(public: PublicEpisode, changes: list[dict[str, Any]]) -> dict[str, Any]:
    """Undo provenance changes in reverse order. Checks every 'after' against the data first."""
    t = copy.deepcopy(tables(public))
    for ch in reversed(changes):
        table = t[ch["table"]]
        if ch["after"] is None:
            assert ch["id"] not in table, ch
        else:
            assert table.get(ch["id"]) == ch["after"], ch
        if ch["before"] is None:
            del table[ch["id"]]
        else:
            table[ch["id"]] = ch["before"]
    return t


def all_public_ids(public: PublicEpisode) -> set[str]:
    return (
        {r.row_id for r in public.ledger}
        | {d.doc_id for d in public.invoices}
        | {d.doc_id for d in public.payments}
        | {f.rate_id for f in public.fx}
    )
