"""The server's episode schedule: which episodes it serves and in which order.

In closed mode the organizer fixes the episodes; the agent only says `reset` and gets the next
one. Two sources, in the order their options appear on the command line:

    --schedule FILE      JSON list of {"seed": int, "config": <path> | <object> | null}.
                         A relative config path is resolved against the schedule file's
                         directory. Configs are validated at startup; episodes are generated
                         lazily, one per reset.
    --episode-dir DIR    a directory holding one episode (it has manifest.json), or a parent
                         directory whose subdirectories are episodes, served in sorted order of
                         their names (dot-directories are skipped). Every episode needs its
                         private/ files. Read lazily, one per reset.

Everything is checked at startup, so a broken schedule stops the server before an agent connects.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from auditgym.contracts.config import ConfigError, normalize_config
from auditgym.contracts.types import Episode

SCHEDULE_KEYS = frozenset({"seed", "config"})


class ScheduleError(ValueError):
    """The schedule cannot be loaded. The message is for the organizer (server stderr)."""


@dataclass(frozen=True)
class ScheduleEntry:
    """One scheduled episode: generated from (seed, config), or read from a directory."""

    seed: int | None = None
    config: dict[str, Any] | None = None
    directory: Path | None = None

    def load(self, generate_fn: Callable[..., Episode] | None = None) -> Episode:
        if self.directory is not None:
            from auditgym.contracts.io import read_episode

            return read_episode(self.directory)
        if generate_fn is None:
            from auditgym.episode.generate import generate_episode

            generate_fn = generate_episode
        return generate_fn(self.seed, self.config)


def _load_json(path: Path, what: str) -> Any:
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        raise ScheduleError(f"cannot read {what} {path}") from exc


def _entry_config(value: Any, base: Path, index: int) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        path = Path(value)
        if not path.is_absolute():
            path = base / path
        value = _load_json(path, f"config of schedule entry {index}")
    if not isinstance(value, dict):
        raise ScheduleError(f"schedule entry {index}: config must be a path, an object or null")
    try:
        return normalize_config(value)
    except ConfigError as exc:
        raise ScheduleError(f"schedule entry {index}: invalid config: {exc}") from exc


def load_schedule_file(path: str | Path) -> list[ScheduleEntry]:
    path = Path(path)
    data = _load_json(path, "schedule file")
    if not isinstance(data, list):
        raise ScheduleError("schedule file must hold a JSON list")
    entries = []
    for index, item in enumerate(data):
        if not isinstance(item, dict) or not set(item) <= SCHEDULE_KEYS or "seed" not in item:
            raise ScheduleError(f"schedule entry {index}: expected {{'seed', 'config'}}")
        seed = item["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ScheduleError(f"schedule entry {index}: seed must be an integer")
        config = _entry_config(item.get("config"), path.parent, index)
        entries.append(ScheduleEntry(seed=seed, config=config))
    return entries


def _is_episode_dir(path: Path) -> bool:
    return (path / "manifest.json").is_file()


def episode_dir_entries(directory: str | Path) -> list[ScheduleEntry]:
    root = Path(directory)
    if not root.is_dir():
        raise ScheduleError(f"episode directory {root} does not exist")
    if _is_episode_dir(root):
        dirs = [root]
    else:
        dirs = sorted(
            (p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")),
            key=lambda p: p.name,
        )
        if not dirs:
            raise ScheduleError(f"no episodes in {root}")
        for d in dirs:
            if not _is_episode_dir(d):
                raise ScheduleError(f"{d} is not an episode directory (no manifest.json)")
    for d in dirs:
        if not (d / "private" / "truth.json").is_file():
            raise ScheduleError(f"{d} has no private/truth.json")
    return [ScheduleEntry(directory=d) for d in dirs]


def build_schedule(sources: Iterable[tuple[str, str]]) -> list[ScheduleEntry]:
    """Entries from ("schedule", FILE) and ("episode_dir", DIR) sources, in the given order."""
    entries: list[ScheduleEntry] = []
    for kind, value in sources:
        if kind == "schedule":
            entries += load_schedule_file(value)
        elif kind == "episode_dir":
            entries += episode_dir_entries(value)
        else:
            raise ScheduleError(f"unknown schedule source {kind!r}")
    if not entries:
        raise ScheduleError("the schedule is empty")
    return entries
