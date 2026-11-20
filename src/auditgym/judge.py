"""The jury adapter: AuditGym behind the single connection protocol of the FinTech checks.

The jury connects to every team the same way (genesis-exe.com/fintech-checks.html, "Единый
способ подключения"; our summary is docs/REQUIREMENTS.md, section 8). `judge_adapter.py` at the
repository root re-exports the three entry points of this module:

    generate_episode(seed, config, output_dir) -> None
    make_env() -> JudgeEnv            # reset(seed, config) / step(action)
    score_episode(episode_dir, findings, inspection_steps) -> dict

The adapter only translates names and envelopes; the environment, the generator and the grader
are the same code that every other entry point uses:

- actions: `{"name": a, "args": {...}}` (the jury's envelope) or `{"action": a, "args": {...}}`
  (ours). For `submit`, `args.findings` is the findings.json object `{episode_id, findings}`;
  a bare list of findings, or an object without `episode_id`, is completed with the current
  episode id;
- observations: `remaining_steps` instead of our `steps_remaining`; results unwrapped:
  list_records -> {"records", "next_cursor"}, get_record -> the record, get_document -> the
  document, calculator -> the decimal string, submit -> {"accepted": true}. Errors stay
  `{"code", "message"}` with a fixed public code;
- reset(seed, config): config = {"mode": "closed" | "train", "episode_dir": dir?,
  "team_config": {...}?}. With `episode_dir` the episode is read from that directory exactly as
  it is (no regeneration, no id changes; every reset starts it again). Without it the generator
  builds the episode from `seed` and `team_config` (our config, docs/DOMAIN.md). "closed"
  returns reward 0.0 to the policy; "train" (also "open") returns the terminal reward;
- time: 60 seconds per episode from the start of reset. A step that arrives later is not
  executed, the episode ends with truncated=true and is graded as an empty answer;
- the grade of every finished episode (and every score_episode call) is appended to
  `JudgeEnv.grade_log` / `SCORE_LOG`, and to the JSONL file named by the AUDITGYM_GRADER_LOG
  environment variable when it is set. These records stay on the jury side; they never enter an
  observation or info.
"""

from __future__ import annotations

import copy
import csv
import io
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from auditgym.contracts.config import normalize_config
from auditgym.contracts.io import read_episode, write_episode
from auditgym.contracts.types import Episode, Truth, fx_rate_id
from auditgym.contracts.versions import ENV_VERSION, GENERATOR_VERSION, GRADER_VERSION
from auditgym.env.session import AuditEnv
from auditgym.grading import grade_submission

TIME_LIMIT_S = 60.0
MODES = {"closed": "closed", "train": "open", "open": "open"}
SCORE_KEYS = ("tp", "fp", "fn", "precision", "recall", "f1", "return")
SCORE_LOG: list[dict[str, Any]] = []
_LOG_ENV = "AUDITGYM_GRADER_LOG"

Clock = Callable[[], float]


def _append_log(sink: list[dict[str, Any]], record: dict[str, Any]) -> None:
    sink.append(record)
    path = os.environ.get(_LOG_ENV)
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


def _team_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Our generator config from the jury's config: the `team_config` part, or the dict itself."""
    if config is None:
        return normalize_config(None)
    if not isinstance(config, dict):
        raise TypeError("config must be a dict or None")
    if "team_config" in config or "mode" in config or "episode_dir" in config:
        return normalize_config(config.get("team_config") or None)
    return normalize_config(config)


# ---------------------------------------------------------------------------------- generate
def generate_episode(seed: int, config: dict[str, Any] | None, output_dir: str) -> None:
    """Write one episode: manifest, policy, ledger, invoices, payments, fx, private/truth.json
    and private/provenance.json. Same seed and config always give the same bytes."""
    from auditgym.episode.generate import generate_episode as generate

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    write_episode(generate(seed, _team_config(config)), output_dir, include_private=True)


# ---------------------------------------------------------------------------------- environment
class JudgeEnv:
    """reset/step with the jury's names, over a fresh AuditEnv for every reset."""

    def __init__(self, *, time_limit_s: float = TIME_LIMIT_S, clock: Clock | None = None) -> None:
        self.grade_log: list[dict[str, Any]] = []
        self._time_limit_s = time_limit_s
        self._clock = clock
        self._env: AuditEnv | None = None
        self._episode_id = ""
        self._started: float | None = None

    def _now(self) -> float:
        if self._clock is None:
            from auditgym.server import host  # the package's only clock

            self._clock = host.MONOTONIC
        return self._clock()

    def reset(
        self, seed: int | None = None, config: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        started = self._now()
        config = {} if config is None else config
        if not isinstance(config, dict):
            raise TypeError("config must be a dict or None")
        mode = MODES.get(str(config.get("mode", "train")))
        if mode is None:
            raise ValueError("config.mode must be 'closed' or 'train'")
        env = AuditEnv(mode=mode, grader_log=lambda r: _append_log(self.grade_log, r))
        episode_dir = config.get("episode_dir")
        if episode_dir is not None:
            episode = read_episode(episode_dir)
            obs, info = env.reset_with_episode(episode)
        else:
            obs, info = env.reset(seed=seed, config=_team_config(config))
        if self._env is not None and self._env.active:
            self._env.abandon()  # a reset never gives a free retry: the old episode is graded
        self._env = env
        self._episode_id = str(obs["manifest"].get("episode_id", ""))
        self._started = started
        return _first_observation(obs), dict(info)

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self._env is None:
            raise RuntimeError("call reset() before step()")
        name = _action_name(action)
        if self._env.active and self._now() - self._started > self._time_limit_s:
            obs, reward, terminated, truncated, info = self._env.truncate()
        else:
            obs, reward, terminated, truncated, info = self._env.step(
                _internal_action(action, self._episode_id)
            )
        return _step_observation(obs, name), reward, terminated, truncated, dict(info)


def make_env(**kwargs: Any) -> JudgeEnv:
    return JudgeEnv(**kwargs)


def _action_name(action: Any) -> str | None:
    if isinstance(action, dict):
        name = action.get("name", action.get("action"))
        return name if isinstance(name, str) else None
    return None


def _internal_action(action: Any, episode_id: str) -> Any:
    """The jury's envelope -> ours. Anything malformed is passed on for the env to reject."""
    if not isinstance(action, dict):
        return action
    out = dict(action)
    if "name" in out and "action" not in out:
        out["action"] = out.pop("name")
    args = out.get("args")
    if out.get("action") == "submit" and isinstance(args, dict) and "findings" in args:
        findings = args["findings"]
        if isinstance(findings, list):
            findings = {"episode_id": episode_id, "findings": findings}
        elif isinstance(findings, dict) and "episode_id" not in findings:
            findings = {"episode_id": episode_id, **findings}
        out["args"] = {**args, "findings": findings}
    return out


def _first_observation(obs: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": obs["task"].replace('{"action": name', '{"name": name'),
        "manifest": obs["manifest"],
        "policy": obs["policy"],
        "remaining_steps": obs["steps_remaining"],
    }


def _step_observation(obs: dict[str, Any], name: str | None) -> dict[str, Any]:
    result = obs.get("result")
    if result is not None:
        if name == "list_records":
            result = {"records": result["records"], "next_cursor": result["next_cursor"]}
        elif name == "get_record":
            result = result["record"]
        elif name == "get_document":
            result = result["document"]
        elif name == "calculator":
            result = result["value"]
        elif name == "submit":
            result = {"accepted": True}
    return {"result": result, "error": obs.get("error"), "remaining_steps": obs["steps_remaining"]}


# ---------------------------------------------------------------------------------- scoring
def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8-sig")
    return [
        {(k or "").strip(): (v or "") for k, v in row.items()}
        for row in csv.DictReader(io.StringIO(text))
    ]


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def open_ids_of_dir(episode_dir: str | Path) -> frozenset[str]:
    """PublicEpisode.open_ids() read leniently from an episode directory.

    The jury hands score_episode its own fixture directories; a missing file contributes no ids
    instead of failing, so scoring never depends on parts of the episode it does not use.
    """
    root = Path(episode_dir)
    ids: set[str] = set()

    def add(value: Any) -> None:
        if isinstance(value, str) and value and value not in ("null", "NULL", "None"):
            ids.add(value)

    for row in _csv(root / "ledger.csv"):
        for key in ("row_id", "entity_id", "invoice_id", "payment_id"):
            add(row.get(key))
    for inv in _jsonl(root / "invoices.jsonl"):
        for key in ("doc_id", "invoice_id", "entity_id"):
            add(inv.get(key))
    for pay in _jsonl(root / "payments.jsonl"):
        for key in ("doc_id", "payment_id"):
            add(pay.get(key))
        for ref in pay.get("invoice_refs") or ():
            add(ref)
    for rate in _csv(root / "fx.csv"):
        if rate.get("date") and rate.get("source_currency") and rate.get("target_currency"):
            add(fx_rate_id(rate["date"], rate["source_currency"], rate["target_currency"]))
    return frozenset(ids)


def _truth_of_dir(root: Path, episode_id: str) -> Truth:
    data = _read_json(root / "private" / "truth.json")
    if isinstance(data, dict) and "episode_id" not in data:
        data = {**data, "episode_id": episode_id}
    return Truth.from_dict(data)


def score_episode(
    episode_dir: str, findings: dict[str, Any] | list[Any], inspection_steps: int
) -> dict[str, Any]:
    """Grade a findings.json object against the closed truth of `episode_dir`.

    Valid submission: {"tp", "fp", "fn", "precision", "recall", "f1", "return"}, with the shared
    reward clip(2*F1 - 1 - 0.002*inspection_steps, -1, 1) as `return`. Structurally invalid
    submission: `return` = -1.0 and the other six values None; the reason code goes to the log.
    """
    root = Path(episode_dir)
    episode_id = str(_read_json(root / "manifest.json").get("episode_id", ""))
    truth = _truth_of_dir(root, episode_id)
    payload = copy.deepcopy(findings)
    if isinstance(payload, list):
        payload = {"episode_id": episode_id, "findings": payload}
    elif isinstance(payload, dict) and "episode_id" not in payload:
        payload = {"episode_id": episode_id, **payload}
    result = grade_submission(
        payload,
        truth,
        episode_id=episode_id,
        inspection_steps=inspection_steps,
        known_ids=open_ids_of_dir(root),
    )
    _append_log(
        SCORE_LOG,
        {
            "episode_id": episode_id,
            "grader_version": GRADER_VERSION,
            "generator_version": GENERATOR_VERSION,
            "env_version": ENV_VERSION,
            **result.to_log(),
        },
    )
    if not result.valid:
        return {**dict.fromkeys(SCORE_KEYS[:-1]), "return": -1.0}
    return {
        "tp": result.tp,
        "fp": result.fp,
        "fn": result.fn,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "return": result.reward,
    }


def episode_from_dir(episode_dir: str | Path) -> Episode:
    """The episode exactly as written in `episode_dir` (what reset loads in closed mode)."""
    return read_episode(episode_dir)
