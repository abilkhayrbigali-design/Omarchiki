"""Helpers for the environment contract and leak tests (K1). Owned by WP3.

Imported by tests/contract/conftest.py and tests/leak/conftest.py; test modules import it as
`envkit`.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from auditgym.contracts.constants import TABLES
from auditgym.contracts.grading_api import GradeResult
from auditgym.contracts.types import Episode, Issue, LedgerRow, PrivateEpisode, Truth
from auditgym.env import AuditEnv
from auditgym.fixtures.tiny import build_tiny_episode

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"


def _registry() -> Registry:
    resources = []
    for path in sorted(SCHEMA_DIR.glob("*.schema.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        resources.append((schema["$id"], Resource.from_contents(schema)))
    return Registry().with_resources(resources)


_REGISTRY = _registry()


def schema_validator(name: str) -> Draft202012Validator:
    schema = json.loads((SCHEMA_DIR / f"{name}.schema.json").read_text(encoding="utf-8"))
    return Draft202012Validator(schema, registry=_REGISTRY)


OBS_SCHEMA = schema_validator("observation")
ACTION_SCHEMA = schema_validator("action")


class RecordingGrader:
    """Stand-in for auditgym.grading.grade_submission: records every call."""

    def __init__(self, reward: float = 0.25, valid: bool = True) -> None:
        self.reward = reward
        self.valid = valid
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, payload, truth, *, episode_id, inspection_steps, viewed_ids=None, known_ids=None
    ):
        self.calls.append(
            {
                "payload": payload,
                "truth": truth,
                "episode_id": episode_id,
                "inspection_steps": inspection_steps,
                "viewed_ids": viewed_ids,
            }
        )
        return GradeResult(
            valid=self.valid,
            error=None if self.valid else "structure",
            reward=self.reward if self.valid else -1.0,
            experimental_reward=None,
            tp=1,
            fp=0,
            fn=len(truth.issues) - 1,
            precision=1.0,
            recall=0.25,
            f1=0.4,
            inspection_steps=inspection_steps,
            matches=((0, truth.issues[0].issue_id),) if truth.issues else (),
        )


def big_episode(extra_rows: int = 110) -> Episode:
    """The tiny fixture with `extra_rows` more ledger rows, so pages of 50 can be tested."""
    tiny = build_tiny_episode()
    rows = list(tiny.public.ledger)
    for i in range(extra_rows):
        side = 1000 + i
        rows.append(
            LedgerRow(
                f"R-X{i:04d}",
                "2026-03-11T12:00:00+05:00",
                "6100" if i % 2 == 0 else "1010",
                side if i % 2 == 0 else 0,
                0 if i % 2 == 0 else side,
                "KZT",
                None,
                None,
                None,
            )
        )
    manifest = json.loads(json.dumps(tiny.public.manifest))
    manifest["counts"]["ledger"] = len(rows)
    public = dataclasses.replace(tiny.public, ledger=tuple(rows), manifest=manifest)
    return Episode(public=public, private=tiny.private)


def with_episode_id(episode: Episode, episode_id: str) -> Episode:
    manifest = json.loads(json.dumps(episode.public.manifest))
    manifest["episode_id"] = episode_id
    return Episode(
        public=dataclasses.replace(episode.public, manifest=manifest), private=episode.private
    )


def act(name: str, **args: Any) -> dict[str, Any]:
    return {"action": name, "args": args}


def list_page(table: str, cursor: str | None = None, limit: int = 50) -> dict[str, Any]:
    return act("list_records", table=table, cursor=cursor, limit=limit)


def oracle_submission(episode: Episode) -> dict[str, Any]:
    return {
        "episode_id": episode.public.episode_id,
        "findings": [
            {
                "type": i.type,
                "affected_ids": list(i.affected_ids),
                "claimed_amount_minor": i.expected_amount_minor,
                "currency": i.currency,
                "evidence_ids": list(i.evidence_sets[0]),
                "explanation": "",
            }
            for i in episode.private.truth.issues
        ],
    }


def scripted_run(env: AuditEnv, *, submit: dict[str, Any] | None) -> list[tuple]:
    """A fixed exploration: page every table, open every document, use every tool, make every
    kind of error, then submit. Returns every (observation, reward, terminated, truncated, info).
    """
    out: list[tuple] = []

    def step(action: Any) -> dict[str, Any]:
        res = env.step(action)
        out.append(res)
        return res[0]

    doc_ids: list[str] = []
    first_ids: dict[str, str] = {}
    for table in TABLES:
        cursor = None
        while True:
            obs = step(list_page(table, cursor, 7))
            records = obs["result"]["records"]
            if table in ("invoices", "payments"):
                doc_ids += [r["doc_id"] for r in records]
            if records and table not in first_ids:
                r = records[0]
                first_ids[table] = (
                    r.get("row_id")
                    or r.get("doc_id")
                    or (f"{r['date']}/{r['source_currency']}/{r['target_currency']}")
                )
            cursor = obs["result"]["next_cursor"]
            if cursor is None:
                break
    for table, rec_id in sorted(first_ids.items()):
        step(act("get_record", table=table, id=rec_id))
    for doc_id in doc_ids:
        step(act("get_document", doc_id=doc_id))
    step(act("calculator", expression="120000 * 504.75 - 120000 * 504.50"))
    step(act("calculator", expression="1/3"))
    # One of each error.
    step(act("calculator", expression="__import__('os')"))
    step(act("teleport"))
    step({"action": "get_record"})
    step(act("list_records", table="secrets"))
    step(act("get_record", table="ledger", id="R-NOPE"))
    step(act("list_records", table="ledger", cursor="c_forged", limit=5))
    step(act("list_records", table="ledger", limit=51))
    if submit is not None:
        step(act("submit", findings=submit))
        step(act("submit", findings=submit))  # after the end: episode_finished
    return out


def env_factory(default_grader: Any):
    """Factory: AuditEnv over the tiny fixture (or a given episode) with a stand-in grader."""

    def _make(
        mode: str = "open",
        episode: Episode | None = None,
        grader_fn: Any = None,
        grader_log: Any = None,
    ) -> AuditEnv:
        def generate(seed: int, config: dict | None = None) -> Episode:
            return episode if episode is not None else build_tiny_episode()

        return AuditEnv(
            mode=mode,
            generate_fn=generate,
            grader=grader_fn if grader_fn is not None else default_grader,
            grader_log=grader_log,
        )

    return _make


def assert_valid_observation(obs: dict[str, Any]) -> None:
    OBS_SCHEMA.validate(obs)
    assert json.loads(json.dumps(obs)) == obs


# ---------------------------------------------------------------------------------- leak helpers
SECRET_REASON = "SECRET-GRADER-REASON"

# Keys of the reference answer, provenance and grade record: none may appear in what the agent
# sees.
FORBIDDEN_KEYS = frozenset(
    {
        "issue_id",
        "issues",
        "expected_amount_minor",
        "amount_tolerance_minor",
        "evidence_sets",
        "provenance",
        "injections",
        "truth",
        "seed",
        "config",
        "config_id",
        "matches",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "valid",
        "experimental_reward",
        "grader_failure",
        "inspection_steps",
        "path",
        "traceback",
    }
)


class TruthGrader:
    """A grader whose output depends on the truth and carries truth data."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self, payload, truth, *, episode_id, inspection_steps, viewed_ids=None, known_ids=None
    ):
        self.calls += 1
        n = len(truth.issues)
        return GradeResult(
            valid=True,
            error=SECRET_REASON,
            reward=round(0.1 * n - 0.002 * inspection_steps, 6),
            experimental_reward=0.01 * n,
            tp=n,
            fp=0,
            fn=0,
            precision=1.0,
            recall=1.0,
            f1=1.0,
            inspection_steps=inspection_steps,
            matches=tuple((i, issue.issue_id) for i, issue in enumerate(truth.issues)),
        )


def with_truth(episode: Episode, truth: Truth) -> Episode:
    return Episode(
        public=episode.public, private=PrivateEpisode(truth=truth, provenance={"swapped": True})
    )


def alternative_truths(episode: Episode) -> list[Truth]:
    """The fixture's truth, an empty truth, and a different, made-up truth."""
    eid = episode.public.episode_id
    made_up = (
        Issue(
            "iss_fake_a",
            "duplicate_invoice",
            ("D-I1", "D-I5"),
            3_000_000,
            "KZT",
            0,
            (("D-I1", "D-I5"),),
        ),
        Issue("iss_fake_b", "period_cutoff", ("D-I6",), 2_500_000, "KZT", 0, (("D-I6", "R-009"),)),
        Issue("iss_fake_c", "fx_conversion", ("D-P5",), 7, "KZT", 1, (("D-P5", "R-024"),)),
    )
    return [episode.private.truth, Truth(eid, ()), Truth(eid, made_up)]


def collect_keys(obj: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(k)
            keys |= collect_keys(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            keys |= collect_keys(v)
    return keys


def truth_markers(episode: Episode) -> list[str]:
    """Strings that only the reference answer or provenance contain."""
    markers = [i.issue_id for i in episode.private.truth.issues]
    markers += ["iss_", "iss_fake", SECRET_REASON, "injections", "hand-built", "credit reduced"]
    return markers
