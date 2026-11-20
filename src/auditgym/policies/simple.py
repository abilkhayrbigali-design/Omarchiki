"""Baseline policies: random explorer, flag-all attack, oracle.

These mirror the organizers' reference policies (case section 06). The reconciliation baseline
lives in reconcile.py.
"""

from __future__ import annotations

from typing import Any

from auditgym.contracts.constants import ISSUE_TYPES, MAX_EVIDENCE_IDS, MAX_FINDINGS, TABLES
from auditgym.contracts.types import Truth
from auditgym.policies.base import (
    calculator,
    episode_id_of,
    finding,
    get_document,
    get_record,
    list_records,
    submit,
)
from auditgym.util.rng import Stream, root_stream


def _ids_of(table: str, rows: list[dict[str, Any]]) -> list[str]:
    key = {"ledger": "row_id", "invoices": "doc_id", "payments": "doc_id"}.get(table)
    if key is None:  # fx
        return [f"{r['date']}/{r['source_currency']}/{r['target_currency']}" for r in rows]
    return [r[key] for r in rows]


class RandomExplorer:
    """Random tool calls with a fixed seed, then an empty submission.

    Measures the floor: exploration alone must earn nothing (E11).
    """

    name = "random_explorer"

    def __init__(self, seed: int = 0, budget: int = 30) -> None:
        self._seed = seed
        self._budget = budget
        self._rng: Stream = root_stream(seed, "policy", "random_explorer")
        self._episode_id = ""
        self._seen: list[tuple[str, str]] = []
        self._steps = 0

    def reset(self, observation: dict[str, Any], info: dict[str, Any]) -> None:
        self._rng = root_stream(self._seed, "policy", "random_explorer")
        self._episode_id = episode_id_of(observation)
        self._seen = []
        self._steps = 0

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        result = observation.get("result")
        if isinstance(result, dict) and isinstance(result.get("records"), list):
            table = result.get("table")
            if table in TABLES:
                self._seen.extend((table, i) for i in _ids_of(table, result["records"]))
        self._steps += 1
        if self._steps > self._budget:
            return submit(self._episode_id, [])
        kind = self._rng.randbelow(4)
        if kind == 0 or not self._seen:
            return list_records(self._rng.choice(TABLES), None, 1 + self._rng.randbelow(50))
        table, record_id = self._rng.choice(self._seen)
        if kind == 1:
            return get_record(table, record_id)
        if kind == 2 and table in ("invoices", "payments"):
            return get_document(record_id)
        return calculator(f"{self._rng.randbelow(10**6)} / {1 + self._rng.randbelow(1000)}")


class FlagAll:
    """Attack: read every table, then accuse as many records as the limits allow.

    Each finding names one record with every issue type in turn and cites that record as
    evidence. Must score badly (S10).
    """

    name = "flag_all"

    def __init__(self) -> None:
        self._episode_id = ""
        self._base = ""
        self._queue: list[tuple[str, str | None]] = []
        self._ids: list[tuple[str, str]] = []

    def reset(self, observation: dict[str, Any], info: dict[str, Any]) -> None:
        self._episode_id = episode_id_of(observation)
        self._base = observation["manifest"]["base_currency"]
        self._queue = [(t, None) for t in ("invoices", "payments", "ledger")]
        self._ids = []

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        result = observation.get("result")
        if isinstance(result, dict) and isinstance(result.get("records"), list):
            table = result["table"]
            self._ids.extend((table, i) for i in _ids_of(table, result["records"]))
            if result.get("next_cursor"):
                self._queue.insert(0, (table, result["next_cursor"]))
        if self._queue and observation.get("steps_remaining", 0) > 1:
            table, cursor = self._queue.pop(0)
            return list_records(table, cursor, 50)
        docs = [i for t, i in self._ids if t in ("invoices", "payments")]
        rows = [i for t, i in self._ids if t == "ledger"]
        targets = docs + rows
        findings = []
        for n, target in enumerate(targets[:MAX_FINDINGS]):
            evidence = [target, *docs[: MAX_EVIDENCE_IDS - 1]]
            findings.append(
                finding(ISSUE_TYPES[n % len(ISSUE_TYPES)], [target], 0, self._base, evidence[:10])
            )
        return submit(self._episode_id, findings)


class Oracle:
    """Grader-side only: submits the reference answer with its first evidence set.

    Checks the maximum reward, not agent ability. Receives the truth from the harness.
    """

    name = "oracle"

    def __init__(self, truth: Truth) -> None:
        self._truth = truth
        self._episode_id = truth.episode_id

    def reset(self, observation: dict[str, Any], info: dict[str, Any]) -> None:
        self._episode_id = episode_id_of(observation)

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        findings = [
            finding(
                issue.type,
                list(issue.affected_ids),
                issue.expected_amount_minor,
                issue.currency,
                list(issue.evidence_sets[0]),
            )
            for issue in self._truth.issues
        ]
        return submit(self._episode_id, findings)
