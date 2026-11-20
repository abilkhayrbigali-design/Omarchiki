"""Policy interface and helpers shared by the baseline policies.

A policy sees only observations (JSON-compatible dicts) and returns actions
`{"action": name, "args": {...}}`. It never touches environment internals.
"""

from __future__ import annotations

from typing import Any, Protocol


class Policy(Protocol):
    name: str

    def reset(self, observation: dict[str, Any], info: dict[str, Any]) -> None: ...

    def act(self, observation: dict[str, Any]) -> dict[str, Any]: ...


def list_records(table: str, cursor: str | None = None, limit: int = 50) -> dict[str, Any]:
    return {"action": "list_records", "args": {"table": table, "cursor": cursor, "limit": limit}}


def get_record(table: str, record_id: str) -> dict[str, Any]:
    return {"action": "get_record", "args": {"table": table, "id": record_id}}


def get_document(doc_id: str) -> dict[str, Any]:
    return {"action": "get_document", "args": {"doc_id": doc_id}}


def calculator(expression: str) -> dict[str, Any]:
    return {"action": "calculator", "args": {"expression": expression}}


def submit(episode_id: str, findings: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "action": "submit",
        "args": {"findings": {"episode_id": episode_id, "findings": findings}},
    }


def finding(
    type_: str,
    affected_ids: list[str],
    amount_minor: int,
    currency: str,
    evidence_ids: list[str],
    explanation: str = "",
) -> dict[str, Any]:
    return {
        "type": type_,
        "affected_ids": list(affected_ids),
        "claimed_amount_minor": int(amount_minor),
        "currency": currency,
        "evidence_ids": list(evidence_ids),
        "explanation": explanation,
    }


def episode_id_of(first_observation: dict[str, Any]) -> str:
    return first_observation["manifest"]["episode_id"]
