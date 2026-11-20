"""Observations of the environment (E01, E04, E08, D12).

This module builds everything the agent sees. It works on a RecordStore (public data only) and
must never import the reference answer or its producers: grading, inject, world, verify, episode
and contracts.grading_api are off limits (checked by tests/leak/test_import_boundary.py).

Step observation: {"result": ..., "error": null | {"code", "message"}, "steps_remaining": n}.
First observation: {"task", "manifest", "policy", "steps_remaining"}.
Result shapes:
    list_records -> {"table", "records": [...], "next_cursor": str | null}
    get_record   -> {"table", "record": {...}}
    get_document -> {"document": {...}}
    calculator   -> {"value": "<plain decimal string>"}
    submit       -> {"submitted": true, "findings_count": n}
On any error "result" is null. Every observation is a fresh plain-JSON object.
"""

from __future__ import annotations

import json
from typing import Any

from auditgym.contracts.constants import (
    ISSUE_TYPES,
    MAX_EVIDENCE_IDS,
    MAX_EXPLANATION_CHARS,
    MAX_FINDINGS,
    MAX_STEPS,
    PAGE_LIMIT_MAX,
    TABLES,
)
from auditgym.contracts.errors import ErrorCode, error_payload
from auditgym.env.actions import Action
from auditgym.env.calculator import CalculatorError, evaluate
from auditgym.env.store import InvalidCursorError, RecordStore, UnknownIdError
from auditgym.util import canon

TASK = (
    "Audit the books of one fictional company for one reporting period and report every "
    "accounting error you can prove. Error types: "
    + ", ".join(ISSUE_TYPES)
    + "; definitions, amounts and the booking rules are in `policy`. A clean company has no "
    'errors. Act with one JSON action per step: {"action": name, "args": {...}}. Tools: '
    f"list_records(table, cursor, limit) pages through a table ({', '.join(TABLES)}), at most "
    f"{PAGE_LIMIT_MAX} rows per page, pass next_cursor (null at the end) to continue; "
    "get_record(table, id) returns one row (id = row_id for ledger, doc_id for invoices and "
    "payments, 'YYYY-MM-DD/SRC/TGT' for fx); get_document(doc_id) returns a full invoice with "
    "line items or a payment; calculator(expression) does exact decimal arithmetic with "
    '+ - * / and parentheses; submit(findings) ends the episode. findings = {"episode_id": '
    'manifest.episode_id, "findings": [{"type", "affected_ids", "claimed_amount_minor", '
    '"currency", "evidence_ids", "explanation"}]}; '
    f"at most {MAX_FINDINGS} findings, {MAX_EVIDENCE_IDS} evidence ids and "
    f"{MAX_EXPLANATION_CHARS} explanation characters per finding. Amounts are integers in minor "
    "units. affected_ids and evidence_ids are record ids. A malformed submission scores -1; a "
    "finding with a wrong type, ids, amount or currency, or citing an id that does not exist, is "
    "a false positive. "
    f"Budget: {MAX_STEPS} steps including submit; an invalid action also costs a step; running "
    "out of steps counts as an empty submission. Exploration gives no reward and no hints."
)


def _fresh(obj: Any) -> Any:
    """Deep, JSON-normalized copy: the caller can mutate it without touching the environment."""
    return json.loads(canon.dumps(obj))


def first_observation(store: RecordStore, steps_remaining: int) -> dict[str, Any]:
    return _fresh(
        {
            "task": TASK,
            "manifest": store.manifest,
            "policy": store.policy,
            "steps_remaining": steps_remaining,
        }
    )


def step_observation(result: Any, error: ErrorCode | None, steps_remaining: int) -> dict[str, Any]:
    return _fresh(
        {
            "result": None if error is not None else result,
            "error": None if error is None else error_payload(error),
            "steps_remaining": steps_remaining,
        }
    )


def findings_count(payload: Any) -> int:
    """Number of findings the agent sent, from its own payload only (nothing about correctness)."""
    if isinstance(payload, dict) and isinstance(payload.get("findings"), list):
        return len(payload["findings"])
    return 0


def submit_result(payload: Any) -> dict[str, Any]:
    return {"submitted": True, "findings_count": findings_count(payload)}


def run_tool(
    store: RecordStore, action: Action
) -> tuple[dict[str, Any] | None, ErrorCode | None, list[str]]:
    """Run one exploration tool: (result, error code, ids of the records shown)."""
    args = action.args
    try:
        if action.name == "list_records":
            table = args["table"]
            records, next_cursor, ids = store.page(table, args["cursor"], args["limit"])
            return {"table": table, "records": records, "next_cursor": next_cursor}, None, ids
        if action.name == "get_record":
            table = args["table"]
            record = store.get(table, args["id"])
            return {"table": table, "record": record}, None, [args["id"]]
        if action.name == "get_document":
            document = store.document(args["doc_id"])
            return {"document": document}, None, [args["doc_id"]]
        if action.name == "calculator":
            return {"value": evaluate(args["expression"])}, None, []
    except UnknownIdError:
        return None, ErrorCode.UNKNOWN_ID, []
    except InvalidCursorError:
        return None, ErrorCode.INVALID_CURSOR, []
    except CalculatorError:
        return None, ErrorCode.CALCULATOR_ERROR, []
    return None, ErrorCode.UNKNOWN_ACTION, []
