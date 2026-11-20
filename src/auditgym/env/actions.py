"""Parsing and validation of step actions (E04, E07; schemas/action.schema.json).

An action is `{"action": name, "args": {...}}` with exactly these two keys. Validation never
looks at episode data: it only decides which fixed error code a malformed action gets.

Order of checks: not a JSON object, or `action` not a string -> invalid_arguments; unknown name
-> unknown_action; missing `args`, extra top-level keys, `args` not an object, missing or extra
argument keys, wrong argument types -> invalid_arguments; table outside the list -> unknown_table;
limit outside 1..50 -> limit_out_of_range. Ids, cursors and expressions are checked when the tool
runs (unknown_id, invalid_cursor, calculator_error).

The `findings` argument of submit is passed to the grader as is: its structure is the grader's
business (S03: a malformed submission ends the episode with reward -1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from auditgym.contracts.constants import ACTIONS, PAGE_LIMIT_MAX, TABLES
from auditgym.contracts.errors import ErrorCode

MAX_ID_CHARS = 64
MAX_JSON_DEPTH = 64

_ARG_KEYS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    # action -> (required keys, optional keys)
    "list_records": (frozenset({"table"}), frozenset({"cursor", "limit"})),
    "get_record": (frozenset({"table", "id"}), frozenset()),
    "get_document": (frozenset({"doc_id"}), frozenset()),
    "calculator": (frozenset({"expression"}), frozenset()),
    "submit": (frozenset({"findings"}), frozenset()),
}


class ActionError(Exception):
    """A malformed action. Only the fixed code reaches the agent."""

    def __init__(self, code: ErrorCode) -> None:
        super().__init__(code.value)
        self.code = code


@dataclass(frozen=True, slots=True)
class Action:
    name: str
    args: dict[str, Any]


def plain_json(value: Any, depth: int = 0) -> Any:
    """A fresh copy of `value` made of plain JSON types only, or ValueError.

    Accepts dict with str keys, list/tuple, str, int, finite float, bool and None. Subclasses are
    converted to the base type, so the environment only ever handles plain JSON data.
    """
    if depth > MAX_JSON_DEPTH:
        raise ValueError("too deep")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return str(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite number")
        return float(value)
    if isinstance(value, (list, tuple)):
        return [plain_json(v, depth + 1) for v in value]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("non-string key")
            out[str(key)] = plain_json(item, depth + 1)
        return out
    raise ValueError("not a JSON value")


def _table(value: Any) -> str:
    if not isinstance(value, str):
        raise ActionError(ErrorCode.INVALID_ARGUMENTS)
    if value not in TABLES:
        raise ActionError(ErrorCode.UNKNOWN_TABLE)
    return value


def _record_id(value: Any) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= MAX_ID_CHARS:
        raise ActionError(ErrorCode.INVALID_ARGUMENTS)
    return value


def _limit(value: Any) -> int:
    if isinstance(value, bool):
        raise ActionError(ErrorCode.INVALID_ARGUMENTS)
    if isinstance(value, float) and value.is_integer():
        value = int(value)  # JSON Schema "integer" accepts 50.0
    if not isinstance(value, int):
        raise ActionError(ErrorCode.INVALID_ARGUMENTS)
    if not 1 <= value <= PAGE_LIMIT_MAX:
        raise ActionError(ErrorCode.LIMIT_OUT_OF_RANGE)
    return value


def parse_action(raw: Any) -> Action:
    """Validate a raw action; return a normalized Action or raise ActionError."""
    try:
        data = plain_json(raw)
    except (ValueError, RecursionError) as exc:
        raise ActionError(ErrorCode.INVALID_ARGUMENTS) from exc
    if not isinstance(data, dict):
        raise ActionError(ErrorCode.INVALID_ARGUMENTS)
    name = data.get("action")
    if not isinstance(name, str):
        raise ActionError(ErrorCode.INVALID_ARGUMENTS)
    if name not in ACTIONS:
        raise ActionError(ErrorCode.UNKNOWN_ACTION)
    if set(data) != {"action", "args"} or not isinstance(data["args"], dict):
        raise ActionError(ErrorCode.INVALID_ARGUMENTS)
    args: dict[str, Any] = data["args"]
    required, optional = _ARG_KEYS[name]
    keys = set(args)
    if not required <= keys or not keys <= required | optional:
        raise ActionError(ErrorCode.INVALID_ARGUMENTS)

    if name == "list_records":
        table = _table(args["table"])
        cursor = args.get("cursor")
        if cursor is not None and not isinstance(cursor, str):
            raise ActionError(ErrorCode.INVALID_ARGUMENTS)
        limit = _limit(args.get("limit", PAGE_LIMIT_MAX))
        return Action(name, {"table": table, "cursor": cursor, "limit": limit})
    if name == "get_record":
        return Action(name, {"table": _table(args["table"]), "id": _record_id(args["id"])})
    if name == "get_document":
        return Action(name, {"doc_id": _record_id(args["doc_id"])})
    if name == "calculator":
        if not isinstance(args["expression"], str):
            raise ActionError(ErrorCode.INVALID_ARGUMENTS)
        return Action(name, {"expression": args["expression"]})
    return Action(name, {"findings": args["findings"]})
