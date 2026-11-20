"""JSON-lines protocol between an agent process and the grader-side environment process.

One request per line, one response per line, UTF-8, canonical JSON (auditgym.util.canon).

Requests:
    {"op": "reset"}                                          next scheduled episode
    {"op": "reset", "seed": int, "config": object | null}   open mode without a schedule only;
                                                             config may be omitted
    {"op": "step", "action": <any JSON value>}
    {"op": "truncate"}                                       give up the episode (graded empty)
    {"op": "close"}
Responses:
    reset:          {"ok": true, "op": "reset", "observation": {...}, "info": {...}}
    step, truncate: {"ok": true, "op": ..., "observation": {...}, "reward": float,
                     "terminated": bool, "truncated": bool, "info": {...}}
    close:          {"ok": true, "op": "close"}
    failure:        {"ok": false, "error": {"code": ..., "message": ...}}

A malformed request line (not JSON, not an object, unknown op, missing or extra fields, wrong
types, longer than MAX_LINE_BYTES) gets an `invalid_request` response and costs no step; the
server keeps running. So does a reset that carries `seed` or `config` when the server serves a
schedule (always in closed mode), or one without `seed` when it does not. A reset after the last
scheduled episode gets `schedule_exhausted`. A malformed *action* inside a valid step request is
the environment's business: it costs a step and returns an error observation. Failure codes and
messages are fixed (PROTOCOL_ERROR_MESSAGES); exception text never leaves the server.

What happens to a running episode on reset, close, EOF and after the time limit is decided by
auditgym.server.host.EpisodeHost.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from auditgym.contracts.config import ConfigError
from auditgym.util import canon

MAX_LINE_BYTES = 1_000_000
OPS = ("reset", "step", "truncate", "close")


class ProtocolCode(StrEnum):
    INVALID_REQUEST = "invalid_request"
    INVALID_CONFIG = "invalid_config"
    SCHEDULE_EXHAUSTED = "schedule_exhausted"
    INTERNAL_ERROR = "internal_error"


PROTOCOL_ERROR_MESSAGES: dict[ProtocolCode, str] = {
    ProtocolCode.INVALID_REQUEST: "Malformed request line.",
    ProtocolCode.INVALID_CONFIG: "Invalid episode config.",
    ProtocolCode.SCHEDULE_EXHAUSTED: "No more scheduled episodes.",
    ProtocolCode.INTERNAL_ERROR: "Internal error.",
}

_REQUEST_KEYS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    # op -> (required keys, optional keys)
    "reset": (frozenset({"op"}), frozenset({"seed", "config"})),
    "step": (frozenset({"op", "action"}), frozenset()),
    "truncate": (frozenset({"op"}), frozenset()),
    "close": (frozenset({"op"}), frozenset()),
}


def error_response(code: ProtocolCode) -> dict[str, Any]:
    return {"ok": False, "error": {"code": code.value, "message": PROTOCOL_ERROR_MESSAGES[code]}}


def encode(message: dict[str, Any]) -> bytes:
    return (canon.dumps(message) + "\n").encode("utf-8")


def decode_request(line: bytes) -> dict[str, Any] | None:
    """A structurally valid request, or None."""
    if len(line) > MAX_LINE_BYTES:
        return None
    try:
        request = canon.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None
    if not isinstance(request, dict):
        return None
    op = request.get("op")
    if not isinstance(op, str) or op not in _REQUEST_KEYS:
        return None
    required, optional = _REQUEST_KEYS[op]
    keys = set(request)
    if not required <= keys or not keys <= required | optional:
        return None
    if op == "reset":
        if "seed" in request:
            seed = request["seed"]
            if isinstance(seed, bool) or not isinstance(seed, int):
                return None
        config = request.get("config")
        if config is not None and not isinstance(config, dict):
            return None
    return request


def step_response(op: str, result: tuple) -> dict[str, Any]:
    obs, reward, terminated, truncated, info = result
    return {
        "ok": True,
        "op": op,
        "observation": obs,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "info": info,
    }


def handle_line(host: Any, line: bytes) -> tuple[dict[str, Any], bool]:
    """Process one request line on an EpisodeHost. Returns (response, close the server)."""
    request = decode_request(line)
    if request is None:
        return error_response(ProtocolCode.INVALID_REQUEST), False
    try:
        return host.handle(request)
    except ConfigError:
        return error_response(ProtocolCode.INVALID_CONFIG), False
    except Exception:
        return error_response(ProtocolCode.INTERNAL_ERROR), False
