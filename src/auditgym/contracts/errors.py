"""Fixed error codes and messages for observations (E07, D12).

Messages are constant per code: they never include IDs, paths, values, seeds or tracebacks,
so an error cannot leak anything about the episode or the reference answer.
"""

from __future__ import annotations

from enum import StrEnum


class ErrorCode(StrEnum):
    UNKNOWN_ACTION = "unknown_action"
    INVALID_ARGUMENTS = "invalid_arguments"
    UNKNOWN_TABLE = "unknown_table"
    UNKNOWN_ID = "unknown_id"
    INVALID_CURSOR = "invalid_cursor"
    LIMIT_OUT_OF_RANGE = "limit_out_of_range"
    CALCULATOR_ERROR = "calculator_error"
    INVALID_SUBMISSION = "invalid_submission"
    EPISODE_FINISHED = "episode_finished"
    INTERNAL_ERROR = "internal_error"


ERROR_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.UNKNOWN_ACTION: "Unknown action.",
    ErrorCode.INVALID_ARGUMENTS: "Invalid or missing arguments for this action.",
    ErrorCode.UNKNOWN_TABLE: "Unknown table.",
    ErrorCode.UNKNOWN_ID: "No record with this id.",
    ErrorCode.INVALID_CURSOR: "Invalid cursor.",
    ErrorCode.LIMIT_OUT_OF_RANGE: "Limit must be an integer from 1 to 50.",
    ErrorCode.CALCULATOR_ERROR: "Expression rejected: only decimal numbers, + - * / and parentheses are allowed.",
    ErrorCode.INVALID_SUBMISSION: "Submission does not match the findings schema.",
    ErrorCode.EPISODE_FINISHED: "Episode has finished. Call reset.",
    ErrorCode.INTERNAL_ERROR: "Internal error.",
}


def error_payload(code: ErrorCode) -> dict[str, str]:
    """The `error` field of an observation."""
    return {"code": code.value, "message": ERROR_MESSAGES[code]}
