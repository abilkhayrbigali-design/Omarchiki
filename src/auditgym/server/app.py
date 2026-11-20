"""Grader-side environment process: `python -m auditgym.server`.

The reference answer lives only in this process. The agent talks to it through the JSON-lines
protocol (auditgym.server.protocol) on stdin/stdout and gets nothing else.

Options:
    --mode open|closed    open returns the terminal reward; closed withholds it (default open)
    --grader-log PATH     JSONL file for grade records (closed default: runs/grader_log.jsonl)
    --episode-dir DIR     serve the episode stored in DIR (contracts.io layout, with private/)
                          on every reset; seed and config are then ignored
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO

from auditgym.env.session import AuditEnv
from auditgym.server.protocol import (
    MAX_LINE_BYTES,
    ProtocolCode,
    encode,
    error_response,
    handle_line,
)

DEFAULT_CLOSED_LOG = Path("runs") / "grader_log.jsonl"


def serve(env: Any, stdin: BinaryIO, stdout: BinaryIO) -> int:
    """Answer request lines until EOF or a close request."""
    while True:
        line = stdin.readline(MAX_LINE_BYTES + 1)
        if not line:
            return 0
        if len(line) > MAX_LINE_BYTES:
            # Drop the rest of an over-long line, answer once, keep serving.
            tail = line
            while tail and not tail.endswith(b"\n"):
                tail = stdin.readline(MAX_LINE_BYTES + 1)
            response, close = error_response(ProtocolCode.INVALID_REQUEST), False
        else:
            response, close = handle_line(env, line)
        stdout.write(encode(response))
        stdout.flush()
        if close:
            return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m auditgym.server", description="AuditGym environment server (JSON lines)"
    )
    parser.add_argument("--mode", choices=("open", "closed"), default="open")
    parser.add_argument("--grader-log", default=None)
    parser.add_argument("--episode-dir", default=None)
    return parser


def _episode_dir_generator(directory: str) -> Callable[..., Any]:
    from auditgym.contracts.io import read_episode

    def generate(seed: int, config: dict[str, Any] | None = None) -> Any:
        return read_episode(directory)

    return generate


def main(
    argv: list[str] | None = None,
    *,
    grader: Any = None,
    generate_fn: Any = None,
    stdin: BinaryIO | None = None,
    stdout: BinaryIO | None = None,
) -> int:
    """Run the server. `grader` and `generate_fn` let tests plug in stand-ins."""
    args = build_parser().parse_args(argv)
    grader_log = args.grader_log
    if grader_log is None and args.mode == "closed":
        grader_log = str(DEFAULT_CLOSED_LOG)
    if generate_fn is None and args.episode_dir is not None:
        generate_fn = _episode_dir_generator(args.episode_dir)
    env = AuditEnv(mode=args.mode, generate_fn=generate_fn, grader=grader, grader_log=grader_log)
    out = stdout if stdout is not None else sys.stdout.buffer
    inp = stdin if stdin is not None else sys.stdin.buffer
    # Keep the protocol channel clean: anything printed by library code goes to stderr.
    saved = sys.stdout
    sys.stdout = sys.stderr
    try:
        return serve(env, inp, out)
    finally:
        sys.stdout = saved
