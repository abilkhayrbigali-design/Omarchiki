"""Grader-side environment process: `python -m auditgym.server [options]`.

Also reachable as `python -m auditgym serve -- [options]` (a leading `--` is dropped).

The reference answer lives only in this process. The agent talks to it through the JSON-lines
protocol (auditgym.server.protocol) on stdin/stdout and gets nothing else. In a real evaluation
the organizer starts this process (as a different OS user than the agent), keeps the grader log
and the episode files where the agent cannot read them, and connects only stdin/stdout to the
agent.

Options:
    --mode open|closed    open returns the terminal reward; closed withholds it (default open)
    --grader-log PATH     JSONL file for grade records. Closed mode: required, absolute path.
                          Open mode: optional (no default).
    --schedule FILE       serve the episodes listed in FILE ([{"seed", "config"}, ...]) in order
    --episode-dir DIR     serve the episode in DIR, or every episode subdirectory of DIR in
                          sorted order (repeatable; sources are served in command-line order)
    --time-limit SECONDS  wall-clock limit per episode from its reset (closed default 60;
                          open default: none)

Closed mode refuses to start without a schedule (--schedule or --episode-dir) or without an
absolute --grader-log. With a schedule, the client's reset carries no seed or config. Without
one (open mode only) the client picks seed and config, as in training.
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import sys
from pathlib import Path
from typing import Any, BinaryIO

from auditgym.env.session import AuditEnv
from auditgym.server.host import Clock, EpisodeHost
from auditgym.server.protocol import (
    MAX_LINE_BYTES,
    ProtocolCode,
    encode,
    error_response,
    handle_line,
)
from auditgym.server.schedule import ScheduleError, build_schedule

DEFAULT_CLOSED_TIME_LIMIT_S = 60.0


def serve(target: Any, stdin: BinaryIO, stdout: BinaryIO) -> int:
    """Answer request lines until EOF or a close request.

    `target` is an EpisodeHost, or an open-mode AuditEnv (served without schedule or time
    limit). However serving stops (close, EOF, a broken pipe, SIGTERM), a running episode is
    ended, graded and logged as abandoned (or timeout) before this returns.
    """
    host = target if isinstance(target, EpisodeHost) else EpisodeHost(target)
    try:
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
                response, close = handle_line(host, line)
            stdout.write(encode(response))
            stdout.flush()
            if close:
                return 0
    finally:
        host.finish()


class _SourceAction(argparse.Action):
    """--schedule and --episode-dir share one ordered list of (kind, value)."""

    def __call__(self, parser, namespace, values, option_string=None) -> None:  # noqa: ANN001
        sources = list(getattr(namespace, self.dest, None) or [])
        sources.append((self.const, values))
        setattr(namespace, self.dest, sources)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m auditgym.server", description="AuditGym environment server (JSON lines)"
    )
    parser.add_argument("--mode", choices=("open", "closed"), default="open")
    parser.add_argument("--grader-log", default=None, metavar="PATH")
    parser.add_argument(
        "--schedule", action=_SourceAction, dest="sources", const="schedule", metavar="FILE"
    )
    parser.add_argument(
        "--episode-dir", action=_SourceAction, dest="sources", const="episode_dir", metavar="DIR"
    )
    parser.add_argument("--time-limit", type=float, default=None, metavar="SECONDS")
    parser.set_defaults(sources=[])
    return parser


def _same_file(path: str, stream: Any) -> bool:
    try:
        a = os.fstat(stream.fileno())
        b = os.stat(path)
    except (OSError, ValueError, AttributeError):
        return False
    return (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)


def _touch(path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8"):
        pass


def _end_on_sigterm() -> None:
    """SIGTERM unwinds through serve(), so a running episode still gets its log record."""

    def handler(signum: int, frame: Any) -> None:
        raise SystemExit(128 + signum)

    try:
        signal.signal(signal.SIGTERM, handler)
    except ValueError:  # not the main thread
        pass


def main(
    argv: list[str] | None = None,
    *,
    grader: Any = None,
    generate_fn: Any = None,
    stdin: BinaryIO | None = None,
    stdout: BinaryIO | None = None,
    clock: Clock | None = None,
) -> int:
    """Run the server. `grader`, `generate_fn` and `clock` let tests plug in stand-ins.

    Configuration errors exit with status 2 and a message on stderr, before any request is read.
    """
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--"]:  # `python -m auditgym serve -- --mode closed ...`
        argv = argv[1:]
    args = parser.parse_args(argv)
    closed = args.mode == "closed"
    out = stdout if stdout is not None else sys.stdout.buffer
    inp = stdin if stdin is not None else sys.stdin.buffer

    grader_log = args.grader_log
    if closed:
        if grader_log is None:
            parser.error("closed mode requires --grader-log with an absolute path")
        if not os.path.isabs(grader_log):
            parser.error("closed mode requires an absolute --grader-log path")
        if not args.sources:
            parser.error("closed mode requires --schedule or --episode-dir")
    if grader_log is not None and _same_file(grader_log, out):
        parser.error("--grader-log must not be the protocol channel")

    time_limit = args.time_limit
    if time_limit is None and closed:
        time_limit = DEFAULT_CLOSED_TIME_LIMIT_S
    if time_limit is not None and not (math.isfinite(time_limit) and time_limit > 0):
        parser.error("--time-limit must be a positive number of seconds")

    schedule = None
    if args.sources:
        try:
            schedule = build_schedule(args.sources)
        except ScheduleError as exc:
            parser.error(str(exc))
    if closed:
        try:
            _touch(grader_log)
        except OSError:
            parser.error("cannot write the grader log")

    env = AuditEnv(mode=args.mode, generate_fn=generate_fn, grader=grader, grader_log=grader_log)
    host = EpisodeHost(
        env, schedule=schedule, generate_fn=generate_fn, time_limit_s=time_limit, clock=clock
    )
    if stdin is None and stdout is None:
        _end_on_sigterm()
    # Keep the protocol channel clean: anything printed by library code goes to stderr.
    saved = sys.stdout
    sys.stdout = sys.stderr
    try:
        return serve(host, inp, out)
    finally:
        sys.stdout = saved
