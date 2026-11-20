"""Command-line entry point: python -m auditgym <command>.

Owned by the integrator. Packages expose functions; commands are wired here.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from auditgym.contracts.versions import (
    ENV_VERSION,
    GENERATOR_VERSION,
    GRADER_VERSION,
    SCHEMA_VERSION,
)


def _cmd_version(_: argparse.Namespace) -> int:
    print(
        f"schema {SCHEMA_VERSION}  generator {GENERATOR_VERSION}  env {ENV_VERSION}  grader {GRADER_VERSION}"
    )
    return 0


def _cmd_fixture(args: argparse.Namespace) -> int:
    from auditgym.contracts.io import fingerprint, write_episode
    from auditgym.fixtures.tiny import build_tiny_episode

    episode = build_tiny_episode()
    write_episode(episode, args.out, include_private=not args.public_only)
    print(f"wrote {args.out}  fingerprint {fingerprint(episode)[:16]}")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    """Single verification command. For now: the test suite. WP8 extends it."""
    cmd = [sys.executable, "-m", "pytest", "-q"]
    if args.fast:
        cmd += ["-m", "not slow"]
    return subprocess.call(cmd, cwd=Path(__file__).resolve().parents[2])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="auditgym", description="AuditGym RL environment")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("version", help="print component versions")
    p.set_defaults(func=_cmd_version)

    p = sub.add_parser("fixture", help="write the tiny hand-built episode to a directory")
    p.add_argument("--out", default="runs/fixture_tiny")
    p.add_argument("--public-only", action="store_true")
    p.set_defaults(func=_cmd_fixture)

    p = sub.add_parser("check", help="run the full verification")
    p.add_argument("--fast", action="store_true", help="skip slow tests")
    p.set_defaults(func=_cmd_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))
