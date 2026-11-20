"""Import boundaries and banned constructs, checked on the source with `ast` (P3, E06, rule 5)."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "auditgym"

# Code that builds what the agent sees, or runs in the agent's process.
OBSERVATION_SIDE = [
    SRC / "env" / "observe.py",
    SRC / "env" / "store.py",
    SRC / "env" / "actions.py",
    SRC / "env" / "calculator.py",
    SRC / "server" / "protocol.py",
    SRC / "client.py",
]
ALL_WP3 = sorted([*(SRC / "env").glob("*.py"), *(SRC / "server").glob("*.py"), SRC / "client.py"])

TRUTH_SIDE = (
    "auditgym.grading",
    "auditgym.inject",
    "auditgym.world",
    "auditgym.verify",
    "auditgym.episode",
    "auditgym.books",
    "auditgym.fixtures",
    "auditgym.policies",
    "auditgym.harness",
    "auditgym.contracts.grading_api",
    "auditgym.contracts.io",
    "auditgym.env.session",
    "auditgym.server.app",
)
NEVER_IN_WP3 = ("auditgym.inject", "auditgym.world", "auditgym.verify", "auditgym.books")
BANNED_MODULES = (
    "random",
    "time",
    "uuid",
    "secrets",
    "socket",
    "urllib",
    "http",
    "pickle",
    "marshal",
    "shelve",
    "ctypes",
    "importlib",
    "shutil",
    "tempfile",
    "multiprocessing",
    "threading",
    "ast",
)
BANNED_CALLS = ("eval", "exec", "compile", "__import__", "breakpoint", "input")
# The one place that may read a clock: the server's episode host, for the wall-clock limit, and
# only through time.monotonic. The generator never reads a clock; the environment reads this
# one, and only when it was given `time_limit_s` (tests/contract/test_env_time_limit.py).
CLOCK_MODULE = SRC / "server" / "host.py"


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                out.append("." * node.level + (node.module or ""))
            else:
                out.append(node.module or "")
                out += [f"{node.module}.{alias.name}" for alias in node.names]
    return out


def _matches(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == p or name.startswith(p + ".") for p in prefixes)


@pytest.mark.parametrize("path", OBSERVATION_SIDE, ids=lambda p: p.name)
def test_K1_observation_code_never_imports_truth_side(path):
    imports = _imports(path)
    assert not [n for n in imports if n.startswith(".")], "relative imports hide the boundary"
    bad = [n for n in imports if _matches(n, TRUTH_SIDE)]
    assert not bad, bad


@pytest.mark.parametrize("path", ALL_WP3, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_K1_wp3_never_imports_generator_internals_or_solver(path):
    bad = [n for n in _imports(path) if _matches(n, NEVER_IN_WP3)]
    assert not bad, bad


@pytest.mark.parametrize("path", ALL_WP3, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_E06_no_eval_exec_or_banned_modules(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in BANNED_CALLS, f"{path.name}: {node.func.id}()"
        if isinstance(node, ast.Attribute):
            assert node.attr not in ("literal_eval", "system", "popen"), f"{path.name}: {node.attr}"
    banned = BANNED_MODULES
    if path == CLOCK_MODULE:
        banned = tuple(m for m in BANNED_MODULES if m != "time")
    bad = [n for n in _imports(path) if _matches(n, banned)]
    assert not bad, bad
    allowed_subprocess = path == SRC / "client.py"
    if not allowed_subprocess:
        assert not [n for n in _imports(path) if _matches(n, ("subprocess",))]


@pytest.mark.parametrize(
    "path", [p for p in OBSERVATION_SIDE if p.parent.name == "env"], ids=lambda p: p.name
)
def test_E08_observation_code_has_no_file_access(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "open"
        if isinstance(node, ast.Attribute):
            assert node.attr not in ("read_text", "read_bytes", "write_text", "write_bytes")
    assert not [n for n in _imports(path) if _matches(n, ("os", "pathlib", "io", "sys"))]


def test_E10_clock_is_read_only_by_the_server_host_via_monotonic():
    """`import time` only in server/host.py, and the only use there is time.monotonic."""
    for path in ALL_WP3:
        if path != CLOCK_MODULE:
            assert not [n for n in _imports(path) if _matches(n, ("time", "datetime"))], path
    tree = ast.parse(CLOCK_MODULE.read_text(encoding="utf-8"))
    imports = _imports(CLOCK_MODULE)
    assert "time" in imports and not [n for n in imports if n.startswith("time.")]
    uses = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "time"
    }
    assert uses == {"monotonic"}


def test_K1_session_imports_grader_and_generator_only_lazily():
    """The controller may reach the grader, but only inside functions (at first use)."""
    path = SRC / "env" / "session.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top_level = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = [n.module for n in top_level if isinstance(n, ast.ImportFrom)]
    assert not [n for n in names if n and _matches(n, ("auditgym.grading", "auditgym.episode"))]


def test_K1_importing_observe_does_not_load_the_session():
    code = (
        "import sys, auditgym.env.observe, auditgym.env.store, auditgym.env.actions\n"
        "print('\\n'.join(sorted(m for m in sys.modules if m.startswith('auditgym'))))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.split()
    assert "auditgym.env.observe" in out
    assert "auditgym.env.session" not in out
    assert "auditgym.contracts.grading_api" not in out


def test_K1_importing_env_and_client_loads_no_truth_side_module():
    code = (
        "import sys, auditgym.env, auditgym.env.observe, auditgym.client, auditgym.server.protocol\n"
        "print('\\n'.join(sorted(m for m in sys.modules if m.startswith('auditgym'))))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.split()
    forbidden = (
        "auditgym.grading",
        "auditgym.inject",
        "auditgym.world",
        "auditgym.verify",
        "auditgym.episode",
        "auditgym.books",
        "auditgym.fixtures",
    )
    assert not [m for m in out if _matches(m, forbidden)], out
