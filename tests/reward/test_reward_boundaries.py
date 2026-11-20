"""S06 and AGENTS.md rules 1, 2, 5: the published grader is small, pure and self-contained.

- `grading/` imports only the standard library, `auditgym.contracts` and `auditgym.util`;
- no randomness, clocks, hashing of objects, eval/exec, file or network access;
- `float` is used only to convert the final metrics and rewards.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

GRADING_DIR = Path(__file__).resolve().parents[2] / "src" / "auditgym" / "grading"
SOURCES = sorted(GRADING_DIR.glob("*.py"))
ALLOWED_INTERNAL = ("auditgym.contracts", "auditgym.util", "auditgym.grading")
FORBIDDEN_MODULES = {
    "random",
    "time",
    "datetime",
    "uuid",
    "os",
    "socket",
    "subprocess",
    "pathlib",
    "io",
    "shutil",
    "urllib",
    "http",
    "pickle",
    "secrets",
}
FORBIDDEN_CALLS = {"eval", "exec", "compile", "hash", "open", "__import__", "input"}


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imports(path: Path) -> list[str]:
    names = []
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
    return names


def test_S06_grading_package_exists_and_is_published():
    assert {p.name for p in SOURCES} >= {"__init__.py", "parse.py", "match.py", "reward.py"}


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_S06_grading_imports_only_stdlib_contracts_and_util(path):
    for name in _imports(path):
        top = name.split(".")[0]
        if top == "auditgym":
            assert name.startswith(ALLOWED_INTERNAL), f"{path.name} imports {name}"
        else:
            assert top in sys.stdlib_module_names or top == "__future__", f"{path.name}: {name}"
        assert top not in FORBIDDEN_MODULES, f"{path.name} imports {name}"


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_S06_grading_has_no_forbidden_calls(path):
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in FORBIDDEN_CALLS, f"{path.name}:{node.lineno}"


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_S08_float_only_converts_final_numbers(path):
    """`float(...)` appears only in grade.py and reward.py, i.e. on final metrics and rewards."""
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "float":
                assert path.name in {"grade.py", "reward.py"}, f"{path.name}:{node.lineno}"
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            # -1.0 parses as USub(1.0): the invalid reward and zero metrics only.
            assert path.name == "grade.py" and node.value in {1.0, 0.0}, path.name
