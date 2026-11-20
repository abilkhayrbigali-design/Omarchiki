"""Independence of the solver: verify/ imports neither the generator nor the presentation layer
(AGENTS.md rule 7, ARCHITECTURE import boundaries), and uses no forbidden nondeterminism."""

import ast
from pathlib import Path

import auditgym.verify

VERIFY_DIR = Path(auditgym.verify.__file__).parent
FORBIDDEN_PACKAGES = ("world", "inject", "episode", "books", "present", "grading", "env")
FORBIDDEN_MODULES = ("random", "time", "uuid", "os", "socket", "subprocess")


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


def test_verify_does_not_import_generator_or_presentation():
    files = sorted(VERIFY_DIR.glob("*.py"))
    assert files
    for path in files:
        for module in _imports(path):
            parts = module.split(".")
            if parts[0] == "auditgym":
                assert parts[1] not in FORBIDDEN_PACKAGES, (path.name, module)
            assert parts[0] not in FORBIDDEN_MODULES, (path.name, module)


def test_verify_has_no_float_eval_or_builtin_hash():
    for path in sorted(VERIFY_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in ("float", "eval", "exec", "hash"), path.name
            assert not (isinstance(node, ast.Constant) and isinstance(node.value, float)), path.name
