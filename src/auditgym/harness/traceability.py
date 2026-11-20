"""Requirement -> implementation -> tests matrix (docs/TRACEABILITY.md), built from pytest's list.

Tests carry the requirement code in their name (`test_<CODE>_...`), so the matrix is derived from
`pytest --collect-only` and cannot drift from the test suite.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

IMPLEMENTED_IN: dict[str, str] = {
    "C01": "вся архитектура: среда, а не приложение",
    "C02": "env/, trajectories/, grading/",
    "C03": "только stdlib в рантайме",
    "C04": "DESIGN_NOTE §8",
    "G01": "inject/ (4 инжектора, config.errors)",
    "G02": "inject/ (2 сценария, config.interactions)",
    "G03": "world/ (X1–X10), config.clean",
    "G04": "world → books → inject, provenance.changes",
    "G05": "truth.evidence_sets, verify/evidence.py, policy.evidence",
    "G06": "world/, present/, policy",
    "G07": "ловушки X1–X10, harness/probes.py",
    "D01": "config.size, max_issues ≤ 12",
    "D02": "contracts/meta.py make_manifest",
    "D03": "util/money.py, util/timeutil.py",
    "D04": "contracts/types.py LedgerRow",
    "D05": "contracts/types.py Invoice",
    "D06": "contracts/types.py Payment",
    "D07": "contracts/types.py FxRate",
    "D08": "contracts/meta.py make_policy",
    "D09": "canon.csv_text, types, policy.formats",
    "D10": "truth.json, provenance.json",
    "D11": "server/ (закрытый режим, расписание), env/session.py",
    "D12": "env/observe.py, contracts/errors.py",
    "E01": "AuditEnv.reset",
    "E02": "AuditEnv.step",
    "E03": "util/rng.py, util/canon.py",
    "E04": "env/actions.py",
    "E05": "env/store.py",
    "E06": "env/calculator.py",
    "E07": "contracts/errors.py",
    "E08": "env/observe.py",
    "E09": "env/session.py",
    "E10": "env/session.py truncate(), server/host.py (лимит времени)",
    "E11": "env/session.py",
    "E12": "mode=closed, grader_log",
    "S01": "grading/parse.py",
    "S02": "grading/parse.py",
    "S03": "grade_submission valid=False",
    "S04": "grading/match.py",
    "S05": "grading/match.py",
    "S06": "docstring grading/",
    "S07": "grading/reward.py",
    "S08": "grading/reward.py",
    "S09": "explanation не читается",
    "S10": "tests/reward",
    "S11": "tests/reward",
    "V01": "configs/splits.json",
    "V02": "схема конфигурации, harness/transform.py",
    "V03": "configs/clean.v1.json, configs/interacting.v1.json",
    "V04": "harness/run.py aggregate",
    "K1": "harness/jury.py k1_checks",
    "K2": "harness/sweep.py",
    "K3": "harness/jury.py k3_checks",
    "K4": "harness/transform.py, k4_checks",
    "K5": "README, configs, trajectories, DESIGN_NOTE",
    "J01": "judge_adapter.py, auditgym/judge.py",
    "J02": "auditgym/judge.py (оболочки и формы результатов)",
    "J03": "JudgeEnv.reset: episode_dir, team_config, mode",
    "J04": "auditgym/judge.py score_episode",
    "J05": "JudgeEnv: 60 с от начала reset",
    "J06": "grading/parse.py, grading/match.py (known_ids)",
    "J07": "grading/match.py match_findings",
    "J08": "contracts/io.py read_episode, harness/judge_check.py block4",
    "J09": "contracts/config.py (variants), harness/judge_check.py d_matrix",
    "J10": "README, раздел «Адаптер жюри»",
}
ARCHITECTURAL = {"C01": "архитектурное требование, см. DESIGN_NOTE §1"}


def collect_tests(repo: Path) -> list[str]:
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return sorted({re.sub(r"\[.*", "", line) for line in out.splitlines() if "::" in line})


def build(repo: Path) -> str:
    tests = collect_tests(repo)
    by_code: dict[str, Counter[str]] = defaultdict(Counter)
    for item in tests:
        path, name = item.split("::", 1)
        for code in IMPLEMENTED_IN:
            if re.search(rf"test_{code}(_|$)", name) or re.search(rf"_{code}_", name):
                by_code[code][Path(path).name] += 1
    lines = [
        "# Трассировка требований",
        "",
        "Каждое требование из [REQUIREMENTS.md](REQUIREMENTS.md): где реализовано и какие тесты его "
        "проверяют (по имени теста, `test_<код>_…`). Файл собирается командой "
        "`python -m auditgym traceability` из списка тестов `pytest --collect-only`.",
        "",
        f"Тестовых функций: {len(tests)}; с параметризацией проверок больше тысячи.",
        "",
        "| Код | Где реализовано | Тесты |",
        "|---|---|---|",
    ]
    for code, where in IMPLEMENTED_IN.items():
        files = by_code.get(code)
        cell = (
            ", ".join(f"`{f}` ({n})" for f, n in sorted(files.items()))
            if files
            else ARCHITECTURAL.get(code, "—")
        )
        lines.append(f"| {code} | {where} | {cell} |")
    return "\n".join(lines) + "\n"
