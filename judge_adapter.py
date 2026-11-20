"""Jury adapter of AuditGym (FinTech checks, "Единый способ подключения").

Run from the repository root (in the container: /app), where `src/` is on the import path:

    from judge_adapter import generate_episode, make_env, score_episode

The implementation and the full description of names, results and errors are in
src/auditgym/judge.py; README.md, section "Адаптер жюри", has the summary.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from auditgym.judge import (  # noqa: E402
    SCORE_LOG,
    JudgeEnv,
    generate_episode,
    make_env,
    score_episode,
)

__all__ = ["SCORE_LOG", "JudgeEnv", "generate_episode", "make_env", "score_episode"]
