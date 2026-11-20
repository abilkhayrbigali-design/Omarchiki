"""Fixtures for the environment contract tests (K1). Owned by WP3."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from envkit import RecordingGrader, env_factory  # noqa: E402


@pytest.fixture
def grader() -> RecordingGrader:
    return RecordingGrader()


@pytest.fixture
def make_env(grader):
    return env_factory(grader)
