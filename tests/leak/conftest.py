"""Fixtures for the leak tests (K1, D11, D12). Owned by WP3."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Shared helpers live next to the contract tests.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "contract"))

from envkit import RecordingGrader, env_factory  # noqa: E402


@pytest.fixture
def grader() -> RecordingGrader:
    return RecordingGrader()


@pytest.fixture
def make_env(grader):
    return env_factory(grader)
