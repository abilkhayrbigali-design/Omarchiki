"""Shared fixtures. Owned by the integrator."""

from __future__ import annotations

from pathlib import Path

import pytest

from auditgym.contracts.config import load_config
from auditgym.contracts.io import write_episode
from auditgym.fixtures.tiny import build_tiny_episode

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
SCHEMA_DIR = REPO_ROOT / "schemas"


@pytest.fixture
def tiny_episode():
    return build_tiny_episode()


@pytest.fixture
def tiny_episode_dir(tmp_path, tiny_episode):
    out = tmp_path / "tiny"
    write_episode(tiny_episode, out)
    return out


@pytest.fixture
def smoke_config():
    return load_config(CONFIG_DIR / "smoke.v1.json")
