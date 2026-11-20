"""The jury simulator's fixed contract (K1) and reward (K3) checks, on the hand-built fixture.

The same checks run on generated episodes in `python -m auditgym jury` and `check`.
"""

import sys
from pathlib import Path

import pytest

from auditgym.contracts.types import Episode, PrivateEpisode, Truth
from auditgym.fixtures.tiny import build_tiny_episode
from auditgym.harness.jury import k1_checks, k3_checks

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "verify"))
from verify_builders import clean_tiny  # noqa: E402

TINY = build_tiny_episode()
_CLEAN_PUB = clean_tiny()
CLEAN = Episode(_CLEAN_PUB, PrivateEpisode(Truth(_CLEAN_PUB.episode_id, ()), {}))


@pytest.mark.parametrize(
    "name,check", k1_checks(TINY), ids=lambda x: x if isinstance(x, str) else ""
)
def test_K1_jury_check(name, check):
    ok, detail = check()
    assert ok, f"{name}: {detail}"


@pytest.mark.parametrize(
    "name,check", k3_checks(TINY, CLEAN, TINY), ids=lambda x: x if isinstance(x, str) else ""
)
def test_K3_jury_check(name, check):
    ok, detail = check()
    assert ok, f"{name}: {detail}"
