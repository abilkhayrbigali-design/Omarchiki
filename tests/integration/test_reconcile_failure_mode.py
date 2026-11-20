"""Measured failure mode: with incomplete books the solver refuses rather than guesses."""

from dataclasses import replace

import pytest

from auditgym.contracts.config import load_config
from auditgym.episode.generate import generate_episode
from auditgym.verify import solve
from auditgym.verify.errors import VerifyError


@pytest.mark.parametrize("seed", range(500, 510))
def test_solver_refuses_on_truncated_ledger(seed):
    pub = generate_episode(seed, load_config("configs/dev.v1.json")).public
    cut = replace(pub, ledger=pub.ledger[: int(len(pub.ledger) * 0.7)])
    with pytest.raises(VerifyError):
        solve(cut)
