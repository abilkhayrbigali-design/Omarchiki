"""Blind swap: the same episode with different truths gives identical observations (D12, K1).

Proves that nothing the agent sees before submit depends on the reference answer. In closed mode
the whole stream, submit and later steps included, must be identical too.
"""

from __future__ import annotations

import pytest
from envkit import (
    TruthGrader,
    act,
    alternative_truths,
    big_episode,
    list_page,
    oracle_submission,
    scripted_run,
    with_truth,
)

from auditgym.env import AuditEnv
from auditgym.fixtures.tiny import build_tiny_episode
from auditgym.util import canon


def _run(episode, mode, payload):
    env = AuditEnv(mode=mode, generate_fn=lambda s, c=None: episode, grader=TruthGrader())
    first = env.reset(11)
    steps = scripted_run(env, submit=payload)
    return first, steps


@pytest.mark.parametrize("make_episode", [build_tiny_episode, lambda: big_episode(70)])
def test_D12_blind_swap_open_mode_identical_until_submit(make_episode):
    base = make_episode()
    payload = oracle_submission(base)
    runs = [_run(with_truth(base, t), "open", payload) for t in alternative_truths(base)]
    first_bytes = {canon.dumps(list(first)) for first, _ in runs}
    assert len(first_bytes) == 1
    n_before_submit = len(runs[0][1]) - 2  # last two steps: submit, then a step after the end
    for i in range(n_before_submit):
        assert len({canon.dumps(list(steps[i])) for _, steps in runs}) == 1, f"step {i + 1}"
    # The swap is real: the grader saw different truths, so the open-mode rewards differ.
    rewards = [steps[-2][1] for _, steps in runs]
    assert len(set(rewards)) == len(rewards)
    # Only the reward differs on submit; the observation itself is the same.
    assert len({canon.dumps(steps[-2][0]) for _, steps in runs}) == 1


@pytest.mark.parametrize("make_episode", [build_tiny_episode, lambda: big_episode(70)])
def test_D12_blind_swap_closed_mode_identical_whole_stream(make_episode):
    base = make_episode()
    payload = oracle_submission(base)
    streams = {
        canon.dumps([list(first), [list(s) for s in steps]])
        for first, steps in (
            _run(with_truth(base, t), "closed", payload) for t in alternative_truths(base)
        )
    }
    assert len(streams) == 1


def test_D12_blind_swap_truncation_closed_mode():
    base = build_tiny_episode()
    streams = set()
    for truth in alternative_truths(base):
        env = AuditEnv(
            mode="closed",
            generate_fn=lambda s, c=None, t=truth: with_truth(base, t),
            grader=TruthGrader(),
        )
        out = [list(env.reset(1))]
        for i in range(101):
            out.append(list(env.step(list_page("fx", None, 1) if i % 3 else act("x"))))
        streams.add(canon.dumps(out))
    assert len(streams) == 1
