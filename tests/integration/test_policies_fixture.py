"""End-to-end: real environment + real grader + baseline policies on the tiny fixture."""

import json

import pytest

from auditgym.harness.run import run_episode
from auditgym.policies.reconcile import Reconcile
from auditgym.policies.simple import FlagAll, Oracle, RandomExplorer


def test_oracle_gets_maximum_reward(tiny_episode):
    run = run_episode(Oracle(tiny_episode.private.truth), tiny_episode)
    assert run.grade["f1"] == 1.0 and run.grade["reward"] == 1.0 and run.terminated


def test_reconcile_solves_fixture_through_tools(tiny_episode):
    run = run_episode(Reconcile(), tiny_episode)
    g = run.grade
    assert (g["tp"], g["fp"], g["fn"]) == (4, 0, 0)
    assert g["reward"] == pytest.approx(1 - 0.002 * g["inspection_steps"])


def test_flag_all_scores_badly(tiny_episode):
    run = run_episode(FlagAll(), tiny_episode)
    assert run.grade["valid"] and run.grade["reward"] < 0


def test_random_explorer_earns_nothing(tiny_episode):
    run = run_episode(RandomExplorer(seed=3, budget=20), tiny_episode)
    assert run.grade["tp"] == 0 and run.grade["reward"] == -1.0


@pytest.mark.parametrize("mode", ["open", "closed"])
def test_trajectory_lines_match_schema_and_hold_no_truth(tiny_episode, mode):
    from pathlib import Path

    from jsonschema import Draft202012Validator

    schema_dir = Path(__file__).resolve().parents[2] / "schemas"
    schema = json.loads((schema_dir / "trajectory_step.schema.json").read_text())
    run = run_episode(Reconcile(), tiny_episode, mode=mode)
    blob = json.dumps([line["observation"] for line in run.trajectory])
    for line in run.trajectory:
        Draft202012Validator(schema).validate(line)
    for marker in ("issue_id", "iss_tiny", "expected_amount", "evidence_sets", "provenance"):
        assert marker not in blob
    if mode == "closed":
        assert all(line["reward"] == 0.0 for line in run.trajectory)
