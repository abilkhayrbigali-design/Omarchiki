"""K1-K4 as the jury runs them: the published check definitions through judge_adapter.py.

auditgym.harness.judge_check rebuilds the four blocks of the FinTech checks (contract C01-C20,
closed episodes D01-D20, matcher and reward T01-T20 with the open examples M01-M12, pairs
P01-P20) on fixtures written to disk and runs them through the same three entry points the jury
calls. Each check is one test here, named after its block's criterion.
"""

from __future__ import annotations

import pytest

from auditgym.harness.judge_check import run_judge_checks

C_IDS = [f"C{k:02d}" for k in range(1, 21)]
D_IDS = [f"D{k:02d}" for k in range(1, 21)]
T_IDS = [f"T{k:02d}" for k in range(1, 21)]
M_IDS = [f"M{k:02d}" for k in range(1, 13)]
P_IDS = [f"P{k:02d}" for k in range(1, 21)]


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    return run_judge_checks(tmp_path_factory.mktemp("judge"))


def _result(report, block, check_id):
    (row,) = [r for r in report[block] if r["id"] == check_id]
    return row


@pytest.mark.parametrize("check_id", C_IDS)
def test_K1_J02_published_contract_check_through_adapter(report, check_id):
    row = _result(report, "block1", check_id)
    assert row["ok"], row["detail"]


@pytest.mark.parametrize("check_id", D_IDS)
def test_K2_J09_published_closed_episode_through_adapter(report, check_id):
    row = _result(report, "block2", check_id)
    assert row["ok"], row["detail"]


@pytest.mark.parametrize("check_id", T_IDS + M_IDS)
def test_K3_J06_published_matcher_and_reward_check_through_adapter(report, check_id):
    row = _result(report, "block3", check_id)
    assert row["ok"], row["detail"]


@pytest.mark.parametrize("check_id", P_IDS)
def test_K4_J08_published_pair_check_through_adapter(report, check_id):
    row = _result(report, "block4", check_id)
    assert row["ok"], row["detail"]


def test_K1_K4_every_published_check_is_run(report):
    assert [r["id"] for r in report["block1"]] == C_IDS
    assert [r["id"] for r in report["block2"]] == D_IDS
    assert [r["id"] for r in report["block3"]] == T_IDS + M_IDS
    assert [r["id"] for r in report["block4"]] == P_IDS
    assert report["summary"]["block1"]["points"] == 25.0


# ------------------------------------------------------------------------------ the checks bite
def _failed(results):
    return [r["id"] for r in results if not r["ok"]]


def test_K3_J06_check_catches_a_grader_without_the_evidence_existence_rule(tmp_path, monkeypatch):
    import auditgym.judge as judge
    from auditgym.harness.judge_check import block3

    real = judge.grade_submission

    def lax(*args, known_ids=None, **kwargs):
        return real(*args, **kwargs)

    monkeypatch.setattr(judge, "grade_submission", lax)
    assert _failed(block3(tmp_path)) == ["M04"]


def test_K1_J05_check_catches_an_adapter_without_the_time_limit(tmp_path, monkeypatch):
    import functools

    import auditgym.harness.judge_check as jc
    import auditgym.judge as judge

    monkeypatch.setattr(jc, "make_env", functools.partial(judge.make_env, time_limit_s=1e9))
    assert _failed(jc.block1(tmp_path)) == ["C18"]


def test_K4_J08_check_catches_a_copy_with_a_stale_truth(tmp_path, monkeypatch):
    import shutil

    import auditgym.harness.judge_check as jc

    real = jc.jury_transform

    def stale(src, dst, **kwargs):
        mapping = real(src, dst, **kwargs)
        shutil.copy(src / "private" / "truth.json", dst / "private" / "truth.json")
        return mapping

    monkeypatch.setattr(jc, "jury_transform", stale)
    # P01-P05 only permute rows, so the old truth still fits; every renaming pair must fail.
    assert _failed(jc.block4(tmp_path)) == [f"P{k:02d}" for k in range(6, 21)]
