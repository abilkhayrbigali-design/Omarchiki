"""J01: judge_adapter.py, the jury's single connection protocol (docs/REQUIREMENTS.md, section 8)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from auditgym.contracts.constants import INTERACTION_SCENARIOS, ISSUE_TYPES, MAX_STEPS
from auditgym.contracts.io import read_episode
from auditgym.judge import SCORE_LOG, generate_episode, make_env, open_ids_of_dir, score_episode

REPO = Path(__file__).resolve().parents[2]
ONE_DUP = {
    "config_id": "t.one",
    "errors": {t: [1, 1] if t == "duplicate_invoice" else [0, 0] for t in ISSUE_TYPES},
    "interactions": {s: [0, 0] for s in INTERACTION_SCENARIOS},
}


@pytest.fixture(scope="module")
def ep_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("ep") / "one"
    generate_episode(42, ONE_DUP, str(d))
    return d


def _oracle(d: Path) -> list[dict]:
    truth = json.loads((d / "private" / "truth.json").read_text("utf-8"))
    return [
        {
            "type": i["type"],
            "affected_ids": i["affected_ids"],
            "claimed_amount_minor": i["expected_amount_minor"],
            "currency": i["currency"],
            "evidence_ids": i["evidence_sets"][0],
            "explanation": "",
        }
        for i in truth["issues"]
    ]


def test_J01_root_module_exposes_the_three_entry_points():
    spec = importlib.util.spec_from_file_location("judge_adapter", REPO / "judge_adapter.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("generate_episode", "make_env", "score_episode"):
        assert callable(getattr(module, name))
    env = module.make_env()
    assert callable(env.reset) and callable(env.step)


def test_J01_generate_episode_writes_public_and_closed_files_reproducibly(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    generate_episode(42, ONE_DUP, str(a))
    generate_episode(42, {"team_config": ONE_DUP}, str(b))  # the jury's wrapper is accepted too
    names = sorted(p.relative_to(a).as_posix() for p in a.rglob("*") if p.is_file())
    assert names == [
        "fx.csv",
        "invoices.jsonl",
        "ledger.csv",
        "manifest.json",
        "payments.jsonl",
        "policy.json",
        "private/provenance.json",
        "private/truth.json",
    ]
    for name in names:
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_J03_closed_reset_loads_the_directory_as_is_and_restarts_it(ep_dir):
    env = make_env()
    obs, info = env.reset(None, {"mode": "closed", "episode_dir": str(ep_dir)})
    assert sorted(obs) == ["manifest", "policy", "remaining_steps", "task"]
    assert obs["manifest"]["episode_id"] == read_episode(ep_dir).public.episode_id
    assert info["mode"] == "closed" and obs["remaining_steps"] == MAX_STEPS
    env.step({"name": "calculator", "args": {"expression": "1+1"}})
    again, _ = env.reset(None, {"mode": "closed", "episode_dir": str(ep_dir)})
    assert again == obs
    assert env.grade_log[-1]["end"] == "abandoned"  # a reset is never a free retry


def test_J03_train_mode_generates_from_seed_and_returns_the_terminal_reward(tmp_path):
    env = make_env()
    obs, _ = env.reset(42, {"mode": "train", "team_config": ONE_DUP})
    d = tmp_path / "same"
    generate_episode(42, ONE_DUP, str(d))
    assert obs["manifest"] == json.loads((d / "manifest.json").read_text("utf-8"))
    eid = obs["manifest"]["episode_id"]
    _, reward, term, trunc, _ = env.step(
        {"name": "submit", "args": {"findings": {"episode_id": eid, "findings": _oracle(d)}}}
    )
    assert term and not trunc and reward == 1.0


def test_J02_both_envelopes_and_short_findings_are_accepted(ep_dir):
    closed = {"mode": "closed", "episode_dir": str(ep_dir)}
    for action in (
        {"name": "submit", "args": {"findings": _oracle(ep_dir)}},  # bare list
        {"action": "submit", "args": {"findings": {"findings": _oracle(ep_dir)}}},  # no episode_id
    ):
        env = make_env()
        env.reset(None, closed)
        obs, reward, term, _, _ = env.step(action)
        assert term and reward == 0.0 and obs["result"] == {"accepted": True}
        assert env.grade_log[-1]["valid"] is True and env.grade_log[-1]["tp"] == 1


def test_J02_results_have_the_published_shapes(ep_dir):
    env = make_env()
    env.reset(None, {"mode": "closed", "episode_dir": str(ep_dir)})
    page, *_ = env.step({"name": "list_records", "args": {"table": "invoices", "cursor": None}})
    assert sorted(page["result"]) == ["next_cursor", "records"]
    doc_id = page["result"]["records"][0]["doc_id"]
    rec, *_ = env.step({"name": "get_record", "args": {"table": "invoices", "id": doc_id}})
    doc, *_ = env.step({"name": "get_document", "args": {"doc_id": doc_id}})
    calc, *_ = env.step({"name": "calculator", "args": {"expression": "10/4"}})
    assert rec["result"]["doc_id"] == doc_id and "line_items" not in rec["result"]
    assert doc["result"]["doc_id"] == doc_id and isinstance(doc["result"]["line_items"], list)
    assert calc["result"] == "2.5"
    for obs in (page, rec, doc, calc):
        assert sorted(obs) == ["error", "remaining_steps", "result"] and obs["error"] is None


def test_J04_score_episode_reports_the_seven_values(ep_dir):
    eid = read_episode(ep_dir).public.episode_id
    good = score_episode(str(ep_dir), {"episode_id": eid, "findings": _oracle(ep_dir)}, 10)
    assert good == {
        "tp": 1,
        "fp": 0,
        "fn": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "return": 0.98,
    }
    bad = score_episode(str(ep_dir), {"episode_id": eid, "findings": [{"type": 1}]}, 10)
    assert bad == {
        "tp": None,
        "fp": None,
        "fn": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "return": -1.0,
    }
    assert SCORE_LOG[-1]["valid"] is False and SCORE_LOG[-1]["error"]


def test_J04_grader_log_file_is_written_only_when_the_jury_names_it(ep_dir, tmp_path, monkeypatch):
    log = tmp_path / "grader.jsonl"
    monkeypatch.setenv("AUDITGYM_GRADER_LOG", str(log))
    eid = read_episode(ep_dir).public.episode_id
    score_episode(str(ep_dir), {"episode_id": eid, "findings": []}, 0)
    env = make_env()
    env.reset(None, {"mode": "closed", "episode_dir": str(ep_dir)})
    env.step({"name": "submit", "args": {"findings": []}})
    records = [json.loads(x) for x in log.read_text("utf-8").splitlines()]
    assert len(records) == 2 and all("reward" in r for r in records)


def test_J01_open_ids_of_a_directory_equal_the_episode_open_ids(ep_dir):
    assert open_ids_of_dir(ep_dir) == read_episode(ep_dir).public.open_ids()


def test_J03_closed_mode_leaks_nothing_through_info_or_reward(ep_dir):
    env = make_env()
    _, info = env.reset(None, {"mode": "closed", "episode_dir": str(ep_dir)})
    out = env.step({"name": "submit", "args": {"findings": _oracle(ep_dir)}})
    assert out[1] == 0.0 and out[4] == {"reward_withheld": True}
    assert set(info) == {"episode_id", "mode", "env_version", "max_steps"}
    with pytest.raises(ValueError):
        env.reset(None, {"mode": "exam", "episode_dir": str(ep_dir)})


def test_J05_time_limit_counts_from_the_start_of_reset(ep_dir, monkeypatch):
    import auditgym.judge as judge

    class Clock:
        now = 0.0

        def __call__(self):
            return self.now

    clock = Clock()
    real_read = judge.read_episode

    def slow_read(d):
        clock.now += 50.0  # loading the episode takes 50 s of the 60
        return real_read(d)

    monkeypatch.setattr(judge, "read_episode", slow_read)
    env = make_env(clock=clock)
    env.reset(None, {"mode": "closed", "episode_dir": str(ep_dir)})
    clock.now = 59.0
    obs, _, term, trunc, _ = env.step({"name": "calculator", "args": {"expression": "1+1"}})
    assert not term and not trunc and obs["remaining_steps"] == MAX_STEPS - 1
    clock.now = 60.5  # 10.5 s after reset returned, 60.5 s after it started
    obs, _, term, trunc, _ = env.step({"name": "submit", "args": {"findings": _oracle(ep_dir)}})
    assert trunc and not term and obs["remaining_steps"] == MAX_STEPS - 1  # not executed
    log = env.grade_log[-1]
    assert log["end"] == "timeout" and (log["tp"], log["fp"]) == (0, 0)


def test_J10_readme_documents_the_adapter():
    text = (REPO / "README.md").read_text("utf-8")
    for needle in (
        "judge_adapter.py",
        "generate_episode",
        "make_env",
        "score_episode",
        "remaining_steps",
        "episode_dir",
        "team_config",
        "AUDITGYM_GRADER_LOG",
        "/app",
        *ISSUE_TYPES,
    ):
        assert needle in text, needle
