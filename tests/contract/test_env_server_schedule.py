"""Closed-mode server: a schedule the agent cannot choose, no free retries (D11, E03, E10, E12).

The server runs in-process over scripted byte streams (`serverkit`), with a stand-in grader.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from envkit import RecordingGrader, act, list_page, oracle_submission, with_episode_id
from serverkit import (
    episode_ids,
    line,
    read_log,
    run_server,
    write_episode_at,
    write_episodes,
)

from auditgym.contracts.config import normalize_config
from auditgym.contracts.constants import MAX_STEPS
from auditgym.fixtures.tiny import build_tiny_episode
from auditgym.server.app import main

INVALID = {"ok": False, "error": {"code": "invalid_request", "message": "Malformed request line."}}
EXHAUSTED = {
    "ok": False,
    "error": {"code": "schedule_exhausted", "message": "No more scheduled episodes."},
}


@pytest.fixture
def log(tmp_path) -> Path:
    return tmp_path / "organizer" / "grader_log.jsonl"


def _closed(log: Path, *sources: str) -> list[str]:
    return ["--mode", "closed", "--grader-log", str(log), *sources]


def _dirs_args(dirs: list[Path]) -> list[str]:
    out: list[str] = []
    for d in dirs:
        out += ["--episode-dir", str(d)]
    return out


def _submit(eid: str, findings=None) -> dict:
    payload = {"episode_id": eid, "findings": findings or []}
    return {"op": "step", "action": act("submit", findings=payload)}


# ------------------------------------------------------------------ client cannot choose
CHOOSING_RESETS = [
    {"op": "reset", "seed": 7},
    {"op": "reset", "seed": 7, "config": None},
    {"op": "reset", "config": None},
    {"op": "reset", "config": {"config_id": "mine"}},
    {"op": "reset", "seed": 0, "config": {"clean": True}},
    {"op": "reset", "episode": "/tmp/my_episode"},
    {"op": "reset", "episode_dir": "/tmp/my_episode"},
    {"op": "reset", "schedule": 3},
]


@pytest.mark.parametrize("request_", CHOOSING_RESETS, ids=range(len(CHOOSING_RESETS)))
def test_D11_closed_client_cannot_choose_seed_config_or_episode(tmp_path, log, request_):
    ids = episode_ids(2)
    dirs = write_episodes(tmp_path / "eps", ids)
    grader = RecordingGrader()
    _, responses, _ = run_server(
        _closed(log, *_dirs_args(dirs)),
        [
            line(request_),  # before any episode: refused, nothing consumed
            line({"op": "reset"}),
            line({"op": "step", "action": list_page("ledger", None, 1)}),
            line(request_),  # mid-episode: refused, the episode goes on untouched
            line({"op": "step", "action": list_page("ledger", None, 1)}),
            line({"op": "close"}),
        ],
        grader=grader,
    )
    assert responses[0] == INVALID and responses[3] == INVALID
    assert responses[1]["info"]["episode_id"] == ids[0]
    assert responses[2]["observation"]["steps_remaining"] == MAX_STEPS - 1
    assert responses[4]["observation"]["steps_remaining"] == MAX_STEPS - 2
    # Only the close ended the episode.
    (record,) = read_log(log)
    assert record["episode_id"] == ids[0] and record["end"] == "abandoned"
    assert record["step_count"] == 2


def test_D11_closed_episode_is_the_scheduled_one_not_the_generator_seed(tmp_path, log):
    """The reviewer's attack: pick seed/config, regenerate, submit the truth. Now refused."""
    ids = episode_ids(1)
    dirs = write_episodes(tmp_path / "eps", ids)
    generated = []

    def generate(seed, config=None):
        generated.append(seed)
        return build_tiny_episode()

    _, responses, _ = run_server(
        _closed(log, *_dirs_args(dirs)),
        [line({"op": "reset", "seed": 1234, "config": None}), line({"op": "reset"})],
        generate_fn=generate,
    )
    assert responses[0] == INVALID
    assert responses[1]["info"]["episode_id"] == ids[0]
    assert generated == []


# ------------------------------------------------------------------ schedule order
def test_D11_schedule_advances_and_exhausts(tmp_path, log):
    ids = episode_ids(3)
    dirs = write_episodes(tmp_path / "eps", ids)
    grader = RecordingGrader()
    _, responses, _ = run_server(
        _closed(log, *_dirs_args(dirs)),
        [
            line({"op": "reset"}),
            line(_submit(ids[0])),
            line({"op": "reset"}),
            line({"op": "step", "action": act("teleport")}),
            line({"op": "reset"}),  # abandons ids[1]
            line({"op": "truncate"}),
            line({"op": "reset"}),  # nothing left
            line({"op": "reset"}),
            line({"op": "step", "action": list_page("ledger")}),
        ],
        grader=grader,
    )
    resets = [r for r in responses if r.get("op") == "reset"]
    assert [r["info"]["episode_id"] for r in resets] == ids
    assert responses[6] == EXHAUSTED and responses[7] == EXHAUSTED
    assert responses[8]["observation"]["error"]["code"] == "episode_finished"
    records = read_log(log)
    assert [(r["episode_id"], r["end"]) for r in records] == [
        (ids[0], "submit"),
        (ids[1], "abandoned"),
        (ids[2], "timeout"),
    ]


def test_D11_parent_episode_dir_served_in_sorted_order(tmp_path, log):
    root = tmp_path / "split"
    ids = ["ep_ZZZ000000001", "ep_AAA000000002", "ep_MMM000000003"]
    for name, eid in zip(["b", "c", "a"], ids, strict=True):
        write_episode_at(root / name, eid)
    (root / ".cache").mkdir()  # dot-directories are skipped
    (root / "README.md").write_text("not an episode", encoding="utf-8")
    _, responses, _ = run_server(
        _closed(log, "--episode-dir", str(root)),
        [line({"op": "reset"})] * 4,
    )
    # a -> MMM, b -> ZZZ, c -> AAA (sorted by directory name, not by episode id)
    assert [r["info"]["episode_id"] for r in responses[:3]] == [ids[2], ids[0], ids[1]]
    assert responses[3] == EXHAUSTED


def test_D11_schedule_file_with_seeds_and_configs(tmp_path, log):
    cfg_dir = tmp_path / "private_cfg"
    cfg_dir.mkdir()
    file_cfg = {"config_id": "hidden-a", "clean": True}
    (cfg_dir / "a.json").write_text(json.dumps(file_cfg), encoding="utf-8")
    inline_cfg = {"config_id": "hidden-b"}
    schedule = cfg_dir / "schedule.json"
    schedule.write_text(
        json.dumps(
            [
                {"seed": 11, "config": "a.json"},  # relative to the schedule file
                {"seed": 12, "config": inline_cfg},
                {"seed": 13},
                {"seed": 14, "config": None},
            ]
        ),
        encoding="utf-8",
    )
    calls = []

    def generate(seed, config=None):
        calls.append((seed, config))
        return with_episode_id(build_tiny_episode(), f"ep_GEN{seed:09d}")

    _, responses, raw = run_server(
        _closed(log, "--schedule", str(schedule)),
        [line({"op": "reset"})] * 5,
        generate_fn=generate,
    )
    assert [r["info"]["episode_id"] for r in responses[:4]] == [
        "ep_GEN000000011",
        "ep_GEN000000012",
        "ep_GEN000000013",
        "ep_GEN000000014",
    ]
    assert responses[4] == EXHAUSTED
    assert calls == [
        (11, normalize_config(file_cfg)),
        (12, normalize_config(inline_cfg)),
        (13, None),
        (14, None),
    ]
    # Seeds and configs never reach the agent.
    text = raw.decode("utf-8")
    assert "hidden-a" not in text and "hidden-b" not in text and str(cfg_dir) not in text


def test_D11_sources_served_in_command_line_order(tmp_path, log):
    a = write_episodes(tmp_path / "a", ["ep_DIRA00000001"])
    b = write_episodes(tmp_path / "b", ["ep_DIRB00000001"])
    schedule = tmp_path / "s.json"
    schedule.write_text(json.dumps([{"seed": 5}]), encoding="utf-8")

    def generate(seed, config=None):
        return with_episode_id(build_tiny_episode(), "ep_SEED00000005")

    _, responses, _ = run_server(
        _closed(
            log, "--episode-dir", str(b[0]), "--schedule", str(schedule), "--episode-dir", str(a[0])
        ),
        [line({"op": "reset"})] * 3,
        generate_fn=generate,
    )
    assert [r["info"]["episode_id"] for r in responses] == [
        "ep_DIRB00000001",
        "ep_SEED00000005",
        "ep_DIRA00000001",
    ]


def test_D11_same_dir_listed_twice_is_two_slots_never_a_retry(tmp_path, log):
    (d,) = write_episodes(tmp_path / "eps", ["ep_TWICE0000001"])
    _, responses, _ = run_server(
        _closed(log, "--episode-dir", str(d), "--episode-dir", str(d)),
        [line({"op": "reset"})] * 3,
    )
    assert [r.get("op") for r in responses] == ["reset", "reset", None]
    assert responses[2] == EXHAUSTED
    assert [r["end"] for r in read_log(log)] == ["abandoned", "abandoned"]


def test_D11_failed_scheduled_episode_is_consumed(tmp_path, log):
    schedule = tmp_path / "s.json"
    schedule.write_text(json.dumps([{"seed": 1}, {"seed": 2}]), encoding="utf-8")

    def generate(seed, config=None):
        if seed == 1:
            raise RuntimeError("generator bug at /secret/path")
        return with_episode_id(build_tiny_episode(), "ep_SEED00000002")

    _, responses, raw = run_server(
        _closed(log, "--schedule", str(schedule)),
        [line({"op": "reset"})] * 3,
        generate_fn=generate,
    )
    assert responses[0]["error"]["code"] == "internal_error"
    assert responses[1]["info"]["episode_id"] == "ep_SEED00000002"
    assert responses[2] == EXHAUSTED
    assert b"/secret" not in raw


# ------------------------------------------------------------------ no free retries
@pytest.mark.parametrize("mode", ["open", "closed"])
@pytest.mark.parametrize("ending", ["reset", "close", "eof"])
def test_E10_reset_close_eof_mid_episode_logs_abandoned_empty_grade(tmp_path, log, mode, ending):
    ids = episode_ids(2)
    dirs = write_episodes(tmp_path / "eps", ids)
    grader = RecordingGrader(reward=0.7)
    requests = [line({"op": "reset"})] + [
        line({"op": "step", "action": list_page("ledger", None, 1)})
    ] * 7
    if ending == "reset":
        requests.append(line({"op": "reset"}))
    elif ending == "close":
        requests.append(line({"op": "close"}))
    argv = ["--mode", mode, "--grader-log", str(log), *_dirs_args(dirs)]
    code, responses, _ = run_server(argv, requests, grader=grader)
    assert code == 0
    if ending == "reset":
        assert responses[-1]["info"]["episode_id"] == ids[1]
    records = read_log(log)
    first = records[0]
    assert first["episode_id"] == ids[0] and first["end"] == "abandoned"
    assert first["step_count"] == 7 and first["inspection_steps"] == 7
    assert first["mode"] == mode and first["reward"] == 0.7
    assert grader.calls[0]["payload"] == {"episode_id": ids[0], "findings": []}
    assert grader.calls[0]["inspection_steps"] == 7
    # With a reset, the new episode is still running when the stream ends: EOF abandons it too.
    if ending == "reset":
        assert [(r["episode_id"], r["end"]) for r in records] == [
            (ids[0], "abandoned"),
            (ids[1], "abandoned"),
        ]
    else:
        assert len(records) == 1


def test_E10_reviewer_attack_five_rounds_then_submit(tmp_path, log):
    """5 x (99 exploration steps, reset), then submit: every round is graded, nothing is retried."""
    ids = episode_ids(6)
    dirs = write_episodes(tmp_path / "eps", ids)
    grader = RecordingGrader()
    requests = []
    for _ in range(5):
        requests.append(line({"op": "reset"}))
        requests += [line({"op": "step", "action": list_page("ledger", None, 1)})] * 99
    requests.append(line({"op": "reset"}))
    payload = oracle_submission(with_episode_id(build_tiny_episode(), ids[0]))
    requests.append(line({"op": "step", "action": act("submit", findings=payload)}))
    requests.append(line({"op": "reset"}))
    _, responses, _ = run_server(_closed(log, *_dirs_args(dirs)), requests, grader=grader)
    resets = [r for r in responses if r.get("op") == "reset"]
    assert [r["info"]["episode_id"] for r in resets] == ids  # never the same episode twice
    assert responses[-1] == EXHAUSTED
    records = read_log(log)
    assert [(r["episode_id"], r["end"], r["inspection_steps"]) for r in records] == [
        *[(eid, "abandoned", 99) for eid in ids[:5]],
        (ids[5], "submit", 0),
    ]
    assert grader.calls[-1]["episode_id"] == ids[5]


def test_E10_open_mode_without_schedule_also_logs_abandoned(tmp_path, log):
    grader = RecordingGrader()

    def generate(seed, config=None):
        return with_episode_id(build_tiny_episode(), f"ep_SEED{seed:08d}")

    _, responses, _ = run_server(
        ["--mode", "open", "--grader-log", str(log)],
        [
            line({"op": "reset", "seed": 1}),
            line({"op": "step", "action": act("teleport")}),
            line({"op": "reset", "seed": 1}),  # same episode again: allowed in training
            line({"op": "reset"}),  # no seed without a schedule: refused, nothing changes
            line({"op": "step", "action": act("teleport")}),
            line({"op": "close"}),
        ],
        grader=grader,
        generate_fn=generate,
    )
    assert responses[3] == INVALID
    assert responses[4]["observation"]["steps_remaining"] == MAX_STEPS - 1
    assert [(r["episode_id"], r["end"], r["step_count"]) for r in read_log(log)] == [
        ("ep_SEED00000001", "abandoned", 1),
        ("ep_SEED00000001", "abandoned", 1),
    ]


def test_E10_open_mode_failed_reset_still_ends_running_episode(tmp_path, log):
    """On the server, a valid reset request ends the running episode before anything else."""
    from auditgym.contracts.config import ConfigError

    def generate(seed, config=None):
        if config is not None:
            raise ConfigError("bad")
        return build_tiny_episode()

    _, responses, _ = run_server(
        ["--mode", "open", "--grader-log", str(log)],
        [
            line({"op": "reset", "seed": 1}),
            line({"op": "reset", "seed": 2, "config": {"x": 1}}),
            line({"op": "step", "action": list_page("ledger")}),
        ],
        generate_fn=generate,
    )
    assert responses[1]["error"]["code"] == "invalid_config"
    assert responses[2]["observation"]["error"]["code"] == "episode_finished"
    assert [r["end"] for r in read_log(log)] == ["abandoned"]


# ------------------------------------------------------------------ grader log stays with the organizer
def test_E12_closed_stdout_carries_nothing_about_grades(tmp_path, log):
    ids = episode_ids(2)
    dirs = write_episodes(tmp_path / "eps", ids)
    grader = RecordingGrader(reward=0.123456)
    payload = oracle_submission(with_episode_id(build_tiny_episode(), ids[0]))
    _, responses, raw = run_server(
        _closed(log, *_dirs_args(dirs)),
        [
            line({"op": "reset"}),
            line({"op": "step", "action": act("submit", findings=payload)}),
            line({"op": "reset"}),
            line({"op": "truncate"}),
        ],
        grader=grader,
    )
    text = raw.decode("utf-8")
    for marker in ("0.123456", '"matches"', '"valid"', '"tp"', '"f1"', "abandoned", str(log)):
        assert marker not in text
    assert all(r.get("reward", 0.0) == 0.0 for r in responses)
    assert [r["reward"] for r in read_log(log)] == [0.123456, 0.123456]


def test_E12_open_mode_writes_no_default_log_in_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_server(
        ["--mode", "open"],
        [line({"op": "reset", "seed": 1}), line({"op": "truncate"})],
        generate_fn=lambda s, c=None: build_tiny_episode(),
    )
    assert list(tmp_path.iterdir()) == []


# ------------------------------------------------------------------ startup refusals
@pytest.mark.parametrize(
    "argv_tail",
    [
        [],  # no grader log, no schedule
        ["--grader-log", "{log}"],  # no schedule
        ["--episode-dir", "{eps}"],  # no grader log
        ["--grader-log", "runs/grader_log.jsonl", "--episode-dir", "{eps}"],  # relative
        ["--grader-log", "grader_log.jsonl", "--episode-dir", "{eps}"],
        ["--grader-log", "{log}", "--episode-dir", "{eps}", "--time-limit", "0"],
        ["--grader-log", "{log}", "--episode-dir", "{eps}", "--time-limit", "-5"],
        ["--grader-log", "{log}", "--episode-dir", "{eps}", "--time-limit", "nan"],
        ["--grader-log", "{log}", "--episode-dir", "{eps}", "--time-limit", "inf"],
        ["--grader-log", "{log}", "--episode-dir", "{missing}"],
        ["--grader-log", "{log}", "--schedule", "{missing}"],
    ],
)
def test_D11_closed_mode_refuses_to_start(tmp_path, monkeypatch, capsys, argv_tail):
    monkeypatch.chdir(tmp_path)
    (d,) = write_episodes(tmp_path / "eps", ["ep_START0000001"])
    values = {"log": str(tmp_path / "abs" / "log.jsonl"), "eps": str(d), "missing": "/nope/x"}
    argv = ["--mode", "closed"] + [a.format(**values) for a in argv_tail]
    with pytest.raises(SystemExit) as exc:
        run_server(argv, [line({"op": "reset"})])
    assert exc.value.code == 2
    assert capsys.readouterr().out == ""
    assert not (tmp_path / "runs").exists() and not (tmp_path / "grader_log.jsonl").exists()


@pytest.mark.parametrize(
    "content",
    [
        "{}",
        "[1]",
        '[{"seed": "1"}]',
        '[{"seed": true}]',
        '[{"config": {}}]',
        '[{"seed": 1, "path": "/etc"}]',
        '[{"seed": 1, "config": 5}]',
        '[{"seed": 1, "config": {"nope": 1}}]',
        '[{"seed": 1, "config": "missing.json"}]',
        "[]",
        "not json",
    ],
)
def test_D11_bad_schedule_file_refuses_to_start(tmp_path, content):
    schedule = tmp_path / "s.json"
    schedule.write_text(content, encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        run_server(
            _closed(tmp_path / "log.jsonl", "--schedule", str(schedule)), [line({"op": "reset"})]
        )
    assert exc.value.code == 2


def test_D11_bad_episode_dir_refuses_to_start(tmp_path):
    root = tmp_path / "split"
    write_episode_at(root / "good", "ep_GOOD00000001")
    (root / "junk").mkdir()  # a subdirectory that is not an episode
    with pytest.raises(SystemExit):
        run_server(_closed(tmp_path / "log.jsonl", "--episode-dir", str(root)), [])
    (no_private,) = write_episodes(tmp_path / "pub", ["ep_PUBLIC000001"])
    for f in (no_private / "private").iterdir():
        f.unlink()
    with pytest.raises(SystemExit):
        run_server(_closed(tmp_path / "log.jsonl", "--episode-dir", str(no_private)), [])
    with pytest.raises(SystemExit):
        run_server(_closed(tmp_path / "log.jsonl", "--episode-dir", str(tmp_path / "empty")), [])


def test_D11_grader_log_cannot_be_the_protocol_channel(tmp_path):
    (d,) = write_episodes(tmp_path / "eps", ["ep_CHAN00000001"])
    channel = tmp_path / "channel.out"
    with channel.open("wb") as out, pytest.raises(SystemExit):
        main(
            _closed(channel, "--episode-dir", str(d)),
            grader=RecordingGrader(),
            stdin=io.BytesIO(b""),
            stdout=out,
        )
    assert channel.read_bytes() == b""


def test_D11_closed_mode_needs_a_schedule_in_the_host_too():
    from auditgym.env import AuditEnv
    from auditgym.server.host import EpisodeHost

    with pytest.raises(ValueError):
        EpisodeHost(AuditEnv(mode="closed"))
