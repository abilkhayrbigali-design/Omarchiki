"""Acceptance tests for case requirements that are not covered by name elsewhere.

Codes refer to docs/REQUIREMENTS.md.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pytest

from auditgym.contracts.config import load_config, normalize_config
from auditgym.contracts.constants import FX_COLUMNS, LEDGER_COLUMNS
from auditgym.contracts.io import public_files
from auditgym.episode.generate import generate_episode
from auditgym.util.canon import read_csv, read_jsonl
from auditgym.util.rng import root_stream

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "auditgym"
TS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$")
SEEDS = range(31_000, 31_012)


@pytest.fixture(scope="module")
def episodes():
    cfgs = [load_config(REPO / "configs" / f"{n}.v1.json") for n in ("train", "dev", "interacting")]
    return [generate_episode(s, cfgs[i % 3]) for i, s in enumerate(SEEDS)]


# ---------------------------------------------------------------------------------- C
def test_C02_rl_interface_trajectories_and_grading_exist():
    from auditgym.env import AuditEnv
    from auditgym.grading import grade_submission

    assert callable(AuditEnv.reset) and callable(AuditEnv.step) and callable(grade_submission)
    path = REPO / "trajectories" / "trajectories.jsonl"
    assert path.exists()
    first = json.loads(path.open(encoding="utf-8").readline())
    assert {
        "episode_id",
        "step",
        "action",
        "observation",
        "reward",
        "terminated",
        "truncated",
    } <= set(first)


# Verification tooling (not the environment) may use the pinned test dependencies.
CHECK_ONLY_IMPORTS = {"harness/jury.py": {"jsonschema", "referencing"}}


def test_C03_runtime_uses_only_stdlib_no_network():
    """The package imports nothing outside the standard library and never opens sockets."""
    base_allowed = set(sys.stdlib_module_names) | {"auditgym"}
    banned = {"socket", "ssl", "urllib", "http", "ftplib", "smtplib", "requests", "asyncio"}
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC).as_posix()
        allowed = base_allowed | CHECK_ONLY_IMPORTS.get(rel, set())
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for name in names:
                top = name.split(".")[0]
                assert top in allowed, f"{path}: non-stdlib import {name}"
                assert top not in banned, f"{path}: network module {name}"


def test_C04_docs_make_no_training_claims():
    note = (REPO / "docs" / "DESIGN_NOTE.md").read_text(encoding="utf-8")
    assert "Обучения весов не было" in note


# ---------------------------------------------------------------------------------- G
def test_G05_every_issue_is_recomputable_from_public_evidence(episodes):
    from auditgym.verify.evidence import check_evidence

    for ep in episodes:
        public_ids = (
            {r.row_id for r in ep.public.ledger}
            | {d.doc_id for d in ep.public.invoices}
            | {d.doc_id for d in ep.public.payments}
            | {f.rate_id for f in ep.public.fx}
        )
        for issue in ep.private.truth.issues:
            assert issue.expected_amount_minor >= 100
            for ev in issue.evidence_sets:
                assert set(ev) <= public_ids, "evidence must be reachable through the tools"
                res = check_evidence(ep.public, issue.type, issue.affected_ids, ev)
                assert res.ok and res.amount_minor == issue.expected_amount_minor, res.reason


# ---------------------------------------------------------------------------------- D
def test_D02_manifest_fields(episodes):
    for ep in episodes:
        m = ep.public.manifest
        for key in (
            "schema_version",
            "episode_id",
            "base_currency",
            "period_start",
            "period_end",
            "timezone",
            "counts",
            "rounding",
            "currencies",
        ):
            assert key in m
        assert m["counts"]["ledger"] == len(ep.public.ledger)
        assert m["counts"]["documents"] == len(ep.public.invoices) + len(ep.public.payments)
        assert m["base_currency"] in m["currencies"]


def test_D03_value_formats(episodes):
    for ep in episodes:
        pub = ep.public
        exps = {c: v["exponent"] for c, v in pub.manifest["currencies"].items()}
        for r in pub.ledger:
            assert (
                TS.match(r.timestamp) and type(r.debit_minor) is int and type(r.credit_minor) is int
            )
            assert r.currency in exps
        for inv in pub.invoices:
            assert TS.match(inv.issued_at) and TS.match(inv.due_at) and type(inv.total_minor) is int
        for p in pub.payments:
            assert TS.match(p.timestamp) and type(p.amount_minor) is int
        for f in pub.fx:
            assert isinstance(f.rate, str) and re.fullmatch(r"\d+(\.\d+)?", f.rate)


def test_D04_D05_D06_D07_file_fields_exactly_as_the_case(episodes):
    files = public_files(episodes[0].public)
    header, _ = read_csv(files["ledger.csv"].decode())
    assert tuple(header) == LEDGER_COLUMNS
    header, _ = read_csv(files["fx.csv"].decode())
    assert tuple(header) == FX_COLUMNS
    inv = read_jsonl(files["invoices.jsonl"].decode())[0]
    assert set(inv) == {
        "doc_id",
        "invoice_id",
        "entity_id",
        "issued_at",
        "due_at",
        "currency",
        "total_minor",
        "line_items",
    }
    pay = read_jsonl(files["payments.jsonl"].decode())[0]
    assert set(pay) == {
        "doc_id",
        "payment_id",
        "timestamp",
        "currency",
        "amount_minor",
        "invoice_refs",
    }


def test_D09_missing_links_are_explicit_nulls(episodes):
    seen_null_ref = seen_null_row = False
    for ep in episodes:
        files = public_files(ep.public)
        for line in files["payments.jsonl"].decode().splitlines():
            if '"invoice_refs":null' in line:
                seen_null_ref = True
        for r in ep.public.ledger:
            if r.invoice_id is None or r.payment_id is None:
                seen_null_row = True
            for v in (r.entity_id, r.invoice_id, r.payment_id):
                assert v is None or v != ""
    assert seen_null_ref and seen_null_row


# ---------------------------------------------------------------------------------- V
def test_V01_open_seeds_and_configs_separate_from_jury_seeds():
    splits = json.loads((REPO / "configs" / "splits.json").read_text())
    open_seeds = {s for split in ("train", "dev") for spec in splits[split] for s in spec["seeds"]}
    stream = root_stream(20261122, "jury", "episodes")
    jury = {10_000_000 + stream.randbelow(80_000_000) for _ in range(20)}
    assert open_seeds.isdisjoint(jury)
    assert max(open_seeds) < 10_000_000
    train = {spec["config"] for spec in splits["train"]}
    dev = {spec["config"] for spec in splits["dev"]}
    assert "train.v1" in train and "dev.v1" in dev and "train.v1" not in dev


def test_V02_new_parameter_combinations_stay_consistent():
    from auditgym.episode.generate import generate_with_clean
    from auditgym.harness.sweep import verify_episode

    stream = root_stream(7, "acceptance", "novel-configs")
    for n in range(12):
        raw = {
            "config_id": f"novel-{n}",
            "size": {
                "customers": [3 + stream.randbelow(10), 14],
                "invoices": [8 + stream.randbelow(40), 70],
            },
            "timezone_pool": [stream.choice(["+14:00", "-12:00", "+05:45", "+00:00", "-03:30"])],
            "period": {
                "year_pool": [2001 + stream.randbelow(98)],
                "margin_days": 7 + stream.randbelow(54),
            },
            "currencies": {
                "base_pool": [stream.choice(["KZT", "USD", "EUR", "JPY", "KWD"])],
                "foreign_count": [1, 2],
            },
        }
        cfg = normalize_config(raw)
        episode, clean = generate_with_clean(1000 + n, cfg)
        result = verify_episode(episode, clean)
        assert result["ok"], (raw, result["problems"])


def test_V03_clean_and_interacting_configs():
    clean = generate_episode(5, load_config(REPO / "configs" / "clean.v1.json"))
    assert clean.private.truth.issues == ()
    inter = generate_episode(5, load_config(REPO / "configs" / "interacting.v1.json"))
    assert len(inter.private.provenance["scenarios"]) == 2


def test_V04_diagnostics_reported_separately():
    report = json.loads((REPO / "reports" / "jury_report.json").read_text())
    row = report["baselines"]["reconcile"]["all"]
    for key in ("reward", "precision", "recall", "f1", "steps", "seconds"):
        assert key in row
