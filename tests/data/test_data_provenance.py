"""Provenance replay, self-check hook, sizes and generation time (G04, D10, K2, D01)."""

from __future__ import annotations

import time

import pytest
from genkit import gen, shipped_config, tables, undo_changes

from auditgym.contracts.types import Issue
from auditgym.episode.generate import (
    MAX_ATTEMPTS,
    GenerationError,
    build_attempt,
    generate_episode,
)

REQUIRED_KEYS = {
    "generator_version",
    "seed",
    "config",
    "attempts",
    "world",
    "distractors",
    "id_map",
    "injections",
    "changes",
    "scenarios",
}


@pytest.mark.parametrize("name", ["train", "dev", "interacting", "smoke", "clean"])
def test_G04_undoing_changes_reproduces_clean_books(name):
    cfg = shipped_config(name)
    for seed in range(20):
        episode, clean = gen(seed, cfg)
        restored = undo_changes(episode.public, episode.private.provenance["changes"])
        assert restored == tables(clean), (name, seed)
        assert episode.public.fx == clean.fx
        assert episode.public.policy == clean.policy


def test_G04_changes_name_their_issues_and_public_ids():
    cfg = shipped_config("interacting")
    for seed in range(10):
        episode, clean = gen(seed, cfg)
        issue_ids = {i.issue_id for i in episode.private.truth.issues}
        changes = episode.private.provenance["changes"]
        assert changes
        covered = set()
        for ch in changes:
            assert ch["table"] in ("ledger", "invoices", "payments")
            assert set(ch["issue_ids"]) <= issue_ids
            covered |= set(ch["issue_ids"])
            assert ch["before"] is not None or ch["after"] is not None
        assert covered == issue_ids  # every issue changed at least one record


def test_D10_provenance_contents():
    cfg = shipped_config("train")
    for seed in range(5):
        episode, _ = gen(seed, cfg)
        prov = episode.private.provenance
        assert REQUIRED_KEYS <= set(prov)
        assert prov["seed"] == seed
        assert prov["attempts"] == 1
        assert prov["config"] == cfg
        issue_ids = {i.issue_id for i in episode.private.truth.issues}
        injected = {i for inj in prov["injections"] for i in inj["issue_ids"]}
        assert injected == issue_ids
        for inj in prov["injections"]:
            assert inj["injector"] and inj["variant"] and inj["affected_keys"]
            assert len(inj["affected_keys"]) == len(inj["affected_ids"])
        assert {d["code"] for d in prov["distractors"]} <= {f"X{n}" for n in range(1, 11)}


def test_verify_fn_agreeing_keeps_first_attempt():
    cfg = shipped_config("smoke")
    plain = generate_episode(3, cfg)
    checked = generate_episode(3, cfg, verify_fn=lambda pub: plain.private.truth.issues)
    assert checked == plain
    assert checked.private.provenance["attempts"] == 1


def test_verify_fn_disagreeing_retries_with_next_attempt():
    cfg = shipped_config("smoke")
    truth_of = {
        build_attempt(3, cfg, a).public.episode_id: build_attempt(3, cfg, a).truth.issues
        for a in range(2)
    }
    bogus = Issue("iss_x", "period_cutoff", ("nope",), 100, "USD", 0, (("nope",),))
    calls: list[str] = []

    def flaky(public) -> tuple[Issue, ...]:
        calls.append(public.episode_id)
        return (bogus,) if len(calls) == 1 else truth_of[public.episode_id]

    episode = generate_episode(3, cfg, verify_fn=flaky)
    assert len(calls) == 2
    assert episode.private.provenance["attempts"] == 2
    assert len(episode.private.provenance["attempt_failures"]) == 1
    assert episode.public.episode_id != generate_episode(3, cfg).public.episode_id


def test_verify_fn_always_disagreeing_fails_closed():
    cfg = shipped_config("train")
    with pytest.raises(GenerationError):
        generate_episode(1, cfg, verify_fn=lambda pub: ())
    assert MAX_ATTEMPTS >= 2


def test_D01_sizes_of_default_and_train_configs():
    for name in ("train",):
        cfg = shipped_config(name)
        for seed in range(40):
            pub = gen(seed, cfg)[0].public
            docs = pub.manifest["counts"]["documents"]
            assert 200 <= len(pub.ledger) <= 700, (seed, len(pub.ledger))
            assert 30 <= docs <= 150, (seed, docs)
    for seed in range(10):
        pub = generate_episode(seed).public
        assert 200 <= len(pub.ledger) <= 700
        assert 30 <= pub.manifest["counts"]["documents"] <= 150


def test_D01_issue_counts_within_case_limits():
    for name in ("train", "dev", "interacting", "smoke"):
        cfg = shipped_config(name)
        for seed in range(20):
            assert len(gen(seed, cfg)[0].private.truth.issues) <= 12


def test_budget_largest_dev_episode_under_two_seconds():
    cfg = shipped_config("dev")
    largest = max(range(30), key=lambda s: len(gen(s, cfg)[0].public.ledger))
    started = time.perf_counter()
    generate_episode(largest, cfg)
    assert time.perf_counter() - started < 2.0
