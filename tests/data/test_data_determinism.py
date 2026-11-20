"""Determinism and per-episode variation of the generator (K4, G06)."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
from genkit import REPO, gen, shipped_config

from auditgym.contracts.io import fingerprint
from auditgym.episode.generate import generate_episode

_PRINT_FP = (
    "import json, sys\n"
    "from auditgym.contracts.config import load_config\n"
    "from auditgym.contracts.io import fingerprint\n"
    "from auditgym.episode.generate import generate_episode\n"
    "cfg = load_config(sys.argv[1])\n"
    "print(json.dumps([fingerprint(generate_episode(s, cfg)) for s in (3, 77, 1001)]))\n"
)


def test_K4_same_seed_same_fingerprint():
    cfg = shipped_config("train")
    for seed in (0, 5, 1234):
        a = generate_episode(seed, cfg)
        b = generate_episode(seed, cfg)
        assert fingerprint(a) == fingerprint(b)
        assert a == b


def test_K4_raw_and_normalized_config_give_same_episode():
    raw = {"config_id": "train.v1"}
    from auditgym.contracts.config import normalize_config

    assert fingerprint(generate_episode(9, raw)) == fingerprint(
        generate_episode(9, normalize_config(raw))
    )


@pytest.mark.parametrize("name", ["train", "interacting"])
def test_K4_fingerprint_stable_across_processes_and_hash_seeds(name):
    path = str(REPO / "configs" / f"{name}.v1.json")
    outputs = []
    for hashseed in ("0", "1", "4242"):
        env = dict(os.environ, PYTHONHASHSEED=hashseed)
        res = subprocess.run(
            [sys.executable, "-c", _PRINT_FP, path],
            capture_output=True,
            text=True,
            env=env,
            cwd=REPO,
            check=True,
        )
        outputs.append(res.stdout.strip())
    assert len(set(outputs)) == 1
    import json

    in_process = [fingerprint(gen(s, shipped_config(name))[0]) for s in (3, 77, 1001)]
    assert json.loads(outputs[0]) == in_process


def test_G06_different_seeds_differ():
    cfg = shipped_config("train")
    fps = {fingerprint(gen(seed, cfg)[0]) for seed in range(20)}
    assert len(fps) == 20


def test_G06_episode_parameters_vary():
    cfg = shipped_config("dev")
    seen: dict[str, set] = {
        k: set()
        for k in (
            "company",
            "base",
            "foreign",
            "timezone",
            "period",
            "accounts",
            "rate_date",
            "rounding",
            "doc_prefix",
            "invoice_prefix",
        )
    }
    for seed in range(40):
        pub = gen(seed, cfg)[0].public
        seen["company"].add(pub.manifest["company"])
        seen["base"].add(pub.manifest["base_currency"])
        seen["foreign"].add(
            tuple(sorted(set(pub.manifest["currencies"]) - {pub.manifest["base_currency"]}))
        )
        seen["timezone"].add(pub.manifest["timezone"])
        seen["period"].add(pub.manifest["period_start"][:7])
        seen["accounts"].add(tuple(sorted(pub.policy["accounts"].items())))
        seen["rate_date"].add(pub.policy["fx"]["rate_date"])
        seen["rounding"].add(pub.policy["fx"]["rounding"])
        seen["doc_prefix"].add(pub.invoices[0].doc_id[:3])
        seen["invoice_prefix"].add(pub.invoices[0].invoice_id[:2])
    assert len(seen["company"]) >= 30
    assert seen["base"] == {"KZT", "EUR", "USD"}
    assert len(seen["foreign"]) >= 10
    assert len(seen["timezone"]) >= 4
    assert len(seen["period"]) >= 15
    assert len(seen["accounts"]) == 40
    assert len(seen["rate_date"]) == 3
    assert len(seen["rounding"]) == 2
    assert len(seen["doc_prefix"]) >= 3
    assert len(seen["invoice_prefix"]) >= 3


def test_G06_amounts_customers_and_issue_mix_vary():
    cfg = shipped_config("train")
    totals, entities, mixes = set(), set(), set()
    for seed in range(20):
        ep = gen(seed, cfg)[0]
        totals.add(tuple(sorted(i.total_minor for i in ep.public.invoices))[:5])
        entities.add(len({i.entity_id for i in ep.public.invoices}))
        mixes.add(tuple(sorted(i.type for i in ep.private.truth.issues)))
    assert len(totals) == 20
    assert len(entities) >= 4
    assert len(mixes) >= 15
