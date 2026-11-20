"""Config `variants` (G01): the variants of each standalone error type can be ordered.

- validation: unknown type, unknown variant, empty list, duplicates are ConfigErrors;
- a config without `variants`, or with exactly the default, keeps the stream key and the episode
  bytes of the generator before the section existed;
- every single forced variant generates on consecutive seeds with zero retries, the injection has
  that variant, and the episode passes the independent check (solver + evidence sufficiency);
- a restricted variant is never replaced by another one, also inside richer configs;
- the interacting scenarios keep their fixed shapes.

The slow loops (`pytest -m slow`) cover many seeds; a fast subset runs by default.
"""

from __future__ import annotations

import itertools
from collections import Counter

import pytest
from genkit import SHIPPED, gen, only, shipped_config

from auditgym.contracts.config import (
    DEFAULT_VARIANTS,
    ISSUE_VARIANTS,
    ConfigError,
    allowed_variants,
    normalize_config,
)
from auditgym.contracts.constants import INTERACTION_SCENARIOS, ISSUE_TYPES
from auditgym.contracts.io import fingerprint
from auditgym.episode.generate import stream_config_hash
from auditgym.harness.sweep import verify_episode
from auditgym.inject.injectors import ALLOC_FALLBACK, ALLOC_VARIANT_WEIGHTS
from auditgym.world.sizing import required_invoices

# stream_config_hash(normalize_config(None)) recorded before the `variants` section was added.
DEFAULT_STREAM_HASH = "83d09f9a2ba2eca90520af78db7588f4ec8d62d2eeff9e84f17b4a5c57b79444"

SINGLE = [(t, v) for t in ISSUE_TYPES for v in ISSUE_VARIANTS[t]]
SUBSETS = [
    (t, sub)
    for t in ISSUE_TYPES
    for k in range(1, len(ISSUE_VARIANTS[t]))
    for sub in itertools.combinations(ISSUE_VARIANTS[t], k)
]
FAST_SEEDS = range(8)
SLOW_SEEDS = range(60)  # at least 50 consecutive seeds per forced variant


def _forced(issue_type: str, variants: list[str], count: int = 1) -> dict:
    return normalize_config(
        {
            "config_id": f"forced-{issue_type}-{'-'.join(variants)}",
            "errors": {t: (count if t == issue_type else 0) for t in ISSUE_TYPES},
            "interactions": {s: 0 for s in INTERACTION_SCENARIOS},
            "variants": {issue_type: variants},
        }
    )


def _standalone(episode, issue_type: str) -> list[str]:
    return [
        inj["variant"]
        for inj in episode.private.provenance["injections"]
        if inj["scenario"] is None and inj["injector"] == issue_type
    ]


def _check_forced(issue_type: str, variant: str, seeds) -> None:
    cfg = _forced(issue_type, [variant])
    for seed in seeds:
        episode, clean = gen(seed, cfg)
        assert episode.private.provenance["attempts"] == 1, (seed, issue_type, variant)
        assert [i.type for i in episode.private.truth.issues] == [issue_type]
        assert _standalone(episode, issue_type) == [variant], seed
        result = verify_episode(episode, clean)
        assert result["ok"], (seed, result["problems"])


# ---------------------------------------------------------------------------------- names
def test_G01_variant_names_cover_every_type_and_match_the_injectors():
    assert tuple(ISSUE_VARIANTS) == ISSUE_TYPES
    assert tuple(ALLOC_VARIANT_WEIGHTS) == ISSUE_VARIANTS["payment_allocation"]
    assert set(ALLOC_FALLBACK) == set(ISSUE_VARIANTS["payment_allocation"])
    assert DEFAULT_VARIANTS == {t: list(ISSUE_VARIANTS[t]) for t in ISSUE_TYPES}


# ---------------------------------------------------------------------------------- validation
@pytest.mark.parametrize(
    "variants",
    [
        None,
        [],
        "partial",
        {"made_up_type": ["x"]},
        {"payment_allocation": ["made_up"]},
        {"payment_allocation": ["wrong_date"]},  # a variant of another type
        {"payment_allocation": []},
        {"payment_allocation": "partial"},
        {"payment_allocation": ["partial", "partial"]},
        {"payment_allocation": [None]},
        {"payment_allocation": [["partial"]]},
        {"fx_conversion": ["wrong_pair", "wrong_date", "wrong_pair"]},
    ],
)
def test_G01_invalid_variants_rejected(variants):
    with pytest.raises(ConfigError):
        normalize_config({"variants": variants})


def test_G01_wrong_pair_alone_needs_room_for_two_foreign_currencies():
    raw = {"variants": {"fx_conversion": ["wrong_pair"]}}
    with pytest.raises(ConfigError, match="two foreign currencies"):
        normalize_config({**raw, "currencies": {"foreign_count": [1, 1]}})
    # without fx errors the restriction is idle and needs nothing
    normalize_config({**raw, "currencies": {"foreign_count": 1}, "errors": {"fx_conversion": 0}})


def test_G01_variants_normalized_to_every_type_in_canonical_order():
    cfg = normalize_config(
        {
            "variants": {
                "payment_allocation": ["other_invoice", "none"],
                "period_cutoff": ["payment_posted_late"],
            }
        }
    )
    assert cfg["variants"] == {
        "duplicate_invoice": ["same_invoice_id", "new_invoice_id"],
        "payment_allocation": ["none", "other_invoice"],
        "fx_conversion": ["wrong_date", "wrong_pair"],
        "period_cutoff": ["payment_posted_late"],
    }
    assert normalize_config(cfg) == cfg
    assert allowed_variants(cfg, "payment_allocation") == ("none", "other_invoice")
    assert allowed_variants(normalize_config({}), "fx_conversion") == ("wrong_date", "wrong_pair")


# ---------------------------------------------------------------------------------- default unchanged
def test_G01_default_variants_keep_the_stream_hash():
    assert stream_config_hash(normalize_config(None)) == DEFAULT_STREAM_HASH
    reordered = {t: list(reversed(names)) for t, names in ISSUE_VARIANTS.items()}
    for variants in (
        {},
        DEFAULT_VARIANTS,
        reordered,
        {"fx_conversion": ["wrong_date", "wrong_pair"]},
    ):
        cfg = normalize_config({"variants": variants})
        assert "variants" not in cfg
        assert cfg == normalize_config(None)
        assert stream_config_hash(cfg) == DEFAULT_STREAM_HASH
    # even a hand-made config that carries the default section keeps the key
    assert stream_config_hash({**normalize_config(None), "variants": DEFAULT_VARIANTS}) == (
        DEFAULT_STREAM_HASH
    )


@pytest.mark.parametrize("name", SHIPPED)
def test_G01_default_variants_give_identical_episodes(name):
    raw_cfg = shipped_config(name)
    with_default = normalize_config({**raw_cfg, "variants": DEFAULT_VARIANTS})
    assert with_default == raw_cfg
    for seed in range(2):
        assert fingerprint(gen(seed, with_default)[0]) == fingerprint(gen(seed, raw_cfg)[0])


def test_G01_restricting_changes_the_stream_hash():
    base = normalize_config(None)
    restricted = normalize_config({"variants": {"payment_allocation": ["partial"]}})
    assert stream_config_hash(restricted) != stream_config_hash(base)


# ---------------------------------------------------------------------------------- forced variants
@pytest.mark.parametrize(("issue_type", "variant"), SINGLE, ids=[f"{t}-{v}" for t, v in SINGLE])
def test_G01_variant_forced_fast(issue_type, variant):
    _check_forced(issue_type, variant, FAST_SEEDS)


@pytest.mark.slow
@pytest.mark.parametrize(("issue_type", "variant"), SINGLE, ids=[f"{t}-{v}" for t, v in SINGLE])
def test_G01_variant_forced_many_seeds(issue_type, variant):
    _check_forced(issue_type, variant, SLOW_SEEDS)


@pytest.mark.parametrize(("issue_type", "variant"), SINGLE, ids=[f"{t}-{v}" for t, v in SINGLE])
def test_G01_variant_forced_several_errors_of_one_type(issue_type, variant):
    cfg = _forced(issue_type, [variant], count=3)
    for seed in range(3):
        episode, clean = gen(seed, cfg)
        assert episode.private.provenance["attempts"] == 1
        assert _standalone(episode, issue_type) == [variant] * 3
        assert verify_episode(episode, clean)["ok"]


def test_G01_wrong_pair_alone_draws_two_foreign_currencies():
    cfg = _forced("fx_conversion", ["wrong_pair"])
    for seed in range(6):
        world = gen(seed, cfg)[0].private.provenance["world"]
        assert len(world["foreign_currencies"]) >= 2


def test_G01_other_invoice_alone_counts_the_second_invoice_in_sizing():
    plain = normalize_config({"errors": {"payment_allocation": [0, 3]}})
    other = normalize_config(
        {
            "errors": {"payment_allocation": [0, 3]},
            "variants": {"payment_allocation": ["other_invoice"]},
        }
    )
    assert required_invoices(other) == required_invoices(plain) + 3


def _check_subset(issue_type: str, subset: tuple[str, ...], seeds) -> Counter:
    """Inside the default config (every type, interactions on), one type restricted."""
    cfg = normalize_config({"config_id": "mixed", "variants": {issue_type: list(subset)}})
    seen: Counter = Counter()
    for seed in seeds:
        episode, clean = gen(seed, cfg)
        assert episode.private.provenance["attempts"] == 1, seed
        got = _standalone(episode, issue_type)
        assert set(got) <= set(subset), (seed, got)
        seen.update(got)
        result = verify_episode(episode, clean)
        assert result["ok"], (seed, result["problems"])
    return seen


@pytest.mark.parametrize(
    ("issue_type", "subset"), SUBSETS, ids=[f"{t}-{'+'.join(s)}" for t, s in SUBSETS]
)
def test_G01_variant_subset_in_mixed_config_fast(issue_type, subset):
    _check_subset(issue_type, subset, range(4))


@pytest.mark.slow
@pytest.mark.parametrize(
    ("issue_type", "subset"), SUBSETS, ids=[f"{t}-{'+'.join(s)}" for t, s in SUBSETS]
)
def test_G01_variant_subset_in_mixed_config_many_seeds(issue_type, subset):
    seen = _check_subset(issue_type, subset, range(80))
    assert set(seen) == set(subset), seen  # every allowed variant still occurs


# ---------------------------------------------------------------------------------- scenarios
SCENARIO_SHAPES = {
    "fx_partial_at_boundary": "posted_across_boundary_converted_at_posting_date_partly_applied",
    "duplicate_misallocated": "copy_under_new_invoice_id_payment_applied_to_copy",
}
SCENARIO_ISSUES = {
    "fx_partial_at_boundary": Counter(["period_cutoff", "fx_conversion", "payment_allocation"]),
    "duplicate_misallocated": Counter(["duplicate_invoice", "payment_allocation"]),
}


@pytest.mark.parametrize("scenario", INTERACTION_SCENARIOS)
def test_G01_variants_do_not_change_interacting_scenarios(scenario):
    restricted = only(
        scenario=scenario,
        variants={
            "duplicate_invoice": ["same_invoice_id"],
            "payment_allocation": ["none"],
            "fx_conversion": ["wrong_pair"],
            "period_cutoff": ["invoice_posted_late"],
        },
    )
    for seed in range(3):
        episode, clean = gen(seed, restricted)
        injections = episode.private.provenance["injections"]
        assert [(inj["scenario"], inj["variant"]) for inj in injections] == [
            (scenario, SCENARIO_SHAPES[scenario])
        ]
        assert Counter(i.type for i in episode.private.truth.issues) == SCENARIO_ISSUES[scenario]
        assert verify_episode(episode, clean)["ok"]
