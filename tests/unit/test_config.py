from pathlib import Path

import pytest

from auditgym.contracts.config import ConfigError, config_hash, load_config, normalize_config
from auditgym.contracts.constants import INTERACTION_SCENARIOS, ISSUE_TYPES

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def test_defaults_are_valid():
    cfg = normalize_config({})
    assert cfg["errors"]["duplicate_invoice"] == [0, 2]


def test_int_counts_become_ranges():
    cfg = normalize_config({"errors": {"fx_conversion": 1}})
    assert cfg["errors"]["fx_conversion"] == [1, 1]


def test_clean_forces_zero_issues():
    cfg = normalize_config({"clean": True, "errors": {"fx_conversion": 2}})
    assert all(cfg["errors"][t] == [0, 0] for t in ISSUE_TYPES)
    assert all(cfg["interactions"][s] == [0, 0] for s in INTERACTION_SCENARIOS)


@pytest.mark.parametrize(
    "raw",
    [
        {"unknown": 1},
        {"errors": {"made_up_type": 1}},
        {"errors": {"fx_conversion": [3, 1]}},
        {"errors": {"fx_conversion": -1}},
        {"errors": {"fx_conversion": True}},
        {"currencies": {"base_pool": ["XXX"]}},
        {"timezone_pool": ["Asia/Almaty"]},
        {"presentation": {"order": "random"}},
        {"errors": {t: 4 for t in ISSUE_TYPES}},  # at least 16 issues > 12
        {"max_issues": 13},
        {
            "max_issues": 4,
            "interactions": {"fx_partial_at_boundary": 1, "duplicate_misallocated": [0, 1]},
        },
    ],
)
def test_invalid_configs_rejected(raw):
    with pytest.raises(ConfigError):
        normalize_config(raw)


def test_hash_is_stable_and_sensitive():
    a = normalize_config({"config_id": "x"})
    b = normalize_config({"config_id": "x"})
    c = normalize_config({"config_id": "y"})
    assert config_hash(a) == config_hash(b) != config_hash(c)


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.v1.json")), ids=lambda p: p.name)
def test_shipped_configs_valid(path):
    cfg = load_config(path)
    assert cfg["config_id"] == path.name.removesuffix(".json")


def test_secret_salt_changes_content_and_is_reproducible():
    from auditgym.contracts.io import fingerprint
    from auditgym.episode.generate import generate_episode

    plain = normalize_config({"config_id": "s"})
    salted = normalize_config({"config_id": "s", "secret_salt": "organizer-private"})
    a, b = generate_episode(5, plain), generate_episode(5, salted)
    assert a.public.episode_id != b.public.episode_id
    assert [r.debit_minor for r in a.public.ledger] != [r.debit_minor for r in b.public.ledger]
    assert fingerprint(generate_episode(5, salted)) == fingerprint(b)


def test_secret_salt_validated():
    with pytest.raises(ConfigError):
        normalize_config({"secret_salt": 5})
