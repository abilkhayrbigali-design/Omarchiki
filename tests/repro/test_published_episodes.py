"""K4 / K5: every published episode is reproduced byte-for-byte from its seed and config."""

import json
from pathlib import Path

import pytest

from auditgym.contracts.config import load_config
from auditgym.contracts.io import fingerprint, read_episode
from auditgym.episode.generate import generate_episode

REPO = Path(__file__).resolve().parents[2]
INDEX = REPO / "episodes" / "index.json"
ENTRIES = json.loads(INDEX.read_text())["episodes"] if INDEX.exists() else []


def test_K5_open_sets_have_ten_train_and_ten_dev_with_clean_and_interacting():
    if not ENTRIES:
        pytest.skip("episodes not exported yet")
    for split in ("train", "dev"):
        rows = [e for e in ENTRIES if e["split"] == split]
        assert len(rows) == 10
        assert any(e["issues"] == 0 for e in rows)
        assert any(e["config_id"] == "interacting.v1" for e in rows)


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: f"{e['split']}-{e['seed']}")
def test_K4_published_episode_regenerates_exactly(entry):
    cfg = load_config(REPO / "configs" / f"{entry['config_id']}.json")
    regenerated = generate_episode(entry["seed"], cfg)
    assert fingerprint(regenerated) == entry["fingerprint"]
    on_disk = read_episode(REPO / "episodes" / entry["split"] / entry["episode_id"])
    assert fingerprint(on_disk) == entry["fingerprint"]
