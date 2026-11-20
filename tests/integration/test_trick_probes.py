"""G07 / S10: common shortcuts must not pay off on generated episodes."""

from auditgym.contracts.config import normalize_config
from auditgym.harness.probes import PROBES, run_probes


def test_G07_every_trick_probe_scores_low():
    cfg = normalize_config({"config_id": "probe-test", "errors": {"payment_allocation": [1, 2]}})
    report = run_probes(cfg, list(range(900, 930)))
    assert set(report) == set(PROBES)
    for name, row in report.items():
        assert row["f1_micro"] < 0.4, (name, row)
        assert row["mean_reward"] < 0.0, (name, row)


def test_G07_probes_find_nothing_real_on_clean_books():
    report = run_probes(
        normalize_config({"config_id": "probe-clean", "clean": True}), list(range(900, 920))
    )
    assert all(row["tp"] == 0 for row in report.values())
    assert sum(row["fp"] for row in report.values()) > 0  # distractors do lure the shortcuts
