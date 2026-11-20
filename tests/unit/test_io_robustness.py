"""K4: loading tolerates benign re-serialization of episode files by a third party."""

import json

import pytest

from auditgym.contracts.io import read_episode


def _rewrite(path, fn):
    path.write_text(fn(path.read_text(encoding="utf-8")), encoding="utf-8")


def test_csv_column_order_bom_blank_lines_and_null_cells(tiny_episode, tiny_episode_dir):
    ledger = tiny_episode_dir / "ledger.csv"
    lines = ledger.read_text(encoding="utf-8").splitlines()
    cols = lines[0].split(",")
    order = list(reversed(range(len(cols))))
    import csv
    import io

    rows = list(csv.reader(io.StringIO("\n".join(lines))))
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    for r in rows:
        w.writerow(["null" if (i in order[:3] and r[i] == "") else r[i] for i in order])
    ledger.write_text("﻿" + out.getvalue() + "\n\n", encoding="utf-8")
    _rewrite(tiny_episode_dir / "fx.csv", lambda t: t + "\n")
    assert read_episode(tiny_episode_dir) == tiny_episode


def test_integral_floats_in_jsonl(tiny_episode, tiny_episode_dir):
    def floats(text):
        out = []
        for line in text.splitlines():
            d = json.loads(line)
            if "total_minor" in d:
                d["total_minor"] = float(d["total_minor"])
            if "amount_minor" in d:
                d["amount_minor"] = float(d["amount_minor"])
            out.append(json.dumps(d))
        return "\n".join(out) + "\n"

    _rewrite(tiny_episode_dir / "invoices.jsonl", floats)
    _rewrite(tiny_episode_dir / "payments.jsonl", floats)
    assert read_episode(tiny_episode_dir) == tiny_episode


def test_missing_provenance_is_allowed(tiny_episode_dir):
    (tiny_episode_dir / "private" / "provenance.json").unlink()
    assert read_episode(tiny_episode_dir).private.provenance == {}


def test_duplicate_ids_across_tables_rejected_clearly(tiny_episode_dir):
    _rewrite(tiny_episode_dir / "payments.jsonl", lambda t: t.replace('"D-P1"', '"D-I1"'))
    with pytest.raises(ValueError, match="unique"):
        read_episode(tiny_episode_dir)
