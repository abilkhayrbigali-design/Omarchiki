"""Presentation layer: opaque IDs, order, no leaks in public files (K4, T6, D11, D12)."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest
from genkit import REPO, gen, shipped_config, tables

from auditgym.contracts.config import normalize_config
from auditgym.contracts.io import fingerprint, public_files, read_episode, write_episode
from auditgym.util.timeutil import parse_ts

SCHEMA_DIR = REPO / "schemas"


def _with_presentation(name: str, **pres: str) -> dict:
    raw = json.loads((REPO / "configs" / f"{name}.v1.json").read_text(encoding="utf-8"))
    raw["presentation"] = pres
    return normalize_config(raw)


def _semantic_truth(episode) -> list:
    """Truth with public ids replaced by internal keys (via the provenance id map)."""
    inverse = {v: k for k, v in episode.private.provenance["id_map"].items()}
    return sorted(
        (
            i.type,
            tuple(sorted(inverse[x] for x in i.affected_ids)),
            i.expected_amount_minor,
            i.currency,
            tuple(sorted(tuple(sorted(inverse.get(x, x) for x in ev)) for ev in i.evidence_sets)),
        )
        for i in episode.private.truth.issues
    )


def _semantic_ledger(episode) -> Counter:
    return Counter(
        (r.timestamp, r.account, r.debit_minor, r.credit_minor, r.currency)
        for r in episode.public.ledger
    )


@pytest.mark.parametrize("name", ["train", "interacting"])
def test_K4_id_salt_changes_ids_not_semantics(name):
    for seed in range(8):
        a, _ = gen(seed, _with_presentation(name, id_salt="", order="chronological"))
        b, _ = gen(seed, _with_presentation(name, id_salt="other-salt", order="chronological"))
        ids_a = {r.row_id for r in a.public.ledger} | {d.doc_id for d in a.public.invoices}
        ids_b = {r.row_id for r in b.public.ledger} | {d.doc_id for d in b.public.invoices}
        assert ids_a.isdisjoint(ids_b)
        assert a.public.episode_id != b.public.episode_id
        assert _semantic_truth(a) == _semantic_truth(b)
        assert _semantic_ledger(a) == _semantic_ledger(b)
        assert a.public.manifest["counts"] == b.public.manifest["counts"]
        assert a.public.fx == b.public.fx


def test_K4_shuffled_order_keeps_ids_and_truth():
    for seed in range(8):
        a, _ = gen(seed, _with_presentation("train", id_salt="", order="chronological"))
        b, _ = gen(seed, _with_presentation("train", id_salt="", order="shuffled"))
        assert a.private.truth == b.private.truth
        assert tables(a.public) == tables(b.public)
        assert set(a.public.fx) == set(b.public.fx)
        assert [r.row_id for r in a.public.ledger] != [r.row_id for r in b.public.ledger]


def test_T6_chronological_order():
    cfg = shipped_config("dev")
    for seed in range(10):
        pub = gen(seed, cfg)[0].public
        for seq in (
            [parse_ts(r.timestamp) for r in pub.ledger],
            [parse_ts(i.issued_at) for i in pub.invoices],
            [parse_ts(p.timestamp) for p in pub.payments],
            [(f.date, f.source_currency) for f in pub.fx],
        ):
            assert seq == sorted(seq)


def test_T6_injected_rows_are_not_positioned_apart():
    """Within the audited period, injected rows are spread like the others.

    Errors are only injected in scope (policy.scope), so positions are measured relative to the
    rows of the period (plus the few days a cut-off posting may land after its end).
    """
    from datetime import timedelta

    cfg = shipped_config("train")
    positions = []
    for seed in range(60):
        episode, _ = gen(seed, cfg)
        pub = episode.public
        start = parse_ts(pub.manifest["period_start"])
        end = parse_ts(pub.manifest["period_end"]) + timedelta(days=4)
        stamps = [parse_ts(r.timestamp) for r in pub.ledger]
        lo = next(n for n, t in enumerate(stamps) if t >= start)
        hi = max(n for n, t in enumerate(stamps) if t < end)
        order = {r.row_id: n for n, r in enumerate(pub.ledger)}
        for ch in episode.private.provenance["changes"]:
            if ch["table"] == "ledger" and ch["after"] is not None:
                positions.append((order[ch["id"]] - lo) / max(hi - lo, 1))
    mean = sum(positions) / len(positions)
    assert 0.3 < mean < 0.7
    assert min(positions) < 0.1 and max(positions) > 0.9


def test_T6_id_formats_are_uniform():
    cfg = shipped_config("interacting")
    for seed in range(10):
        episode, _ = gen(seed, cfg)
        pub = episode.public
        groups = {
            "doc": [d.doc_id for d in pub.invoices] + [p.doc_id for p in pub.payments],
            "invoice": [i.invoice_id for i in pub.invoices],
            "payment": [p.payment_id for p in pub.payments],
            "row": [r.row_id for r in pub.ledger],
            "entity": [i.entity_id for i in pub.invoices],
            "order": [li.order_ref for i in pub.invoices for li in i.line_items],
        }
        fmt = episode.private.provenance["presentation"]["id_formats"]
        for kind, values in groups.items():
            prefix, chars = fmt[kind]["prefix"], fmt[kind]["chars"]
            pattern = re.compile("^" + re.escape(prefix) + "[0-9A-Z]{" + str(chars) + "}$")
            assert all(pattern.match(v) for v in values), kind


def test_T6_timestamps_and_due_dates_look_alike():
    cfg = shipped_config("train")
    whole_second = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$")
    for seed in range(20):
        pub = gen(seed, cfg)[0].public
        for inv in pub.invoices:
            assert whole_second.match(inv.issued_at) and whole_second.match(inv.due_at)
            term = parse_ts(inv.due_at) - parse_ts(inv.issued_at)
            assert term.seconds == 0 and term.days in (14, 30, 45, 60)
        for row in pub.ledger:
            assert whole_second.match(row.timestamp)


def test_D12_public_files_hold_no_truth_or_internal_keys(tmp_path: Path):
    for name in ("train", "interacting"):
        cfg = shipped_config(name)
        for seed in range(5):
            episode, _ = gen(seed, cfg)
            blob = b"".join(public_files(episode.public).values())
            for marker in (b"iss_", b"expected_amount", b"evidence_sets", b"provenance"):
                assert marker not in blob
            for key in episode.private.provenance["id_map"]:
                assert key.encode() not in blob
            assert b"inject" not in blob.lower() and b"distractor" not in blob.lower()
            assert b'"seed"' not in blob and b"attempt" not in blob


def test_D12_episode_id_does_not_contain_seed():
    cfg = shipped_config("train")
    for seed in (12345, 987654):
        assert str(seed) not in gen(seed, cfg)[0].public.episode_id


def test_D01_disk_roundtrip_is_exact(tmp_path: Path):
    episode, _ = gen(4, shipped_config("interacting"))
    write_episode(episode, tmp_path / "ep")
    again = read_episode(tmp_path / "ep")
    assert again == episode
    assert fingerprint(again) == fingerprint(episode)


def test_D04_generated_files_match_schemas():
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource

    resources = []
    for path in SCHEMA_DIR.glob("*.schema.json"):
        schema = json.loads(path.read_text(encoding="utf-8"))
        resources.append((schema["$id"], Resource.from_contents(schema)))
    registry = Registry().with_resources(resources)

    def validator(name: str) -> Draft202012Validator:
        schema = json.loads((SCHEMA_DIR / f"{name}.schema.json").read_text(encoding="utf-8"))
        return Draft202012Validator(schema, registry=registry)

    v = {
        n: validator(n)
        for n in ("manifest", "policy", "ledger_row", "invoice", "payment", "fx_rate", "truth")
    }
    for name in ("train", "interacting", "clean"):
        for seed in range(3):
            episode, _ = gen(seed, shipped_config(name))
            pub = episode.public
            v["manifest"].validate(pub.manifest)
            v["policy"].validate(pub.policy)
            for row in pub.ledger:
                v["ledger_row"].validate(row.to_dict())
            for inv in pub.invoices:
                v["invoice"].validate(inv.to_dict())
            for pay in pub.payments:
                v["payment"].validate(pay.to_dict())
            for fx in pub.fx:
                v["fx_rate"].validate(fx.to_dict())
            v["truth"].validate(episode.private.truth.to_dict())
