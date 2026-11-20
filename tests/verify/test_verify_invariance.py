"""K4 / T4 / T6: shuffling every table and consistently renaming every ID leaves the answer
unchanged up to the renaming. No tie is broken by ID spelling or record position."""

from __future__ import annotations

import random
from dataclasses import replace

import pytest
import verify_builders as vb
from hypothesis import given, settings
from hypothesis import strategies as st

from auditgym.verify import check_evidence, solve


def _all_ids(public) -> list[str]:
    ids = set()
    for inv in public.invoices:
        ids |= {inv.doc_id, inv.invoice_id, inv.entity_id}
        ids |= {li.order_ref for li in inv.line_items}
    for p in public.payments:
        ids |= {p.doc_id, p.payment_id, *(p.invoice_refs or ())}
    for r in public.ledger:
        ids |= {r.row_id} | {x for x in (r.entity_id, r.invoice_id, r.payment_id) if x}
    return sorted(ids)


STYLES = ("num", "hex", "rev", "pad")


def _names(n: int, rng: random.Random, style: str) -> list[str]:
    """n distinct new IDs in a random order, spelled so that lexicographic order of the new
    names is unrelated to (or the reverse of) the old one."""
    if style == "num":
        names = [str(k) for k in rng.sample(range(1, 50 * n), n)]
    elif style == "hex":
        names = set()
        while len(names) < n:
            names.add(f"{rng.getrandbits(48):012x}")
        names = sorted(names)
        rng.shuffle(names)
    elif style == "rev":
        names = [f"Z{n - k:05d}" for k in range(n)]
    else:
        names = [f"id{'_' * rng.randint(0, 3)}{k}" for k in rng.sample(range(n), n)]
    return list(names)


def transform(public, rng: random.Random, style: str):
    ids = _all_ids(public)
    m = dict(zip(ids, _names(len(ids), rng, style), strict=True))

    def opt(x):
        return None if x is None else m[x]

    invoices = [
        replace(
            i,
            doc_id=m[i.doc_id],
            invoice_id=m[i.invoice_id],
            entity_id=m[i.entity_id],
            line_items=tuple(replace(li, order_ref=m[li.order_ref]) for li in i.line_items),
        )
        for i in public.invoices
    ]
    payments = [
        replace(
            p,
            doc_id=m[p.doc_id],
            payment_id=m[p.payment_id],
            invoice_refs=None if p.invoice_refs is None else tuple(m[x] for x in p.invoice_refs),
        )
        for p in public.payments
    ]
    ledger = [
        replace(
            r,
            row_id=m[r.row_id],
            entity_id=opt(r.entity_id),
            invoice_id=opt(r.invoice_id),
            payment_id=opt(r.payment_id),
        )
        for r in public.ledger
    ]
    fx = list(public.fx)
    for table in (invoices, payments, ledger, fx):
        rng.shuffle(table)
    out = replace(
        public,
        invoices=tuple(invoices),
        payments=tuple(payments),
        ledger=tuple(ledger),
        fx=tuple(fx),
    )
    return out, m


def project(issues, m=None):
    def f(x):
        return x if m is None else m.get(x, x)  # FX rate ids are not renamed

    return {
        (
            i.type,
            frozenset(f(x) for x in i.affected_ids),
            i.expected_amount_minor,
            i.currency,
            i.amount_tolerance_minor,
            frozenset(frozenset(f(x) for x in s) for s in i.evidence_sets),
        )
        for i in issues
    }


def _bases():
    return {
        "tiny": vb.tiny_public(),
        "kitchen_sink": vb.kitchen_sink()[0],
        "west": vb.west_case(21_800, "payment_local_date"),
    }


BASES = _bases()
BASE_ANSWERS = {name: solve(p) for name, p in BASES.items()}


@pytest.mark.parametrize("name", sorted(BASES))
@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 2**32 - 1), style=st.sampled_from(STYLES))
def test_K4_shuffle_and_rename_preserve_answer(name, seed, style):
    public, m = transform(BASES[name], random.Random(seed), style)
    assert project(solve(public)) == project(BASE_ANSWERS[name], m)


@settings(max_examples=10, deadline=None)
@given(seed=st.integers(0, 2**32 - 1), style=st.sampled_from(STYLES))
def test_K4_evidence_check_survives_shuffle_and_rename(seed, style):
    base = BASES["tiny"]
    public, m = transform(base, random.Random(seed), style)
    for issue in BASE_ANSWERS["tiny"]:
        for ev in issue.evidence_sets:
            check = check_evidence(
                public, issue.type, [m[x] for x in issue.affected_ids], [m.get(x, x) for x in ev]
            )
            assert check.ok and check.amount_minor == issue.expected_amount_minor


@settings(max_examples=15, deadline=None)
@given(seed=st.integers(0, 2**32 - 1))
def test_T6_shuffle_alone_gives_identical_output(seed):
    rng = random.Random(seed)
    base = BASES["kitchen_sink"]
    tables = {k: list(getattr(base, k)) for k in ("invoices", "payments", "ledger", "fx")}
    for rows in tables.values():
        rng.shuffle(rows)
    shuffled = replace(base, **{k: tuple(v) for k, v in tables.items()})
    assert solve(shuffled) == BASE_ANSWERS["kitchen_sink"]
