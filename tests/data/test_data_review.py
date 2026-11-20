"""Review findings on the generator: scope, period variety, allocation difficulty, config
fidelity of size.invoices, FX materiality (K2, G06, G07, D01)."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import timedelta

import pytest
from genkit import SHIPPED, gen, shipped_config

from auditgym.contracts.config import ConfigError, normalize_config
from auditgym.contracts.constants import ISSUE_TYPES
from auditgym.contracts.types import fx_rate_id
from auditgym.episode.generate import CapacityError, generate_episode
from auditgym.util.money import convert_minor
from auditgym.util.timeutil import local_date, parse_offset, parse_ts, utc_date
from auditgym.world.sizing import required_invoices


def _rows(pub):
    by_invoice, by_payment = defaultdict(list), defaultdict(list)
    for r in pub.ledger:
        if r.payment_id:
            by_payment[r.payment_id].append(r)
        elif r.invoice_id:
            by_invoice[r.invoice_id].append(r)
    return by_invoice, by_payment


def _period(pub):
    return parse_ts(pub.manifest["period_start"]), parse_ts(pub.manifest["period_end"])


def _scope_checker(pub):
    """doc_id -> True when the document date or its posting lies inside the period."""
    start, end = _period(pub)
    invs = {i.doc_id: i for i in pub.invoices}
    pays = {p.doc_id: p for p in pub.payments}
    by_invoice, by_payment = _rows(pub)

    def inside(ts: str) -> bool:
        return start <= parse_ts(ts) < end

    def in_scope(doc_id: str) -> bool:
        if doc_id in invs:
            d = invs[doc_id]
            return inside(d.issued_at) or any(inside(r.timestamp) for r in by_invoice[d.invoice_id])
        p = pays[doc_id]
        return inside(p.timestamp) or any(inside(r.timestamp) for r in by_payment[p.payment_id])

    return in_scope


def _class_and_month(pub):
    tz = parse_offset(pub.manifest["timezone"])
    start, end = _period(pub)

    def cls(ts: str) -> int:
        t = parse_ts(ts)
        return 0 if t < start else 1 if t < end else 2

    def month(ts: str) -> tuple[int, int]:
        d = local_date(ts, tz)
        return d.year, d.month

    return cls, month


# ---------------------------------------------------------------------------------- 1. scope
@pytest.mark.parametrize("name", ["train", "dev", "interacting", "smoke"])
def test_K2_every_issue_is_in_the_reporting_period(name):
    """The erroneous record has its document date or its posting inside the period."""
    cfg = shipped_config(name)
    for seed in range(40):
        episode, _ = gen(seed, cfg)
        pub = episode.public
        in_scope = _scope_checker(pub)
        pays = {p.doc_id for p in pub.payments}
        for issue in episode.private.truth.issues:
            targets = list(issue.affected_ids)
            if issue.type in ("payment_allocation", "fx_conversion"):
                targets = [d for d in targets if d in pays]
            assert any(in_scope(d) for d in targets), (name, seed, issue)


# ---------------------------------------------------------------------------------- 2. periods
def test_G06_period_kinds_vary_and_boundaries_are_local_midnights():
    cfg = shipped_config("train")
    kinds = Counter()
    for seed in range(120):
        pub = gen(seed, cfg)[0].public
        kind = gen(seed, cfg)[0].private.provenance["world"]["period_kind"]
        kinds[kind] += 1
        tz = parse_offset(pub.manifest["timezone"])
        start, end = _period(pub)
        for b in (start, end):
            local = b.astimezone(tz)
            assert (local.hour, local.minute, local.second) == (0, 0, 0)
        days = (end - start).days
        if kind == "month":
            assert start.day == 1 and 28 <= days <= 31
        elif kind == "quarter":
            assert start.day == 1 and start.month in (1, 4, 7, 10) and 89 <= days <= 92
        else:
            assert days == (28 if kind == "fiscal_4w" else 35)
    assert set(kinds) == {"month", "quarter", "fiscal_4w", "fiscal_5w"}
    assert kinds["month"] > max(kinds["quarter"], kinds["fiscal_4w"], kinds["fiscal_5w"])


def test_G07_clean_postings_cross_calendar_months_inside_a_period():
    """A different calendar month is not a different period: some clean postings show it."""
    cfg = shipped_config("train")
    crossing = 0
    for seed in range(80):
        _, clean = gen(seed, cfg)
        cls, month = _class_and_month(clean)
        by_invoice, by_payment = _rows(clean)
        docs = [(i.issued_at, by_invoice[i.invoice_id]) for i in clean.invoices]
        docs += [(p.timestamp, by_payment[p.payment_id]) for p in clean.payments]
        for ts, rows in docs:
            for r in rows[:1]:
                assert cls(ts) == cls(r.timestamp)  # clean: same period class
                crossing += month(ts) != month(r.timestamp)
    assert crossing >= 10


# ---------------------------------------------------------------------------------- 3. allocation
def test_G07_allocation_errors_on_later_instalments_and_multi_reference_payments():
    shapes = Counter()
    below_shortcut = 0
    for name in ("train", "dev"):
        cfg = shipped_config(name)
        for seed in range(80):
            for inj in gen(seed, cfg)[0].private.provenance["injections"]:
                if inj["injector"] != "payment_allocation":
                    continue
                d = inj["details"]
                shapes[d["shape"]] += 1
                if d["expected_applied"] < min(d["payment_amount"], d["invoice_total"]):
                    below_shortcut += 1
    assert set(shapes) == {"first", "later", "multi_ref"}
    assert shapes["later"] + shapes["multi_ref"] > shapes["first"]
    assert below_shortcut >= 10  # expected = open balance < min(payment, invoice total)


def test_G07_clean_payments_applied_below_min_of_payment_and_total():
    """Clean distractor: an instalment applied only up to the remaining open balance."""
    found = 0
    cfg = shipped_config("train")
    for seed in range(40):
        _, clean = gen(seed, cfg)
        roles = {code: role for role, code in clean.policy["accounts"].items()}
        first_doc = {}
        for inv in sorted(clean.invoices, key=lambda i: parse_ts(i.issued_at)):
            first_doc.setdefault(inv.invoice_id, inv)
        _, by_payment = _rows(clean)
        for p in clean.payments:
            if not p.invoice_refs or p.amount_minor <= 0 or p.invoice_refs[0] not in first_doc:
                continue
            inv = first_doc[p.invoice_refs[0]]
            applied = sum(
                r.credit_minor - r.debit_minor
                for r in by_payment[p.payment_id]
                if roles[r.account] == "AR" and r.invoice_id == inv.invoice_id
            )
            if 0 < applied < min(p.amount_minor, inv.total_minor):
                found += 1
    assert found >= 10


# ---------------------------------------------------------------------------------- 4. sizes
@pytest.mark.parametrize("name", [*SHIPPED, "stress"])
def test_D01_invoice_documents_within_size_invoices(name):
    cfg = shipped_config(name)
    lo, hi = cfg["size"]["invoices"]
    assert required_invoices(cfg) <= hi
    counts = Counter()
    for seed in range(40 if name != "stress" else 8):
        n = len(gen(seed, cfg)[0].public.invoices)
        assert lo <= n <= hi, (name, seed, n)
        counts[n] += 1
    assert len(counts) > 1 or lo == hi


def test_D01_exact_invoice_count_when_min_equals_max():
    raw = {
        "config_id": "exact-20",
        "size": {"invoices": [20, 20]},
        "errors": {t: [0, 1] for t in ISSUE_TYPES},
        "interactions": {"fx_partial_at_boundary": 0, "duplicate_misallocated": 0},
    }
    cfg = normalize_config(raw)
    for seed in range(30):
        episode, clean = gen(seed, cfg)
        copies = sum(
            len(i.affected_ids) - 1
            for i in episode.private.truth.issues
            if i.type == "duplicate_invoice"
        )
        assert len(episode.public.invoices) == 20
        assert len(clean.invoices) == 20 - copies


def test_D01_too_small_size_fails_fast_with_config_error():
    with pytest.raises(CapacityError) as err:
        generate_episode(1, {"config_id": "tiny", "size": {"invoices": [4, 6]}})
    assert isinstance(err.value, ConfigError)
    assert "at least" in str(err.value)
    # Without errors and distractors, a small company fits.
    tiny = {
        "config_id": "tiny-clean",
        "clean": True,
        "size": {"invoices": [4, 6], "customers": [3, 3]},
        "distractors": {k: 0 for k in shipped_config("train")["distractors"]},
    }
    for seed in range(10):
        assert 4 <= len(generate_episode(seed, tiny).public.invoices) <= 6


# ---------------------------------------------------------------------------------- 5. FX materiality
def test_K2_fx_errors_are_material():
    """|booked - correct| >= max(100, 0.5% of the correct converted amount), recomputed here."""
    for name in ("train", "dev", "interacting"):
        cfg = shipped_config(name)
        for seed in range(40):
            episode, _ = gen(seed, cfg)
            pub = episode.public
            pol = pub.policy
            tz = parse_offset(pol["timezone"])
            base = pol["base_currency"]
            rates = {f.rate_id: f.rate for f in pub.fx}
            pays = {p.doc_id: p for p in pub.payments}
            for issue in episode.private.truth.issues:
                if issue.type != "fx_conversion":
                    continue
                p = pays[issue.affected_ids[0]]
                if pol["fx"]["rate_date"] == "payment_local_date":
                    day = local_date(p.timestamp, tz)
                elif pol["fx"]["rate_date"] == "payment_local_date_minus_1":
                    day = local_date(p.timestamp, tz) - timedelta(days=1)
                else:
                    day = utc_date(p.timestamp)
                rate = rates[fx_rate_id(day.isoformat(), p.currency, base)]
                right = convert_minor(p.amount_minor, p.currency, base, rate, pol["fx"]["rounding"])
                diff = issue.expected_amount_minor
                assert diff >= 100
                assert diff * 200 >= abs(right), (name, seed, diff, right)
