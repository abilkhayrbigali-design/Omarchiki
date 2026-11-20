"""Independent sufficiency of evidence (DOMAIN section 6, "Допустимые наборы доказательств"; K2).

`check_evidence` decides from the records named in `evidence_ids`, plus manifest and policy,
whether they establish an issue of the given type on the given affected IDs, and recomputes
its amount. The tables of the episode are touched in exactly one place, `restrict`, which keeps
only the named records; every check below sees nothing else. A record outside the evidence set
therefore cannot influence the answer.

What a subset of records cannot show is absence: that no further copy, earlier payment or
posting row exists outside the set. The checks take the set at its word on that and verify
everything else, including that the payment rows in a set are complete enough to balance.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from auditgym.contracts.constants import ISSUE_TYPES, MAX_EVIDENCE_IDS
from auditgym.contracts.types import FxRate, Invoice, LedgerRow, Payment, PublicEpisode
from auditgym.verify.allocation import expected_allocations
from auditgym.verify.errors import ID_NOT_UNIQUE, VerifyError
from auditgym.verify.rules import (
    AR,
    CASH,
    FX_CLEARING,
    UNAPPLIED,
    Rules,
    check_invoice,
    check_payment,
    check_rate,
    check_row,
    load_rules,
    net_debit,
    sign,
)
from auditgym.verify.solver import duplicate_amount, duplicate_groups


@dataclass(frozen=True, slots=True)
class EvidenceCheck:
    ok: bool
    amount_minor: int | None
    currency: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class EvidenceView:
    """The records named by an evidence set, and nothing else."""

    invoices: tuple[Invoice, ...]
    payments: tuple[Payment, ...]
    rows: tuple[LedgerRow, ...]
    rates: tuple[FxRate, ...]


class _Reject(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def restrict(public: PublicEpisode, ids: frozenset[str]) -> EvidenceView:
    """Keep only the records whose public ID is in `ids`. The single read of the tables."""
    return EvidenceView(
        invoices=tuple(i for i in public.invoices if i.doc_id in ids),
        payments=tuple(p for p in public.payments if p.doc_id in ids),
        rows=tuple(r for r in public.ledger if r.row_id in ids),
        rates=tuple(f for f in public.fx if f.rate_id in ids),
    )


def check_evidence(
    public: PublicEpisode,
    issue_type: str,
    affected_ids: Iterable[str],
    evidence_ids: Iterable[str],
) -> EvidenceCheck:
    """Do the evidence records alone establish `issue_type` on `affected_ids`? With what amount?

    Returns ok=False with a reason when they do not. Raises VerifyError when the policy or a
    named record is malformed.
    """
    rules = load_rules(public.manifest, public.policy)
    evidence = list(evidence_ids)
    affected = list(affected_ids)
    if not all(isinstance(x, str) for x in evidence + affected):
        return _fail("ids must be strings")
    if issue_type not in ISSUE_TYPES:
        return _fail("unknown issue type")
    if len(evidence) > MAX_EVIDENCE_IDS:
        return _fail(f"more than {MAX_EVIDENCE_IDS} evidence ids")
    if not affected:
        return _fail("no affected ids")
    ev = frozenset(evidence)
    if not set(affected) <= ev:
        return _fail("an affected id is not in the evidence set")

    view = restrict(public, ev)
    _validate(view, rules)
    try:
        amount, currency = _CHECKS[issue_type](view, rules, frozenset(affected))
    except _Reject as rej:
        return _fail(rej.reason)
    return EvidenceCheck(True, amount, currency, "established")


def _fail(reason: str) -> EvidenceCheck:
    return EvidenceCheck(False, None, None, reason)


def _validate(view: EvidenceView, rules: Rules) -> None:
    for inv in view.invoices:
        check_invoice(inv, rules)
    for pay in view.payments:
        check_payment(pay, rules)
    for row in view.rows:
        check_row(row, rules)
    for rate in view.rates:
        check_rate(rate)
    ids = (
        [i.doc_id for i in view.invoices]
        + [p.doc_id for p in view.payments]
        + [r.row_id for r in view.rows]
        + [f.rate_id for f in view.rates]
    )
    if len(set(ids)) != len(ids):
        raise VerifyError(ID_NOT_UNIQUE, "an evidence id names more than one record")


def _one(items: list, what: str):
    if len(items) != 1:
        raise _Reject(f"needs exactly one {what} among affected ids")
    return items[0]


def _invoice_document(view: EvidenceView, invoice_id: str, rules: Rules) -> Invoice | None:
    """Earliest-issued document of an invoice_id among the evidence, or None."""
    docs = [i for i in view.invoices if i.invoice_id == invoice_id]
    if not docs:
        return None
    instants = [rules.instant(d.issued_at) for d in docs]
    first = min(instants)
    if instants.count(first) > 1:
        raise _Reject(f"documents of {invoice_id} share issued_at")
    return docs[instants.index(first)]


# ---------------------------------------------------------------------------------- per type
def _duplicate(view: EvidenceView, rules: Rules, affected: frozenset[str]) -> tuple[int, str]:
    if len(affected) < 2:
        raise _Reject("a duplicate needs at least two documents")
    if not affected <= {i.doc_id for i in view.invoices}:
        raise _Reject("an affected id is not an invoice document")
    for group in duplicate_groups(view.invoices):
        ids = {i.doc_id for i in group}
        if not affected & ids:
            continue
        if ids != affected:
            raise _Reject("affected documents are not exactly one group of linked copies")
        try:
            return duplicate_amount(group)
        except VerifyError as exc:
            raise _Reject(exc.detail) from exc
    raise _Reject("documents share neither invoice_id nor order_ref")


def _allocation(view: EvidenceView, rules: Rules, affected: frozenset[str]) -> tuple[int, str]:
    if len(affected) != 2:
        raise _Reject("payment_allocation needs a payment and an invoice")
    p = _one([x for x in view.payments if x.doc_id in affected], "payment document")
    inv = _one([x for x in view.invoices if x.doc_id in affected], "invoice document")
    invoice_id = inv.invoice_id
    if invoice_id not in (p.invoice_refs or ()):
        raise _Reject("the payment does not reference this invoice")
    if inv.currency != p.currency:
        raise _Reject("payment and invoice currencies differ")
    document = _invoice_document(view, invoice_id, rules)
    if document is None or document.doc_id != inv.doc_id:
        raise _Reject("an earlier document with the same invoice_id is in evidence")

    source_role = CASH if p.currency == rules.base_currency else FX_CLEARING
    rows = [r for r in view.rows if r.payment_id == p.payment_id and r.currency == p.currency]
    if any(rules.role(r) not in (source_role, AR, UNAPPLIED) for r in rows):
        raise _Reject("unexpected account in the payment posting")
    source = [r for r in rows if rules.role(r) == source_role]
    if net_debit(source) != p.amount_minor or net_debit(rows) != 0:
        raise _Reject("payment rows in the payment currency are incomplete or unbalanced")

    opening: dict[str, int] = {}
    for iid in sorted({i.invoice_id for i in view.invoices}):
        doc = _invoice_document(view, iid, rules)
        assert doc is not None
        opening[iid] = doc.total_minor
    allocation = expected_allocations(view.payments, opening, rules, partial=True)
    expected = allocation[p.payment_id].by_invoice.get(invoice_id)
    if expected is None:
        raise _Reject(
            "insufficient: the expected allocation depends on invoices or payments "
            "that are not in the evidence set"
        )
    actual = sum(
        r.credit_minor - r.debit_minor
        for r in rows
        if rules.role(r) == AR and r.invoice_id == invoice_id
    )
    shortfall = sign(p.amount_minor) * (expected - actual)
    if shortfall <= 0:
        raise _Reject("the invoice received at least its expected allocation")
    return shortfall, p.currency


def _fx(view: EvidenceView, rules: Rules, affected: frozenset[str]) -> tuple[int, str]:
    if len(affected) != 1:
        raise _Reject("fx_conversion has exactly one affected payment")
    p = _one([x for x in view.payments if x.doc_id in affected], "payment document")
    base = rules.base_currency
    if p.currency == base:
        raise _Reject("payment is in the base currency")
    rate_id = rules.rate_id_for(p)
    rate = next((f for f in view.rates if f.rate_id == rate_id), None)
    if rate is None:
        raise _Reject("the policy rate for this payment is not in evidence")
    expected = rules.to_base(p.amount_minor, p.currency, rate.rate)

    own = [r for r in view.rows if r.payment_id == p.payment_id]
    cash = [r for r in own if rules.role(r) == CASH]
    clearing = [r for r in own if rules.role(r) == FX_CLEARING and r.currency == base]
    if any(r.currency != base for r in cash):
        raise _Reject("CASH row not in the base currency")
    booked = []
    if cash:
        booked.append(net_debit(cash))
    if clearing:
        booked.append(-net_debit(clearing))
    if not booked:
        raise _Reject("no CASH or base-currency FX_CLEARING row of the payment in evidence")
    if len(set(booked)) != 1:
        raise _Reject("CASH and FX_CLEARING rows in evidence disagree")
    diff = booked[0] - expected
    if diff == 0:
        raise _Reject("the booked conversion equals the policy conversion")
    return abs(diff), base


def _cutoff(view: EvidenceView, rules: Rules, affected: frozenset[str]) -> tuple[int, str]:
    if len(affected) != 1:
        raise _Reject("period_cutoff has exactly one affected document")
    invs = [x for x in view.invoices if x.doc_id in affected]
    pays = [x for x in view.payments if x.doc_id in affected]
    if invs:
        inv = invs[0]
        if sum(1 for x in view.invoices if x.invoice_id == inv.invoice_id) > 1:
            raise _Reject("several documents share this invoice_id; rows cannot be attributed")
        rows = [r for r in view.rows if r.invoice_id == inv.invoice_id and r.payment_id is None]
        doc_ts, amount, currency = inv.issued_at, abs(inv.total_minor), inv.currency
    elif pays:
        p = pays[0]
        rows = [r for r in view.rows if r.payment_id == p.payment_id]
        doc_ts, amount, currency = p.timestamp, abs(p.amount_minor), p.currency
    else:
        raise _Reject("the affected id is not a document")
    if not rows:
        raise _Reject("no row of the document's posting in evidence")
    if len({rules.instant(r.timestamp) for r in rows}) != 1:
        raise _Reject("rows of the posting disagree on the timestamp")
    if rules.period_of(doc_ts) == rules.period_of(rows[0].timestamp):
        raise _Reject("the posting is in the same period as the document")
    return amount, currency


_CHECKS = {
    "duplicate_invoice": _duplicate,
    "payment_allocation": _allocation,
    "fx_conversion": _fx,
    "period_cutoff": _cutoff,
}
