"""Independent solver: recompute the expected books from documents, FX and policy, report
every discrepancy as an Issue (docs/DOMAIN.md, sections 4-9).

Written from DOMAIN.md and the public data only. It does not import world/, inject/,
episode/, books/ or present/.

Expected values never come from other ledger rows (T1). Every comparison is keyed by
document links, never by record position or by the spelling of IDs (T4). Anything that differs
from the expected books but fits none of the four issue types raises VerifyError: by T5 the
public data holds no discrepancy outside the reference answer.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from auditgym.contracts.constants import MAX_EVIDENCE_IDS
from auditgym.contracts.types import FxRate, Invoice, Issue, LedgerRow, Payment, PublicEpisode
from auditgym.verify.allocation import PaymentAllocation, expected_allocations
from auditgym.verify.errors import (
    COUNTS_MISMATCH,
    CUTOFF_AMBIGUOUS,
    DUPLICATE_GROUP_INVALID,
    EVIDENCE_TOO_LARGE,
    FX_RATE_MISSING,
    ID_NOT_UNIQUE,
    POSTING_INVALID,
    UNEXPLAINED_DISCREPANCY,
    UNKNOWN_REFERENCE,
    UNPOSTED_DOCUMENT,
    VerifyError,
)
from auditgym.verify.rules import (
    AR,
    CASH,
    EXPENSE,
    FX_CLEARING,
    REVENUE,
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


def solve(public: PublicEpisode) -> tuple[Issue, ...]:
    """All discrepancies in the public data, sorted by (type, affected_ids)."""
    rules = load_rules(public.manifest, public.policy)
    book = Book.build(public, rules)
    issues: list[Issue] = []
    issues += duplicate_issues(book, rules)
    issues += invoice_posting_issues(book, rules)
    issues += payment_issues(book, rules)
    check_background(book, rules)
    check_ledger_balance(public.ledger)
    keys = [i.key() for i in issues]
    if len(set(keys)) != len(keys):
        raise VerifyError(UNEXPLAINED_DISCREPANCY, "two findings share (type, affected_ids)")
    return tuple(sorted(issues, key=lambda i: i.key()))


# ---------------------------------------------------------------------------------- index
@dataclass(frozen=True, slots=True)
class Book:
    invoices: tuple[Invoice, ...]
    payments: tuple[Payment, ...]
    docs_by_invoice_id: dict[str, list[Invoice]]  # sorted by issued_at instant
    payment_by_id: dict[str, Payment]
    rates: dict[str, FxRate]
    rows_by_payment: dict[str, list[LedgerRow]]
    rows_by_invoice: dict[str, list[LedgerRow]]  # rows with invoice_id and payment_id = null
    background: list[LedgerRow]  # rows with neither link

    @staticmethod
    def build(public: PublicEpisode, rules: Rules) -> Book:
        check_counts(public)
        for inv in public.invoices:
            check_invoice(inv, rules)
        for pay in public.payments:
            check_payment(pay, rules)
        for row in public.ledger:
            check_row(row, rules)
        for rate in public.fx:
            check_rate(rate)
        check_ids(public)

        docs_by_invoice_id: dict[str, list[Invoice]] = defaultdict(list)
        for inv in public.invoices:
            docs_by_invoice_id[inv.invoice_id].append(inv)
        for invoice_id, docs in docs_by_invoice_id.items():
            docs.sort(key=lambda d: rules.instant(d.issued_at))
            instants = [rules.instant(d.issued_at) for d in docs]
            if len(set(instants)) != len(instants):
                raise VerifyError(
                    DUPLICATE_GROUP_INVALID,
                    f"documents of {invoice_id} share issued_at; the invoice document is undefined",
                )

        rows_by_payment: dict[str, list[LedgerRow]] = defaultdict(list)
        rows_by_invoice: dict[str, list[LedgerRow]] = defaultdict(list)
        background: list[LedgerRow] = []
        for row in public.ledger:
            if row.payment_id is not None:
                rows_by_payment[row.payment_id].append(row)
            elif row.invoice_id is not None:
                rows_by_invoice[row.invoice_id].append(row)
            else:
                background.append(row)

        return Book(
            invoices=public.invoices,
            payments=public.payments,
            docs_by_invoice_id=dict(docs_by_invoice_id),
            payment_by_id={p.payment_id: p for p in public.payments},
            rates={r.rate_id: r for r in public.fx},
            rows_by_payment=dict(rows_by_payment),
            rows_by_invoice=dict(rows_by_invoice),
            background=background,
        )

    def invoice_document(self, invoice_id: str) -> Invoice:
        """The document of an invoice_id: the earliest issued_at (DOMAIN 6, Уточнения)."""
        try:
            return self.docs_by_invoice_id[invoice_id][0]
        except KeyError as exc:
            raise VerifyError(UNKNOWN_REFERENCE, f"no invoice document for {invoice_id}") from exc


def check_counts(public: PublicEpisode) -> None:
    counts = public.manifest.get("counts")
    actual = {
        "ledger": len(public.ledger),
        "invoices": len(public.invoices),
        "payments": len(public.payments),
        "fx": len(public.fx),
        "documents": len(public.invoices) + len(public.payments),
    }
    if not isinstance(counts, dict) or any(counts.get(k) != v for k, v in actual.items()):
        raise VerifyError(COUNTS_MISMATCH, "manifest counts differ from the tables")


def check_ids(public: PublicEpisode) -> None:
    """All IDs are unique, also across tables (DOMAIN 2). invoice_id may repeat across
    documents only as a duplicate, which is reported, not rejected."""
    kinds: dict[str, list[str]] = {
        "doc_id": [d.doc_id for d in public.invoices] + [p.doc_id for p in public.payments],
        "row_id": [r.row_id for r in public.ledger],
        "payment_id": [p.payment_id for p in public.payments],
        "rate_id": [r.rate_id for r in public.fx],
    }
    owner: dict[str, str] = {}
    for kind, ids in sorted(kinds.items()):
        if len(set(ids)) != len(ids):
            raise VerifyError(ID_NOT_UNIQUE, f"repeated {kind}")
        for value in ids:
            if value in owner:
                raise VerifyError(ID_NOT_UNIQUE, f"{value} is both {owner[value]} and {kind}")
            owner[value] = kind
    for value in sorted({d.invoice_id for d in public.invoices}):
        if value in owner:
            raise VerifyError(ID_NOT_UNIQUE, f"{value} is both {owner[value]} and invoice_id")


# ---------------------------------------------------------------------------------- issues
def make_issue(
    rules: Rules,
    issue_type: str,
    affected: Iterable[str],
    amount: int,
    currency: str,
    evidence_sets: Iterable[Iterable[str]],
) -> Issue:
    affected_ids = tuple(sorted(set(affected)))
    sets = tuple(tuple(sorted(set(s))) for s in evidence_sets)
    for s in sets:
        if len(s) > MAX_EVIDENCE_IDS:
            raise VerifyError(
                EVIDENCE_TOO_LARGE,
                f"{issue_type} on {list(affected_ids)} needs {len(s)} evidence ids",
            )
    digest = hashlib.sha256("\0".join((issue_type, *affected_ids)).encode("utf-8")).hexdigest()
    return Issue(
        issue_id=f"vfy_{digest[:16]}",
        type=issue_type,
        affected_ids=affected_ids,
        expected_amount_minor=amount,
        currency=currency,
        amount_tolerance_minor=rules.tolerance(issue_type),
        evidence_sets=sets,
    )


# ---------------------------------------------------------------------------------- duplicates
def duplicate_groups(invoices: Iterable[Invoice]) -> list[list[Invoice]]:
    """Connected groups of invoice documents linked by a shared invoice_id or order_ref (I7).

    Groups of one are not returned. Each group is sorted by doc_id for stable output only.
    """
    docs = {inv.doc_id: inv for inv in invoices}
    parent = {doc_id: doc_id for doc_id in docs}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)  # representative choice does not affect groups

    first_by_key: dict[tuple[str, str], str] = {}
    for doc_id in sorted(docs):
        inv = docs[doc_id]
        keys = {("invoice_id", inv.invoice_id)} | {
            ("order_ref", li.order_ref) for li in inv.line_items
        }
        for key in sorted(keys):
            if key in first_by_key:
                union(first_by_key[key], doc_id)
            else:
                first_by_key[key] = doc_id

    groups: dict[str, list[Invoice]] = defaultdict(list)
    for doc_id in sorted(docs):
        groups[find(doc_id)].append(docs[doc_id])
    return [g for _, g in sorted(groups.items()) if len(g) > 1]


def duplicate_amount(group: list[Invoice]) -> tuple[int, str]:
    """(k - 1) * |total_minor| in the invoice currency. Copies must agree on both."""
    currencies = {inv.currency for inv in group}
    totals = {inv.total_minor for inv in group}
    if len(currencies) != 1 or len(totals) != 1:
        raise VerifyError(
            DUPLICATE_GROUP_INVALID,
            f"documents {[i.doc_id for i in group]} share an invoice_id or order_ref "
            "but differ in currency or total",
        )
    return (len(group) - 1) * abs(group[0].total_minor), group[0].currency


def duplicate_issues(book: Book, rules: Rules) -> list[Issue]:
    out = []
    for group in duplicate_groups(book.invoices):
        amount, currency = duplicate_amount(group)
        doc_ids = [inv.doc_id for inv in group]
        out.append(make_issue(rules, "duplicate_invoice", doc_ids, amount, currency, [doc_ids]))
    return out


# ---------------------------------------------------------------------------------- invoices
def _group_by_instant(
    rows: list[LedgerRow], rules: Rules
) -> list[tuple[datetime, list[LedgerRow]]]:
    groups: dict[datetime, list[LedgerRow]] = defaultdict(list)
    for row in rows:
        groups[rules.instant(row.timestamp)].append(row)
    return sorted(groups.items(), key=lambda kv: kv[0])


def check_invoice_posting(rows: list[LedgerRow], inv: Invoice, rules: Rules) -> None:
    """One recognition posting: Dr AR T, Cr REVENUE T in the invoice currency (I3).
    A credit note (T < 0) swaps the sides."""
    total = inv.total_minor
    ar = [r for r in rows if rules.role(r) == AR]
    revenue = [r for r in rows if rules.role(r) == REVENUE]
    where = f"posting of invoice {inv.invoice_id}"
    if len(ar) + len(revenue) != len(rows):
        raise VerifyError(POSTING_INVALID, f"{where}: account other than AR/REVENUE")
    if any(r.currency != inv.currency for r in rows):
        raise VerifyError(POSTING_INVALID, f"{where}: row not in the invoice currency")
    ar_is_debit = total > 0
    if any((r.debit_minor > 0) != ar_is_debit for r in ar) or any(
        (r.debit_minor > 0) == ar_is_debit for r in revenue
    ):
        raise VerifyError(POSTING_INVALID, f"{where}: AR/REVENUE on the wrong side")
    if net_debit(ar) != total or net_debit(revenue) != -total:
        raise VerifyError(
            UNEXPLAINED_DISCREPANCY, f"{where}: amount differs from the document total"
        )


def invoice_posting_issues(book: Book, rules: Rules) -> list[Issue]:
    for invoice_id in sorted(book.rows_by_invoice):
        if invoice_id not in book.docs_by_invoice_id:
            raise VerifyError(UNKNOWN_REFERENCE, f"ledger rows for unknown invoice {invoice_id}")
    out = []
    for invoice_id, docs in sorted(book.docs_by_invoice_id.items()):
        rows = book.rows_by_invoice.get(invoice_id, [])
        if not rows:
            raise VerifyError(UNPOSTED_DOCUMENT, f"invoice {invoice_id} has no posting")
        if len({(d.currency, d.total_minor) for d in docs}) != 1:
            raise VerifyError(DUPLICATE_GROUP_INVALID, f"documents of {invoice_id} differ")
        postings = _group_by_instant(rows, rules)
        for _, posting in postings:
            check_invoice_posting(posting, docs[0], rules)
        if len(postings) != len(docs):
            raise VerifyError(
                POSTING_INVALID,
                f"invoice {invoice_id}: {len(docs)} documents but {len(postings)} postings",
            )
        doc_periods = sorted(rules.period_of(d.issued_at) for d in docs)
        posting_periods = sorted(rules.period_of(p[0].timestamp) for _, p in postings)
        if doc_periods == posting_periods:
            continue
        if len(docs) > 1:
            raise VerifyError(
                CUTOFF_AMBIGUOUS,
                f"invoice {invoice_id}: periods of postings do not match its {len(docs)} documents",
            )
        inv = docs[0]
        out.append(
            make_issue(
                rules,
                "period_cutoff",
                [inv.doc_id],
                abs(inv.total_minor),
                inv.currency,
                [[inv.doc_id, r.row_id] for r in rows],
            )
        )
    return out


# ---------------------------------------------------------------------------------- payments
def _check_payment_rows(p: Payment, rows: list[LedgerRow], book: Book, rules: Rules) -> None:
    """Shape of a payment posting (DOMAIN 4): accounts, currencies, sides, links."""
    base, cur = rules.base_currency, p.currency
    foreign = cur != base
    positive = p.amount_minor > 0
    where = f"posting of payment {p.payment_id}"
    for r in rows:
        role = rules.role(r)
        is_debit = r.debit_minor > 0
        if role == CASH:
            ok = r.currency == base and is_debit == positive
        elif role == FX_CLEARING and foreign and r.currency == cur:
            ok = is_debit == positive
        elif role == FX_CLEARING and foreign and r.currency == base:
            ok = is_debit != positive
        elif role in (AR, UNAPPLIED):
            ok = r.currency == cur and is_debit != positive
        else:
            ok = False
        if not ok:
            raise VerifyError(
                POSTING_INVALID, f"{where}: row {r.row_id} has unexpected account/currency/side"
            )
        if role == UNAPPLIED and r.invoice_id is not None:
            raise VerifyError(POSTING_INVALID, f"{where}: UNAPPLIED row {r.row_id} has invoice_id")
        if role == AR:
            if r.invoice_id is None:
                raise VerifyError(POSTING_INVALID, f"{where}: AR row {r.row_id} without invoice_id")
            if book.invoice_document(r.invoice_id).currency != cur:
                raise VerifyError(POSTING_INVALID, f"{where}: AR row on invoice in other currency")
    by_currency: dict[str, int] = defaultdict(int)
    for r in rows:
        by_currency[r.currency] += r.debit_minor - r.credit_minor
    if any(v != 0 for v in by_currency.values()):
        raise VerifyError(POSTING_INVALID, f"{where}: debits differ from credits (I1)")


def payment_issues(book: Book, rules: Rules) -> list[Issue]:
    for payment_id in sorted(book.rows_by_payment):
        if payment_id not in book.payment_by_id:
            raise VerifyError(UNKNOWN_REFERENCE, f"ledger rows for unknown payment {payment_id}")
    for p in book.payments:
        for ref in p.invoice_refs or ():
            if book.invoice_document(ref).currency != p.currency:
                raise VerifyError(
                    UNKNOWN_REFERENCE, f"payment {p.doc_id} references {ref} in another currency"
                )
    opening = {iid: docs[0].total_minor for iid, docs in book.docs_by_invoice_id.items()}
    expected = expected_allocations(book.payments, opening, rules)

    out: list[Issue] = []
    for p in sorted(book.payments, key=lambda x: x.doc_id):
        rows = book.rows_by_payment.get(p.payment_id, [])
        if not rows:
            raise VerifyError(UNPOSTED_DOCUMENT, f"payment {p.payment_id} has no posting")
        if len({rules.instant(r.timestamp) for r in rows}) != 1:
            raise VerifyError(POSTING_INVALID, f"payment {p.payment_id}: rows differ in timestamp")
        _check_payment_rows(p, rows, book, rules)
        out += _fx_issue(p, rows, book, rules)
        out += _allocation_issues(p, rows, expected[p.payment_id], book, rules)
        if rules.period_of(p.timestamp) != rules.period_of(rows[0].timestamp):
            out.append(
                make_issue(
                    rules,
                    "period_cutoff",
                    [p.doc_id],
                    abs(p.amount_minor),
                    p.currency,
                    [[p.doc_id, r.row_id] for r in rows],
                )
            )
    return out


def _fx_issue(p: Payment, rows: list[LedgerRow], book: Book, rules: Rules) -> list[Issue]:
    base, cur = rules.base_currency, p.currency
    source = [r for r in rows if r.currency == cur and rules.role(r) in (CASH, FX_CLEARING)]
    if net_debit(source) != p.amount_minor:
        raise VerifyError(
            UNEXPLAINED_DISCREPANCY,
            f"payment {p.payment_id}: received amount booked differs from the document",
        )
    if cur == base:
        return []
    rate_id = rules.rate_id_for(p)
    rate = book.rates.get(rate_id)
    if rate is None:
        raise VerifyError(FX_RATE_MISSING, f"payment {p.doc_id} needs rate {rate_id}")
    expected = rules.to_base(p.amount_minor, cur, rate.rate)
    cash = [r for r in rows if rules.role(r) == CASH]
    clearing_base = [r for r in rows if rules.role(r) == FX_CLEARING and r.currency == base]
    actual = net_debit(cash)
    if actual == expected:
        return []
    ids = [p.doc_id, rate_id]
    return [
        make_issue(
            rules,
            "fx_conversion",
            [p.doc_id],
            abs(actual - expected),
            base,
            [ids + [r.row_id for r in cash], ids + [r.row_id for r in clearing_base]],
        )
    ]


def _allocation_issues(
    p: Payment, rows: list[LedgerRow], expected: PaymentAllocation, book: Book, rules: Rules
) -> list[Issue]:
    """One payment_allocation per referenced invoice that received less than the rule gives.
    Excess elsewhere (another invoice or UNAPPLIED) is where the shortfall went, not a finding;
    an UNAPPLIED shortfall fits no issue type."""
    if expected.unapplied is None:
        raise VerifyError(UNKNOWN_REFERENCE, f"payment {p.payment_id}: allocation undetermined")
    direction = sign(p.amount_minor)
    actual: dict[str, int] = defaultdict(int)
    for r in rows:
        if rules.role(r) == AR:
            actual[r.invoice_id] += r.credit_minor - r.debit_minor
    actual_unapplied = sum(
        r.credit_minor - r.debit_minor for r in rows if rules.role(r) == UNAPPLIED
    )

    unapplied_diff = direction * (actual_unapplied - expected.unapplied)
    if unapplied_diff < 0:
        raise VerifyError(
            UNEXPLAINED_DISCREPANCY,
            f"payment {p.payment_id}: less booked to UNAPPLIED than the allocation rule leaves",
        )

    out = []
    in_payment_currency = [r.row_id for r in rows if r.currency == p.currency]
    for invoice_id in sorted(set(expected.by_invoice) | set(actual)):
        shortfall = direction * (expected.by_invoice.get(invoice_id, 0) - actual.get(invoice_id, 0))
        if shortfall <= 0:
            continue
        inv = book.invoice_document(invoice_id)
        instant = rules.instant(p.timestamp)
        earlier = [
            q.doc_id
            for q in book.payments
            if invoice_id in (q.invoice_refs or ()) and rules.instant(q.timestamp) < instant
        ]
        out.append(
            make_issue(
                rules,
                "payment_allocation",
                [p.doc_id, inv.doc_id],
                shortfall,
                p.currency,
                [[p.doc_id, inv.doc_id, *in_payment_currency, *earlier]],
            )
        )
    return out


# ---------------------------------------------------------------------------------- the rest
def check_background(book: Book, rules: Rules) -> None:
    """Rows linked to no document: EXPENSE/CASH entries in base currency, each balanced (I1)."""
    for _, rows in _group_by_instant(book.background, rules):
        for r in rows:
            if rules.role(r) not in (EXPENSE, CASH) or r.currency != rules.base_currency:
                raise VerifyError(
                    POSTING_INVALID,
                    f"row {r.row_id}: not linked to a document but not a base EXPENSE/CASH row",
                )
        if net_debit(rows) != 0:
            raise VerifyError(
                POSTING_INVALID, f"background entry at {rows[0].timestamp} unbalanced"
            )


def check_ledger_balance(ledger: Iterable[LedgerRow]) -> None:
    """I1 over the whole ledger, per currency."""
    total: dict[str, int] = defaultdict(int)
    for r in ledger:
        total[r.currency] += r.debit_minor - r.credit_minor
    if any(v != 0 for v in total.values()):
        raise VerifyError(POSTING_INVALID, "ledger debits differ from credits")
