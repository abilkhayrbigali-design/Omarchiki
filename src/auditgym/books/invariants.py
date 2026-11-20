"""Invariants I1-I7 of the books (docs/DOMAIN.md, section 5), computed from public data only.

`check(public)` returns every violation it finds. On clean books it returns nothing. On books with
injected errors every violation corresponds to one reference issue (see
auditgym.episode.selfcheck for the exact correspondence). Nothing here reads the generator's
internal state: the expected books are recomputed from the documents, FX rows and policy.

Semantics chosen where DOMAIN leaves room (also listed in the generator report):
- "Period" of a timestamp is its position relative to [period_start, period_end): before, in, after.
- I3 is checked per invoice_id: exactly one recognition entry (rows with that invoice_id and no
  payment_id, one timestamp) whose AR amount equals total_minor of the invoice's document (the
  earliest document carrying that invoice_id). A repeated invoice_id therefore breaks I3 and I7;
  a copy under a new invoice_id with the same order_ref breaks I7 only.
- I5 reports only under-allocation per (payment, invoice) pair; over-allocation to another invoice
  is where the money went, not a separate violation (DOMAIN 6).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from auditgym.contracts.types import Invoice, LedgerRow, Payment, PublicEpisode, fx_rate_id
from auditgym.util.money import convert_minor
from auditgym.util.timeutil import local_date, parse_offset, parse_ts, period_class, utc_date


@dataclass(frozen=True, slots=True)
class Violation:
    code: str  # I1 ... I7
    ids: tuple[str, ...]  # public record IDs, sorted
    amount_minor: int | None = None
    currency: str | None = None
    detail: str = ""

    def signature(self) -> tuple[str, tuple[str, ...], int | None, str | None]:
        return (self.code, self.ids, self.amount_minor, self.currency)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "ids": list(self.ids),
            "amount_minor": self.amount_minor,
            "currency": self.currency,
            "detail": self.detail,
        }


def _v(
    code: str,
    ids: Any,
    amount: int | None = None,
    currency: str | None = None,
    detail: str = "",
) -> Violation:
    return Violation(code, tuple(sorted(set(ids))), amount, currency, detail)


def _sort_key(v: Violation) -> tuple[Any, ...]:
    return (
        v.code,
        v.ids,
        -1 if v.amount_minor is None else v.amount_minor,
        v.currency or "",
        v.detail,
    )


class _Books:
    """Indexes over the public episode."""

    def __init__(self, public: PublicEpisode) -> None:
        pol = public.policy
        self.public = public
        self.tz = parse_offset(pol["timezone"])
        self.start = parse_ts(pol["period"]["start"])
        self.end = parse_ts(pol["period"]["end"])
        self.base = pol["base_currency"]
        self.rate_rule = pol["fx"]["rate_date"]
        self.rounding = pol["fx"]["rounding"]
        self.role_of = {code: role for role, code in pol["accounts"].items()}
        self.rates = {f.rate_id: f.rate for f in public.fx}

        self.docs_by_number: dict[str, list[Invoice]] = defaultdict(list)
        for inv in public.invoices:
            self.docs_by_number[inv.invoice_id].append(inv)
        for docs in self.docs_by_number.values():
            docs.sort(key=lambda d: parse_ts(d.issued_at))

        self.recognition: dict[str, list[LedgerRow]] = defaultdict(list)
        self.by_payment: dict[str, list[LedgerRow]] = defaultdict(list)
        for row in public.ledger:
            if row.payment_id is not None:
                self.by_payment[row.payment_id].append(row)
            elif row.invoice_id is not None:
                self.recognition[row.invoice_id].append(row)

    def role(self, row: LedgerRow) -> str | None:
        return self.role_of.get(row.account)

    def invoice_doc(self, number: str) -> Invoice | None:
        docs = self.docs_by_number.get(number)
        return docs[0] if docs else None

    def period(self, ts: str) -> str:
        return period_class(ts, self.start, self.end)

    def rate_date(self, ts: str) -> date:
        if self.rate_rule == "payment_local_date":
            return local_date(ts, self.tz)
        if self.rate_rule == "payment_local_date_minus_1":
            return local_date(ts, self.tz) - timedelta(days=1)
        if self.rate_rule == "payment_utc_date":
            return utc_date(ts)
        raise ValueError(f"unknown rate date rule {self.rate_rule!r}")


# ---------------------------------------------------------------------------------- I1
def _i1(b: _Books) -> list[Violation]:
    out = []
    groups: dict[tuple[str, Any], list[LedgerRow]] = defaultdict(list)
    for row in b.public.ledger:
        if (
            row.debit_minor < 0
            or row.credit_minor < 0
            or (row.debit_minor > 0) == (row.credit_minor > 0)
        ):
            out.append(_v("I1", [row.row_id], detail="row must have exactly one positive side"))
        if row.payment_id is not None:
            key: tuple[str, Any] = ("payment", row.payment_id)
        elif row.invoice_id is not None:
            key = ("invoice", row.invoice_id)
        else:
            key = ("time", parse_ts(row.timestamp))
        groups[key].append(row)
    for rows in groups.values():
        totals: dict[str, int] = defaultdict(int)
        for row in rows:
            totals[row.currency] += row.debit_minor - row.credit_minor
        for cur, total in sorted(totals.items()):
            if total != 0:
                out.append(
                    _v("I1", [r.row_id for r in rows], abs(total), cur, "entry does not balance")
                )
    return out


# ---------------------------------------------------------------------------------- I2
def _i2(b: _Books) -> list[Violation]:
    out = []
    for inv in b.public.invoices:
        lines_ok = all(
            li.amount_minor == li.quantity * li.unit_price_minor for li in inv.line_items
        )
        total = sum(li.amount_minor for li in inv.line_items)
        if not lines_ok or total != inv.total_minor:
            out.append(_v("I2", [inv.doc_id], abs(total - inv.total_minor), inv.currency))
    return out


# ---------------------------------------------------------------------------------- I3
def _i3(b: _Books) -> list[Violation]:
    out = []
    for number, docs in b.docs_by_number.items():
        doc = docs[0]
        rows = b.recognition.get(number, [])
        ar = sum(r.debit_minor - r.credit_minor for r in rows if b.role(r) == "AR")
        rev = sum(r.debit_minor - r.credit_minor for r in rows if b.role(r) == "REVENUE")
        foreign_rows = [
            r for r in rows if r.currency != doc.currency or b.role(r) not in ("AR", "REVENUE")
        ]
        entries = {parse_ts(r.timestamp) for r in rows}
        if len(entries) != 1 or ar != doc.total_minor or rev != -doc.total_minor or foreign_rows:
            out.append(
                _v(
                    "I3",
                    [d.doc_id for d in docs],
                    abs(ar - doc.total_minor),
                    doc.currency,
                    f"{len(entries)} recognition entries",
                )
            )
    for number, rows in b.recognition.items():
        if number not in b.docs_by_number:
            out.append(_v("I3", [r.row_id for r in rows], detail="rows for an unknown invoice_id"))
    return out


# ---------------------------------------------------------------------------------- I4
def _i4(b: _Books) -> list[Violation]:
    out = []
    for number, docs in b.docs_by_number.items():
        rows = b.recognition.get(number, [])
        if not rows:
            continue  # missing entry is an I3 violation
        classes = {b.period(d.issued_at) for d in docs} | {b.period(r.timestamp) for r in rows}
        if len(classes) > 1:
            doc = docs[0]
            out.append(_v("I4", [d.doc_id for d in docs], abs(doc.total_minor), doc.currency))
    for pay in b.public.payments:
        rows = b.by_payment.get(pay.payment_id, [])
        if any(b.period(r.timestamp) != b.period(pay.timestamp) for r in rows):
            out.append(_v("I4", [pay.doc_id], abs(pay.amount_minor), pay.currency))
    return out


# ---------------------------------------------------------------------------------- I5
def expected_allocation(
    b: _Books,
) -> dict[str, tuple[dict[str, int], int]]:
    """payment_id -> ({invoice_id: expected applied amount}, expected unapplied), by the rule
    listed_order_up_to_open_balance over documents only."""
    open_balance = {number: docs[0].total_minor for number, docs in b.docs_by_number.items()}
    out: dict[str, tuple[dict[str, int], int]] = {}
    payments: list[Payment] = sorted(b.public.payments, key=lambda p: parse_ts(p.timestamp))
    for pay in payments:
        sign = 1 if pay.amount_minor > 0 else -1
        remaining = pay.amount_minor
        applied: dict[str, int] = {}
        for number in pay.invoice_refs or ():
            doc = b.invoice_doc(number)
            ob = open_balance.get(number, 0)
            if doc is None or doc.currency != pay.currency or remaining == 0 or ob == 0:
                continue
            if (ob > 0) != (remaining > 0):
                continue
            part = sign * min(abs(remaining), abs(ob))
            applied[number] = applied.get(number, 0) + part
            open_balance[number] = ob - part
            remaining -= part
        out[pay.payment_id] = (applied, remaining)
    return out


def _i5(b: _Books) -> list[Violation]:
    out = []
    expected = expected_allocation(b)
    for pay in b.public.payments:
        applied, unapplied = expected[pay.payment_id]
        sign = 1 if pay.amount_minor > 0 else -1
        actual: dict[str | None, int] = defaultdict(int)
        actual_unapplied = 0
        for row in b.by_payment.get(pay.payment_id, []):
            role = b.role(row)
            if role == "AR":
                actual[row.invoice_id] += row.credit_minor - row.debit_minor
            elif role == "UNAPPLIED":
                actual_unapplied += row.credit_minor - row.debit_minor
        for number, part in applied.items():
            under = (part - actual.get(number, 0)) * sign
            if under > 0:
                doc = b.invoice_doc(number)
                assert doc is not None
                out.append(_v("I5", [pay.doc_id, doc.doc_id], under, pay.currency))
        under_unapplied = (unapplied - actual_unapplied) * sign
        if under_unapplied > 0:
            out.append(
                _v("I5", [pay.doc_id], under_unapplied, pay.currency, "unapplied remainder short")
            )
    return out


# ---------------------------------------------------------------------------------- I6
def _i6(b: _Books) -> list[Violation]:
    out = []
    for pay in b.public.payments:
        rows = b.by_payment.get(pay.payment_id, [])
        cash = sum(r.debit_minor - r.credit_minor for r in rows if b.role(r) == "CASH")
        if pay.currency == b.base:
            expected = pay.amount_minor
        else:
            day = b.rate_date(pay.timestamp)
            rid = fx_rate_id(day.isoformat(), pay.currency, b.base)
            if rid not in b.rates:
                out.append(_v("I6", [pay.doc_id], detail="no FX rate for the policy date"))
                continue
            expected = convert_minor(
                pay.amount_minor, pay.currency, b.base, b.rates[rid], b.rounding
            )
        if cash != expected:
            out.append(_v("I6", [pay.doc_id], abs(cash - expected), b.base))
    return out


# ---------------------------------------------------------------------------------- I7
def _i7(b: _Books) -> list[Violation]:
    invoices = b.public.invoices
    parent = {inv.doc_id: inv.doc_id for inv in invoices}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    first_by_key: dict[tuple[str, str], str] = {}
    for inv in invoices:
        keys = [("invoice_id", inv.invoice_id)] + [
            ("order_ref", li.order_ref) for li in inv.line_items
        ]
        for key in keys:
            if key in first_by_key:
                union(first_by_key[key], inv.doc_id)
            else:
                first_by_key[key] = inv.doc_id
    groups: dict[str, list[Invoice]] = defaultdict(list)
    for inv in invoices:
        groups[find(inv.doc_id)].append(inv)
    out = []
    for docs in groups.values():
        if len(docs) > 1:
            docs = sorted(docs, key=lambda d: parse_ts(d.issued_at))
            extra = sum(abs(d.total_minor) for d in docs[1:])
            out.append(_v("I7", [d.doc_id for d in docs], extra, docs[0].currency))
    return out


def check(public: PublicEpisode) -> list[Violation]:
    """All violations of I1-I7 in the public episode, in canonical order."""
    b = _Books(public)
    out = _i1(b) + _i2(b) + _i3(b) + _i4(b) + _i5(b) + _i6(b) + _i7(b)
    return sorted(out, key=_sort_key)


def id_problems(public: PublicEpisode) -> list[str]:
    """Structural problems outside I1-I7: duplicate IDs, dangling references, bad timestamps."""
    problems = []
    doc_ids = [d.doc_id for d in public.invoices] + [p.doc_id for p in public.payments]
    row_ids = [r.row_id for r in public.ledger]
    pay_ids = [p.payment_id for p in public.payments]
    all_ids = doc_ids + row_ids + pay_ids + [f.rate_id for f in public.fx]
    if len(set(all_ids)) != len(all_ids):
        problems.append("duplicate record id")
    numbers = {inv.invoice_id for inv in public.invoices}
    if numbers & set(doc_ids + row_ids + pay_ids):
        problems.append("invoice_id collides with another id")
    for pay in public.payments:
        for ref in pay.invoice_refs or ():
            if ref not in numbers:
                problems.append(f"payment references unknown invoice_id {ref}")
    for row in public.ledger:
        if row.payment_id is not None and row.payment_id not in set(pay_ids):
            problems.append("ledger row references unknown payment_id")
            break
    stamps: list[str] = [r.timestamp for r in public.ledger]
    stamps += [d.issued_at for d in public.invoices] + [d.due_at for d in public.invoices]
    stamps += [p.timestamp for p in public.payments]
    for ts in stamps:
        try:
            dt: datetime = parse_ts(ts)
        except ValueError:
            problems.append(f"bad timestamp {ts}")
            continue
        if dt.utcoffset() is None:
            problems.append(f"naive timestamp {ts}")
    return problems
