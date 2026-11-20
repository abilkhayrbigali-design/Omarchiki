"""Edits and evidence sets shared by the injectors (docs/DOMAIN.md, section 6)."""

from __future__ import annotations

import copy
from datetime import date, datetime, timedelta

from auditgym.books.posting import WRow, make_row, post_invoice
from auditgym.contracts.types import fx_rate_id
from auditgym.inject.state import InjectState
from auditgym.util.rng import Stream
from auditgym.world.candidates import earlier_payments
from auditgym.world.fx import FX_TOLERANCE_MINOR, rate_date
from auditgym.world.model import WInvoice, WPayment
from auditgym.world.timeline import HOUR

__all__ = ["FX_TOLERANCE_MINOR", "InjectState"]


# ---------------------------------------------------------------------------------- rows
def ar_row(state: InjectState, pay: WPayment, number: str) -> WRow:
    rows = [r for r in state.entry(pay.key) if r.role == "AR" and r.invoice == number]
    if len(rows) != 1:
        raise RuntimeError("expected exactly one AR row for the pair")
    return rows[0]


def role_row(state: InjectState, entry: str, role: str, currency: str) -> WRow:
    rows = [r for r in state.entry(entry) if r.role == role and r.currency == currency]
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one {role} row in {currency}")
    return rows[0]


def unapplied_row(state: InjectState, pay: WPayment) -> WRow | None:
    rows = [r for r in state.entry(pay.key) if r.role == "UNAPPLIED"]
    if len(rows) > 1:
        raise RuntimeError("payment has several UNAPPLIED rows")
    return rows[0] if rows else None


def _book_unapplied(state: InjectState, pay: WPayment, amount: int, ts: datetime) -> None:
    """Credit `amount` more to UNAPPLIED: on the payment's UNAPPLIED row, or on a new one."""
    existing = unapplied_row(state, pay)
    if existing is not None:
        state.touch_row(existing).set_signed(existing.signed - amount)
        return
    state.add_row(
        make_row(
            f"{pay.key}#UNAP",
            pay.key,
            ts,
            "UNAPPLIED",
            -amount,
            pay.currency,
            pay.payer,
            None,
            pay.number,
        )
    )


def apply_partial(state: InjectState, pay: WPayment, number: str, short: int) -> None:
    """Apply `short` less to the invoice and book it to UNAPPLIED instead."""
    row = state.touch_row(ar_row(state, pay, number))
    row.set_signed(row.signed + short)  # AR is a credit: a smaller credit
    _book_unapplied(state, pay, short, row.ts)


def apply_none(state: InjectState, pay: WPayment, number: str) -> None:
    """Book the whole part meant for the invoice to UNAPPLIED. Without an UNAPPLIED row the AR
    row itself is re-booked (same row id); otherwise the AR row goes and the UNAPPLIED row grows,
    so a payment never shows two UNAPPLIED rows."""
    row = ar_row(state, pay, number)
    if unapplied_row(state, pay) is None:
        state.touch_row(row)
        row.role = "UNAPPLIED"
        row.invoice = None
        return
    part = -row.signed
    state.remove_row(row)
    _book_unapplied(state, pay, part, row.ts)


def move_posting(state: InjectState, doc: WInvoice | WPayment, new_ts: datetime) -> datetime:
    old = doc.post_ts
    for row in state.entry(doc.key):
        state.touch_row(row).ts = new_ts
    doc.post_ts = new_ts
    return old


def rebook_conversion(state: InjectState, pay: WPayment, new_base: int) -> int:
    """Change the base-currency amount of a foreign payment (CASH and FX_CLEARING base rows)."""
    base = state.world.base
    cash = role_row(state, pay.key, "CASH", base)
    fxb = role_row(state, pay.key, "FX_CLEARING", base)
    old = cash.signed
    state.touch_row(cash).set_signed(new_base)
    state.touch_row(fxb).set_signed(-new_base)
    return old


def add_copy(
    state: InjectState, original: WInvoice, issued: datetime, new_number: bool
) -> WInvoice:
    """A copy of an invoice document (same order_ref and lines), entered and posted at `issued`.

    The copy keeps the original's payment term, so its due date looks like any other invoice's.
    """
    world = state.world
    copy_inv = WInvoice(
        key=world.mint("inv"),
        number=world.mint("invno") if new_number else original.number,
        customer=original.customer,
        issued=issued,
        due=issued + (original.due - original.issued),
        currency=original.currency,
        lines=copy.deepcopy(original.lines),
        display_tz=world.tl.tz,
        post_ts=issued,
        tag="injected",
    )
    state.add_invoice(copy_inv)
    for row in post_invoice(copy_inv):
        state.add_row(row)
    return copy_inv


def posting_after(
    state: InjectState, s: Stream, boundary: datetime, doc_time: datetime, day_offset: int
) -> datetime | None:
    """A business-hours posting time on the local date boundary + day_offset, after the document."""
    tl = state.world.tl
    day = tl.local_date(boundary) + timedelta(days=day_offset)
    t = tl.business_time(s, day)
    if t <= doc_time + HOUR:
        return None
    return t


# ---------------------------------------------------------------------------------- evidence
def pay_currency_rows(state: InjectState, pay: WPayment) -> list[str]:
    return [r.key for r in state.entry(pay.key) if r.currency == pay.currency]


def alloc_evidence(state: InjectState, pay: WPayment, inv: WInvoice) -> list[list[str]]:
    earlier = [q.key for q in earlier_payments(state.world, inv.number, pay)]
    return [[pay.key, inv.key, *pay_currency_rows(state, pay), *earlier]]


def cutoff_evidence(state: InjectState, doc_key: str) -> list[list[str]]:
    return [[doc_key, r.key] for r in state.entry(doc_key)]


def fx_evidence(state: InjectState, pay: WPayment) -> list[list[str]]:
    world = state.world
    rid = correct_rate_id(state, pay)
    cash = role_row(state, pay.key, "CASH", world.base)
    fxb = role_row(state, pay.key, "FX_CLEARING", world.base)
    return [[pay.key, cash.key, rid], [pay.key, fxb.key, rid]]


def correct_rate_id(state: InjectState, pay: WPayment) -> str:
    day: date = rate_date(state.world, pay.ts)
    return fx_rate_id(day.isoformat(), pay.currency, state.world.base)
