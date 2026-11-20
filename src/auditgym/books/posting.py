"""Posting rules (docs/DOMAIN.md, section 4): clean ledger rows from the hidden world.

Rows carry internal keys. A row key is "<entry key>#<suffix>", where the entry key is the key of
the document (or expense entry) the row belongs to.

Allocation follows the story recorded in the world (`WPayment.plan`), not a recomputation of the
rule: the invariant check (auditgym.books.invariants) recomputes the rule from the public
documents, so a world whose story disagrees with the rule is caught there.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from auditgym.world.fx import conversion
from auditgym.world.model import WExpense, WInvoice, World, WPayment


@dataclass
class WRow:
    key: str
    entry: str
    ts: datetime
    role: str
    debit: int
    credit: int
    currency: str
    entity: str | None
    invoice: str | None  # invoice number key
    payment: str | None  # payment number key

    @property
    def signed(self) -> int:
        return self.debit - self.credit

    def set_signed(self, value: int) -> None:
        if value == 0:
            raise ValueError("ledger rows never carry a zero amount")
        self.debit = max(value, 0)
        self.credit = max(-value, 0)


def make_row(
    key: str,
    entry: str,
    ts: datetime,
    role: str,
    signed: int,
    currency: str,
    entity: str | None,
    invoice: str | None,
    payment: str | None,
) -> WRow:
    row = WRow(key, entry, ts, role, 0, 0, currency, entity, invoice, payment)
    row.set_signed(signed)
    return row


# ---------------------------------------------------------------------------------- entries
def post_invoice(inv: WInvoice) -> list[WRow]:
    """Dr AR / Cr REVENUE for the invoice total; sides swap for a credit note."""
    e, t = inv.key, inv.post_ts
    return [
        make_row(f"{e}#AR", e, t, "AR", inv.total, inv.currency, inv.customer, inv.number, None),
        make_row(
            f"{e}#REV", e, t, "REVENUE", -inv.total, inv.currency, inv.customer, inv.number, None
        ),
    ]


def post_payment(world: World, pay: WPayment) -> list[WRow]:
    e, t, c = pay.key, pay.post_ts, pay.currency
    who, num = pay.payer, pay.number
    rows: list[WRow] = []
    if c == world.base:
        rows.append(make_row(f"{e}#CASH", e, t, "CASH", pay.amount, c, who, None, num))
    else:
        b = conversion(world, pay)
        rows.append(make_row(f"{e}#FXF", e, t, "FX_CLEARING", pay.amount, c, who, None, num))
        rows.append(make_row(f"{e}#FXB", e, t, "FX_CLEARING", -b, world.base, who, None, num))
        rows.append(make_row(f"{e}#CASH", e, t, "CASH", b, world.base, who, None, num))
    applied = 0
    for i, (number, part) in enumerate(pay.plan):
        rows.append(make_row(f"{e}#AR{i}", e, t, "AR", -part, c, who, number, num))
        applied += part
    rest = pay.amount - applied
    if rest:
        rows.append(make_row(f"{e}#UNAP", e, t, "UNAPPLIED", -rest, c, who, None, num))
    return rows


def post_expense(world: World, x: WExpense) -> list[WRow]:
    e, t = x.key, x.ts
    return [
        make_row(f"{e}#EXP", e, t, "EXPENSE", x.amount, world.base, None, None, None),
        make_row(f"{e}#CASH", e, t, "CASH", -x.amount, world.base, None, None, None),
    ]


def post_all(world: World) -> list[WRow]:
    rows: list[WRow] = []
    for inv in world.invoices:
        rows.extend(post_invoice(inv))
    for pay in world.payments:
        rows.extend(post_payment(world, pay))
    for x in world.expenses:
        rows.extend(post_expense(world, x))
    return rows
