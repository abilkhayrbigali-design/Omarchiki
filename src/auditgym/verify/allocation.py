"""Expected allocation of payments to invoices: rule `listed_order_up_to_open_balance`.

DOMAIN section 4: payments are processed in time order; each payment goes to the invoices of
`invoice_refs` in the listed order, to each at most its open balance; the remainder goes to
UNAPPLIED. Open balances come from documents (the earliest-issued document of an invoice_id),
never from the ledger. Refunds (negative payments) apply the same rule to negative balances.

The engine also runs on a partial set of records (evidence check). There an invoice whose
document is missing has an unknown balance, and every amount that depends on it is unknown
(`None`) instead of guessed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from auditgym.contracts.types import Payment
from auditgym.verify.errors import ORDER_AMBIGUOUS, UNKNOWN_REFERENCE, VerifyError
from auditgym.verify.rules import Rules


@dataclass(frozen=True, slots=True)
class PaymentAllocation:
    """Expected split of one payment. Signed amounts in the payment currency; None = unknown."""

    by_invoice: dict[str, int | None]  # invoice_id -> amount applied (summed if listed twice)
    unapplied: int | None


def apply_amount(remaining: int, open_balance: int) -> int:
    """Part of `remaining` applied to an invoice with `open_balance`; signs must agree."""
    if remaining > 0 and open_balance > 0:
        return min(remaining, open_balance)
    if remaining < 0 and open_balance < 0:
        return max(remaining, open_balance)
    return 0


def _check_ties(payments: list[Payment], rules: Rules) -> None:
    """Payments at the same instant must not share an invoice (DOMAIN 4, Уточнения)."""
    by_instant: dict[object, list[Payment]] = {}
    for p in payments:
        by_instant.setdefault(rules.instant(p.timestamp), []).append(p)
    for group in by_instant.values():
        if len(group) < 2:
            continue
        seen: dict[str, str] = {}
        for p in group:
            for ref in sorted(set(p.invoice_refs or ())):
                if ref in seen:
                    raise VerifyError(
                        ORDER_AMBIGUOUS,
                        f"payments {seen[ref]} and {p.doc_id} reference {ref} at the same instant",
                    )
                seen[ref] = p.doc_id


def expected_allocations(
    payments: Iterable[Payment],
    opening: Mapping[str, int],
    rules: Rules,
    *,
    partial: bool = False,
) -> dict[str, PaymentAllocation]:
    """payment_id -> expected allocation.

    `opening` maps invoice_id -> opening balance (signed total of its invoice document).
    With `partial=False` a reference to an invoice_id missing from `opening` is an error;
    with `partial=True` that invoice's balance is unknown.
    """
    plist = list(payments)
    _check_ties(plist, rules)
    # Sorting by instant only; equal instants never share an invoice (checked above), so their
    # relative order cannot change any balance.
    ordered = sorted(plist, key=lambda p: rules.instant(p.timestamp))

    balance: dict[str, int | None] = dict(opening)
    out: dict[str, PaymentAllocation] = {}
    for p in ordered:
        remaining: int | None = p.amount_minor
        by_invoice: dict[str, int | None] = {}
        for ref in p.invoice_refs or ():
            if ref not in balance:
                if not partial:
                    raise VerifyError(
                        UNKNOWN_REFERENCE, f"payment {p.doc_id} references unknown invoice {ref}"
                    )
                balance[ref] = None
            open_balance = balance[ref]
            if remaining == 0 or open_balance == 0:
                applied: int | None = 0
            elif remaining is None or open_balance is None:
                applied = None
            else:
                applied = apply_amount(remaining, open_balance)
            prior = by_invoice.get(ref, 0)
            by_invoice[ref] = None if applied is None or prior is None else prior + applied
            if applied is None:
                remaining = None
                balance[ref] = None
            else:
                if remaining is not None:
                    remaining -= applied
                if open_balance is not None:
                    balance[ref] = open_balance - applied
        out[p.payment_id] = PaymentAllocation(by_invoice=by_invoice, unapplied=remaining)
    return out
