"""Builders for manifest.json and policy.json, so every producer emits the same shape.

The policy is the agent's rulebook: everything needed to recompute expected books from the
documents (docs/DOMAIN.md, section 4). It is shown in the first observation.
"""

from __future__ import annotations

from typing import Any

from auditgym.contracts.constants import ALLOCATION_RULE, FX_RATE_DATE_RULES, ROUNDING_RULES
from auditgym.contracts.versions import GENERATOR_VERSION, SCHEMA_VERSION
from auditgym.util.money import exponent

ISSUE_TYPE_DESCRIPTIONS: dict[str, str] = {
    "duplicate_invoice": (
        "The same sale is invoiced and booked more than once: two or more invoice documents share "
        "an invoice_id, or share an order_ref. affected_ids = doc_id of every copy. "
        "Amount = (copies - 1) * |total_minor|, invoice currency."
    ),
    "payment_allocation": (
        "A payment is not applied to an invoice it references as the allocation rule requires: "
        "missing, partial, or booked against another invoice. One finding per (payment, invoice) "
        "pair that received less than expected. affected_ids = [payment doc_id, invoice doc_id]. "
        "Amount = expected minus actual amount applied, payment currency."
    ),
    "fx_conversion": (
        "A foreign-currency payment is converted with the wrong rate (wrong date or wrong pair). "
        "affected_ids = [payment doc_id]. Amount = |booked CASH - expected CASH|, base currency."
    ),
    "period_cutoff": (
        "A document's ledger entry is posted in a different reporting period than the document "
        "date (local date in the company timezone). affected_ids = [document doc_id]. "
        "Amount = |document amount|, document currency."
    ),
}

EVIDENCE_RULES: dict[str, str] = {
    "duplicate_invoice": "doc_id of every copy in the duplicate group.",
    "payment_allocation": (
        "doc_id of the payment, doc_id of the invoice, row_id of every ledger row of the payment's "
        "entry in the payment currency (CASH or FX_CLEARING, AR, UNAPPLIED), and doc_id of every "
        "earlier payment that references the same invoice."
    ),
    "fx_conversion": (
        "doc_id of the payment, row_id of the payment's CASH row (or of its FX_CLEARING row in the "
        "base currency), and the id of the correct fx row: 'YYYY-MM-DD/<payment currency>/<base>'."
    ),
    "period_cutoff": "doc_id of the document and the row_id of any one row of its ledger entry.",
}

ACCOUNT_ROLE_DESCRIPTIONS: dict[str, str] = {
    "CASH": "bank account, base currency",
    "AR": "accounts receivable, invoice currency",
    "REVENUE": "sales revenue, invoice currency",
    "UNAPPLIED": "received cash not applied to any invoice, payment currency",
    "FX_CLEARING": "clearing account for currency conversion of payments",
    "EXPENSE": "operating expenses, base currency, not linked to documents",
}


def make_policy(
    *,
    base_currency: str,
    timezone: str,
    period_start: str,
    period_end: str,
    accounts: dict[str, str],
    fx_rate_date: str,
    rounding: str,
    fx_tolerance_minor: int = 1,
) -> dict[str, Any]:
    if fx_rate_date not in FX_RATE_DATE_RULES:
        raise ValueError(f"unknown fx rate date rule {fx_rate_date!r}")
    if rounding not in ROUNDING_RULES:
        raise ValueError(f"unknown rounding rule {rounding!r}")
    return {
        "schema_version": SCHEMA_VERSION,
        "base_currency": base_currency,
        "timezone": timezone,
        "period": {
            "start": period_start,
            "end": period_end,
            "boundaries": "start_inclusive_end_exclusive",
            "document_date": "local date of the document timestamp in the company timezone",
            "posting_date": "local date of the ledger entry timestamp in the company timezone",
            "classes": (
                "each timestamp is before, in or after the reporting period; a cut-off error is a "
                "document and its ledger entry in different classes"
            ),
        },
        "accounts": dict(sorted(accounts.items())),
        "account_roles": dict(sorted(ACCOUNT_ROLE_DESCRIPTIONS.items())),
        "fx": {
            "rate_meaning": "1 unit of source_currency = rate units of target_currency",
            "rate_date": fx_rate_date,
            "rate_date_options": {
                "payment_local_date": "local date of the payment timestamp in the company timezone",
                "payment_local_date_minus_1": "the day before the local date of the payment",
                "payment_utc_date": "UTC date of the payment timestamp",
            },
            "conversion": "base_minor = round(amount_minor * 10^(exp_base - exp_payment) * rate)",
            "rounding": rounding,
            "booking": (
                "foreign payment: Dr FX_CLEARING and Cr AR/UNAPPLIED in payment currency; "
                "Dr CASH and Cr FX_CLEARING in base currency for the converted amount"
            ),
        },
        "allocation": {
            "rule": ALLOCATION_RULE,
            "description": (
                "Payments are processed in timestamp order. A payment is applied to the invoices "
                "in invoice_refs in the listed order, to each at most its open balance computed "
                "from documents. The remainder is booked to UNAPPLIED. invoice_refs = null means "
                "the whole amount goes to UNAPPLIED. Negative payments (refunds) apply the same "
                "rule to negative open balances (credit notes)."
            ),
            "open_balance": (
                "total_minor of the invoice minus amounts applied to it by earlier payments, both "
                "taken from documents. If several documents share an invoice_id (a duplicate), the "
                "invoice is counted once, from the document with the earliest issued_at."
            ),
            "currency": "a payment references only invoices in its own currency",
            "applied_in_books": (
                "credits to AR (debits for refunds) with the payment_id and the invoice_id; an "
                "amount booked to another invoice is where the shortfall went, not a second issue"
            ),
        },
        "duplicates": {
            "rule": "one_invoice_per_order_ref",
            "description": "Each order_ref is invoiced once; each invoice_id belongs to one document.",
            "grouping": (
                "documents linked by a shared invoice_id or a shared order_ref form one group, "
                "transitively (A~B, B~C -> one group of 3 copies)"
            ),
        },
        "rounding": {
            "money": "integer_minor_units",
            "fx": rounding,
            "note": "half_up rounds halves away from zero, also for negative amounts",
        },
        "tolerances": {"default_minor": 0, "fx_conversion_minor": fx_tolerance_minor},
        "issue_types": dict(sorted(ISSUE_TYPE_DESCRIPTIONS.items())),
        "evidence": {
            "rule": (
                "a finding counts only if evidence_ids contain at least one complete evidence set "
                "for its issue; extra ids are allowed up to 10"
            ),
            "sets": dict(sorted(EVIDENCE_RULES.items())),
        },
    }


def make_manifest(
    *,
    episode_id: str,
    company: str,
    base_currency: str,
    currencies: list[str],
    period_start: str,
    period_end: str,
    timezone: str,
    rounding: str,
    counts: dict[str, int],
) -> dict[str, Any]:
    required = {"ledger", "invoices", "payments", "fx"}
    if set(counts) != required:
        raise ValueError(f"counts must have exactly {sorted(required)}")
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "episode_id": episode_id,
        "company": company,
        "base_currency": base_currency,
        "currencies": {c: {"exponent": exponent(c)} for c in sorted(set(currencies))},
        "period_start": period_start,
        "period_end": period_end,
        "timezone": timezone,
        "counts": {
            **dict(sorted(counts.items())),
            "documents": counts["invoices"] + counts["payments"],
        },
        "rounding": {"money": "integer_minor_units", "fx": rounding},
    }
