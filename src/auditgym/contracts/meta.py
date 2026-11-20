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
        },
        "duplicates": {
            "rule": "one_invoice_per_order_ref",
            "description": "Each order_ref is invoiced once; each invoice_id belongs to one document.",
        },
        "rounding": {"money": "integer_minor_units", "fx": rounding},
        "tolerances": {"default_minor": 0, "fx_conversion_minor": fx_tolerance_minor},
        "issue_types": dict(sorted(ISSUE_TYPE_DESCRIPTIONS.items())),
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
