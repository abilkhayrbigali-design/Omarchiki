"""Episode rules read from manifest.json and policy.json, plus record-level validation.

Everything that varies between episodes comes from here: base currency, company timezone,
reporting period, account codes, FX rate-date rule, rounding mode, tolerances
(docs/DOMAIN.md sections 1, 3, 4, 6). Both the solver and the evidence checker use it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta, timezone
from typing import Any

from auditgym.contracts.constants import (
    ACCOUNT_ROLES,
    ALLOCATION_RULE,
    FX_RATE_DATE_RULES,
    ROUNDING_RULES,
)
from auditgym.contracts.types import FxRate, Invoice, LedgerRow, Payment, fx_rate_id
from auditgym.util.money import CURRENCY_EXPONENTS, MoneyError, convert_minor, parse_rate
from auditgym.util.timeutil import TimeError, parse_date, parse_offset, parse_ts
from auditgym.verify.errors import POLICY_INVALID, RECORD_INVALID, VerifyError

BEFORE, IN, AFTER = "before", "in", "after"

CASH, AR, REVENUE, UNAPPLIED, FX_CLEARING, EXPENSE = (
    "CASH",
    "AR",
    "REVENUE",
    "UNAPPLIED",
    "FX_CLEARING",
    "EXPENSE",
)

DUPLICATES_RULE = "one_invoice_per_order_ref"
PERIOD_BOUNDARIES = "start_inclusive_end_exclusive"


@dataclass(frozen=True, slots=True)
class Rules:
    base_currency: str
    tz: timezone
    period_start: date  # local date in the company timezone, inclusive
    period_end: date  # local date in the company timezone, exclusive
    role_by_code: dict[str, str]
    rate_date_rule: str
    rounding: str
    fx_tolerance_minor: int
    default_tolerance_minor: int
    currencies: frozenset[str]

    # ------------------------------------------------------------------ time
    def instant(self, ts: str) -> datetime:
        try:
            return parse_ts(ts)
        except TimeError as exc:
            raise VerifyError(RECORD_INVALID, str(exc)) from exc

    def local_date(self, ts: str) -> date:
        return self.instant(ts).astimezone(self.tz).date()

    def period_of(self, ts: str) -> str:
        """before / in / after the reporting period, by local date in the company timezone."""
        day = self.local_date(ts)
        if day < self.period_start:
            return BEFORE
        if day < self.period_end:
            return IN
        return AFTER

    # ------------------------------------------------------------------ fx
    def rate_date(self, payment_ts: str) -> date:
        """Date of the FX rate for a payment, from the document timestamp (DOMAIN 4)."""
        instant = self.instant(payment_ts)
        if self.rate_date_rule == "payment_local_date":
            return instant.astimezone(self.tz).date()
        if self.rate_date_rule == "payment_local_date_minus_1":
            return instant.astimezone(self.tz).date() - timedelta(days=1)
        if self.rate_date_rule == "payment_utc_date":
            return instant.astimezone(UTC).date()
        raise VerifyError(POLICY_INVALID, f"unknown rate date rule {self.rate_date_rule!r}")

    def rate_id_for(self, payment: Payment) -> str:
        day = self.rate_date(payment.timestamp)
        return fx_rate_id(day.isoformat(), payment.currency, self.base_currency)

    def to_base(self, amount_minor: int, currency: str, rate: str) -> int:
        try:
            return convert_minor(amount_minor, currency, self.base_currency, rate, self.rounding)
        except MoneyError as exc:
            raise VerifyError(RECORD_INVALID, f"conversion failed: {exc}") from exc

    # ------------------------------------------------------------------ accounts
    def role(self, row: LedgerRow) -> str:
        try:
            return self.role_by_code[row.account]
        except KeyError as exc:
            raise VerifyError(
                RECORD_INVALID, f"row {row.row_id}: unknown account code {row.account!r}"
            ) from exc

    def tolerance(self, issue_type: str) -> int:
        if issue_type == "fx_conversion":
            return self.fx_tolerance_minor
        return self.default_tolerance_minor


# ---------------------------------------------------------------------------------- loading
def _need(container: Any, key: str, where: str) -> Any:
    if not isinstance(container, dict) or key not in container:
        raise VerifyError(POLICY_INVALID, f"{where}: missing {key!r}")
    return container[key]


def _nonneg_int(value: Any, where: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise VerifyError(POLICY_INVALID, f"{where}: expected a non-negative integer")
    return value


def _midnight_date(ts: str, tz: timezone, where: str) -> tuple[datetime, date]:
    try:
        instant = parse_ts(ts)
    except TimeError as exc:
        raise VerifyError(POLICY_INVALID, f"{where}: {exc}") from exc
    local = instant.astimezone(tz)
    if local.time() != time(0, 0):
        raise VerifyError(
            POLICY_INVALID,
            f"{where}: period boundary is not local midnight in the company timezone",
        )
    return instant, local.date()


def load_rules(manifest: dict[str, Any], policy: dict[str, Any]) -> Rules:
    """Read and cross-check manifest and policy. Raises VerifyError(policy_invalid)."""
    if not isinstance(manifest, dict) or not isinstance(policy, dict):
        raise VerifyError(POLICY_INVALID, "manifest and policy must be JSON objects")

    base = _need(policy, "base_currency", "policy")
    if base != _need(manifest, "base_currency", "manifest"):
        raise VerifyError(POLICY_INVALID, "base currency differs between manifest and policy")

    currencies = _need(manifest, "currencies", "manifest")
    if not isinstance(currencies, dict) or not currencies:
        raise VerifyError(POLICY_INVALID, "manifest.currencies: expected a non-empty object")
    for code, info in sorted(currencies.items()):
        if code not in CURRENCY_EXPONENTS:
            raise VerifyError(POLICY_INVALID, f"manifest.currencies: unknown currency {code!r}")
        if _need(info, "exponent", f"manifest.currencies.{code}") != CURRENCY_EXPONENTS[code]:
            raise VerifyError(POLICY_INVALID, f"manifest.currencies.{code}: wrong exponent")
    if base not in currencies:
        raise VerifyError(POLICY_INVALID, "base currency is not listed in manifest.currencies")

    tz_text = _need(policy, "timezone", "policy")
    if tz_text != _need(manifest, "timezone", "manifest"):
        raise VerifyError(POLICY_INVALID, "timezone differs between manifest and policy")
    try:
        tz = parse_offset(tz_text)
    except TimeError as exc:
        raise VerifyError(POLICY_INVALID, f"policy.timezone: {exc}") from exc

    period = _need(policy, "period", "policy")
    boundaries = _need(period, "boundaries", "policy.period")
    if boundaries != PERIOD_BOUNDARIES:
        raise VerifyError(POLICY_INVALID, f"policy.period.boundaries: {boundaries!r}")
    start_i, start_d = _midnight_date(_need(period, "start", "policy.period"), tz, "period.start")
    end_i, end_d = _midnight_date(_need(period, "end", "policy.period"), tz, "period.end")
    m_start_i, _ = _midnight_date(_need(manifest, "period_start", "manifest"), tz, "period_start")
    m_end_i, _ = _midnight_date(_need(manifest, "period_end", "manifest"), tz, "period_end")
    if (start_i, end_i) != (m_start_i, m_end_i):
        raise VerifyError(POLICY_INVALID, "reporting period differs between manifest and policy")
    if not start_d < end_d:
        raise VerifyError(POLICY_INVALID, "reporting period is empty")

    accounts = _need(policy, "accounts", "policy")
    if not isinstance(accounts, dict) or set(accounts) != set(ACCOUNT_ROLES):
        raise VerifyError(POLICY_INVALID, f"policy.accounts: need exactly {sorted(ACCOUNT_ROLES)}")
    role_by_code: dict[str, str] = {}
    for role, code in sorted(accounts.items()):
        if not isinstance(code, str) or not code:
            raise VerifyError(POLICY_INVALID, f"policy.accounts.{role}: expected a non-empty code")
        if code in role_by_code:
            raise VerifyError(POLICY_INVALID, f"policy.accounts: code {code!r} used twice")
        role_by_code[code] = role

    fx = _need(policy, "fx", "policy")
    rate_date_rule = _need(fx, "rate_date", "policy.fx")
    if rate_date_rule not in FX_RATE_DATE_RULES:
        raise VerifyError(POLICY_INVALID, f"policy.fx.rate_date: {rate_date_rule!r}")
    rounding = _need(fx, "rounding", "policy.fx")
    if rounding not in ROUNDING_RULES:
        raise VerifyError(POLICY_INVALID, f"policy.fx.rounding: {rounding!r}")
    for where, section in (
        ("policy.rounding", policy.get("rounding")),
        ("manifest.rounding", manifest.get("rounding")),
    ):
        if not isinstance(section, dict) or section.get("fx") != rounding:
            raise VerifyError(POLICY_INVALID, f"{where}.fx differs from policy.fx.rounding")

    allocation = _need(policy, "allocation", "policy")
    if _need(allocation, "rule", "policy.allocation") != ALLOCATION_RULE:
        raise VerifyError(POLICY_INVALID, "policy.allocation.rule: unsupported rule")
    duplicates = _need(policy, "duplicates", "policy")
    if _need(duplicates, "rule", "policy.duplicates") != DUPLICATES_RULE:
        raise VerifyError(POLICY_INVALID, "policy.duplicates.rule: unsupported rule")

    tolerances = _need(policy, "tolerances", "policy")
    fx_tol = _nonneg_int(_need(tolerances, "fx_conversion_minor", "policy.tolerances"), "fx tol")
    default_tol = _nonneg_int(_need(tolerances, "default_minor", "policy.tolerances"), "tol")

    return Rules(
        base_currency=base,
        tz=tz,
        period_start=start_d,
        period_end=end_d,
        role_by_code=role_by_code,
        rate_date_rule=rate_date_rule,
        rounding=rounding,
        fx_tolerance_minor=fx_tol,
        default_tolerance_minor=default_tol,
        currencies=frozenset(currencies),
    )


# ---------------------------------------------------------------------------------- records
def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _id(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise VerifyError(RECORD_INVALID, f"{where}: expected a non-empty string id")
    return value


def check_invoice(inv: Invoice, rules: Rules) -> None:
    """Document format and I2: total = sum of lines, line amount = quantity * unit price."""
    where = f"invoice {inv.doc_id}"
    _id(inv.doc_id, "invoice doc_id")
    _id(inv.invoice_id, f"{where} invoice_id")
    rules.instant(inv.issued_at)
    if inv.currency not in rules.currencies:
        raise VerifyError(RECORD_INVALID, f"{where}: currency {inv.currency!r} not in manifest")
    if not _is_int(inv.total_minor) or inv.total_minor == 0:
        raise VerifyError(RECORD_INVALID, f"{where}: total_minor must be a non-zero integer")
    if not inv.line_items:
        raise VerifyError(RECORD_INVALID, f"{where}: no line items")
    total = 0
    for item in inv.line_items:
        if not all(_is_int(v) for v in (item.quantity, item.unit_price_minor, item.amount_minor)):
            raise VerifyError(RECORD_INVALID, f"{where}: line amounts must be integers")
        if item.amount_minor != item.quantity * item.unit_price_minor:
            raise VerifyError(RECORD_INVALID, f"{where}: line amount != quantity * unit price (I2)")
        _id(item.order_ref, f"{where} order_ref")
        total += item.amount_minor
    if total != inv.total_minor:
        raise VerifyError(RECORD_INVALID, f"{where}: total_minor != sum of lines (I2)")


def check_payment(pay: Payment, rules: Rules) -> None:
    where = f"payment {pay.doc_id}"
    _id(pay.doc_id, "payment doc_id")
    _id(pay.payment_id, f"{where} payment_id")
    rules.instant(pay.timestamp)
    if pay.currency not in rules.currencies:
        raise VerifyError(RECORD_INVALID, f"{where}: currency {pay.currency!r} not in manifest")
    if not _is_int(pay.amount_minor) or pay.amount_minor == 0:
        raise VerifyError(RECORD_INVALID, f"{where}: amount_minor must be a non-zero integer")
    if pay.invoice_refs is not None:
        if not isinstance(pay.invoice_refs, tuple) or not pay.invoice_refs:
            raise VerifyError(RECORD_INVALID, f"{where}: invoice_refs must be null or non-empty")
        for ref in pay.invoice_refs:
            _id(ref, f"{where} invoice_refs")


def check_row(row: LedgerRow, rules: Rules) -> None:
    """Row format and the row part of I1: exactly one side non-zero, both non-negative."""
    _id(row.row_id, "ledger row_id")
    rules.instant(row.timestamp)
    rules.role(row)
    if row.currency not in rules.currencies:
        raise VerifyError(RECORD_INVALID, f"row {row.row_id}: currency not in manifest")
    if not (_is_int(row.debit_minor) and _is_int(row.credit_minor)):
        raise VerifyError(RECORD_INVALID, f"row {row.row_id}: amounts must be integers")
    if row.debit_minor < 0 or row.credit_minor < 0:
        raise VerifyError(RECORD_INVALID, f"row {row.row_id}: negative amount (I1)")
    if (row.debit_minor > 0) == (row.credit_minor > 0):
        raise VerifyError(
            RECORD_INVALID, f"row {row.row_id}: exactly one side must be non-zero (I1)"
        )


def check_rate(rate: FxRate) -> None:
    try:
        parse_date(rate.date)
        parse_rate(rate.rate)
    except (TimeError, MoneyError) as exc:
        raise VerifyError(RECORD_INVALID, f"fx {rate.rate_id}: {exc}") from exc
    for code in (rate.source_currency, rate.target_currency):
        if code not in CURRENCY_EXPONENTS:
            raise VerifyError(RECORD_INVALID, f"fx {rate.rate_id}: unknown currency {code!r}")
    if rate.source_currency == rate.target_currency:
        raise VerifyError(RECORD_INVALID, f"fx {rate.rate_id}: source equals target")


def net_debit(rows: list[LedgerRow] | tuple[LedgerRow, ...]) -> int:
    return sum(r.debit_minor - r.credit_minor for r in rows)


def sign(value: int) -> int:
    return (value > 0) - (value < 0)
