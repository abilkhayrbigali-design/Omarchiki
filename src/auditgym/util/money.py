"""Money in integer minor units, exact FX conversion with Decimal.

Rules (see docs/DOMAIN.md):
- amounts are `int` in the currency's minor units;
- a currency's exponent is the number of minor-unit digits (KZT 2, JPY 0, KWD 3);
- FX rates are decimal strings: 1 unit of the source currency = `rate` units of the target;
- conversion never uses float.
"""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Context, Decimal, InvalidOperation

# Real ISO 4217 codes and exponents. All companies, people and amounts in episodes are fictional.
CURRENCY_EXPONENTS: dict[str, int] = {
    "AED": 2,
    "CNY": 2,
    "EUR": 2,
    "GBP": 2,
    "JPY": 0,
    "KWD": 3,
    "KZT": 2,
    "USD": 2,
}

ROUNDING_MODES: dict[str, str] = {
    "half_even": ROUND_HALF_EVEN,
    "half_up": ROUND_HALF_UP,
}

_CTX = Context(prec=60)


class MoneyError(ValueError):
    """Invalid currency, amount, rate or rounding mode."""


def exponent(currency: str) -> int:
    try:
        return CURRENCY_EXPONENTS[currency]
    except KeyError as exc:
        raise MoneyError(f"unknown currency {currency!r}") from exc


def parse_rate(rate: str | Decimal) -> Decimal:
    """Parse a decimal-string FX rate. Must be finite and positive."""
    if isinstance(rate, float):
        raise MoneyError("FX rates must be decimal strings, not float")
    try:
        value = Decimal(str(rate))
    except InvalidOperation as exc:
        raise MoneyError(f"invalid rate {rate!r}") from exc
    if not value.is_finite() or value <= 0:
        raise MoneyError(f"rate must be finite and positive, got {rate!r}")
    return value


def format_rate(rate: Decimal, places: int) -> str:
    """Render a rate as a fixed-point decimal string with `places` digits after the point."""
    return str(rate.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_EVEN, context=_CTX))


def round_to_int(value: Decimal, rounding: str) -> int:
    try:
        mode = ROUNDING_MODES[rounding]
    except KeyError as exc:
        raise MoneyError(f"unknown rounding mode {rounding!r}") from exc
    return int(value.quantize(Decimal(1), rounding=mode, context=_CTX))


def convert_minor(
    amount_minor: int,
    source_currency: str,
    target_currency: str,
    rate: str | Decimal,
    rounding: str,
) -> int:
    """Convert integer minor units of `source_currency` into minor units of `target_currency`.

    result = round(amount_minor * 10**(exp_target - exp_source) * rate)
    """
    if not isinstance(amount_minor, int) or isinstance(amount_minor, bool):
        raise MoneyError("amount_minor must be int")
    shift = exponent(target_currency) - exponent(source_currency)
    value = _CTX.multiply(Decimal(amount_minor).scaleb(shift, context=_CTX), parse_rate(rate))
    return round_to_int(value, rounding)


def to_major_str(amount_minor: int, currency: str) -> str:
    """Human-readable amount: 123456 KZT -> '1234.56'. For logs and explanations only."""
    exp = exponent(currency)
    return str(Decimal(amount_minor).scaleb(-exp, context=_CTX))
