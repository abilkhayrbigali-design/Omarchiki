"""Amount draws in integer minor units. No float anywhere.

Amounts are drawn in a reference unit (roughly US dollars) and converted with a fixed table of
reference values, so that a 0-digit currency (JPY) and a 3-digit one (KWD) get plausible sizes.
Roundness varies: each amount is rounded to a random number of significant digits, the same way
for every record, so injected values are not recognisable by their roundness.
"""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Decimal

from auditgym.util.money import exponent
from auditgym.util.rng import Stream

# Reference value of one major unit. Synthetic, only used to scale amounts and start FX walks.
REFERENCE_VALUE: dict[str, Decimal] = {
    "AED": Decimal("0.2723"),
    "CNY": Decimal("0.138"),
    "EUR": Decimal("1.08"),
    "GBP": Decimal("1.27"),
    "JPY": Decimal("0.0067"),
    "KWD": Decimal("3.25"),
    "KZT": Decimal("0.0021"),
    "USD": Decimal("1"),
}

MIN_AMOUNT = 100  # every document amount is at least 100 minor units (no grey zone, DOMAIN 6)


def round_sig(value: int, digits: int) -> int:
    """Round a positive integer to `digits` significant digits (half up)."""
    if value <= 0:
        raise ValueError("round_sig needs a positive value")
    length = len(str(value))
    if length <= digits:
        return value
    factor = 10 ** (length - digits)
    return (value + factor // 2) // factor * factor


def sig_digits(s: Stream) -> int:
    return s.weighted_choice([2, 3, 4, 5, 6], [2, 3, 3, 2, 1])


def _reference_amount(s: Stream, lo: int, hi: int) -> Decimal:
    """Roughly log-uniform reference amount in [lo, hi]: pick a decade, then uniform inside it."""
    edges = [lo]
    edge = 10
    while edge < hi:
        if edge > lo:
            edges.append(edge)
        edge *= 10
    edges.append(hi)
    i = s.randbelow(len(edges) - 1)
    return s.decimal(edges[i], edges[i + 1], 2)


def to_minor(currency: str, reference: Decimal) -> int:
    major = reference / REFERENCE_VALUE[currency]
    minor = (major * (Decimal(10) ** exponent(currency))).to_integral_value(ROUND_HALF_EVEN)
    return int(minor)


def draw_minor(s: Stream, currency: str, lo: int, hi: int) -> int:
    """A positive amount in minor units worth about [lo, hi] reference units."""
    raw = max(to_minor(currency, _reference_amount(s, lo, hi)), MIN_AMOUNT)
    return max(round_sig(raw, sig_digits(s)), MIN_AMOUNT)


def can_split(total: int) -> bool:
    """Whether split_part can split `total` (decided without drawing)."""
    return total >= 3 * MIN_AMOUNT


def split_part(s: Stream, total: int, lo_pct: int = 20, hi_pct: int = 80) -> int | None:
    """A part of `total` between lo_pct and hi_pct percent, with random roundness.

    Both the part and the rest are at least MIN_AMOUNT. None if `total` is too small to split.
    """
    if not can_split(total):
        return None
    pct = s.randint(lo_pct, hi_pct)
    raw = max(total * pct // 100, 1)
    part = round_sig(raw, sig_digits(s))
    return min(max(part, MIN_AMOUNT), total - MIN_AMOUNT)
