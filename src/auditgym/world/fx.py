"""Synthetic daily FX rates: foreign currency -> base currency, one row per calendar day.

Each currency follows its own random walk (own child stream), so adding a currency does not shift
the others. Most days move by at most 0.4%; about one day in five moves by 0.6-1.5% (a jump). A
spike (distractor X9) multiplies the level by 3-8% on one day; the new level persists.

An injected FX error is material: the wrong conversion differs from the right one by at least
max(100 minor units, 0.5% of the right converted amount).
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import ROUND_HALF_EVEN, Context, Decimal

from auditgym.util.money import convert_minor
from auditgym.util.rng import Stream
from auditgym.util.timeutil import daterange
from auditgym.world.amounts import REFERENCE_VALUE
from auditgym.world.model import World, WPayment

_CTX = Context(prec=24, rounding=ROUND_HALF_EVEN)
DAILY_MOVE = ("-0.004", "0.004")
JUMP_MOVE = ("0.006", "0.015")  # magnitude of a jump day, either direction
JUMP_ODDS = (1, 5)
MIN_DISCREPANCY = 100
MIN_RELATIVE_PER_MILLE = 5  # 0.5% of the right converted amount
FX_TOLERANCE_MINOR = 1  # policy.tolerances.fx_conversion_minor


# ---------------------------------------------------------------------------------- conversion
def rate_date_by(world: World, rule: str, ts: datetime) -> date:
    if rule == "payment_local_date":
        return world.tl.local_date(ts)
    if rule == "payment_local_date_minus_1":
        return world.tl.local_date(ts) - timedelta(days=1)
    if rule == "payment_utc_date":
        return ts.astimezone(UTC).date()
    raise ValueError(f"unknown fx rate date rule {rule!r}")


def rate_date(world: World, ts: datetime) -> date:
    """Date of the FX rate for a payment at `ts`, by the episode policy."""
    return rate_date_by(world, world.fx_rule, ts)


def convert_on(world: World, amount: int, currency: str, day: date) -> int | None:
    """Convert with the currency's rate of `day`; None if there is no such FX row."""
    rate = world.fx.get((day.isoformat(), currency))
    if rate is None:
        return None
    return convert_minor(amount, currency, world.base, rate, world.rounding)


def conversion(world: World, pay: WPayment) -> int:
    """Expected base-currency amount of a payment (signed), by the policy."""
    if pay.currency == world.base:
        return pay.amount
    value = convert_on(world, pay.amount, pay.currency, rate_date(world, pay.ts))
    if value is None:
        raise ValueError("no FX rate for the policy date")
    return value


def is_material(right: int, wrong: int | None) -> bool:
    """|wrong - right| >= max(100, 0.5% of |right|), in exact integer arithmetic."""
    if wrong is None or wrong <= 0:
        return False
    diff = abs(wrong - right)
    return diff >= MIN_DISCREPANCY and diff * 1000 >= MIN_RELATIVE_PER_MILLE * abs(right)


def material(world: World, pay: WPayment, wrong: int | None) -> bool:
    """A wrong conversion of `pay` that an auditor cannot dismiss as rounding or noise."""
    return is_material(conversion(world, pay), wrong)


def wrong_rate_dates(world: World, pay: WPayment) -> list[date]:
    """Dates within a week of the correct one whose rate gives a material difference."""
    right = rate_date(world, pay.ts)
    out = []
    for k in range(-7, 8):
        d = right + timedelta(days=k)
        if k and material(world, pay, convert_on(world, pay.amount, pay.currency, d)):
            out.append(d)
    return out


def format_level(level: Decimal) -> str:
    """Six significant digits, at least four decimals, never scientific notation."""
    places = max(4, 5 - level.adjusted())
    q = level.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_EVEN, context=_CTX)
    return format(q, "f")


def build_fx(
    s: Stream,
    base: str,
    foreign: list[str],
    first: date,
    last: date,
    spikes: dict[tuple[str, date], Decimal],
) -> dict[tuple[str, str], str]:
    """(YYYY-MM-DD, currency) -> rate string, for every day in [first, last]."""
    out: dict[tuple[str, str], str] = {}
    for cur in foreign:
        cs = s.child(f"fx/{cur}")
        level = _CTX.multiply(
            _CTX.divide(REFERENCE_VALUE[cur], REFERENCE_VALUE[base]), cs.decimal("0.92", "1.08", 4)
        )
        for day in daterange(first, last):
            if day != first:
                if cs.chance(*JUMP_ODDS):
                    move = cs.decimal(JUMP_MOVE[0], JUMP_MOVE[1], 5)
                    if cs.chance(1, 2):
                        move = -move
                else:
                    move = cs.decimal(DAILY_MOVE[0], DAILY_MOVE[1], 5)
                level = _CTX.multiply(level, _CTX.add(Decimal(1), move))
            factor = spikes.get((cur, day))
            if factor is not None:
                level = _CTX.multiply(level, factor)
            out[(day.isoformat(), cur)] = format_level(level)
    return out
