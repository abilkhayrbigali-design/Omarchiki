from datetime import date, timedelta, timezone
from decimal import Decimal

import pytest

from auditgym.util import canon
from auditgym.util.money import MoneyError, convert_minor, parse_rate, round_to_int, to_major_str
from auditgym.util.timeutil import (
    TimeError,
    format_offset,
    format_ts,
    local_date,
    parse_offset,
    parse_ts,
    period_class,
    utc_date,
)


# ------------------------------------------------------------------ money (D03)
def test_convert_same_exponent():
    assert convert_minor(120_000, "USD", "KZT", "504.75", "half_even") == 60_570_000


def test_convert_zero_exponent_source():
    # 1000 JPY at 3.21 KZT per JPY = 3210.00 KZT = 321000 minor
    assert convert_minor(1000, "JPY", "KZT", "3.21", "half_even") == 321_000


def test_convert_three_exponent_target():
    # 10.00 USD at 0.3071 KWD = 3.071 KWD = 3071 minor
    assert convert_minor(1000, "USD", "KWD", "0.3071", "half_even") == 3071


@pytest.mark.parametrize(
    "value,mode,expected",
    [
        ("2.5", "half_even", 2),
        ("3.5", "half_even", 4),
        ("2.5", "half_up", 3),
        ("-2.5", "half_up", -3),
    ],
)
def test_rounding_modes(value, mode, expected):
    assert round_to_int(Decimal(value), mode) == expected


def test_rates_must_not_be_float():
    with pytest.raises(MoneyError):
        parse_rate(1.5)  # type: ignore[arg-type]
    with pytest.raises(MoneyError):
        parse_rate("0")
    with pytest.raises(MoneyError):
        parse_rate("NaN")


def test_unknown_currency():
    with pytest.raises(MoneyError):
        convert_minor(1, "XXX", "KZT", "1", "half_even")


def test_to_major_str():
    assert to_major_str(123_456, "KZT") == "1234.56"
    assert to_major_str(500, "JPY") == "500"


# ------------------------------------------------------------------ time (D03)
def test_offsets_roundtrip():
    tz = parse_offset("-04:30")
    assert tz == timezone(-timedelta(hours=4, minutes=30))
    assert format_offset(tz) == "-04:30"


def test_naive_timestamp_rejected():
    with pytest.raises(TimeError):
        parse_ts("2026-03-01T00:00:00")


def test_local_date_differs_from_utc_date():
    ts = "2026-03-31T21:30:00+00:00"
    assert utc_date(ts) == date(2026, 3, 31)
    assert local_date(ts, parse_offset("+05:00")) == date(2026, 4, 1)


def test_format_ts_converts_offset():
    dt = parse_ts("2026-03-20T11:00:00+00:00")
    assert format_ts(dt, parse_offset("+05:00")) == "2026-03-20T16:00:00+05:00"


@pytest.mark.parametrize(
    "ts,expected",
    [
        ("2026-02-28T23:59:59+05:00", "before"),
        ("2026-03-01T00:00:00+05:00", "in"),
        ("2026-03-31T23:59:59+05:00", "in"),
        ("2026-04-01T00:00:00+05:00", "after"),
        ("2026-03-31T19:30:00+00:00", "after"),  # = 2026-04-01T00:30+05:00
    ],
)
def test_period_class(ts, expected):
    assert period_class(ts, "2026-03-01T00:00:00+05:00", "2026-04-01T00:00:00+05:00") == expected


# ------------------------------------------------------------------ canonical output
def test_dumps_sorted_and_compact():
    assert canon.dumps({"b": 1, "a": [1, None]}) == '{"a":[1,null],"b":1}'


def test_dumps_rejects_nan():
    with pytest.raises(ValueError):
        canon.dumps({"x": float("nan")})


def test_csv_none_is_empty_and_roundtrips():
    text = canon.csv_text(["a", "b"], [{"a": "x", "b": None}, {"a": "y,z", "b": 5}])
    assert text == 'a,b\nx,\n"y,z",5\n'
    header, rows = canon.read_csv(text)
    assert header == ["a", "b"]
    assert rows == [{"a": "x", "b": ""}, {"a": "y,z", "b": "5"}]
