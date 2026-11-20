"""Calculator: exact decimal arithmetic only, never code (E06)."""

from __future__ import annotations

import re
from decimal import Decimal

import pytest
from envkit import act
from hypothesis import given, settings
from hypothesis import strategies as st

from auditgym.contracts.constants import CALCULATOR_MAX_CHARS, CALCULATOR_MAX_DIGITS
from auditgym.env.calculator import MAX_NESTING, CalculatorError, evaluate

PLAIN = re.compile(r"^-?(0|[1-9][0-9]*)(\.[0-9]*[1-9])?$")


@pytest.mark.parametrize(
    "expression,value",
    [
        ("1+2", "3"),
        ("2*3+4", "10"),
        ("2*(3+4)", "14"),
        ("10-2-3", "5"),
        ("100/10/2", "5"),
        ("10/4", "2.5"),
        ("-5+2", "-3"),
        ("--5", "5"),
        ("3*-2", "-6"),
        ("3--2", "5"),
        ("-(2+3)*2", "-10"),
        ("0.1+0.2", "0.3"),
        ("1.50*2", "3"),
        ("2.5-2.5", "0"),
        ("-0", "0"),
        ("0.0", "0"),
        ("007", "7"),
        ("120000 * 504.75", "60570000"),
        ("120000 * 504.75 - 120000 * 504.50", "30000"),
        ("2500000 - 1500000", "1000000"),
        ("1/3", "0.3333333333333333333333333333333333333333"),
        ("2/3", "0.6666666666666666666666666666666666666667"),
        ("1000000000000000000000000000000000000000/10", "100000000000000000000000000000000000000"),
        ("0.0000000001*0.0000000001", "0.00000000000000000001"),
        ("\t1 +  2 ", "3"),
        ("(((((1)))))", "1"),
        ("9" * 20 + "*" + "9" * 20, "9999999999999999999800000000000000000001"),
    ],
)
def test_E06_calculator_values(expression, value):
    assert evaluate(expression) == value
    assert PLAIN.match(value)


@pytest.mark.parametrize(
    "expression",
    [
        "",
        "   ",
        "__import__('os').system('ls')",
        "open('/etc/passwd').read()",
        "exec('1')",
        "eval('1')",
        "x",
        "x + 1",
        "abs(-1)",
        "pi",
        "2**3",
        "2^3",
        "7//2",
        "7%2",
        "1e5",
        "1E5",
        "inf",
        "nan",
        "NaN",
        "Infinity",
        "0x10",
        "1_000",
        "5.",
        ".5",
        "1..2",
        "1.2.3",
        "+5",
        "(1",
        "1)",
        "()",
        "1 2",
        "2(3)",
        "1/0",
        "0/0",
        "1/(2-2)",
        "1,5",
        "١٢",  # non-ASCII digits
        "１",  # full-width digit
        "1;2",
        "[1]",
        "{1}",
        "'1'",
        '"1"',
        "1 # comment",
        "lambda: 1",
        "\n1",
        "1" * (CALCULATOR_MAX_DIGITS + 1),
        "0." + "1" * CALCULATOR_MAX_DIGITS,
        "9" * CALCULATOR_MAX_DIGITS + "*10",
        ("9" * 39 + "*") * 4 + "9",
        "1" + "+1" * (CALCULATOR_MAX_CHARS // 2),
        "(" * (MAX_NESTING + 1) + "1" + ")" * (MAX_NESTING + 1),
        "(" * 200 + "1" + ")" * 200,
        "-" * 300 + "1",
    ],
)
def test_E06_calculator_rejects(expression):
    with pytest.raises(CalculatorError):
        evaluate(expression)


def test_E06_limits_are_inclusive():
    assert evaluate("1" * CALCULATOR_MAX_DIGITS) == "1" * CALCULATOR_MAX_DIGITS
    expr = "(" * MAX_NESTING + "1" + ")" * MAX_NESTING
    assert evaluate(expr) == "1"
    long_ok = "0" + "+".join(["1"] * (CALCULATOR_MAX_CHARS // 2))
    assert len(long_ok) == CALCULATOR_MAX_CHARS
    assert evaluate(long_ok) == str(CALCULATOR_MAX_CHARS // 2)
    assert evaluate("-" * 200 + "1") == "1"  # unary minus is a loop, not recursion


def test_E06_rejects_non_string():
    for value in (1, None, b"1+1", ["1"]):
        with pytest.raises(CalculatorError):
            evaluate(value)  # type: ignore[arg-type]


def test_E06_calculator_tool_errors_are_fixed(make_env):
    env = make_env()
    env.reset(0)
    for expr in ("__import__('os')", "1/0", "9" * 300, "x" * 5000):
        obs = env.step(act("calculator", expression=expr))[0]
        assert obs["result"] is None
        assert obs["error"]["code"] == "calculator_error"
        assert expr[:5] not in obs["error"]["message"]


MONEY = st.integers(min_value=-(10**15), max_value=10**15)


@settings(max_examples=300, deadline=None)
@given(MONEY, MONEY)
def test_E06_integer_arithmetic_is_exact(a, b):
    assert evaluate(f"{a} + {b}") == str(a + b)
    assert evaluate(f"{a} - {b}") == str(a - b)
    assert evaluate(f"{a} * {b}") == str(a * b)


DECIMALS = st.decimals(
    min_value=Decimal("-1000000"), max_value=Decimal("1000000"), places=6, allow_nan=False
)


@settings(max_examples=300, deadline=None)
@given(DECIMALS, DECIMALS)
def test_E06_decimal_arithmetic_is_exact(x, y):
    sx, sy = format(x, "f"), format(y, "f")
    assert Decimal(evaluate(f"{sx} * {sy}")) == x * y
    assert Decimal(evaluate(f"{sx} + {sy}")) == x + y
    if y != 0:
        out = evaluate(f"({sx}) / ({sy})")
        assert PLAIN.match(out)
        assert out == evaluate(f"({sx}) / ({sy})")  # deterministic


@settings(max_examples=500, deadline=None)
@given(st.text(alphabet="0123456789.+-*/() \tabcdeEx_^%,'\"[]", max_size=60))
def test_E06_fuzz_never_crashes(expression):
    try:
        out = evaluate(expression)
    except CalculatorError:
        return
    assert PLAIN.match(out)
    assert "e" not in out.lower()


@settings(max_examples=200, deadline=None)
@given(st.text(max_size=300))
def test_E06_arbitrary_unicode_never_crashes(expression):
    try:
        out = evaluate(expression)
    except CalculatorError:
        return
    assert PLAIN.match(out)
