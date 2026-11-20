"""Safe exact calculator for the `calculator(expression)` tool (E06).

Own tokenizer and recursive-descent parser over `decimal.Decimal`. No eval, exec, ast or names:
the only accepted characters are ASCII digits, '.', '+', '-', '*', '/', '(', ')', space and tab.

Grammar:
    expression := term (('+' | '-') term)*
    term       := unary (('*' | '/') unary)*
    unary      := ('+' | '-')* primary
    primary    := NUMBER | '(' expression ')'
    NUMBER     := DIGITS ('.' DIGITS?)? | '.' DIGITS

So `+5`, `5.` and `.5` are accepted, as in ordinary decimal notation. Exponents (`1e5`), `**`,
underscores, names and non-ASCII digits are not.

Limits: at most CALCULATOR_MAX_CHARS characters, at most CALCULATOR_MAX_DIGITS digits per number,
at most MAX_NESTING levels of parentheses, every intermediate value below
10**CALCULATOR_MAX_DIGITS in absolute value. Arithmetic uses CALCULATOR_MAX_DIGITS significant
digits with ROUND_HALF_EVEN, so `1/3` is rounded deterministically. The result is a plain decimal
string without exponent and without trailing fractional zeros ("60570000", "0.5", "-2.25").
Anything else, including division by zero, raises CalculatorError.
"""

from __future__ import annotations

from decimal import (
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DecimalException,
    DivisionByZero,
    InvalidOperation,
    Overflow,
)

from auditgym.contracts.constants import CALCULATOR_MAX_CHARS, CALCULATOR_MAX_DIGITS

MAX_NESTING = 32
_OPERATORS = frozenset("+-*/()")
_SPACE = frozenset(" \t")
_DIGITS = frozenset("0123456789")


class CalculatorError(ValueError):
    """The expression is rejected. Carries no detail: the agent only sees calculator_error."""


def _context() -> Context:
    # A fresh context per evaluation: flags never leak between calls.
    return Context(
        prec=CALCULATOR_MAX_DIGITS,
        rounding=ROUND_HALF_EVEN,
        Emax=999_999,
        Emin=-999_999,
        capitals=1,
        clamp=0,
        flags=[],
        traps=[InvalidOperation, DivisionByZero, Overflow],
    )


def _tokenize(expression: str) -> list[str | Decimal]:
    tokens: list[str | Decimal] = []
    i, n = 0, len(expression)
    while i < n:
        ch = expression[i]
        if ch in _SPACE:
            i += 1
            continue
        if ch in _OPERATORS:
            tokens.append(ch)
            i += 1
            continue
        if ch in _DIGITS or ch == ".":
            j = i
            while j < n and expression[j] in _DIGITS:
                j += 1
            int_digits = j - i
            frac_digits = 0
            if j < n and expression[j] == ".":
                k = j + 1
                while k < n and expression[k] in _DIGITS:
                    k += 1
                frac_digits = k - j - 1
                j = k
            if int_digits + frac_digits == 0:  # a lone '.'
                raise CalculatorError
            if int_digits + frac_digits > CALCULATOR_MAX_DIGITS:
                raise CalculatorError
            # Exact: built from an ASCII digit string ("5.", ".5" and "5.5" are all valid).
            tokens.append(Decimal(expression[i:j]))
            i = j
            continue
        raise CalculatorError
    return tokens


class _Parser:
    __slots__ = ("_tokens", "_pos", "_ctx")

    def __init__(self, tokens: list[str | Decimal], ctx: Context) -> None:
        self._tokens = tokens
        self._pos = 0
        self._ctx = ctx

    def _peek(self) -> str | Decimal | None:
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _take(self) -> str | Decimal:
        token = self._peek()
        if token is None:
            raise CalculatorError
        self._pos += 1
        return token

    def _bounded(self, value: Decimal) -> Decimal:
        if not value.is_finite():
            raise CalculatorError
        if not value.is_zero() and value.adjusted() >= CALCULATOR_MAX_DIGITS:
            raise CalculatorError
        return value

    def parse(self) -> Decimal:
        value = self._expression(0)
        if self._peek() is not None:
            raise CalculatorError
        return value

    def _expression(self, depth: int) -> Decimal:
        value = self._term(depth)
        while self._peek() in ("+", "-"):
            op = self._take()
            right = self._term(depth)
            if op == "+":
                value = self._bounded(self._ctx.add(value, right))
            else:
                value = self._bounded(self._ctx.subtract(value, right))
        return value

    def _term(self, depth: int) -> Decimal:
        value = self._unary(depth)
        while self._peek() in ("*", "/"):
            op = self._take()
            right = self._unary(depth)
            if op == "*":
                value = self._bounded(self._ctx.multiply(value, right))
            else:
                if right.is_zero():
                    raise CalculatorError
                value = self._bounded(self._ctx.divide(value, right))
        return value

    def _unary(self, depth: int) -> Decimal:
        negate = False
        while self._peek() in ("+", "-"):
            if self._take() == "-":
                negate = not negate
        value = self._primary(depth)
        return self._ctx.minus(value) if negate else value

    def _primary(self, depth: int) -> Decimal:
        token = self._take()
        if isinstance(token, Decimal):
            return token
        if token == "(":
            if depth + 1 > MAX_NESTING:
                raise CalculatorError
            value = self._expression(depth + 1)
            if self._take() != ")":
                raise CalculatorError
            return value
        raise CalculatorError


def _plain(value: Decimal, ctx: Context) -> str:
    if value.is_zero():
        return "0"
    return format(value.normalize(ctx), "f")


def evaluate(expression: str) -> str:
    """Evaluate an arithmetic expression; return a plain decimal string or raise CalculatorError."""
    if not isinstance(expression, str) or len(expression) > CALCULATOR_MAX_CHARS:
        raise CalculatorError
    tokens = _tokenize(expression)
    if not tokens:
        raise CalculatorError
    ctx = _context()
    try:
        value = _Parser(tokens, ctx).parse()
        return _plain(value, ctx)
    except (DecimalException, ArithmeticError, RecursionError) as exc:
        raise CalculatorError from exc
