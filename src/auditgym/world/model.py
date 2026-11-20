"""Hidden company history.

Every record carries internal keys of the form "k:<kind>:<n>". They never leave the generator:
the presentation layer (auditgym.present) maps them to opaque public IDs. Keys are only used for
identity, never for ordering anything that reaches the public files.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from auditgym.world.timeline import Timeline

KEY_PREFIX = "k:"


class WorldError(RuntimeError):
    """The world could not satisfy a construction constraint for this attempt."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass
class Customer:
    key: str
    name: str
    currency: str


@dataclass
class WLine:
    description: str
    quantity: int
    unit_price: int
    order: str  # internal order key
    service_period: str | None = None

    @property
    def amount(self) -> int:
        return self.quantity * self.unit_price


@dataclass
class WInvoice:
    key: str  # document key
    number: str  # invoice_id key
    customer: str
    issued: datetime
    due: datetime
    currency: str
    lines: list[WLine]
    display_tz: timezone  # offset the document timestamps are written in
    post_ts: datetime  # timestamp of the recognition entry
    tag: str = "regular"  # "regular", a distractor kind, or "injected"

    @property
    def total(self) -> int:
        return sum(li.amount for li in self.lines)


@dataclass
class WPayment:
    key: str  # document key
    number: str  # payment_id key
    payer: str
    ts: datetime
    currency: str
    amount: int
    refs: list[str] | None  # invoice number keys in the payer's order
    plan: list[tuple[str, int]]  # intended allocation of the story: (invoice number key, amount)
    display_tz: timezone
    post_ts: datetime
    tag: str = "regular"


@dataclass
class WExpense:
    key: str
    ts: datetime
    amount: int  # base currency


@dataclass
class DistractorRec:
    kind: str
    code: str
    docs: list[str]
    note: str = ""


@dataclass
class World:
    company: str
    base: str
    foreign: list[str]
    tl: Timeline
    accounts: dict[str, str]
    fx_rule: str
    rounding: str
    fx: dict[tuple[str, str], str] = field(default_factory=dict)
    fx_first: date | None = None
    fx_last: date | None = None
    customers: list[Customer] = field(default_factory=list)
    invoices: list[WInvoice] = field(default_factory=list)
    payments: list[WPayment] = field(default_factory=list)
    expenses: list[WExpense] = field(default_factory=list)
    distractors: list[DistractorRec] = field(default_factory=list)
    spikes: list[dict] = field(default_factory=list)
    topups: dict[str, int] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)
    taken: set[datetime] = field(default_factory=set)  # membership only, never iterated

    # ------------------------------------------------------------------ keys and instants
    def mint(self, kind: str) -> str:
        n = self.counters.get(kind, 0) + 1
        self.counters[kind] = n
        return f"{KEY_PREFIX}{kind}:{n:05d}"

    def claim(self, draw: Callable[[], datetime]) -> datetime:
        """Draw a document instant not used by any other document or expense entry."""
        for _ in range(64):
            t = draw()
            if t not in self.taken:
                self.taken.add(t)
                return t
        raise WorldError("no_free_instant")

    # ------------------------------------------------------------------ lookups
    @property
    def currencies(self) -> list[str]:
        return [self.base, *self.foreign]

    def customer(self, key: str) -> Customer:
        for c in self.customers:
            if c.key == key:
                return c
        raise KeyError(key)

    def invoice_docs(self, number: str) -> list[WInvoice]:
        return [inv for inv in self.invoices if inv.number == number]

    def number_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for inv in self.invoices:
            counts[inv.number] = counts.get(inv.number, 0) + 1
        return counts

    def invoice_of_number(self, number: str) -> WInvoice:
        """The document of an invoice_id: the earliest one (DOMAIN 6, clarifications)."""
        docs = self.invoice_docs(number)
        if not docs:
            raise KeyError(number)
        return min(docs, key=lambda d: d.issued)
