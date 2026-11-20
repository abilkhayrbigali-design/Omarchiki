"""Record types of an episode. Immutable; `to_dict` gives the exact public JSON shape.

Field names follow the case (docs/REQUIREMENTS.md D04-D10). Money is int minor units,
timestamps are ISO-8601 strings with offset, FX rates are decimal strings, missing links are None.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_NULL_SPELLINGS = ("", "null", "NULL", "None")


def _opt_str(value: Any) -> str | None:
    """Optional id: None, or an empty/'null' CSV cell, is a missing link (D09)."""
    if value is None or value in _NULL_SPELLINGS:
        return None
    if not isinstance(value, str):
        raise TypeError(f"expected string or null, got {type(value).__name__}")
    return value


def _int(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError("bool is not an integer amount")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer() and abs(value) <= 2**53:
        return int(value)  # JSON writers may emit 123.0 for an integer
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    raise TypeError(f"expected integer, got {value!r}")


# ---------------------------------------------------------------------------------- documents
@dataclass(frozen=True, slots=True)
class LineItem:
    description: str
    quantity: int
    unit_price_minor: int
    amount_minor: int
    order_ref: str
    service_period: str | None = None  # "YYYY-MM" for subscriptions, else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "quantity": self.quantity,
            "unit_price_minor": self.unit_price_minor,
            "amount_minor": self.amount_minor,
            "order_ref": self.order_ref,
            "service_period": self.service_period,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LineItem:
        return cls(
            description=d["description"],
            quantity=_int(d["quantity"]),
            unit_price_minor=_int(d["unit_price_minor"]),
            amount_minor=_int(d["amount_minor"]),
            order_ref=d["order_ref"],
            service_period=_opt_str(d.get("service_period")),
        )


@dataclass(frozen=True, slots=True)
class Invoice:
    """A sales invoice. Negative total_minor means a credit note."""

    doc_id: str
    invoice_id: str
    entity_id: str
    issued_at: str
    due_at: str
    currency: str
    total_minor: int
    line_items: tuple[LineItem, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "invoice_id": self.invoice_id,
            "entity_id": self.entity_id,
            "issued_at": self.issued_at,
            "due_at": self.due_at,
            "currency": self.currency,
            "total_minor": self.total_minor,
            "line_items": [li.to_dict() for li in self.line_items],
        }

    def summary(self) -> dict[str, Any]:
        """Table view used by list_records/get_record: header fields without line items."""
        return {
            "doc_id": self.doc_id,
            "invoice_id": self.invoice_id,
            "entity_id": self.entity_id,
            "issued_at": self.issued_at,
            "due_at": self.due_at,
            "currency": self.currency,
            "total_minor": self.total_minor,
            "line_item_count": len(self.line_items),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Invoice:
        return cls(
            doc_id=d["doc_id"],
            invoice_id=d["invoice_id"],
            entity_id=d["entity_id"],
            issued_at=d["issued_at"],
            due_at=d["due_at"],
            currency=d["currency"],
            total_minor=_int(d["total_minor"]),
            line_items=tuple(LineItem.from_dict(x) for x in d["line_items"]),
        )


@dataclass(frozen=True, slots=True)
class Payment:
    """A bank payment. Negative amount_minor means money paid out (refund)."""

    doc_id: str
    payment_id: str
    timestamp: str
    currency: str
    amount_minor: int
    invoice_refs: tuple[str, ...] | None  # payer's references in the payer's order, or None

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "payment_id": self.payment_id,
            "timestamp": self.timestamp,
            "currency": self.currency,
            "amount_minor": self.amount_minor,
            "invoice_refs": None if self.invoice_refs is None else list(self.invoice_refs),
        }

    def summary(self) -> dict[str, Any]:
        return self.to_dict()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Payment:
        refs = d.get("invoice_refs")
        return cls(
            doc_id=d["doc_id"],
            payment_id=d["payment_id"],
            timestamp=d["timestamp"],
            currency=d["currency"],
            amount_minor=_int(d["amount_minor"]),
            invoice_refs=None if refs is None else tuple(refs),
        )


# ---------------------------------------------------------------------------------- tables
@dataclass(frozen=True, slots=True)
class LedgerRow:
    """One general-ledger line. Exactly one of debit_minor / credit_minor is non-zero."""

    row_id: str
    timestamp: str
    account: str
    debit_minor: int
    credit_minor: int
    currency: str
    entity_id: str | None
    invoice_id: str | None
    payment_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "timestamp": self.timestamp,
            "account": self.account,
            "debit_minor": self.debit_minor,
            "credit_minor": self.credit_minor,
            "currency": self.currency,
            "entity_id": self.entity_id,
            "invoice_id": self.invoice_id,
            "payment_id": self.payment_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LedgerRow:
        return cls(
            row_id=d["row_id"],
            timestamp=d["timestamp"],
            account=d["account"],
            debit_minor=_int(d["debit_minor"]),
            credit_minor=_int(d["credit_minor"]),
            currency=d["currency"],
            entity_id=_opt_str(d.get("entity_id")),
            invoice_id=_opt_str(d.get("invoice_id")),
            payment_id=_opt_str(d.get("payment_id")),
        )


@dataclass(frozen=True, slots=True)
class FxRate:
    """1 unit of source_currency = `rate` units of target_currency on `date`."""

    date: str  # YYYY-MM-DD
    source_currency: str
    target_currency: str
    rate: str  # decimal string

    @property
    def rate_id(self) -> str:
        """Public id of an FX row, used by get_record and in evidence."""
        return fx_rate_id(self.date, self.source_currency, self.target_currency)

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "source_currency": self.source_currency,
            "target_currency": self.target_currency,
            "rate": self.rate,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FxRate:
        return cls(
            date=d["date"],
            source_currency=d["source_currency"],
            target_currency=d["target_currency"],
            rate=str(d["rate"]),
        )


def fx_rate_id(date: str, source_currency: str, target_currency: str) -> str:
    return f"{date}/{source_currency}/{target_currency}"


# ---------------------------------------------------------------------------------- reference answer
def _sorted_ids(ids: Any) -> tuple[str, ...]:
    return tuple(sorted(set(ids)))


@dataclass(frozen=True, slots=True)
class Issue:
    """One reference discrepancy (truth.json). IDs and evidence sets are stored sorted."""

    issue_id: str
    type: str
    affected_ids: tuple[str, ...]
    expected_amount_minor: int
    currency: str
    amount_tolerance_minor: int
    evidence_sets: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "affected_ids", _sorted_ids(self.affected_ids))
        sets = {_sorted_ids(s) for s in self.evidence_sets}
        object.__setattr__(self, "evidence_sets", tuple(sorted(sets)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "type": self.type,
            "affected_ids": list(self.affected_ids),
            "expected_amount_minor": self.expected_amount_minor,
            "currency": self.currency,
            "amount_tolerance_minor": self.amount_tolerance_minor,
            "evidence_sets": [list(s) for s in self.evidence_sets],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Issue:
        return cls(
            issue_id=d["issue_id"],
            type=d["type"],
            affected_ids=tuple(d["affected_ids"]),
            expected_amount_minor=_int(d["expected_amount_minor"]),
            currency=d["currency"],
            amount_tolerance_minor=_int(d["amount_tolerance_minor"]),
            evidence_sets=tuple(tuple(s) for s in d["evidence_sets"]),
        )

    def key(self) -> tuple[str, tuple[str, ...]]:
        """(type, affected_ids): unique within an episode (DOMAIN T2)."""
        return (self.type, self.affected_ids)


@dataclass(frozen=True, slots=True)
class Truth:
    episode_id: str
    issues: tuple[Issue, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "issues", tuple(sorted(self.issues, key=lambda i: i.key())))

    def to_dict(self) -> dict[str, Any]:
        return {"episode_id": self.episode_id, "issues": [i.to_dict() for i in self.issues]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Truth:
        return cls(
            episode_id=d["episode_id"], issues=tuple(Issue.from_dict(x) for x in d["issues"])
        )


@dataclass(frozen=True, slots=True)
class Finding:
    """One finding submitted by an agent, after structural validation."""

    type: str
    affected_ids: tuple[str, ...]
    claimed_amount_minor: int
    currency: str
    evidence_ids: tuple[str, ...]
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "affected_ids": list(self.affected_ids),
            "claimed_amount_minor": self.claimed_amount_minor,
            "currency": self.currency,
            "evidence_ids": list(self.evidence_ids),
            "explanation": self.explanation,
        }


# ---------------------------------------------------------------------------------- episode
@dataclass(frozen=True, slots=True)
class PublicEpisode:
    """Everything the agent may see, through the tool interface only."""

    manifest: dict[str, Any]
    policy: dict[str, Any]
    ledger: tuple[LedgerRow, ...]
    invoices: tuple[Invoice, ...]
    payments: tuple[Payment, ...]
    fx: tuple[FxRate, ...]

    @property
    def episode_id(self) -> str:
        return self.manifest["episode_id"]


@dataclass(frozen=True, slots=True)
class PrivateEpisode:
    """Grader-only data. Never reachable from the agent process."""

    truth: Truth
    provenance: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Episode:
    public: PublicEpisode
    private: PrivateEpisode
