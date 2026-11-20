"""Working copy of the books during injection, with truth and record-level change tracking.

An injector opens a group (`begin`), touches or adds records, registers its issues and closes the
group (`end`). For every record touched in a group the state keeps a deep copy before the first
change and after the last one, so provenance can replay the injection record by record.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from auditgym.books.posting import WRow
from auditgym.world.model import WInvoice, World, WPayment


class CandidateError(RuntimeError):
    """No record can carry the requested injection in this attempt."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass
class IssueRec:
    key: str
    type: str
    affected: list[str]  # internal document keys
    amount: int
    currency: str
    tolerance: int
    evidence: list[list[str]]  # internal keys and FX rate ids


@dataclass
class Snapshot:
    table: str  # "ledger" | "invoices" | "payments"
    key: str
    obj: Any  # live record
    before: Any  # deep copy before the first change, None if added in this group
    after: Any = None  # deep copy after the group, None if removed


@dataclass
class Group:
    injector: str
    scenario: str | None
    variant: str = ""
    issue_keys: list[str] = field(default_factory=list)
    affected_keys: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    snapshots: list[Snapshot] = field(default_factory=list)


class InjectState:
    def __init__(self, world: World, rows: list[WRow]) -> None:
        self.world = copy.deepcopy(world)
        self.rows = copy.deepcopy(rows)
        self.used: set[str] = set()  # membership only
        self.issues: list[IssueRec] = []
        self.groups: list[Group] = []
        self._open: Group | None = None
        self._seen: dict[str, Snapshot] = {}

    # ------------------------------------------------------------------ groups
    def begin(self, injector: str, scenario: str | None = None) -> None:
        if self._open is not None:
            raise RuntimeError("injection group already open")
        self._open = Group(injector, scenario)
        self._seen = {}

    @property
    def group(self) -> Group:
        if self._open is None:
            raise RuntimeError("no open injection group")
        return self._open

    def end(self, variant: str, affected: list[str], **details: Any) -> None:
        g = self.group
        g.variant = variant
        g.affected_keys = list(affected)
        g.details = details
        for snap in g.snapshots:
            snap.after = copy.deepcopy(snap.obj)
        self.used.update(affected)
        self.groups.append(g)
        self._open = None
        self._seen = {}

    def abort(self) -> None:
        """Close a group that made no change (candidate rejected before any edit)."""
        if self._open is not None and self._open.snapshots:
            raise RuntimeError("cannot abort a group that changed records")
        self._open = None
        self._seen = {}

    # ------------------------------------------------------------------ record changes
    def _touch(self, table: str, key: str, obj: Any, added: bool) -> None:
        if key in self._seen:
            return
        snap = Snapshot(table, key, obj, None if added else copy.deepcopy(obj))
        self._seen[key] = snap
        self.group.snapshots.append(snap)

    def touch_row(self, row: WRow) -> WRow:
        self._touch("ledger", row.key, row, added=False)
        return row

    def add_row(self, row: WRow) -> None:
        self._touch("ledger", row.key, row, added=True)
        self.rows.append(row)

    def add_invoice(self, inv: WInvoice) -> None:
        self._touch("invoices", inv.key, inv, added=True)
        self.world.invoices.append(inv)

    def touch_invoice(self, inv: WInvoice) -> WInvoice:
        self._touch("invoices", inv.key, inv, added=False)
        return inv

    def touch_payment(self, pay: WPayment) -> WPayment:
        self._touch("payments", pay.key, pay, added=False)
        return pay

    # ------------------------------------------------------------------ truth
    def issue(
        self,
        type_: str,
        affected: list[str],
        amount: int,
        currency: str,
        evidence: list[list[str]],
        tolerance: int = 0,
    ) -> IssueRec:
        if amount < 100:
            raise CandidateError(f"{type_}_amount_below_100")
        for ev in evidence:
            if len(set(ev)) > 10:
                raise CandidateError(f"{type_}_evidence_too_long")
        rec = IssueRec(
            self.world.mint("iss"), type_, list(affected), amount, currency, tolerance, evidence
        )
        self.issues.append(rec)
        self.group.issue_keys.append(rec.key)
        return rec

    # ------------------------------------------------------------------ lookups
    def entry(self, key: str) -> list[WRow]:
        return [r for r in self.rows if r.entry == key]

    def invoice(self, key: str) -> WInvoice:
        for inv in self.world.invoices:
            if inv.key == key:
                return inv
        raise KeyError(key)
