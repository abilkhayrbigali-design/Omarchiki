"""Reconciliation baseline: read the books through the tools, then apply the published rules.

Plan, within the 100-step budget:
1. page through payments, invoices (summaries), FX and the ledger with list_records;
2. fetch full documents only for invoices that could be duplicates (same customer, currency
   and total, or a shared invoice_id) — line items are needed to compare order_ref;
3. rebuild the public records it has seen and run the rule-based solver (auditgym.verify)
   on them; submit what it finds with the evidence the solver cites.

It sees nothing the tools do not show. Invoices whose documents were not fetched get a single
placeholder line item with a unique order_ref, so they can never be reported as duplicates by
order_ref. If the budget does not allow reading everything, it reads what it can and submits.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from auditgym.contracts.constants import MAX_FINDINGS, MAX_STEPS, PAGE_LIMIT_MAX
from auditgym.contracts.types import FxRate, Invoice, LedgerRow, LineItem, Payment, PublicEpisode
from auditgym.policies.base import episode_id_of, finding, get_document, list_records, submit

_TABLE_ORDER = ("payments", "invoices", "fx", "ledger")


class Reconcile:
    name = "reconcile"

    def __init__(self) -> None:
        self._reset_state()

    def _reset_state(self) -> None:
        self._first: dict[str, Any] = {}
        self._episode_id = ""
        self._rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._docs: dict[str, dict[str, Any]] = {}
        self._pending_tables: list[tuple[str, str | None]] = []
        self._pending_docs: list[str] | None = None
        self._last: tuple[str, str] | None = None

    def reset(self, observation: dict[str, Any], info: dict[str, Any]) -> None:
        self._reset_state()
        self._first = observation
        self._episode_id = episode_id_of(observation)
        self._pending_tables = [(t, None) for t in _TABLE_ORDER]

    # ------------------------------------------------------------------ acting
    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        self._absorb(observation)
        remaining = int(observation.get("steps_remaining", MAX_STEPS))
        if remaining <= 1:
            return self._submit()
        if self._pending_tables:
            table, cursor = self._pending_tables.pop(0)
            self._last = ("table", table)
            return list_records(table, cursor, PAGE_LIMIT_MAX)
        if self._pending_docs is None:
            self._pending_docs = self._duplicate_candidates()
        if self._pending_docs:
            doc_id = self._pending_docs.pop(0)
            self._last = ("doc", doc_id)
            return get_document(doc_id)
        return self._submit()

    def _absorb(self, observation: dict[str, Any]) -> None:
        result = observation.get("result")
        if not isinstance(result, dict):
            return
        if isinstance(result.get("records"), list):
            table = result.get("table")
            self._rows[table].extend(result["records"])
            if result.get("next_cursor"):
                self._pending_tables.insert(0, (table, result["next_cursor"]))
        elif isinstance(result.get("document"), dict):
            doc = result["document"]
            self._docs[doc["doc_id"]] = doc

    def _duplicate_candidates(self) -> list[str]:
        groups: dict[tuple[Any, ...], list[str]] = defaultdict(list)
        by_invoice_id: dict[str, list[str]] = defaultdict(list)
        for inv in self._rows["invoices"]:
            groups[(inv["entity_id"], inv["currency"], inv["total_minor"])].append(inv["doc_id"])
            by_invoice_id[inv["invoice_id"]].append(inv["doc_id"])
        wanted: set[str] = set()
        for docs in list(groups.values()) + list(by_invoice_id.values()):
            if len(docs) > 1:
                wanted.update(docs)
        return sorted(wanted)

    # ------------------------------------------------------------------ solving
    def _rebuild(self) -> PublicEpisode:
        invoices = []
        for summary in self._rows["invoices"]:
            full = self._docs.get(summary["doc_id"])
            if full is not None:
                invoices.append(Invoice.from_dict(full))
                continue
            total = summary["total_minor"]
            placeholder = LineItem("unread", 1, total, total, f"unread:{summary['doc_id']}", None)
            invoices.append(
                Invoice(
                    doc_id=summary["doc_id"],
                    invoice_id=summary["invoice_id"],
                    entity_id=summary["entity_id"],
                    issued_at=summary["issued_at"],
                    due_at=summary["due_at"],
                    currency=summary["currency"],
                    total_minor=total,
                    line_items=(placeholder,),
                )
            )
        manifest = dict(self._first["manifest"])
        return PublicEpisode(
            manifest=manifest,
            policy=self._first["policy"],
            ledger=tuple(LedgerRow.from_dict(r) for r in self._rows["ledger"]),
            invoices=tuple(invoices),
            payments=tuple(Payment.from_dict(p) for p in self._rows["payments"]),
            fx=tuple(FxRate.from_dict(f) for f in self._rows["fx"]),
        )

    def _submit(self) -> dict[str, Any]:
        findings: list[dict[str, Any]] = []
        try:
            from auditgym.verify import solve

            issues = solve(self._rebuild())
        except Exception:  # incomplete or unexpected data: submit nothing rather than guess
            issues = ()
        for issue in issues[:MAX_FINDINGS]:
            evidence = list(issue.evidence_sets[0]) if issue.evidence_sets else []
            findings.append(
                finding(
                    issue.type,
                    list(issue.affected_ids),
                    issue.expected_amount_minor,
                    issue.currency,
                    evidence[:10],
                )
            )
        return submit(self._episode_id, findings)
