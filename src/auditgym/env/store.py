"""Read-only record store over the public part of one episode (D11: narrow read-only interface).

The store only ever receives a PublicEpisode: it has no reference to the reference answer.
Records are immutable dataclasses; every view is built as a new dict on each call.

Record ids: ledger -> row_id, invoices and payments -> doc_id, fx -> FxRate.rate_id
("YYYY-MM-DD/SRC/TGT"). Tables keep the order of the episode files.

Cursors are opaque and stable: the cursor that starts a page at row `offset` of `table` is a
keyed hash of (episode_id, table, offset). The same position always gets the same token; a token
from another table or another episode is rejected.
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any

from auditgym.contracts.types import FxRate, Invoice, LedgerRow, Payment, PublicEpisode

_CURSOR_DOMAIN = "auditgym/cursor/v1"
_CURSOR_PREFIX = "c_"
_CURSOR_HEX = 24

Record = LedgerRow | Invoice | Payment | FxRate


class UnknownIdError(LookupError):
    pass


class InvalidCursorError(LookupError):
    pass


def record_id(record: Record) -> str:
    if isinstance(record, LedgerRow):
        return record.row_id
    if isinstance(record, (Invoice, Payment)):
        return record.doc_id
    return record.rate_id


def record_view(record: Record) -> dict[str, Any]:
    """Table view of one record: invoices without line items, everything else in full."""
    if isinstance(record, Invoice):
        return record.summary()
    return record.to_dict()


class RecordStore:
    """Public records of one episode, indexed for the tools."""

    def __init__(self, public: PublicEpisode) -> None:
        self.manifest: dict[str, Any] = copy.deepcopy(dict(public.manifest))
        self.policy: dict[str, Any] = copy.deepcopy(dict(public.policy))
        self.episode_id = str(self.manifest.get("episode_id", ""))
        self._tables: dict[str, tuple[Record, ...]] = {
            "ledger": tuple(public.ledger),
            "invoices": tuple(public.invoices),
            "payments": tuple(public.payments),
            "fx": tuple(public.fx),
        }
        self._index: dict[str, dict[str, Record]] = {}
        for table, rows in self._tables.items():
            index: dict[str, Record] = {}
            for rec in rows:
                index.setdefault(record_id(rec), rec)  # first occurrence wins
            self._index[table] = index
        self._documents: dict[str, Invoice | Payment] = {}
        for doc in (*public.invoices, *public.payments):
            self._documents.setdefault(doc.doc_id, doc)
        self._cursors: dict[str, dict[str, int]] = {}

    # ------------------------------------------------------------------ cursors
    def cursor_for(self, table: str, offset: int) -> str:
        key = f"{_CURSOR_DOMAIN}|{self.episode_id}|{table}|{offset}".encode()
        return _CURSOR_PREFIX + hashlib.sha256(key).hexdigest()[:_CURSOR_HEX]

    def _cursor_map(self, table: str) -> dict[str, int]:
        if table not in self._cursors:
            n = len(self._tables[table])
            self._cursors[table] = {self.cursor_for(table, off): off for off in range(1, n)}
        return self._cursors[table]

    # ------------------------------------------------------------------ queries
    def page(
        self, table: str, cursor: str | None, limit: int
    ) -> tuple[list[dict[str, Any]], str | None, list[str]]:
        """(record views, next cursor or None, ids shown) for one page."""
        rows = self._tables[table]
        if cursor is None:
            offset = 0
        else:
            offset = self._cursor_map(table).get(cursor, -1)
            if offset < 0:
                raise InvalidCursorError
        chunk = rows[offset : offset + limit]
        end = offset + len(chunk)
        next_cursor = self.cursor_for(table, end) if end < len(rows) else None
        return [record_view(r) for r in chunk], next_cursor, [record_id(r) for r in chunk]

    def get(self, table: str, rec_id: str) -> dict[str, Any]:
        rec = self._index[table].get(rec_id)
        if rec is None:
            raise UnknownIdError
        return record_view(rec)

    def document(self, doc_id: str) -> dict[str, Any]:
        doc = self._documents.get(doc_id)
        if doc is None:
            raise UnknownIdError
        return doc.to_dict()
