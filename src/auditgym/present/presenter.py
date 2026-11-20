"""Public IDs and row order (pipeline step 4, rule T6).

All public IDs are opaque keyed tokens: `prefix + keyed_token(secret, kind|internal_key)`. The
secret comes from the presentation stream and `config.presentation.id_salt`, so IDs reveal neither
generation order nor which records were injected, and a new salt renames everything without
touching the semantics. Formats (prefix and length per ID kind) vary per episode (G06).

Rows are ordered chronologically by instant; ties are broken by a keyed token of the record (from
the presentation secret), never by the ID string or the generation position. With
`order = "shuffled"` every table is a keyed pseudo-random permutation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
from typing import Any, TypeVar

from auditgym.util.rng import Stream, keyed_token

T = TypeVar("T")

ID_KINDS = ("doc", "invoice", "payment", "row", "entity", "order")

FORMATS: dict[str, tuple[tuple[str, int], ...]] = {
    "doc": (("DOC-", 10), ("D", 11), ("REC-", 9), ("DC", 10)),
    "invoice": (("INV-", 8), ("SI-", 8), ("INV", 9), ("F-", 8)),
    "payment": (("PAY-", 8), ("BNK-", 8), ("PMT", 9), ("RCPT-", 8)),
    "row": (("R-", 10), ("GL-", 10), ("JE", 10), ("L-", 10)),
    "entity": (("C-", 6), ("CUST-", 6), ("E-", 6), ("CL", 7)),
    "order": (("ORD-", 8), ("SO-", 8), ("PO", 9), ("ORDER-", 7)),
}

_KIND_OF_PREFIX = {
    "k:inv:": "doc",
    "k:pay:": "doc",
    "k:invno:": "invoice",
    "k:payno:": "payment",
    "k:cust:": "entity",
    "k:ord:": "order",
}


def kind_of(key: str) -> str:
    """ID kind of an internal key. Ledger row keys contain '#'."""
    if "#" in key:
        return "row"
    for prefix, kind in _KIND_OF_PREFIX.items():
        if key.startswith(prefix):
            return kind
    raise ValueError(f"not a public record key: {key!r}")


class Presenter:
    def __init__(self, stream: Stream, presentation: dict[str, Any]) -> None:
        material = stream.child("secret").token(32).encode("ascii")
        salt = presentation["id_salt"].encode("utf-8")
        self._id_secret = hashlib.sha256(b"auditgym/present/ids|" + material + b"|" + salt).digest()
        self._order_secret = hashlib.sha256(
            b"auditgym/present/order|" + material + b"|" + salt
        ).digest()
        fs = stream.child("formats")
        self.formats = {kind: fs.choice(FORMATS[kind]) for kind in ID_KINDS}
        self.order = presentation["order"]

    # ------------------------------------------------------------------ ids
    def episode_id(self) -> str:
        return "ep_" + keyed_token(self._id_secret, "episode", 12)

    def issue_id(self, key: str) -> str:
        return "iss_" + keyed_token(self._id_secret, "issue|" + key, 10)

    def assign_ids(self, keys: Iterable[str]) -> dict[str, str]:
        """Internal key -> public ID for every key, globally unique."""
        out: dict[str, str] = {}
        taken: set[str] = set()
        for key in sorted(set(keys)):
            kind = kind_of(key)
            prefix, nchars = self.formats[kind]
            n = 0
            while True:
                label = f"{kind}|{key}" if n == 0 else f"{kind}|{key}|{n}"
                pid = prefix + keyed_token(self._id_secret, label, nchars)
                if pid not in taken:
                    break
                n += 1
            taken.add(pid)
            out[key] = pid
        return out

    # ------------------------------------------------------------------ order
    def _tie(self, label: str) -> str:
        return keyed_token(self._order_secret, label, 13)

    def arrange(
        self,
        items: Sequence[T],
        *,
        label: Callable[[T], str],
        instant: Callable[[T], datetime] | None = None,
        group: Callable[[T], str] | None = None,
    ) -> list[T]:
        """Chronological order with keyed tie-breaks, or a keyed shuffle."""
        if self.order == "shuffled" or instant is None:
            return sorted(items, key=lambda x: self._tie("item|" + label(x)))

        def key(x: T) -> tuple[datetime, str, str]:
            g = self._tie("group|" + group(x)) if group is not None else ""
            return (instant(x), g, self._tie("item|" + label(x)))

        return sorted(items, key=key)
