"""Benign transformations of a finished episode for paired checks (K4, V02).

`transform_episode` permutes the rows of every table and renames every opaque ID (rows,
documents, invoices, payments, entities, episode), consistently in the public data and in the
truth. The semantics of the episode do not change, so:
- the oracle's answer, mapped through the renaming, gets the same reward;
- an order- and spelling-independent solver returns the same answer up to the renaming.

FX record ids ("YYYY-MM-DD/SRC/TGT") are semantic, not opaque, and are kept.
Provenance is grader-only; its ID map is rewritten too, so it stays consistent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any

from auditgym.contracts.types import (
    Episode,
    FxRate,
    Issue,
    PrivateEpisode,
    PublicEpisode,
    Truth,
)
from auditgym.util.rng import Stream, keyed_token, root_stream

_FX_ID = re.compile(r"^\d{4}-\d{2}-\d{2}/[A-Z]{3}/[A-Z]{3}$")


@dataclass(frozen=True)
class Transform:
    episode: Episode
    id_map: dict[str, str]  # old id -> new id (opaque ids only)

    def map_id(self, value: str) -> str:
        return self.id_map.get(value, value)

    def map_ids(self, values: Any) -> list[str]:
        return [self.map_id(v) for v in values]


def _collect_ids(public: PublicEpisode) -> dict[str, str]:
    """Every opaque id -> its kind prefix used for the new id."""
    kinds: dict[str, str] = {public.episode_id: "ep"}
    for r in public.ledger:
        kinds[r.row_id] = "GL"
        if r.entity_id:
            kinds[r.entity_id] = "CU"
        if r.invoice_id:
            kinds[r.invoice_id] = "IV"
        if r.payment_id:
            kinds[r.payment_id] = "PY"
    for inv in public.invoices:
        kinds[inv.doc_id] = "DI"
        kinds[inv.invoice_id] = "IV"
        kinds[inv.entity_id] = "CU"
    for pay in public.payments:
        kinds[pay.doc_id] = "DP"
        kinds[pay.payment_id] = "PY"
        for ref in pay.invoice_refs or ():
            kinds[ref] = "IV"
    return kinds


def _new_ids(kinds: dict[str, str], secret: bytes) -> dict[str, str]:
    out: dict[str, str] = {}
    used: set[str] = set()
    for old in sorted(kinds):
        prefix = kinds[old]
        n = 0
        while True:
            label = f"{old}#{n}"
            new = f"{prefix}-{keyed_token(secret, label, 10)}"
            if prefix == "ep":
                new = f"ep_{keyed_token(secret, label, 12)}"
            if new not in used and new not in kinds:
                break
            n += 1
        used.add(new)
        out[old] = new
    return out


def _opt(m: dict[str, str], value: str | None) -> str | None:
    return None if value is None else m[value]


def transform_episode(
    episode: Episode, salt: str, *, rename: bool = True, permute: bool = True
) -> Transform:
    """Return the transformed episode and the id map. Deterministic in (episode, salt)."""
    public, private = episode.public, episode.private
    stream: Stream = root_stream(0, "transform", salt)
    secret = stream.child("ids").token(32).encode("ascii")

    kinds = _collect_ids(public)
    m = _new_ids(kinds, secret) if rename else {k: k for k in kinds}

    ledger = [
        replace(
            r,
            row_id=m[r.row_id],
            entity_id=_opt(m, r.entity_id),
            invoice_id=_opt(m, r.invoice_id),
            payment_id=_opt(m, r.payment_id),
        )
        for r in public.ledger
    ]
    invoices = [
        replace(inv, doc_id=m[inv.doc_id], invoice_id=m[inv.invoice_id], entity_id=m[inv.entity_id])
        for inv in public.invoices
    ]
    payments = [
        replace(
            pay,
            doc_id=m[pay.doc_id],
            payment_id=m[pay.payment_id],
            invoice_refs=None
            if pay.invoice_refs is None
            else tuple(m[x] for x in pay.invoice_refs),
        )
        for pay in public.payments
    ]
    fx: list[FxRate] = list(public.fx)
    if permute:
        ledger = stream.child("ledger").shuffled(ledger)
        invoices = stream.child("invoices").shuffled(invoices)
        payments = stream.child("payments").shuffled(payments)
        fx = stream.child("fx").shuffled(fx)

    new_episode_id = m[public.episode_id]
    manifest = dict(public.manifest)
    manifest["episode_id"] = new_episode_id

    def map_any(value: str) -> str:
        if _FX_ID.match(value):
            return value
        return m[value]

    issues = tuple(
        Issue(
            issue_id=i.issue_id,
            type=i.type,
            affected_ids=tuple(map_any(x) for x in i.affected_ids),
            expected_amount_minor=i.expected_amount_minor,
            currency=i.currency,
            amount_tolerance_minor=i.amount_tolerance_minor,
            evidence_sets=tuple(tuple(map_any(x) for x in s) for s in i.evidence_sets),
        )
        for i in private.truth.issues
    )
    provenance = _rewrite_provenance(private.provenance, m)
    provenance["transform"] = {"salt": salt, "renamed": rename, "permuted": permute}

    new = Episode(
        public=PublicEpisode(
            manifest=manifest,
            policy=public.policy,
            ledger=tuple(ledger),
            invoices=tuple(invoices),
            payments=tuple(payments),
            fx=tuple(fx),
        ),
        private=PrivateEpisode(truth=Truth(new_episode_id, issues), provenance=provenance),
    )
    return Transform(episode=new, id_map=m)


def _rewrite_provenance(value: Any, m: dict[str, str]) -> Any:
    """Replace every string that is exactly an old opaque id; leave everything else."""
    if isinstance(value, dict):
        return {k: _rewrite_provenance(v, m) for k, v in value.items()}
    if isinstance(value, list):
        return [_rewrite_provenance(v, m) for v in value]
    if isinstance(value, str):
        return m.get(value, value)
    return value


def map_issue_key(issue: Issue, t: Transform) -> tuple[str, tuple[str, ...], int, str]:
    """Comparable key of an issue after mapping its ids through the transform."""
    return (
        issue.type,
        tuple(sorted(t.map_id(x) for x in issue.affected_ids)),
        issue.expected_amount_minor,
        issue.currency,
    )


def issue_key(issue: Issue) -> tuple[str, tuple[str, ...], int, str]:
    return (
        issue.type,
        tuple(sorted(issue.affected_ids)),
        issue.expected_amount_minor,
        issue.currency,
    )
