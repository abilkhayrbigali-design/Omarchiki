"""The generator's own consistency check (pipeline step 6, first half).

The invariant violations found in the public books must correspond exactly to the reference
answer:

| issue type          | expected violation(s), same ids, amount and currency          |
|---------------------|---------------------------------------------------------------|
| duplicate_invoice   | I7; plus I3 when all copies share one invoice_id               |
| payment_allocation  | I5 on (payment doc, invoice doc)                              |
| fx_conversion       | I6 on the payment doc                                         |
| period_cutoff       | I4 on the document                                            |

Clean books must have no violation at all. Any difference means the generator produced data whose
reference answer is not what an exact auditor would find, and the attempt is retried.
"""

from __future__ import annotations

from collections.abc import Iterable

from auditgym.books.invariants import check, id_problems
from auditgym.contracts.types import Issue, PublicEpisode, Truth

Signature = tuple[str, tuple[str, ...], int | None, str | None]


def expected_violations(public: PublicEpisode, truth: Truth) -> list[Signature]:
    number_of = {inv.doc_id: inv.invoice_id for inv in public.invoices}
    out: list[Signature] = []
    for issue in truth.issues:
        sig = (issue.affected_ids, issue.expected_amount_minor, issue.currency)
        if issue.type == "duplicate_invoice":
            out.append(("I7", *sig))
            if len({number_of[d] for d in issue.affected_ids}) == 1:
                out.append(("I3", *sig))
        elif issue.type == "payment_allocation":
            out.append(("I5", *sig))
        elif issue.type == "fx_conversion":
            out.append(("I6", *sig))
        elif issue.type == "period_cutoff":
            out.append(("I4", *sig))
        else:
            raise ValueError(f"unknown issue type {issue.type!r}")
    return sorted(out, key=_key)


def _key(sig: Signature) -> tuple[object, ...]:
    code, ids, amount, currency = sig
    return (code, ids, -1 if amount is None else amount, currency or "")


def found_violations(public: PublicEpisode) -> list[Signature]:
    return sorted((v.signature() for v in check(public)), key=_key)


def problems(public: PublicEpisode, truth: Truth, clean: PublicEpisode) -> list[str]:
    """Empty when the books, the clean books and the truth agree."""
    out = []
    clean_violations = check(clean)
    if clean_violations:
        out.append(f"clean books violate {sorted({v.code for v in clean_violations})}")
    out += [f"clean: {p}" for p in id_problems(clean)]
    out += id_problems(public)
    if found_violations(public) != expected_violations(public, truth):
        out.append("invariant violations differ from truth")
    ids = _all_ids(public)
    for issue in truth.issues:
        if not set(issue.affected_ids) <= ids:
            out.append(f"{issue.issue_id}: unknown affected id")
        if issue.expected_amount_minor < 100:
            out.append(f"{issue.issue_id}: discrepancy below 100 minor units")
        for ev in issue.evidence_sets:
            if len(ev) > 10 or not set(ev) <= ids:
                out.append(f"{issue.issue_id}: bad evidence set")
    if len({i.key() for i in truth.issues}) != len(truth.issues):
        out.append("duplicate (type, affected_ids) in truth")
    return out


def _all_ids(public: PublicEpisode) -> set[str]:
    return (
        {r.row_id for r in public.ledger}
        | {d.doc_id for d in public.invoices}
        | {d.doc_id for d in public.payments}
        | {f.rate_id for f in public.fx}
    )


def issue_signatures(issues: Iterable[Issue]) -> list[tuple[str, tuple[str, ...], int, str]]:
    """What the self-check compares with an independent solver."""
    return sorted(
        (i.type, tuple(sorted(i.affected_ids)), i.expected_amount_minor, i.currency) for i in issues
    )
