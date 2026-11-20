"""Trick probes: common shortcuts that the environment must not reward (G07, S10).

Each probe reads the whole public episode directly (a best case for the shortcut: no step
budget, no paging) and turns one heuristic into findings. The findings are graded by the real
grader with zero inspection steps. A good environment gives every probe a low F1, because
the distractors (X1-X10) are built to look like errors to exactly these heuristics.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from datetime import timedelta
from typing import Any

from auditgym.contracts.types import PublicEpisode
from auditgym.util.money import convert_minor
from auditgym.util.timeutil import local_date, parse_offset, utc_date

Probe = Callable[[PublicEpisode], list[dict[str, Any]]]


def _finding(
    t: str, affected: list[str], amount: int, cur: str, evidence: list[str]
) -> dict[str, Any]:
    return {
        "type": t,
        "affected_ids": sorted(set(affected)),
        "claimed_amount_minor": abs(int(amount)),
        "currency": cur,
        "evidence_ids": sorted(set(evidence))[:10],
        "explanation": "",
    }


def _roles(pub: PublicEpisode) -> dict[str, str]:
    return {code: role for role, code in pub.policy["accounts"].items()}


def same_amount_duplicates(pub: PublicEpisode) -> list[dict[str, Any]]:
    """Invoices of one customer with the same currency and total are duplicates."""
    groups: dict[tuple[str, str, int], list[Any]] = defaultdict(list)
    for inv in pub.invoices:
        if inv.total_minor > 0:
            groups[(inv.entity_id, inv.currency, inv.total_minor)].append(inv)
    out = []
    for (_, cur, total), invs in groups.items():
        if len(invs) > 1:
            docs = [i.doc_id for i in invs]
            out.append(_finding("duplicate_invoice", docs, (len(invs) - 1) * total, cur, docs))
    return out


def unapplied_means_error(pub: PublicEpisode) -> list[dict[str, Any]]:
    """Any UNAPPLIED amount on a payment that has references is an allocation error."""
    roles = _roles(pub)
    inv_doc = {}
    for inv in sorted(pub.invoices, key=lambda i: i.issued_at):
        inv_doc.setdefault(inv.invoice_id, inv.doc_id)
    rows_by_payment: dict[str, list[Any]] = defaultdict(list)
    for r in pub.ledger:
        if r.payment_id:
            rows_by_payment[r.payment_id].append(r)
    payments_by_invoice: dict[str, list[Any]] = defaultdict(list)
    for p in pub.payments:
        for ref in p.invoice_refs or ():
            payments_by_invoice[ref].append(p)
    out = []
    for p in pub.payments:
        if not p.invoice_refs or p.amount_minor <= 0:
            continue
        rows = [r for r in rows_by_payment[p.payment_id] if r.currency == p.currency]
        unapplied = sum(
            r.credit_minor - r.debit_minor for r in rows if roles.get(r.account) == "UNAPPLIED"
        )
        if unapplied > 0 and p.invoice_refs[0] in inv_doc:
            ref = p.invoice_refs[0]
            earlier = [q.doc_id for q in payments_by_invoice[ref] if q.timestamp < p.timestamp]
            evidence = [p.doc_id, inv_doc[ref], *[r.row_id for r in rows], *earlier]
            out.append(
                _finding(
                    "payment_allocation", [p.doc_id, inv_doc[ref]], unapplied, p.currency, evidence
                )
            )
    return out


def utc_date_cutoff(pub: PublicEpisode) -> list[dict[str, Any]]:
    """Compare UTC dates of documents and postings with the period, ignoring the company offset."""
    start, end = utc_date(pub.manifest["period_start"]), utc_date(pub.manifest["period_end"])

    def cls(d: Any) -> str:
        return "before" if d < start else "in" if d < end else "after"

    out = []
    first_row: dict[str, Any] = {}
    for r in pub.ledger:
        key = r.payment_id or r.invoice_id
        if key and key not in first_row:
            first_row[key] = r
    for inv in pub.invoices:
        r = first_row.get(inv.invoice_id)
        if r and cls(utc_date(inv.issued_at)) != cls(utc_date(r.timestamp)):
            out.append(
                _finding(
                    "period_cutoff",
                    [inv.doc_id],
                    inv.total_minor,
                    inv.currency,
                    [inv.doc_id, r.row_id],
                )
            )
    for p in pub.payments:
        r = first_row.get(p.payment_id)
        if r and cls(utc_date(p.timestamp)) != cls(utc_date(r.timestamp)):
            out.append(
                _finding(
                    "period_cutoff", [p.doc_id], p.amount_minor, p.currency, [p.doc_id, r.row_id]
                )
            )
    return out


def posting_date_fx(pub: PublicEpisode) -> list[dict[str, Any]]:
    """Recompute FX with the rate of the posting date, ignoring the policy's rate-date rule."""
    roles = _roles(pub)
    tz = parse_offset(pub.manifest["timezone"])
    base = pub.manifest["base_currency"]
    rounding = pub.policy["fx"]["rounding"]
    rates = {(f.date, f.source_currency): f for f in pub.fx if f.target_currency == base}
    cash: dict[str, list[Any]] = defaultdict(list)
    for r in pub.ledger:
        if r.payment_id and roles.get(r.account) == "CASH":
            cash[r.payment_id].append(r)
    out = []
    for p in pub.payments:
        if p.currency == base or not cash.get(p.payment_id):
            continue
        row = cash[p.payment_id][0]
        day = local_date(row.timestamp, tz).isoformat()
        rate = rates.get((day, p.currency))
        if rate is None:
            continue
        expected = convert_minor(p.amount_minor, p.currency, base, rate.rate, rounding)
        booked = sum(r.debit_minor - r.credit_minor for r in cash[p.payment_id])
        if booked != expected:
            out.append(
                _finding(
                    "fx_conversion",
                    [p.doc_id],
                    booked - expected,
                    base,
                    [p.doc_id, row.row_id, rate.rate_id],
                )
            )
    return out


def big_fx_moves(pub: PublicEpisode) -> list[dict[str, Any]]:
    """Flag foreign payments made on days when the rate jumped by more than 2%."""
    base = pub.manifest["base_currency"]
    tz = parse_offset(pub.manifest["timezone"])
    rate = {(f.date, f.source_currency): float(f.rate) for f in pub.fx if f.target_currency == base}
    out = []
    for p in pub.payments:
        if p.currency == base:
            continue
        d = local_date(p.timestamp, tz)
        today, yesterday = (
            rate.get((d.isoformat(), p.currency)),
            rate.get(((d - timedelta(days=1)).isoformat(), p.currency)),
        )
        if today and yesterday and abs(today / yesterday - 1) > 0.02:
            out.append(_finding("fx_conversion", [p.doc_id], 100, base, [p.doc_id]))
    return out


PROBES: dict[str, Probe] = {
    "same_amount_duplicates": same_amount_duplicates,
    "unapplied_means_error": unapplied_means_error,
    "utc_date_cutoff": utc_date_cutoff,
    "posting_date_fx": posting_date_fx,
    "big_fx_moves": big_fx_moves,
}


def run_probes(config: dict[str, Any] | None, seeds: list[int]) -> dict[str, Any]:
    """Grade every probe on every episode; per probe: TP/FP/FN totals, micro P/R/F1, mean reward."""
    from auditgym.episode.generate import generate_episode
    from auditgym.grading import grade_submission

    totals = {name: {"tp": 0, "fp": 0, "fn": 0, "reward": 0.0, "episodes": 0} for name in PROBES}
    for seed in seeds:
        ep = generate_episode(seed, config)
        for name, probe in PROBES.items():
            findings = probe(ep.public)[:30]
            g = grade_submission(
                {"episode_id": ep.public.episode_id, "findings": findings},
                ep.private.truth,
                episode_id=ep.public.episode_id,
                inspection_steps=0,
            )
            t = totals[name]
            t["tp"] += g.tp
            t["fp"] += g.fp
            t["fn"] += g.fn
            t["reward"] += g.reward
            t["episodes"] += 1
    out = {}
    for name, t in totals.items():
        tp, fp, fn = t["tp"], t["fp"], t["fn"]
        out[name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(tp / (tp + fp), 3) if tp + fp else 0.0,
            "recall": round(tp / (tp + fn), 3) if tp + fn else 0.0,
            "f1_micro": round(2 * tp / (2 * tp + fp + fn), 3) if tp + fp + fn else 1.0,
            "mean_reward": round(t["reward"] / max(t["episodes"], 1), 3),
        }
    return out
