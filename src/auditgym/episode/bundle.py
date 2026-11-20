"""Bundle (pipeline step 5): public files, truth and provenance of one generation attempt."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from auditgym.books.posting import WRow
from auditgym.contracts.types import Issue, PublicEpisode, Truth
from auditgym.contracts.versions import GENERATOR_VERSION
from auditgym.inject.plan import Plan
from auditgym.inject.state import InjectState, IssueRec, Snapshot
from auditgym.present.presenter import Presenter
from auditgym.present.render import render_invoice, render_payment, render_public, render_row
from auditgym.world.build import world_summary
from auditgym.world.model import World


@dataclass
class Built:
    public: PublicEpisode
    clean: PublicEpisode
    truth: Truth
    provenance: dict[str, Any]


def _keys(world: World, rows: list[WRow]) -> list[str]:
    keys: list[str] = [c.key for c in world.customers]
    for inv in world.invoices:
        keys += [inv.key, inv.number, *(li.order for li in inv.lines)]
    for pay in world.payments:
        keys += [pay.key, pay.number]
    keys += [r.key for r in rows]
    return keys


def _public_ref(ids: dict[str, str], ref: str) -> str:
    """Internal key -> public ID; FX rate ids are already public."""
    return ids.get(ref, ref)


def _issue(rec: IssueRec, ids: dict[str, str], presenter: Presenter) -> Issue:
    return Issue(
        issue_id=presenter.issue_id(rec.key),
        type=rec.type,
        affected_ids=tuple(ids[k] for k in rec.affected),
        expected_amount_minor=rec.amount,
        currency=rec.currency,
        amount_tolerance_minor=rec.tolerance,
        evidence_sets=tuple(tuple(_public_ref(ids, x) for x in ev) for ev in rec.evidence),
    )


def _render_snapshot(world: World, snap: Snapshot, obj: Any, ids: dict[str, str]) -> Any:
    if obj is None:
        return None
    if snap.table == "ledger":
        return render_row(world, obj, ids).to_dict()
    if snap.table == "invoices":
        return render_invoice(obj, ids).to_dict()
    return render_payment(obj, ids).to_dict()


def bundle(
    *,
    seed: int,
    cfg: dict[str, Any],
    stream_config_hash: str,
    plan: Plan,
    world: World,
    clean_rows: list[WRow],
    state: InjectState,
    presenter: Presenter,
) -> Built:
    ids = presenter.assign_ids(_keys(world, clean_rows) + _keys(state.world, state.rows))
    episode_id = presenter.episode_id()
    public = render_public(state.world, state.rows, ids, presenter, episode_id)
    clean = render_public(world, clean_rows, ids, presenter, episode_id)
    truth = Truth(episode_id, tuple(_issue(rec, ids, presenter) for rec in state.issues))

    def issue_ids(keys: list[str]) -> list[str]:
        return [presenter.issue_id(k) for k in keys]

    injections = []
    changes = []
    scenarios = []
    for group in state.groups:
        injections.append(
            {
                "issue_ids": issue_ids(group.issue_keys),
                "injector": group.injector,
                "scenario": group.scenario,
                "variant": group.variant,
                "affected_keys": list(group.affected_keys),
                "affected_ids": [ids[k] for k in group.affected_keys],
                "details": group.details,
            }
        )
        if group.scenario is not None:
            scenarios.append({"scenario": group.scenario, "issue_ids": issue_ids(group.issue_keys)})
        for snap in group.snapshots:
            changes.append(
                {
                    "issue_ids": issue_ids(group.issue_keys),
                    "table": snap.table,
                    "id": ids[snap.key],
                    "before": _render_snapshot(state.world, snap, snap.before, ids),
                    "after": _render_snapshot(state.world, snap, snap.after, ids),
                }
            )

    provenance: dict[str, Any] = {
        "generator": "auditgym.episode.generate",
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "config": cfg,
        "stream_config_hash": stream_config_hash,
        "attempts": None,  # filled in by generate
        "attempt_failures": [],
        "plan": plan.to_dict(),
        "world": world_summary(world),
        "presentation": {
            "order": presenter.order,
            "id_formats": {
                k: {"prefix": p, "chars": n} for k, (p, n) in sorted(presenter.formats.items())
            },
        },
        "distractors": [
            {
                "kind": d.kind,
                "code": d.code,
                "keys": list(d.docs),
                "ids": [ids[k] for k in d.docs],
                "note": d.note,
            }
            for d in world.distractors
        ],
        "injections": injections,
        "scenarios": scenarios,
        "changes": changes,
        "id_map": dict(sorted(ids.items())),
    }
    return Built(public, clean, truth, provenance)
