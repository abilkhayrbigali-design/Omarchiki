"""Render hidden-world records as public records (contracts.types) with public IDs."""

from __future__ import annotations

from typing import Any

from auditgym.books.posting import WRow
from auditgym.contracts.meta import make_manifest, make_policy
from auditgym.contracts.types import FxRate, Invoice, LedgerRow, LineItem, Payment, PublicEpisode
from auditgym.present.presenter import Presenter
from auditgym.util.timeutil import format_ts
from auditgym.world.fx import FX_TOLERANCE_MINOR
from auditgym.world.model import WInvoice, World, WPayment


def render_invoice(inv: WInvoice, ids: dict[str, str]) -> Invoice:
    return Invoice(
        doc_id=ids[inv.key],
        invoice_id=ids[inv.number],
        entity_id=ids[inv.customer],
        issued_at=format_ts(inv.issued, inv.display_tz),
        due_at=format_ts(inv.due, inv.display_tz),
        currency=inv.currency,
        total_minor=inv.total,
        line_items=tuple(
            LineItem(
                description=li.description,
                quantity=li.quantity,
                unit_price_minor=li.unit_price,
                amount_minor=li.amount,
                order_ref=ids[li.order],
                service_period=li.service_period,
            )
            for li in inv.lines
        ),
    )


def render_payment(pay: WPayment, ids: dict[str, str]) -> Payment:
    return Payment(
        doc_id=ids[pay.key],
        payment_id=ids[pay.number],
        timestamp=format_ts(pay.ts, pay.display_tz),
        currency=pay.currency,
        amount_minor=pay.amount,
        invoice_refs=None if pay.refs is None else tuple(ids[r] for r in pay.refs),
    )


def render_row(world: World, row: WRow, ids: dict[str, str]) -> LedgerRow:
    return LedgerRow(
        row_id=ids[row.key],
        timestamp=format_ts(row.ts, world.tl.tz),
        account=world.accounts[row.role],
        debit_minor=row.debit,
        credit_minor=row.credit,
        currency=row.currency,
        entity_id=None if row.entity is None else ids[row.entity],
        invoice_id=None if row.invoice is None else ids[row.invoice],
        payment_id=None if row.payment is None else ids[row.payment],
    )


def render_fx(world: World) -> list[FxRate]:
    return [FxRate(day, cur, world.base, rate) for (day, cur), rate in sorted(world.fx.items())]


def policy_of(world: World) -> dict[str, Any]:
    tl = world.tl
    return make_policy(
        base_currency=world.base,
        timezone=tl.tz_text,
        period_start=format_ts(tl.period_start, tl.tz),
        period_end=format_ts(tl.period_end, tl.tz),
        accounts=world.accounts,
        fx_rate_date=world.fx_rule,
        rounding=world.rounding,
        fx_tolerance_minor=FX_TOLERANCE_MINOR,
    )


def render_public(
    world: World, rows: list[WRow], ids: dict[str, str], presenter: Presenter, episode_id: str
) -> PublicEpisode:
    tl = world.tl
    invoices = presenter.arrange(world.invoices, label=lambda i: i.key, instant=lambda i: i.issued)
    payments = presenter.arrange(world.payments, label=lambda p: p.key, instant=lambda p: p.ts)
    ledger = presenter.arrange(
        rows, label=lambda r: r.key, instant=lambda r: r.ts, group=lambda r: r.entry
    )
    fx = render_fx(world)
    if presenter.order == "shuffled":
        fx = presenter.arrange(fx, label=lambda f: f.rate_id)
    public_ledger = tuple(render_row(world, r, ids) for r in ledger)
    public_invoices = tuple(render_invoice(i, ids) for i in invoices)
    public_payments = tuple(render_payment(p, ids) for p in payments)
    manifest = make_manifest(
        episode_id=episode_id,
        company=world.company,
        base_currency=world.base,
        currencies=world.currencies,
        period_start=format_ts(tl.period_start, tl.tz),
        period_end=format_ts(tl.period_end, tl.tz),
        timezone=tl.tz_text,
        rounding=world.rounding,
        counts={
            "ledger": len(public_ledger),
            "invoices": len(public_invoices),
            "payments": len(public_payments),
            "fx": len(fx),
        },
    )
    return PublicEpisode(
        manifest=manifest,
        policy=policy_of(world),
        ledger=public_ledger,
        invoices=public_invoices,
        payments=public_payments,
        fx=tuple(fx),
    )
