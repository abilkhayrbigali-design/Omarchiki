"""A tiny, hand-checked episode with one issue of each type.

Lets the environment, grader and solver be tested before the generator exists.
All amounts below are computed by hand; tests/unit/test_tiny_fixture.py re-checks them.

Company timezone +05:00, base KZT, foreign USD. Period: March 2026.
FX: USD->KZT rate on day d = 500.00 + 0.25 * (d - 2026-03-01).
Policy: rate date = local date of the payment, rounding half_even.

Issues:
1. duplicate_invoice  - INV-1003 entered twice (docs D-I3, D-I4). 7500000 KZT.
2. fx_conversion      - PAY-2002 (USD 1200.00) converted with the 2026-03-19 rate (504.50)
                        instead of 2026-03-20 (504.75). 60540000 vs 60570000 -> 30000 KZT.
3. payment_allocation - PAY-2003 pays INV-1006 in full (2500000) but only 1500000 is applied,
                        1000000 is booked to UNAPPLIED. 1000000 KZT.
4. period_cutoff      - INV-1005 issued 2026-03-31 20:00 (March), posted 2026-04-01 09:00 (April).
                        3000000 KZT.
Distractors: lagged posting (INV-1006), partial payment leaving INV-1003 open, unapplied
payment without references (PAY-2005), payment timestamp written in UTC (PAY-2002).
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from auditgym.contracts.meta import make_manifest, make_policy
from auditgym.contracts.types import (
    Episode,
    FxRate,
    Invoice,
    Issue,
    LedgerRow,
    LineItem,
    Payment,
    PrivateEpisode,
    PublicEpisode,
    Truth,
)
from auditgym.util.timeutil import daterange, format_date

EPISODE_ID = "ep_TINY000001"
TZ = "+05:00"
PERIOD_START = "2026-03-01T00:00:00+05:00"
PERIOD_END = "2026-04-01T00:00:00+05:00"
ACCOUNTS = {
    "AR": "1200",
    "CASH": "1010",
    "EXPENSE": "6100",
    "FX_CLEARING": "1900",
    "REVENUE": "4000",
    "UNAPPLIED": "2300",
}


def _fx_rates() -> tuple[FxRate, ...]:
    out = []
    origin = date(2026, 3, 1)
    for day in daterange(date(2026, 2, 25), date(2026, 4, 5)):
        rate = Decimal("500.00") + Decimal("0.25") * (day - origin).days
        out.append(FxRate(format_date(day), "USD", "KZT", f"{rate:.2f}"))
    return tuple(out)


def _invoice(doc_id, invoice_id, entity, issued, due, cur, qty, unit, order_ref, desc):
    item = LineItem(desc, qty, unit, qty * unit, order_ref, None)
    return Invoice(doc_id, invoice_id, entity, issued, due, cur, qty * unit, (item,))


def _row(row_id, ts, role, dr, cr, cur, entity, inv, pay):
    return LedgerRow(row_id, ts, ACCOUNTS[role], dr, cr, cur, entity, inv, pay)


def build_tiny_episode() -> Episode:
    invoices = (
        _invoice(
            "D-I1",
            "INV-1001",
            "E-ALFA",
            "2026-03-03T10:00:00+05:00",
            "2026-04-02T10:00:00+05:00",
            "KZT",
            2,
            5_000_000,
            "ORD-501",
            "Consulting, day rate",
        ),
        _invoice(
            "D-I2",
            "INV-1002",
            "E-BETA",
            "2026-03-05T12:00:00+05:00",
            "2026-04-04T12:00:00+05:00",
            "USD",
            1,
            120_000,
            "ORD-502",
            "Software licence",
        ),
        _invoice(
            "D-I3",
            "INV-1003",
            "E-GAMMA",
            "2026-03-10T09:30:00+05:00",
            "2026-04-09T09:30:00+05:00",
            "KZT",
            3,
            2_500_000,
            "ORD-503",
            "Equipment rental",
        ),
        # Duplicate entry of the same invoice two days later.
        _invoice(
            "D-I4",
            "INV-1003",
            "E-GAMMA",
            "2026-03-12T11:00:00+05:00",
            "2026-04-09T09:30:00+05:00",
            "KZT",
            3,
            2_500_000,
            "ORD-503",
            "Equipment rental",
        ),
        _invoice(
            "D-I5",
            "INV-1005",
            "E-ALFA",
            "2026-03-31T20:00:00+05:00",
            "2026-04-30T20:00:00+05:00",
            "KZT",
            1,
            3_000_000,
            "ORD-505",
            "Training session",
        ),
        _invoice(
            "D-I6",
            "INV-1006",
            "E-BETA",
            "2026-03-08T15:00:00+05:00",
            "2026-04-07T15:00:00+05:00",
            "KZT",
            2,
            1_250_000,
            "ORD-506",
            "Support hours",
        ),
    )
    payments = (
        Payment("D-P1", "PAY-2001", "2026-03-15T14:00:00+05:00", "KZT", 10_000_000, ("INV-1001",)),
        # Timestamp written in UTC: 11:00Z = 16:00 local, local date 2026-03-20.
        Payment("D-P2", "PAY-2002", "2026-03-20T11:00:00+00:00", "USD", 120_000, ("INV-1002",)),
        Payment("D-P3", "PAY-2003", "2026-03-18T10:00:00+05:00", "KZT", 2_500_000, ("INV-1006",)),
        Payment("D-P4", "PAY-2004", "2026-03-25T09:00:00+05:00", "KZT", 4_000_000, ("INV-1003",)),
        Payment("D-P5", "PAY-2005", "2026-03-26T10:00:00+05:00", "KZT", 500_000, None),
    )
    ledger = (
        # Invoice postings: Dr AR, Cr REVENUE.
        _row(
            "R-001",
            "2026-03-03T10:00:00+05:00",
            "AR",
            10_000_000,
            0,
            "KZT",
            "E-ALFA",
            "INV-1001",
            None,
        ),
        _row(
            "R-002",
            "2026-03-03T10:00:00+05:00",
            "REVENUE",
            0,
            10_000_000,
            "KZT",
            "E-ALFA",
            "INV-1001",
            None,
        ),
        _row(
            "R-003",
            "2026-03-05T12:00:00+05:00",
            "AR",
            120_000,
            0,
            "USD",
            "E-BETA",
            "INV-1002",
            None,
        ),
        _row(
            "R-004",
            "2026-03-05T12:00:00+05:00",
            "REVENUE",
            0,
            120_000,
            "USD",
            "E-BETA",
            "INV-1002",
            None,
        ),
        _row(
            "R-005",
            "2026-03-10T09:30:00+05:00",
            "AR",
            7_500_000,
            0,
            "KZT",
            "E-GAMMA",
            "INV-1003",
            None,
        ),
        _row(
            "R-006",
            "2026-03-10T09:30:00+05:00",
            "REVENUE",
            0,
            7_500_000,
            "KZT",
            "E-GAMMA",
            "INV-1003",
            None,
        ),
        _row(
            "R-007",
            "2026-03-12T11:00:00+05:00",
            "AR",
            7_500_000,
            0,
            "KZT",
            "E-GAMMA",
            "INV-1003",
            None,
        ),
        _row(
            "R-008",
            "2026-03-12T11:00:00+05:00",
            "REVENUE",
            0,
            7_500_000,
            "KZT",
            "E-GAMMA",
            "INV-1003",
            None,
        ),
        # Lagged posting, same period: correct.
        _row(
            "R-009",
            "2026-03-09T10:00:00+05:00",
            "AR",
            2_500_000,
            0,
            "KZT",
            "E-BETA",
            "INV-1006",
            None,
        ),
        _row(
            "R-010",
            "2026-03-09T10:00:00+05:00",
            "REVENUE",
            0,
            2_500_000,
            "KZT",
            "E-BETA",
            "INV-1006",
            None,
        ),
        # Cut-off error: March invoice posted in April.
        _row(
            "R-011",
            "2026-04-01T09:00:00+05:00",
            "AR",
            3_000_000,
            0,
            "KZT",
            "E-ALFA",
            "INV-1005",
            None,
        ),
        _row(
            "R-012",
            "2026-04-01T09:00:00+05:00",
            "REVENUE",
            0,
            3_000_000,
            "KZT",
            "E-ALFA",
            "INV-1005",
            None,
        ),
        # PAY-2001: correct.
        _row(
            "R-013",
            "2026-03-15T14:00:00+05:00",
            "CASH",
            10_000_000,
            0,
            "KZT",
            "E-ALFA",
            None,
            "PAY-2001",
        ),
        _row(
            "R-014",
            "2026-03-15T14:00:00+05:00",
            "AR",
            0,
            10_000_000,
            "KZT",
            "E-ALFA",
            "INV-1001",
            "PAY-2001",
        ),
        # PAY-2002: foreign payment converted with the wrong day's rate.
        _row(
            "R-015",
            "2026-03-20T16:00:00+05:00",
            "FX_CLEARING",
            120_000,
            0,
            "USD",
            "E-BETA",
            None,
            "PAY-2002",
        ),
        _row(
            "R-016",
            "2026-03-20T16:00:00+05:00",
            "AR",
            0,
            120_000,
            "USD",
            "E-BETA",
            "INV-1002",
            "PAY-2002",
        ),
        _row(
            "R-017",
            "2026-03-20T16:00:00+05:00",
            "CASH",
            60_540_000,
            0,
            "KZT",
            "E-BETA",
            None,
            "PAY-2002",
        ),
        _row(
            "R-018",
            "2026-03-20T16:00:00+05:00",
            "FX_CLEARING",
            0,
            60_540_000,
            "KZT",
            "E-BETA",
            None,
            "PAY-2002",
        ),
        # PAY-2003: partial allocation, remainder wrongly to UNAPPLIED.
        _row(
            "R-019",
            "2026-03-18T10:00:00+05:00",
            "CASH",
            2_500_000,
            0,
            "KZT",
            "E-BETA",
            None,
            "PAY-2003",
        ),
        _row(
            "R-020",
            "2026-03-18T10:00:00+05:00",
            "AR",
            0,
            1_500_000,
            "KZT",
            "E-BETA",
            "INV-1006",
            "PAY-2003",
        ),
        _row(
            "R-021",
            "2026-03-18T10:00:00+05:00",
            "UNAPPLIED",
            0,
            1_000_000,
            "KZT",
            "E-BETA",
            None,
            "PAY-2003",
        ),
        # PAY-2004: correct partial payment, INV-1003 stays open.
        _row(
            "R-022",
            "2026-03-25T09:00:00+05:00",
            "CASH",
            4_000_000,
            0,
            "KZT",
            "E-GAMMA",
            None,
            "PAY-2004",
        ),
        _row(
            "R-023",
            "2026-03-25T09:00:00+05:00",
            "AR",
            0,
            4_000_000,
            "KZT",
            "E-GAMMA",
            "INV-1003",
            "PAY-2004",
        ),
        # PAY-2005: no references, correctly unapplied.
        _row(
            "R-024",
            "2026-03-26T10:00:00+05:00",
            "CASH",
            500_000,
            0,
            "KZT",
            "E-ALFA",
            None,
            "PAY-2005",
        ),
        _row(
            "R-025",
            "2026-03-26T10:00:00+05:00",
            "UNAPPLIED",
            0,
            500_000,
            "KZT",
            "E-ALFA",
            None,
            "PAY-2005",
        ),
        # Background expense.
        _row("R-026", "2026-03-11T12:00:00+05:00", "EXPENSE", 150_000, 0, "KZT", None, None, None),
        _row("R-027", "2026-03-11T12:00:00+05:00", "CASH", 0, 150_000, "KZT", None, None, None),
    )
    fx = _fx_rates()
    policy = make_policy(
        base_currency="KZT",
        timezone=TZ,
        period_start=PERIOD_START,
        period_end=PERIOD_END,
        accounts=ACCOUNTS,
        fx_rate_date="payment_local_date",
        rounding="half_even",
    )
    manifest = make_manifest(
        episode_id=EPISODE_ID,
        company="Tiny Fixture Trading LLP",
        base_currency="KZT",
        currencies=["KZT", "USD"],
        period_start=PERIOD_START,
        period_end=PERIOD_END,
        timezone=TZ,
        rounding="half_even",
        counts={
            "ledger": len(ledger),
            "invoices": len(invoices),
            "payments": len(payments),
            "fx": len(fx),
        },
    )
    issues = (
        Issue(
            "iss_tiny_dup",
            "duplicate_invoice",
            ("D-I3", "D-I4"),
            7_500_000,
            "KZT",
            0,
            (("D-I3", "D-I4"),),
        ),
        Issue(
            "iss_tiny_fx",
            "fx_conversion",
            ("D-P2",),
            30_000,
            "KZT",
            1,
            (("D-P2", "R-017", "2026-03-20/USD/KZT"), ("D-P2", "R-018", "2026-03-20/USD/KZT")),
        ),
        Issue(
            "iss_tiny_alloc",
            "payment_allocation",
            ("D-P3", "D-I6"),
            1_000_000,
            "KZT",
            0,
            (("D-P3", "D-I6", "R-019", "R-020", "R-021"),),
        ),
        Issue(
            "iss_tiny_cutoff",
            "period_cutoff",
            ("D-I5",),
            3_000_000,
            "KZT",
            0,
            (("D-I5", "R-011"), ("D-I5", "R-012")),
        ),
    )
    provenance = {
        "generator": "fixtures.tiny",
        "note": "hand-built fixture, see module docstring",
        "injections": [
            {
                "issue_id": "iss_tiny_dup",
                "change": "document D-I4 added as a copy of D-I3, rows R-007/R-008 posted",
            },
            {
                "issue_id": "iss_tiny_fx",
                "change": "R-017/R-018 use the 2026-03-19 rate 504.50 instead of 504.75",
                "before": 60_570_000,
                "after": 60_540_000,
            },
            {
                "issue_id": "iss_tiny_alloc",
                "change": "R-020 credit reduced from 2500000 to 1500000, R-021 added",
                "before": 2_500_000,
                "after": 1_500_000,
            },
            {
                "issue_id": "iss_tiny_cutoff",
                "change": "R-011/R-012 timestamp moved from 2026-03-31T20:00:00+05:00",
                "before": "2026-03-31T20:00:00+05:00",
                "after": "2026-04-01T09:00:00+05:00",
            },
        ],
    }
    return Episode(
        public=PublicEpisode(
            manifest=manifest,
            policy=policy,
            ledger=ledger,
            invoices=invoices,
            payments=payments,
            fx=fx,
        ),
        private=PrivateEpisode(truth=Truth(EPISODE_ID, issues), provenance=provenance),
    )
