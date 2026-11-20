"""Hand-built episodes for the solver tests.

Every amount below is written out by hand (conversions were worked out with plain Decimal
arithmetic, not with the solver). Scenarios are additive patches over a corrected copy of the
tiny fixture, each with its own ID prefix, so they can be applied alone or all together.

USD->KZT rate on day d (tiny fixture): 500.00 + 0.25 * (d - 2026-03-01).
  03-14 503.25 | 03-19 504.50 | 03-20 504.75 | 03-21 505.00 | 03-22 505.25 | 03-23 505.50
  03-31 507.50 | 04-01 507.75
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from decimal import Decimal

from auditgym.contracts.meta import make_manifest, make_policy
from auditgym.contracts.types import FxRate, Invoice, LedgerRow, LineItem, Payment, PublicEpisode
from auditgym.fixtures.tiny import ACCOUNTS, build_tiny_episode

Expected = tuple[str, tuple[str, ...], int, str]


def ts(day: str, hour: int = 10, minute: int = 0, offset: str = "+05:00") -> str:
    """ts("03-14", 9) -> '2026-03-14T09:00:00+05:00'."""
    return f"2026-{day}T{hour:02d}:{minute:02d}:00{offset}"


# ---------------------------------------------------------------------------------- records
def invoice(
    doc_id: str,
    invoice_id: str,
    issued: str,
    cur: str,
    total: int,
    order_ref: str,
    *,
    entity: str = "E-X",
    service_period: str | None = None,
    desc: str = "Goods",
) -> Invoice:
    item = LineItem(desc, 1, total, total, order_ref, service_period)
    return Invoice(doc_id, invoice_id, entity, issued, issued, cur, total, (item,))


def payment(
    doc_id: str, payment_id: str, when: str, cur: str, amount: int, refs: tuple[str, ...] | None
) -> Payment:
    return Payment(doc_id, payment_id, when, cur, amount, refs)


def _sides(net_debit: int) -> tuple[int, int]:
    return (net_debit, 0) if net_debit > 0 else (0, -net_debit)


class Rows:
    """Ledger row factory with explicit amounts. Row IDs: <prefix>01, <prefix>02, ..."""

    def __init__(self, prefix: str, base: str = "KZT", accounts: dict[str, str] = ACCOUNTS):
        self.prefix, self.base, self.accounts, self.n = prefix, base, accounts, 0

    def row(self, when, role, net_debit, cur, *, entity="E-X", inv=None, pay=None) -> LedgerRow:
        self.n += 1
        dr, cr = _sides(net_debit)
        return LedgerRow(
            f"{self.prefix}{self.n:02d}", when, self.accounts[role], dr, cr, cur, entity, inv, pay
        )

    def invoice(self, inv: Invoice, when: str | None = None, amount: int | None = None):
        """Dr AR T / Cr REVENUE T (credit note: the other way round)."""
        when = when or inv.issued_at
        total = inv.total_minor if amount is None else amount
        kw = {"entity": inv.entity_id, "inv": inv.invoice_id}
        return [
            self.row(when, "AR", total, inv.currency, **kw),
            self.row(when, "REVENUE", -total, inv.currency, **kw),
        ]

    def payment(
        self,
        p: Payment,
        *,
        ar: dict[str, int] | None = None,
        unapplied: int = 0,
        cash: int | None = None,
        when: str | None = None,
    ):
        """Book a payment. `ar` and `unapplied` are signed applied amounts (credit when the
        payment is positive); `cash` is the booked base amount of a foreign payment."""
        when = when or p.timestamp
        kw = {"entity": "E-X", "pay": p.payment_id}
        cur = p.currency
        out = []
        if cur == self.base:
            out.append(self.row(when, "CASH", p.amount_minor, cur, **kw))
        else:
            out.append(self.row(when, "FX_CLEARING", p.amount_minor, cur, **kw))
        for invoice_id, amount in (ar or {}).items():
            out.append(self.row(when, "AR", -amount, cur, inv=invoice_id, **kw))
        if unapplied:
            out.append(self.row(when, "UNAPPLIED", -unapplied, cur, **kw))
        if cur != self.base:
            assert cash is not None
            out.append(self.row(when, "CASH", cash, self.base, **kw))
            out.append(self.row(when, "FX_CLEARING", -cash, self.base, **kw))
        return out


def daily_rates(src: str, tgt: str, rate_of_day, first=date(2026, 2, 25), last=date(2026, 4, 5)):
    out = []
    day = first
    while day <= last:
        out.append(FxRate(day.isoformat(), src, tgt, rate_of_day(day)))
        day += timedelta(days=1)
    return tuple(out)


# ---------------------------------------------------------------------------------- episodes
def assemble(
    *,
    invoices,
    payments,
    ledger,
    fx,
    base="KZT",
    currencies=("KZT", "USD"),
    tz="+05:00",
    period=("2026-03-01T00:00:00+05:00", "2026-04-01T00:00:00+05:00"),
    accounts=ACCOUNTS,
    rate_date="payment_local_date",
    rounding="half_even",
    episode_id="ep_VERIFY00001",
) -> PublicEpisode:
    policy = make_policy(
        base_currency=base,
        timezone=tz,
        period_start=period[0],
        period_end=period[1],
        accounts=dict(accounts),
        fx_rate_date=rate_date,
        rounding=rounding,
    )
    manifest = make_manifest(
        episode_id=episode_id,
        company="Verify Fixture Trading LLP",
        base_currency=base,
        currencies=list(currencies),
        period_start=period[0],
        period_end=period[1],
        timezone=tz,
        rounding=rounding,
        counts={
            "ledger": len(ledger),
            "invoices": len(invoices),
            "payments": len(payments),
            "fx": len(fx),
        },
    )
    return PublicEpisode(
        manifest=manifest,
        policy=policy,
        ledger=tuple(ledger),
        invoices=tuple(invoices),
        payments=tuple(payments),
        fx=tuple(fx),
    )


def rebuild(public: PublicEpisode, **overrides) -> PublicEpisode:
    """Re-assemble an episode with some tables or policy options replaced."""
    pol, man = public.policy, public.manifest
    args = {
        "invoices": public.invoices,
        "payments": public.payments,
        "ledger": public.ledger,
        "fx": public.fx,
        "base": pol["base_currency"],
        "currencies": tuple(sorted(man["currencies"])),
        "tz": pol["timezone"],
        "period": (pol["period"]["start"], pol["period"]["end"]),
        "accounts": pol["accounts"],
        "rate_date": pol["fx"]["rate_date"],
        "rounding": pol["fx"]["rounding"],
        "episode_id": man["episode_id"],
    }
    args.update(overrides)
    return assemble(**args)


def tiny_public() -> PublicEpisode:
    return build_tiny_episode().public


def clean_tiny() -> PublicEpisode:
    """The tiny fixture with its four injected errors undone."""
    pub = tiny_public()
    ledger = []
    for r in pub.ledger:
        if r.row_id in ("R-007", "R-008", "R-021"):  # duplicate posting, misbooked remainder
            continue
        if r.row_id in ("R-011", "R-012"):  # back to the document date
            r = replace(r, timestamp="2026-03-31T20:00:00+05:00")
        if r.row_id == "R-017":  # 120000 * 504.75
            r = replace(r, debit_minor=60_570_000)
        if r.row_id == "R-018":
            r = replace(r, credit_minor=60_570_000)
        if r.row_id == "R-020":
            r = replace(r, credit_minor=2_500_000)
        ledger.append(r)
    invoices = [i for i in pub.invoices if i.doc_id != "D-I4"]
    return rebuild(pub, invoices=invoices, ledger=ledger)


def clean_tiny_base_only() -> PublicEpisode:
    """Corrected tiny without its USD invoice and payment, for policy-option cases: the
    tiny USD payment is booked for the local-date rule only."""
    pub = clean_tiny()
    drop = {"D-I2", "D-P2", "R-003", "R-004", "R-015", "R-016", "R-017", "R-018"}
    return rebuild(
        pub,
        invoices=[i for i in pub.invoices if i.doc_id not in drop],
        payments=[p for p in pub.payments if p.doc_id not in drop],
        ledger=[r for r in pub.ledger if r.row_id not in drop],
    )


@dataclass
class Patch:
    """Records added to an episode, rates added or replaced, and the issues they carry."""

    name: str
    invoices: list[Invoice] = field(default_factory=list)
    payments: list[Payment] = field(default_factory=list)
    rows: list[LedgerRow] = field(default_factory=list)
    rates: tuple[FxRate, ...] = ()
    currencies: tuple[str, ...] = ()
    expected: tuple[Expected, ...] = ()


def apply(public: PublicEpisode, *patches: Patch, **overrides) -> PublicEpisode:
    rates = {f.rate_id: f for f in public.fx}
    invoices, payments, ledger = list(public.invoices), list(public.payments), list(public.ledger)
    currencies = set(public.manifest["currencies"])
    for p in patches:
        invoices += p.invoices
        payments += p.payments
        ledger += p.rows
        for f in p.rates:
            rates[f.rate_id] = f
        currencies |= set(p.currencies)
    return rebuild(
        public,
        invoices=invoices,
        payments=payments,
        ledger=ledger,
        fx=tuple(sorted(rates.values(), key=lambda f: f.rate_id)),
        currencies=tuple(sorted(currencies)),
        **overrides,
    )


def expected_of(*patches: Patch) -> set[Expected]:
    return {e for p in patches for e in p.expected}


def exp(issue_type: str, affected, amount: int, currency: str) -> Expected:
    return (issue_type, tuple(sorted(affected)), amount, currency)


# ---------------------------------------------------------------------------------- distractors
def x1_split_payment() -> Patch:
    r = Rows("RS-")
    inv = invoice("D-S1", "INV-S1", ts("03-04"), "KZT", 900_000, "ORD-S1")
    p1 = payment("D-S2", "PAY-S2", ts("03-06"), "KZT", 300_000, ("INV-S1",))
    p2 = payment("D-S3", "PAY-S3", ts("03-07"), "KZT", 400_000, ("INV-S1",))
    p3 = payment("D-S4", "PAY-S4", ts("03-08"), "KZT", 200_000, ("INV-S1",))
    rows = r.invoice(inv)
    rows += r.payment(p1, ar={"INV-S1": 300_000})
    rows += r.payment(p2, ar={"INV-S1": 400_000})
    rows += r.payment(p3, ar={"INV-S1": 200_000})
    return Patch("X1", [inv], [p1, p2, p3], rows)


def x2_multi_invoice_payment() -> Patch:
    r = Rows("RM-")
    i1 = invoice("D-M1", "INV-M1", ts("03-05"), "KZT", 400_000, "ORD-M1")
    i2 = invoice("D-M2", "INV-M2", ts("03-05", 11), "KZT", 250_000, "ORD-M2")
    p = payment("D-M3", "PAY-M3", ts("03-08"), "KZT", 650_000, ("INV-M2", "INV-M1"))
    rows = r.invoice(i1) + r.invoice(i2) + r.payment(p, ar={"INV-M2": 250_000, "INV-M1": 400_000})
    return Patch("X2", [i1, i2], [p], rows)


def x3_partial_open() -> Patch:
    r = Rows("RPO-")
    inv = invoice("D-PO1", "INV-PO1", ts("03-06"), "KZT", 1_000_000, "ORD-PO1")
    p = payment("D-PO2", "PAY-PO2", ts("03-09"), "KZT", 400_000, ("INV-PO1",))
    return Patch("X3", [inv], [p], r.invoice(inv) + r.payment(p, ar={"INV-PO1": 400_000}))


def x4_credit_note_refund() -> Patch:
    r = Rows("RC-")
    i1 = invoice("D-C1", "INV-C1", ts("03-05"), "KZT", 800_000, "ORD-C1")
    p1 = payment("D-C2", "PAY-C2", ts("03-07"), "KZT", 800_000, ("INV-C1",))
    cn = invoice("D-C3", "INV-C3", ts("03-10"), "KZT", -300_000, "ORD-C3", desc="Return")
    refund = payment("D-C4", "PAY-C4", ts("03-12"), "KZT", -300_000, ("INV-C3",))
    # Foreign credit note and refund: -10000 USD at 503.25 (03-14) = -5032500 KZT.
    i2 = invoice("D-C7", "INV-C7", ts("03-03"), "USD", 30_000, "ORD-C7")
    cn2 = invoice("D-C5", "INV-C5", ts("03-13"), "USD", -10_000, "ORD-C5", desc="Return")
    refund2 = payment("D-C6", "PAY-C6", ts("03-14"), "USD", -10_000, ("INV-C5",))
    rows = r.invoice(i1) + r.payment(p1, ar={"INV-C1": 800_000})
    rows += r.invoice(cn) + r.payment(refund, ar={"INV-C3": -300_000})
    rows += r.invoice(i2) + r.invoice(cn2)
    rows += r.payment(refund2, ar={"INV-C5": -10_000}, cash=-5_032_500)
    return Patch("X4", [i1, cn, i2, cn2], [p1, refund, refund2], rows)


def x5_subscription() -> Patch:
    r = Rows("RSB-")
    invs = [
        invoice(
            f"D-SB{k}",
            f"INV-SB{k}",
            when,
            "KZT",
            150_000,
            f"ORD-SB{k}",
            entity="E-SUB",
            service_period=sp,
            desc="Monthly subscription",
        )
        for k, when, sp in (
            (1, ts("02-15"), "2026-02"),
            (2, ts("03-15"), "2026-03"),
            (3, ts("04-15"), "2026-04"),
        )
    ]
    p = payment("D-SB4", "PAY-SB4", ts("03-20"), "KZT", 150_000, ("INV-SB2",))
    rows = [row for inv in invs for row in r.invoice(inv)] + r.payment(p, ar={"INV-SB2": 150_000})
    return Patch("X5", invs, [p], rows)


def x6_same_amount_orders() -> Patch:
    r = Rows("RSA-")
    i1 = invoice("D-SA1", "INV-SA1", ts("03-17", 10), "KZT", 275_000, "ORD-SA1", entity="E-SA")
    i2 = invoice("D-SA2", "INV-SA2", ts("03-17", 15), "KZT", 275_000, "ORD-SA2", entity="E-SA")
    return Patch("X6", [i1, i2], [], r.invoice(i1) + r.invoice(i2))


def x7_unapplied_and_overpayment() -> Patch:
    r = Rows("RU-")
    p1 = payment("D-U1", "PAY-U1", ts("03-21"), "KZT", 120_000, None)
    inv = invoice("D-U2", "INV-U2", ts("03-02"), "KZT", 500_000, "ORD-U2")
    p2 = payment("D-U3", "PAY-U3", ts("03-03"), "KZT", 700_000, ("INV-U2",))
    rows = r.payment(p1, unapplied=120_000) + r.invoice(inv)
    rows += r.payment(p2, ar={"INV-U2": 500_000}, unapplied=200_000)
    return Patch("X7", [inv], [p1, p2], rows)


def x8_boundary_offset() -> Patch:
    r = Rows("RB-")
    # 2026-03-31T21:30Z is 2026-04-01 02:30 local: April, after the period.
    i1 = invoice("D-B1", "INV-B1", "2026-03-31T21:30:00+00:00", "KZT", 640_000, "ORD-B1")
    # 2026-02-28T20:00Z is 2026-03-01 01:00 local: March, in the period.
    i2 = invoice("D-B2", "INV-B2", "2026-02-28T20:00:00+00:00", "KZT", 410_000, "ORD-B2")
    # 2026-03-31T19:30Z is 2026-04-01 00:30 local.
    p1 = payment("D-B3", "PAY-B3", "2026-03-31T19:30:00+00:00", "KZT", 410_000, ("INV-B2",))
    # Foreign: 2026-03-31T20:00Z is 2026-04-01 01:00 local; local rate 04-01 507.75.
    i3 = invoice("D-B4", "INV-B4", ts("03-20"), "USD", 10_000, "ORD-B4")
    p2 = payment("D-B5", "PAY-B5", "2026-03-31T20:00:00+00:00", "USD", 10_000, ("INV-B4",))
    rows = r.invoice(i1, "2026-04-01T02:30:00+05:00") + r.invoice(i2, "2026-03-01T09:00:00+05:00")
    rows += r.payment(p1, ar={"INV-B2": 410_000}, when="2026-04-01T00:30:00+05:00")
    rows += r.invoice(i3)
    rows += r.payment(p2, ar={"INV-B4": 10_000}, cash=5_077_500, when="2026-04-01T01:00:00+05:00")
    return Patch("X8", [i1, i2, i3], [p1, p2], rows)


def x9_fx_spike() -> Patch:
    r = Rows("RFS-")
    spike = FxRate("2026-03-16", "USD", "KZT", "651.30")
    inv = invoice("D-FS1", "INV-FS1", ts("03-10"), "USD", 50_000, "ORD-FS1")
    p = payment("D-FS2", "PAY-FS2", ts("03-16", 12), "USD", 50_000, ("INV-FS1",))
    rows = r.invoice(inv) + r.payment(p, ar={"INV-FS1": 50_000}, cash=32_565_000)  # 50000*651.30
    return Patch("X9", [inv], [p], rows, rates=(spike,))


def x10_posting_lag() -> Patch:
    r = Rows("RL-")
    inv = invoice("D-L1", "INV-L1", ts("03-02", 9), "KZT", 333_000, "ORD-L1")
    p = payment("D-L2", "PAY-L2", ts("03-05"), "KZT", 333_000, ("INV-L1",))
    rows = r.invoice(inv, ts("03-20", 9)) + r.payment(
        p, ar={"INV-L1": 333_000}, when=ts("03-09", 12)
    )
    return Patch("X10", [inv], [p], rows)


def jpy_payment() -> Patch:
    r = Rows("RJ-")
    rates = daily_rates("JPY", "KZT", lambda d: "3.4150")
    inv = invoice("D-J1", "INV-J1", ts("03-11"), "JPY", 125_000, "ORD-J1")
    p = payment("D-J2", "PAY-J2", ts("03-12"), "JPY", 125_000, ("INV-J1",))
    # 125000 * 10^(2-0) * 3.4150 = 42687500
    rows = r.invoice(inv) + r.payment(p, ar={"INV-J1": 125_000}, cash=42_687_500)
    return Patch("JPY", [inv], [p], rows, rates=rates, currencies=("JPY",))


def kwd_payment() -> Patch:
    r = Rows("RK-")
    rates = daily_rates("KWD", "KZT", lambda d: "1625.40")
    inv = invoice("D-K1", "INV-K1", ts("03-13"), "KWD", 1_234_567, "ORD-K1")
    p = payment("D-K2", "PAY-K2", ts("03-14"), "KWD", 1_234_567, ("INV-K1",))
    # 1234567 * 10^(2-3) * 1625.40 = 200666520.18 -> 200666520
    rows = r.invoice(inv) + r.payment(p, ar={"INV-K1": 1_234_567}, cash=200_666_520)
    return Patch("KWD", [inv], [p], rows, rates=rates, currencies=("KWD",))


DISTRACTORS = (
    x1_split_payment,
    x2_multi_invoice_payment,
    x3_partial_open,
    x4_credit_note_refund,
    x5_subscription,
    x6_same_amount_orders,
    x7_unapplied_and_overpayment,
    x8_boundary_offset,
    x9_fx_spike,
    x10_posting_lag,
    jpy_payment,
    kwd_payment,
)


# ---------------------------------------------------------------------------------- issues
def dup_same_invoice_id() -> Patch:
    r = Rows("RDS-")
    i1 = invoice("D-DS1", "INV-DS1", ts("03-04"), "KZT", 420_000, "ORD-DS1")
    i2 = replace(i1, doc_id="D-DS2", issued_at=ts("03-06", 12))
    return Patch(
        "dup_same_id",
        [i1, i2],
        [],
        r.invoice(i1) + r.invoice(i2),
        expected=(exp("duplicate_invoice", ["D-DS1", "D-DS2"], 420_000, "KZT"),),
    )


def dup_same_order_ref() -> Patch:
    r = Rows("RDO-")
    i1 = invoice("D-DO1", "INV-DO1", ts("03-05"), "KZT", 515_000, "ORD-DO1")
    i2 = replace(i1, doc_id="D-DO2", invoice_id="INV-DO2", issued_at=ts("03-07"))
    return Patch(
        "dup_order_ref",
        [i1, i2],
        [],
        r.invoice(i1) + r.invoice(i2),
        expected=(exp("duplicate_invoice", ["D-DO1", "D-DO2"], 515_000, "KZT"),),
    )


def dup_triple() -> Patch:
    r = Rows("RDT-")
    i1 = invoice("D-DT1", "INV-DT1", ts("03-02"), "USD", 70_000, "ORD-DT1")
    i2 = replace(i1, doc_id="D-DT2", issued_at=ts("03-03"))  # same invoice_id
    i3 = replace(i1, doc_id="D-DT3", invoice_id="INV-DT3", issued_at=ts("03-04"))  # same order
    return Patch(
        "dup_triple",
        [i1, i2, i3],
        [],
        r.invoice(i1) + r.invoice(i2) + r.invoice(i3),
        expected=(exp("duplicate_invoice", ["D-DT1", "D-DT2", "D-DT3"], 140_000, "USD"),),
    )


def alloc_none() -> Patch:
    r = Rows("RAN-")
    inv = invoice("D-AN1", "INV-AN1", ts("03-03"), "KZT", 610_000, "ORD-AN1")
    p = payment("D-AN2", "PAY-AN2", ts("03-09"), "KZT", 610_000, ("INV-AN1",))
    return Patch(
        "alloc_none",
        [inv],
        [p],
        r.invoice(inv) + r.payment(p, unapplied=610_000),
        expected=(exp("payment_allocation", ["D-AN2", "D-AN1"], 610_000, "KZT"),),
    )


def alloc_partial_after_earlier_payment() -> Patch:
    r = Rows("RAP-")
    inv = invoice("D-AP1", "INV-AP1", ts("03-02"), "KZT", 900_000, "ORD-AP1")
    p1 = payment("D-AP2", "PAY-AP2", ts("03-04"), "KZT", 300_000, ("INV-AP1",))
    p2 = payment("D-AP3", "PAY-AP3", ts("03-06"), "KZT", 700_000, ("INV-AP1",))
    # A later payment on the same invoice: by documents the invoice is closed after PAY-AP3,
    # so all of it is unapplied. It is not evidence for PAY-AP3.
    p3 = payment("D-AP4", "PAY-AP4", ts("03-08"), "KZT", 50_000, ("INV-AP1",))
    # Expected for PAY-AP3: 600000 to INV-AP1 (its open balance), 100000 unapplied.
    rows = r.invoice(inv) + r.payment(p1, ar={"INV-AP1": 300_000})
    rows += r.payment(p2, ar={"INV-AP1": 500_000}, unapplied=200_000)
    rows += r.payment(p3, unapplied=50_000)
    return Patch(
        "alloc_partial",
        [inv],
        [p1, p2, p3],
        rows,
        expected=(exp("payment_allocation", ["D-AP3", "D-AP1"], 100_000, "KZT"),),
    )


def alloc_other_invoice() -> Patch:
    r = Rows("RAO-")
    i1 = invoice("D-AO1", "INV-AO1", ts("03-03"), "KZT", 380_000, "ORD-AO1")
    i2 = invoice("D-AO2", "INV-AO2", ts("03-04"), "KZT", 380_000, "ORD-AO2")
    p = payment("D-AO3", "PAY-AO3", ts("03-10"), "KZT", 380_000, ("INV-AO1",))
    return Patch(
        "alloc_other",
        [i1, i2],
        [p],
        r.invoice(i1) + r.invoice(i2) + r.payment(p, ar={"INV-AO2": 380_000}),
        expected=(exp("payment_allocation", ["D-AO3", "D-AO1"], 380_000, "KZT"),),
    )


def alloc_listed_order() -> Patch:
    r = Rows("RAL-")
    i1 = invoice("D-AL1", "INV-AL1", ts("03-03"), "KZT", 400_000, "ORD-AL1")
    i2 = invoice("D-AL2", "INV-AL2", ts("03-03", 11), "KZT", 250_000, "ORD-AL2")
    p = payment("D-AL3", "PAY-AL3", ts("03-11"), "KZT", 500_000, ("INV-AL2", "INV-AL1"))
    # Rule: listed order, 250000 to INV-AL2 first, then 250000 to INV-AL1.
    # Booked in ID order instead: 400000 to INV-AL1, 100000 to INV-AL2.
    rows = r.invoice(i1) + r.invoice(i2)
    rows += r.payment(p, ar={"INV-AL1": 400_000, "INV-AL2": 100_000})
    return Patch(
        "alloc_order",
        [i1, i2],
        [p],
        rows,
        expected=(exp("payment_allocation", ["D-AL3", "D-AL2"], 150_000, "KZT"),),
    )


def alloc_second_ref() -> Patch:
    """Shortfall on the second invoice of a two-invoice payment. The DOMAIN evidence set
    lacks the first invoice's document, on which the expected amount depends."""
    r = Rows("RAG-")
    i1 = invoice("D-AG1", "INV-AG1", ts("03-03"), "KZT", 400_000, "ORD-AG1")
    i2 = invoice("D-AG2", "INV-AG2", ts("03-03", 12), "KZT", 250_000, "ORD-AG2")
    p = payment("D-AG3", "PAY-AG3", ts("03-12"), "KZT", 650_000, ("INV-AG1", "INV-AG2"))
    rows = r.invoice(i1) + r.invoice(i2)
    rows += r.payment(p, ar={"INV-AG1": 400_000, "INV-AG2": 100_000}, unapplied=150_000)
    return Patch(
        "alloc_second_ref",
        [i1, i2],
        [p],
        rows,
        expected=(exp("payment_allocation", ["D-AG3", "D-AG2"], 150_000, "KZT"),),
    )


def alloc_refund() -> Patch:
    r = Rows("RAR-")
    cn = invoice("D-AR1", "INV-AR1", ts("03-08"), "KZT", -200_000, "ORD-AR1", desc="Return")
    p = payment("D-AR2", "PAY-AR2", ts("03-10"), "KZT", -200_000, ("INV-AR1",))
    rows = r.invoice(cn) + r.payment(p, ar={"INV-AR1": -50_000}, unapplied=-150_000)
    return Patch(
        "alloc_refund",
        [cn],
        [p],
        rows,
        expected=(exp("payment_allocation", ["D-AR2", "D-AR1"], 150_000, "KZT"),),
    )


def fx_wrong_date() -> Patch:
    r = Rows("RFD-")
    inv = invoice("D-FD1", "INV-FD1", ts("03-05"), "USD", 80_000, "ORD-FD1")
    p = payment("D-FD2", "PAY-FD2", ts("03-22"), "USD", 80_000, ("INV-FD1",))
    # Policy: 03-22 rate 505.25 -> 40420000. Booked with 03-21 rate 505.00 -> 40400000.
    rows = r.invoice(inv) + r.payment(p, ar={"INV-FD1": 80_000}, cash=40_400_000)
    return Patch(
        "fx_date",
        [inv],
        [p],
        rows,
        expected=(exp("fx_conversion", ["D-FD2"], 20_000, "KZT"),),
    )


def fx_wrong_pair() -> Patch:
    r = Rows("RFP-")
    rates = daily_rates("EUR", "KZT", lambda d: "548.20")
    inv = invoice("D-FP1", "INV-FP1", ts("03-06"), "USD", 50_000, "ORD-FP1")
    p = payment("D-FP2", "PAY-FP2", ts("03-23"), "USD", 50_000, ("INV-FP1",))
    # Policy: USD 03-23 505.50 -> 25275000. Booked with EUR/KZT 548.20 -> 27410000.
    rows = r.invoice(inv) + r.payment(p, ar={"INV-FP1": 50_000}, cash=27_410_000)
    return Patch(
        "fx_pair",
        [inv],
        [p],
        rows,
        rates=rates,
        currencies=("EUR",),
        expected=(exp("fx_conversion", ["D-FP2"], 2_135_000, "KZT"),),
    )


def cutoff_invoice_late() -> Patch:
    r = Rows("RCL-")
    inv = invoice("D-CL1", "INV-CL1", ts("03-31", 18), "KZT", 730_000, "ORD-CL1")
    return Patch(
        "cutoff_late",
        [inv],
        [],
        r.invoice(inv, ts("04-01", 10)),
        expected=(exp("period_cutoff", ["D-CL1"], 730_000, "KZT"),),
    )


def cutoff_invoice_early() -> Patch:
    r = Rows("RCE-")
    inv = invoice("D-CE1", "INV-CE1", ts("04-01", 9), "USD", 45_000, "ORD-CE1")
    return Patch(
        "cutoff_early",
        [inv],
        [],
        r.invoice(inv, ts("03-31", 17)),
        expected=(exp("period_cutoff", ["D-CE1"], 45_000, "USD"),),
    )


def cutoff_from_before_period() -> Patch:
    r = Rows("RCB-")
    inv = invoice("D-CB1", "INV-CB1", ts("02-27", 12), "KZT", 260_000, "ORD-CB1")
    return Patch(
        "cutoff_before",
        [inv],
        [],
        r.invoice(inv, ts("03-02", 9)),
        expected=(exp("period_cutoff", ["D-CB1"], 260_000, "KZT"),),
    )


def cutoff_payment() -> Patch:
    r = Rows("RCP-")
    inv = invoice("D-CP1", "INV-CP1", ts("03-20"), "KZT", 150_000, "ORD-CP1")
    p = payment("D-CP2", "PAY-CP2", ts("03-31", 23), "KZT", 150_000, ("INV-CP1",))
    rows = r.invoice(inv) + r.payment(p, ar={"INV-CP1": 150_000}, when=ts("04-01", 8))
    return Patch(
        "cutoff_payment",
        [inv],
        [p],
        rows,
        expected=(exp("period_cutoff", ["D-CP2"], 150_000, "KZT"),),
    )


def fx_partial_at_boundary() -> Patch:
    """G02: posted after the boundary, converted at the posting-date rate, partly applied."""
    r = Rows("RIB-")
    inv = invoice("D-IB1", "INV-IB1", ts("03-20"), "USD", 200_000, "ORD-IB1")
    p = payment("D-IB2", "PAY-IB2", ts("03-31", 22), "USD", 200_000, ("INV-IB1",))
    # Policy: 03-31 rate 507.50 -> 101500000. Booked at 04-01 rate 507.75 -> 101550000.
    rows = r.invoice(inv) + r.payment(
        p, ar={"INV-IB1": 120_000}, unapplied=80_000, cash=101_550_000, when=ts("04-01", 10)
    )
    return Patch(
        "fx_partial_at_boundary",
        [inv],
        [p],
        rows,
        expected=(
            exp("period_cutoff", ["D-IB2"], 200_000, "USD"),
            exp("fx_conversion", ["D-IB2"], 50_000, "KZT"),
            exp("payment_allocation", ["D-IB2", "D-IB1"], 80_000, "USD"),
        ),
    )


def duplicate_misallocated() -> Patch:
    """G02: invoice copied under another invoice_id; payment references the original but is
    applied to the copy."""
    r = Rows("RIM-")
    i1 = invoice("D-IM1", "INV-IM1", ts("03-06"), "KZT", 480_000, "ORD-IM1")
    i2 = replace(i1, doc_id="D-IM2", invoice_id="INV-IM2", issued_at=ts("03-08"))
    p = payment("D-IM3", "PAY-IM3", ts("03-12"), "KZT", 480_000, ("INV-IM1",))
    rows = r.invoice(i1) + r.invoice(i2) + r.payment(p, ar={"INV-IM2": 480_000})
    return Patch(
        "duplicate_misallocated",
        [i1, i2],
        [p],
        rows,
        expected=(
            exp("duplicate_invoice", ["D-IM1", "D-IM2"], 480_000, "KZT"),
            exp("payment_allocation", ["D-IM3", "D-IM1"], 480_000, "KZT"),
        ),
    )


ISSUES = (
    dup_same_invoice_id,
    dup_same_order_ref,
    dup_triple,
    alloc_none,
    alloc_partial_after_earlier_payment,
    alloc_other_invoice,
    alloc_listed_order,
    alloc_second_ref,
    alloc_refund,
    fx_wrong_date,
    fx_wrong_pair,
    cutoff_invoice_late,
    cutoff_invoice_early,
    cutoff_from_before_period,
    cutoff_payment,
)
INTERACTIONS = (fx_partial_at_boundary, duplicate_misallocated)

# Issues whose DOMAIN evidence set is not independently sufficient (see alloc_second_ref).
EVIDENCE_GAP_KEYS = {("payment_allocation", ("D-AG2", "D-AG3"))}


def kitchen_sink() -> tuple[PublicEpisode, set[Expected]]:
    """Clean tiny + every distractor + every issue scenario + both interactions."""
    patches = [f() for f in DISTRACTORS + ISSUES + INTERACTIONS]
    return apply(clean_tiny(), *patches), expected_of(*patches)


# ---------------------------------------------------------------------------------- policy cases
def rate_date_case(cash: int, rule: str) -> PublicEpisode:
    """+05:00. Payment at 2026-03-10T21:00Z = 03-11 02:00 local.
    local -> 03-11 (502.50) 10050000; local-1 -> 03-10 (502.25) 10045000; utc -> 03-10."""
    r = Rows("RRD-")
    inv = invoice("D-RD1", "INV-RD1", ts("03-02"), "USD", 20_000, "ORD-RD1")
    p = payment("D-RD2", "PAY-RD2", "2026-03-10T21:00:00+00:00", "USD", 20_000, ("INV-RD1",))
    rows = r.invoice(inv) + r.payment(
        p, ar={"INV-RD1": 20_000}, cash=cash, when="2026-03-11T02:00:00+05:00"
    )
    return apply(clean_tiny_base_only(), Patch("rd", [inv], [p], rows), rate_date=rule)


WEST_ACCOUNTS = {
    "AR": "A-1100",
    "CASH": "A-1000",
    "EXPENSE": "A-6000",
    "FX_CLEARING": "A-1950",
    "REVENUE": "A-4100",
    "UNAPPLIED": "A-2350",
}


def west_case(cash: int, rule: str, rounding: str = "half_even") -> PublicEpisode:
    """Base USD, foreign EUR, timezone -04:00, other account codes.
    EUR->USD on day d: 1.0800 + 0.0010 * (d - 03-01).
    Payment 2026-03-10T21:30-04:00 = 2026-03-11T01:30Z:
      local 03-10 (1.0890) -> 21780; local-1 03-09 (1.0880) -> 21760; utc 03-11 (1.0900) -> 21800.
    """
    origin = date(2026, 3, 1)
    fx = daily_rates(
        "EUR", "USD", lambda d: f"{Decimal('1.0800') + Decimal('0.0010') * (d - origin).days:.4f}"
    )
    r = Rows("W-", base="USD", accounts=WEST_ACCOUNTS)
    off = "-04:00"
    inv = invoice("WD-1", "WINV-1", ts("03-02", offset=off), "EUR", 20_000, "WORD-1")
    p = payment("WD-2", "WPAY-2", ts("03-10", 21, 30, off), "EUR", 20_000, ("WINV-1",))
    usd_inv = invoice("WD-3", "WINV-3", ts("03-03", offset=off), "USD", 99_900, "WORD-3")
    usd_pay = payment("WD-4", "WPAY-4", ts("03-31", 23, offset=off), "USD", 99_900, ("WINV-3",))
    rows = r.invoice(inv) + r.payment(p, ar={"WINV-1": 20_000}, cash=cash)
    rows += r.invoice(usd_inv) + r.payment(usd_pay, ar={"WINV-3": 99_900})
    rows.append(r.row(ts("03-15", offset=off), "EXPENSE", 5_000, "USD"))
    rows.append(r.row(ts("03-15", offset=off), "CASH", -5_000, "USD"))
    return assemble(
        invoices=[inv, usd_inv],
        payments=[p, usd_pay],
        ledger=rows,
        fx=fx,
        base="USD",
        currencies=("EUR", "USD"),
        tz=off,
        period=("2026-03-01T00:00:00-04:00", "2026-04-01T00:00:00-04:00"),
        accounts=WEST_ACCOUNTS,
        rate_date=rule,
        rounding=rounding,
        episode_id="ep_VERIFYWEST1",
    )


def rounding_case(cash: int, refund_cash: int, rounding: str) -> PublicEpisode:
    """USD 10001 at 504.50 (03-19) = 5045504.5: half_even 5045504, half_up 5045505.
    Refund -10001 the same day = -5045504.5: half_even -5045504, half_up -5045505."""
    r = Rows("RRN-")
    inv = invoice("D-RN1", "INV-RN1", ts("03-02"), "USD", 10_001, "ORD-RN1")
    p = payment("D-RN2", "PAY-RN2", ts("03-19", 12), "USD", 10_001, ("INV-RN1",))
    cn = invoice("D-RN3", "INV-RN3", ts("03-15"), "USD", -10_001, "ORD-RN3", desc="Return")
    refund = payment("D-RN4", "PAY-RN4", ts("03-19", 13), "USD", -10_001, ("INV-RN3",))
    rows = r.invoice(inv) + r.payment(p, ar={"INV-RN1": 10_001}, cash=cash)
    rows += r.invoice(cn) + r.payment(refund, ar={"INV-RN3": -10_001}, cash=refund_cash)
    return apply(
        clean_tiny_base_only(), Patch("rn", [inv, cn], [p, refund], rows), rounding=rounding
    )
