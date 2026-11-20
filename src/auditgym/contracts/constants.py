"""Limits and enumerations fixed by the case (docs/REQUIREMENTS.md, sections 03-05)."""

from decimal import Decimal

# Episode contract (E04, E05, E09).
MAX_STEPS = 100
PAGE_LIMIT_MAX = 50
TABLES = ("ledger", "invoices", "payments", "fx")
ACTIONS = ("list_records", "get_record", "get_document", "calculator", "submit")

# Submission limits (S02).
MAX_FINDINGS = 30
MAX_EVIDENCE_IDS = 10
MAX_EXPLANATION_CHARS = 1000

# Episode size (D01).
MAX_ERRORS_PER_EPISODE = 12

# Calculator limits (E06).
CALCULATOR_MAX_CHARS = 256
CALCULATOR_MAX_DIGITS = 40

# Reward (S08): clip(2*F1 - 1 - STEP_COST*inspection_steps, -1, 1).
STEP_COST = Decimal("0.002")

# Issue types (G01) and interacting scenarios (G02). See docs/DOMAIN.md, sections 6-7.
ISSUE_TYPES = ("duplicate_invoice", "payment_allocation", "fx_conversion", "period_cutoff")
INTERACTION_SCENARIOS = ("fx_partial_at_boundary", "duplicate_misallocated")
INTERACTION_ISSUE_COUNTS = {"fx_partial_at_boundary": 3, "duplicate_misallocated": 2}

# Distractors (G03). See docs/DOMAIN.md, section 8.
DISTRACTOR_KINDS = (
    "split_payment",  # X1
    "multi_invoice_payment",  # X2
    "partial_open",  # X3
    "credit_note_refund",  # X4
    "subscription",  # X5
    "same_amount_orders",  # X6
    "unapplied_payment",  # X7
    "boundary_offset",  # X8
    "fx_spike",  # X9
    "posting_lag",  # X10
)

# Chart of accounts roles. Codes per episode are in policy.json["accounts"].
ACCOUNT_ROLES = ("CASH", "AR", "REVENUE", "UNAPPLIED", "FX_CLEARING", "EXPENSE")

# Policy option values.
FX_RATE_DATE_RULES = ("payment_local_date", "payment_local_date_minus_1", "payment_utc_date")
ROUNDING_RULES = ("half_even", "half_up")
ALLOCATION_RULE = "listed_order_up_to_open_balance"

# CSV column order (D04, D07).
LEDGER_COLUMNS = (
    "row_id",
    "timestamp",
    "account",
    "debit_minor",
    "credit_minor",
    "currency",
    "entity_id",
    "invoice_id",
    "payment_id",
)
FX_COLUMNS = ("date", "source_currency", "target_currency", "rate")

# Public and private file names (D02-D10).
PUBLIC_FILES = (
    "manifest.json",
    "policy.json",
    "ledger.csv",
    "invoices.jsonl",
    "payments.jsonl",
    "fx.csv",
)
PRIVATE_FILES = ("truth.json", "provenance.json")
