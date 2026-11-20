"""Fail-closed errors of the independent solver.

The solver never skips data it does not understand. Malformed records, a policy it cannot
read, or a discrepancy that none of the four issue types explains all raise `VerifyError`
with a fixed code, so a sweep can count failures by kind. Messages name public IDs only.
"""

from __future__ import annotations

# Codes. Fixed list; a sweep groups failures by them.
POLICY_INVALID = "policy_invalid"  # manifest/policy unreadable or inconsistent with each other
COUNTS_MISMATCH = "counts_mismatch"  # manifest counts differ from table sizes
ID_NOT_UNIQUE = "id_not_unique"  # an ID repeats or collides across tables
RECORD_INVALID = "record_invalid"  # a document, row or FX rate breaks its own format or I2
UNKNOWN_REFERENCE = "unknown_reference"  # a reference to an invoice/payment that does not exist
ORDER_AMBIGUOUS = "order_ambiguous"  # equal timestamps where the order changes the answer
FX_RATE_MISSING = "fx_rate_missing"  # no FX row for the date and pair the policy requires
UNPOSTED_DOCUMENT = "unposted_document"  # a document without ledger rows
POSTING_INVALID = "posting_invalid"  # rows of one posting break I1/I3 or the posting shape
DUPLICATE_GROUP_INVALID = "duplicate_group_invalid"  # copies with different totals/currency
CUTOFF_AMBIGUOUS = "cutoff_ambiguous"  # cut-off on an invoice_id that has several documents
UNEXPLAINED_DISCREPANCY = "unexplained_discrepancy"  # differs from expected, fits no issue type
EVIDENCE_TOO_LARGE = "evidence_too_large"  # an evidence set prescribed by DOMAIN exceeds 10 IDs

CODES = (
    POLICY_INVALID,
    COUNTS_MISMATCH,
    ID_NOT_UNIQUE,
    RECORD_INVALID,
    UNKNOWN_REFERENCE,
    ORDER_AMBIGUOUS,
    FX_RATE_MISSING,
    UNPOSTED_DOCUMENT,
    POSTING_INVALID,
    DUPLICATE_GROUP_INVALID,
    CUTOFF_AMBIGUOUS,
    UNEXPLAINED_DISCREPANCY,
    EVIDENCE_TOO_LARGE,
)


class VerifyError(ValueError):
    """Public data is malformed or contradictory; the solver refuses to answer."""

    def __init__(self, code: str, message: str) -> None:
        if code not in CODES:
            raise ValueError(f"unknown VerifyError code {code!r}")
        super().__init__(f"{code}: {message}")
        self.code = code
        self.detail = message
