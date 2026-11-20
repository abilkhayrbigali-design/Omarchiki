"""Shared contracts: record types, limits, error codes, config schema, episode IO.

Only the integrator changes this package (see AGENTS.md). Other packages import from here.
"""

from auditgym.contracts.constants import (
    ACCOUNT_ROLES,
    ACTIONS,
    INTERACTION_ISSUE_COUNTS,
    INTERACTION_SCENARIOS,
    ISSUE_TYPES,
    MAX_ERRORS_PER_EPISODE,
    MAX_EVIDENCE_IDS,
    MAX_EXPLANATION_CHARS,
    MAX_FINDINGS,
    MAX_STEPS,
    PAGE_LIMIT_MAX,
    TABLES,
)
from auditgym.contracts.errors import ErrorCode, error_payload
from auditgym.contracts.types import (
    Episode,
    Finding,
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
from auditgym.contracts.versions import (
    ENV_VERSION,
    GENERATOR_VERSION,
    GRADER_VERSION,
    SCHEMA_VERSION,
)

__all__ = [
    "ACCOUNT_ROLES",
    "ACTIONS",
    "ENV_VERSION",
    "GENERATOR_VERSION",
    "GRADER_VERSION",
    "INTERACTION_ISSUE_COUNTS",
    "INTERACTION_SCENARIOS",
    "ISSUE_TYPES",
    "MAX_ERRORS_PER_EPISODE",
    "MAX_EVIDENCE_IDS",
    "MAX_EXPLANATION_CHARS",
    "MAX_FINDINGS",
    "MAX_STEPS",
    "PAGE_LIMIT_MAX",
    "SCHEMA_VERSION",
    "TABLES",
    "Episode",
    "ErrorCode",
    "Finding",
    "FxRate",
    "Invoice",
    "Issue",
    "LedgerRow",
    "LineItem",
    "Payment",
    "PrivateEpisode",
    "PublicEpisode",
    "Truth",
    "error_payload",
]
