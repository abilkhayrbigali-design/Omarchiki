"""AuditGym: reproducible RL environment for training financial-audit agents."""

from typing import Any

from auditgym.contracts.versions import ENV_VERSION, GENERATOR_VERSION, SCHEMA_VERSION

__all__ = ["ENV_VERSION", "GENERATOR_VERSION", "SCHEMA_VERSION", "make"]


def make(**kwargs: Any) -> Any:
    """Create an environment: auditgym.make(mode="open") -> AuditEnv."""
    from auditgym.env import AuditEnv

    return AuditEnv(**kwargs)
