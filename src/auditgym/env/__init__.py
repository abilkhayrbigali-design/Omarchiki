"""WP3. Environment: store, actions, calculator, observations, episode session.

    from auditgym.env import AuditEnv
    env = AuditEnv(mode="open")
    obs, info = env.reset(seed=123, config=None)
    obs, reward, terminated, truncated, info = env.step(
        {"action": "list_records", "args": {"table": "ledger", "cursor": None, "limit": 50}}
    )

`AuditEnv` is loaded on first access, so importing the observation modules (observe, store,
actions, calculator) never loads the session controller that holds the reference answer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from auditgym.env.session import AuditEnv

__all__ = ["AuditEnv"]


def __getattr__(name: str) -> Any:
    if name == "AuditEnv":
        from auditgym.env.session import AuditEnv

        return AuditEnv
    raise AttributeError(name)
