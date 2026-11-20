"""Episode configuration: schema, defaults, validation, hashing.

A config is JSON. Every count is either an integer or an inclusive [min, max] pair; after
normalization every count is a [min, max] list. Unknown keys are rejected.

Error types are configured independently (G01). `clean: true` forces a clean company (G03).
Interacting scenarios are configured separately (G02); each one adds a fixed number of issues
(INTERACTION_ISSUE_COUNTS).

The `presentation` section (`id_salt`, `order`) changes only public IDs and row order, never the
content of the episode: the generator's random streams are keyed without it.

`secret_salt` is part of the stream key. The generator is open source, so with an empty salt
anyone who knows the config can regenerate an episode for every candidate seed and recover the
truth by brute force. For a closed evaluation the organizer sets a private `secret_salt` in the
config given to the grader process; the agent never sees the config, so the episode cannot be
regenerated. Same seed + same config (including the salt) still reproduces the episode exactly.

`max_issues` (<= 12, D01) caps the total. The generator samples interaction counts first, then
per-type error counts, and if the total exceeds `max_issues` it removes randomly chosen
per-type errors (never interactions) until it fits. Validation requires that the minimum
counts alone fit under `max_issues`.

`variants` (G01) restricts, per error type, the variants the standalone injector of that type may
produce, e.g. {"variants": {"payment_allocation": ["partial"]}}. Names: ISSUE_VARIANTS. A type
left out keeps all its variants. It applies to the per-type errors of `errors` only; the
interacting scenarios keep their fixed shapes. A restricted variant is never replaced by another
one: the injector picks a record that can carry an allowed variant, or the attempt fails and the
generator retries. `fx_conversion: ["wrong_pair"]` alone needs a second foreign currency, so the
world then draws `currencies.foreign_count` from [max(min, 2), max].

The normalized config carries `variants` only when it differs from the default (every variant
allowed), and then for every type with names in canonical order. A config without `variants`,
or with exactly the default, normalizes to the same dict as before the section existed, so its
episodes, stream key and provenance stay byte-identical.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from auditgym.contracts.constants import (
    DISTRACTOR_KINDS,
    INTERACTION_ISSUE_COUNTS,
    INTERACTION_SCENARIOS,
    ISSUE_TYPES,
    MAX_ERRORS_PER_EPISODE,
)
from auditgym.util.canon import hash_obj
from auditgym.util.money import CURRENCY_EXPONENTS
from auditgym.util.timeutil import parse_offset


class ConfigError(ValueError):
    """Invalid episode config."""


DEFAULT_CONFIG: dict[str, Any] = {
    "config_id": "default",
    "size": {
        "customers": [6, 12],
        "invoices": [24, 60],
        "background_entries": [40, 160],
    },
    "currencies": {
        "base_pool": ["KZT", "USD", "EUR"],
        "foreign_pool": ["USD", "EUR", "GBP", "CNY", "JPY", "KWD", "AED"],
        "foreign_count": [1, 2],
    },
    "timezone_pool": ["+05:00", "+03:00", "+06:00", "+01:00", "-04:00"],
    "period": {"year_pool": [2025, 2026], "margin_days": 21},
    "clean": False,
    "secret_salt": "",
    "max_issues": MAX_ERRORS_PER_EPISODE,
    "errors": {t: [0, 2] for t in ISSUE_TYPES},
    "interactions": {s: [0, 1] for s in INTERACTION_SCENARIOS},
    "distractors": {
        "split_payment": [1, 3],
        "multi_invoice_payment": [1, 2],
        "partial_open": [1, 3],
        "credit_note_refund": [0, 2],
        "subscription": [1, 3],
        "same_amount_orders": [0, 2],
        "unapplied_payment": [0, 2],
        "boundary_offset": [1, 3],
        "fx_spike": [0, 1],
        "posting_lag": [2, 6],
    },
    "presentation": {"id_salt": "", "order": "chronological"},
}

_PRESENTATION_ORDERS = ("chronological", "shuffled")

# Variants of each standalone error type, in canonical order (docs/DOMAIN.md, section 6). The
# injectors draw among the allowed ones in this order, so the default draws stay unchanged.
ISSUE_VARIANTS: dict[str, tuple[str, ...]] = {
    "duplicate_invoice": ("same_invoice_id", "new_invoice_id"),
    "payment_allocation": ("none", "partial", "other_invoice"),
    "fx_conversion": ("wrong_date", "wrong_pair"),
    "period_cutoff": ("invoice_posted_late", "payment_posted_late"),
}
DEFAULT_VARIANTS: dict[str, list[str]] = {t: list(ISSUE_VARIANTS[t]) for t in ISSUE_TYPES}


def _range(value: Any, where: str) -> list[int]:
    if isinstance(value, bool):
        raise ConfigError(f"{where}: expected integer or [min, max], got bool")
    if isinstance(value, int):
        lo = hi = value
    elif (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(v, int) and not isinstance(v, bool) for v in value)
    ):
        lo, hi = value
    else:
        raise ConfigError(f"{where}: expected integer or [min, max], got {value!r}")
    if lo < 0 or hi < lo:
        raise ConfigError(f"{where}: need 0 <= min <= max, got [{lo}, {hi}]")
    return [lo, hi]


def _check_keys(section: dict[str, Any], allowed: tuple[str, ...] | list[str], where: str) -> None:
    if not isinstance(section, dict):
        raise ConfigError(f"{where}: expected an object")
    unknown = sorted(set(section) - set(allowed))
    if unknown:
        raise ConfigError(f"{where}: unknown keys {unknown}")


def _merge(base: dict[str, Any], override: dict[str, Any], where: str) -> dict[str, Any]:
    _check_keys(override, list(base), where)
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(base[key], dict):
            out[key] = _merge(base[key], value, f"{where}.{key}")
        else:
            out[key] = copy.deepcopy(value)
    return out


def _currency_list(value: Any, where: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{where}: expected a non-empty list of currency codes")
    for code in value:
        if code not in CURRENCY_EXPONENTS:
            raise ConfigError(f"{where}: unknown currency {code!r}")
    if len(set(value)) != len(value):
        raise ConfigError(f"{where}: duplicate currency codes")
    return list(value)


def _variants(value: Any) -> dict[str, list[str]] | None:
    """Validated `variants` for every type in canonical order; None when it is the default."""
    _check_keys(value, ISSUE_TYPES, "config.variants")
    out: dict[str, list[str]] = {}
    for t in ISSUE_TYPES:
        names = ISSUE_VARIANTS[t]
        if t not in value:
            out[t] = list(names)
            continue
        where = f"config.variants.{t}"
        chosen = value[t]
        if not isinstance(chosen, list) or not chosen:
            raise ConfigError(f"{where}: expected a non-empty list of variants of {list(names)}")
        for name in chosen:
            if not isinstance(name, str) or name not in names:
                raise ConfigError(
                    f"{where}: unknown variant {name!r}, expected one of {list(names)}"
                )
        if len(set(chosen)) != len(chosen):
            raise ConfigError(f"{where}: duplicate variant names")
        out[t] = [name for name in names if name in chosen]
    return None if out == DEFAULT_VARIANTS else out


def allowed_variants(cfg: dict[str, Any], issue_type: str) -> tuple[str, ...]:
    """Variants the standalone injector of `issue_type` may produce, in canonical order."""
    chosen = cfg.get("variants")
    if chosen is None:
        return ISSUE_VARIANTS[issue_type]
    return tuple(chosen[issue_type])


def needs_two_foreign(cfg: dict[str, Any]) -> bool:
    """`fx_conversion: ["wrong_pair"]` alone takes the wrong rate from a second foreign currency."""
    return (
        allowed_variants(cfg, "fx_conversion") == ("wrong_pair",)
        and cfg["errors"]["fx_conversion"][1] > 0
    )


def normalize_config(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge `raw` over DEFAULT_CONFIG, validate, and return the normalized config."""
    raw = raw or {}
    variants = None
    if isinstance(raw, dict) and "variants" in raw:
        variants = _variants(raw["variants"])
        raw = {k: v for k, v in raw.items() if k != "variants"}
    cfg = _merge(DEFAULT_CONFIG, raw, "config")
    if variants is not None:
        cfg["variants"] = variants

    if not isinstance(cfg["config_id"], str) or not cfg["config_id"]:
        raise ConfigError("config.config_id: expected a non-empty string")

    for key in cfg["size"]:
        cfg["size"][key] = _range(cfg["size"][key], f"config.size.{key}")
    if cfg["size"]["customers"][0] < 3:
        raise ConfigError("config.size.customers: need at least 3 customers")
    if cfg["size"]["invoices"][0] < 4:
        raise ConfigError("config.size.invoices: need at least 4 invoices")

    cur = cfg["currencies"]
    cur["base_pool"] = _currency_list(cur["base_pool"], "config.currencies.base_pool")
    cur["foreign_pool"] = _currency_list(cur["foreign_pool"], "config.currencies.foreign_pool")
    cur["foreign_count"] = _range(cur["foreign_count"], "config.currencies.foreign_count")
    if cur["foreign_count"][1] > len(cur["foreign_pool"]) - 1:
        raise ConfigError(
            "config.currencies.foreign_count: max must leave room for the base currency in the pool"
        )

    pool = cfg["timezone_pool"]
    if not isinstance(pool, list) or not pool:
        raise ConfigError("config.timezone_pool: expected a non-empty list of offsets")
    for offset in pool:
        try:
            parse_offset(offset)
        except ValueError as exc:
            raise ConfigError(f"config.timezone_pool: {exc}") from exc

    years = cfg["period"]["year_pool"]
    if (
        not isinstance(years, list)
        or not years
        or not all(isinstance(y, int) and 2000 <= y <= 2100 for y in years)
    ):
        raise ConfigError("config.period.year_pool: expected a list of years 2000..2100")
    margin = cfg["period"]["margin_days"]
    if not isinstance(margin, int) or isinstance(margin, bool) or not 7 <= margin <= 60:
        raise ConfigError("config.period.margin_days: expected an integer 7..60")

    if not isinstance(cfg["clean"], bool):
        raise ConfigError("config.clean: expected true or false")
    if not isinstance(cfg["secret_salt"], str) or len(cfg["secret_salt"]) > 256:
        raise ConfigError("config.secret_salt: expected a string of at most 256 characters")

    _check_keys(cfg["errors"], ISSUE_TYPES, "config.errors")
    for key in ISSUE_TYPES:
        cfg["errors"][key] = _range(cfg["errors"][key], f"config.errors.{key}")
    _check_keys(cfg["interactions"], INTERACTION_SCENARIOS, "config.interactions")
    for key in INTERACTION_SCENARIOS:
        cfg["interactions"][key] = _range(cfg["interactions"][key], f"config.interactions.{key}")
    _check_keys(cfg["distractors"], DISTRACTOR_KINDS, "config.distractors")
    for key in DISTRACTOR_KINDS:
        cfg["distractors"][key] = _range(cfg["distractors"][key], f"config.distractors.{key}")

    if cfg["clean"]:
        for key in ISSUE_TYPES:
            cfg["errors"][key] = [0, 0]
        for key in INTERACTION_SCENARIOS:
            cfg["interactions"][key] = [0, 0]

    cap = cfg["max_issues"]
    if not isinstance(cap, int) or isinstance(cap, bool) or not 0 <= cap <= MAX_ERRORS_PER_EPISODE:
        raise ConfigError(f"config.max_issues: expected an integer 0..{MAX_ERRORS_PER_EPISODE}")
    min_issues = sum(lo for lo, _ in cfg["errors"].values()) + sum(
        cfg["interactions"][s][0] * INTERACTION_ISSUE_COUNTS[s] for s in INTERACTION_SCENARIOS
    )
    if min_issues > cap:
        raise ConfigError(f"config: at least {min_issues} issues required, max_issues is {cap}")
    max_interaction_issues = sum(
        cfg["interactions"][s][1] * INTERACTION_ISSUE_COUNTS[s] for s in INTERACTION_SCENARIOS
    )
    if max_interaction_issues > cap:
        raise ConfigError(
            f"config: interactions alone may produce {max_interaction_issues} issues, max_issues is {cap}"
        )
    needs_fx = (
        cfg["errors"]["fx_conversion"][1] > 0
        or cfg["interactions"]["fx_partial_at_boundary"][1] > 0
        or cfg["distractors"]["fx_spike"][1] > 0
    )
    if needs_fx and cur["foreign_count"][0] < 1:
        raise ConfigError(
            "config.currencies.foreign_count: min must be >= 1 when FX errors or FX distractors are enabled"
        )
    if needs_two_foreign(cfg) and cur["foreign_count"][1] < 2:
        raise ConfigError(
            "config.variants.fx_conversion: wrong_pair alone needs two foreign currencies, "
            "currencies.foreign_count max must be >= 2"
        )

    pres = cfg["presentation"]
    if not isinstance(pres["id_salt"], str):
        raise ConfigError("config.presentation.id_salt: expected a string")
    if pres["order"] not in _PRESENTATION_ORDERS:
        raise ConfigError(f"config.presentation.order: expected one of {_PRESENTATION_ORDERS}")
    return cfg


def config_hash(cfg: dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON of a normalized config."""
    return hash_obj(cfg)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return normalize_config(raw)
