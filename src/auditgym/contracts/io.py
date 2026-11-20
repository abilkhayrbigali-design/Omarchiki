"""Episode files on disk and their canonical bytes.

Layout of one episode directory:
    manifest.json policy.json ledger.csv invoices.jsonl payments.jsonl fx.csv   (public)
    private/truth.json private/provenance.json                                  (grader only)

Serialization is canonical (auditgym.util.canon), so the same episode always gives the same
bytes and the same fingerprint.
"""

from __future__ import annotations

from pathlib import Path

from auditgym.contracts.constants import FX_COLUMNS, LEDGER_COLUMNS
from auditgym.contracts.types import (
    Episode,
    FxRate,
    Invoice,
    LedgerRow,
    Payment,
    PrivateEpisode,
    PublicEpisode,
    Truth,
)
from auditgym.util import canon


def public_files(public: PublicEpisode) -> dict[str, bytes]:
    """Public file name -> canonical bytes."""
    files = {
        "manifest.json": canon.dumps_pretty(public.manifest),
        "policy.json": canon.dumps_pretty(public.policy),
        "ledger.csv": canon.csv_text(LEDGER_COLUMNS, (r.to_dict() for r in public.ledger)),
        "invoices.jsonl": canon.jsonl(i.to_dict() for i in public.invoices),
        "payments.jsonl": canon.jsonl(p.to_dict() for p in public.payments),
        "fx.csv": canon.csv_text(FX_COLUMNS, (f.to_dict() for f in public.fx)),
    }
    return {name: text.encode("utf-8") for name, text in files.items()}


def private_files(private: PrivateEpisode) -> dict[str, bytes]:
    files = {
        "truth.json": canon.dumps_pretty(private.truth.to_dict()),
        "provenance.json": canon.dumps_pretty(private.provenance),
    }
    return {name: text.encode("utf-8") for name, text in files.items()}


def _text(data: bytes) -> str:
    """UTF-8 text without a byte-order mark."""
    return data.decode("utf-8-sig")


def _csv_rows(text: str, columns: tuple[str, ...], name: str) -> list[dict[str, str]]:
    """Rows keyed by column name. Column order may differ; blank lines are ignored."""
    lines = "\n".join(line for line in text.splitlines() if line.strip())
    header, rows = canon.read_csv(lines)
    header = [h.strip() for h in header]
    if sorted(header) != sorted(columns):
        raise ValueError(f"{name}: expected columns {list(columns)}, got {header}")
    return [{h.strip(): v for h, v in row.items()} for row in rows]


def check_unique_ids(public: PublicEpisode) -> None:
    """Record ids must be globally unique (get_record/get_document and evidence rely on it)."""
    seen: dict[str, str] = {}
    for kind, ids in (
        ("ledger row", (r.row_id for r in public.ledger)),
        ("invoice document", (d.doc_id for d in public.invoices)),
        ("payment document", (d.doc_id for d in public.payments)),
        ("fx row", (f.rate_id for f in public.fx)),
    ):
        for value in ids:
            if value in seen:
                raise ValueError(
                    f"id {value!r} is used by a {seen[value]} and a {kind}; ids must be unique"
                )
            seen[value] = kind


def parse_public(files: dict[str, bytes]) -> PublicEpisode:
    text = {name: _text(data) for name, data in files.items()}
    ledger_rows = _csv_rows(text["ledger.csv"], LEDGER_COLUMNS, "ledger.csv")
    fx_rows = _csv_rows(text["fx.csv"], FX_COLUMNS, "fx.csv")
    public = PublicEpisode(
        manifest=canon.loads(text["manifest.json"]),
        policy=canon.loads(text["policy.json"]),
        ledger=tuple(LedgerRow.from_dict(r) for r in ledger_rows),
        invoices=tuple(Invoice.from_dict(d) for d in canon.read_jsonl(text["invoices.jsonl"])),
        payments=tuple(Payment.from_dict(d) for d in canon.read_jsonl(text["payments.jsonl"])),
        fx=tuple(FxRate.from_dict(r) for r in fx_rows),
    )
    check_unique_ids(public)
    return public


def parse_private(files: dict[str, bytes]) -> PrivateEpisode:
    provenance = files.get("provenance.json")
    return PrivateEpisode(
        truth=Truth.from_dict(canon.loads(_text(files["truth.json"]))),
        provenance=canon.loads(_text(provenance)) if provenance is not None else {},
    )


def write_episode(episode: Episode, directory: str | Path, include_private: bool = True) -> None:
    """Write an episode. Private files go to <directory>/private/ only if include_private."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    for name, data in public_files(episode.public).items():
        (root / name).write_bytes(data)
    if include_private:
        priv = root / "private"
        priv.mkdir(exist_ok=True)
        for name, data in private_files(episode.private).items():
            (priv / name).write_bytes(data)


def read_public(directory: str | Path) -> PublicEpisode:
    root = Path(directory)
    names = (
        "manifest.json",
        "policy.json",
        "ledger.csv",
        "invoices.jsonl",
        "payments.jsonl",
        "fx.csv",
    )
    return parse_public({name: (root / name).read_bytes() for name in names})


def read_private(directory: str | Path) -> PrivateEpisode:
    priv = Path(directory) / "private"
    files = {
        name: (priv / name).read_bytes()
        for name in ("truth.json", "provenance.json")
        if (priv / name).exists()
    }
    return parse_private(files)


def read_episode(directory: str | Path) -> Episode:
    return Episode(public=read_public(directory), private=read_private(directory))


def fingerprint(episode: Episode, include_private: bool = True) -> str:
    """SHA-256 over the canonical bytes of all files, in a fixed order."""
    parts = []
    for name, data in sorted(public_files(episode.public).items()):
        parts.append(name.encode("utf-8") + b"\0" + canon.sha256_hex(data).encode("ascii"))
    if include_private:
        for name, data in sorted(private_files(episode.private).items()):
            parts.append(
                b"private/" + name.encode("utf-8") + b"\0" + canon.sha256_hex(data).encode("ascii")
            )
    return canon.sha256_hex(b"\n".join(parts))
