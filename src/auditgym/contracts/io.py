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


def parse_public(files: dict[str, bytes]) -> PublicEpisode:
    text = {name: data.decode("utf-8") for name, data in files.items()}
    header, ledger_rows = canon.read_csv(text["ledger.csv"])
    if tuple(header) != LEDGER_COLUMNS:
        raise ValueError("ledger.csv: unexpected header")
    fx_header, fx_rows = canon.read_csv(text["fx.csv"])
    if tuple(fx_header) != FX_COLUMNS:
        raise ValueError("fx.csv: unexpected header")
    return PublicEpisode(
        manifest=canon.loads(text["manifest.json"]),
        policy=canon.loads(text["policy.json"]),
        ledger=tuple(LedgerRow.from_dict(r) for r in ledger_rows),
        invoices=tuple(Invoice.from_dict(d) for d in canon.read_jsonl(text["invoices.jsonl"])),
        payments=tuple(Payment.from_dict(d) for d in canon.read_jsonl(text["payments.jsonl"])),
        fx=tuple(FxRate.from_dict(r) for r in fx_rows),
    )


def parse_private(files: dict[str, bytes]) -> PrivateEpisode:
    return PrivateEpisode(
        truth=Truth.from_dict(canon.loads(files["truth.json"].decode("utf-8"))),
        provenance=canon.loads(files["provenance.json"].decode("utf-8")),
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
    return parse_private(
        {name: (priv / name).read_bytes() for name in ("truth.json", "provenance.json")}
    )


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
