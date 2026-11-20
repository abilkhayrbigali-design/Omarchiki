"""Canonical serialization: identical data always produces identical bytes.

- JSON: sorted keys, fixed separators, UTF-8, no NaN/Infinity, trailing newline for files.
- CSV: fixed column order, "\\n" line endings, None written as an empty field.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def dumps(obj: Any) -> str:
    """Compact canonical JSON (one line). Used for hashing, JSONL and protocol messages."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def dumps_pretty(obj: Any) -> str:
    """Indented canonical JSON with a trailing newline. Used for human-facing files."""
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def loads(text: str) -> Any:
    return json.loads(text, parse_constant=_reject_constant)


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite JSON constant {name} is not allowed")


def jsonl(rows: Iterable[Any]) -> str:
    return "".join(dumps(row) + "\n" for row in rows)


def read_jsonl(text: str) -> list[Any]:
    return [loads(line) for line in text.splitlines() if line.strip()]


def csv_text(header: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> str:
    """CSV with a fixed header. Values: None -> empty, int -> decimal digits, str as is."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        out = []
        for col in header:
            value = row[col]
            if value is None:
                out.append("")
            elif isinstance(value, bool):
                raise TypeError(f"bool is not allowed in CSV column {col!r}")
            elif isinstance(value, (int, str)):
                out.append(str(value))
            else:
                raise TypeError(f"unsupported CSV value type {type(value).__name__} in {col!r}")
        writer.writerow(out)
    return buf.getvalue()


def read_csv(text: str) -> tuple[list[str], list[dict[str, str]]]:
    """Header and rows as strings. Empty fields stay '' — callers map them to None."""
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        raise ValueError("empty CSV")
    header = rows[0]
    return header, [dict(zip(header, r, strict=True)) for r in rows[1:]]


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def hash_obj(obj: Any) -> str:
    """SHA-256 of the canonical JSON of `obj`."""
    return sha256_hex(dumps(obj))
