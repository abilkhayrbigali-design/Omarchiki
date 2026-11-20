"""Named, platform-independent random streams.

Every source of randomness in the generator is a `Stream`. Streams are derived by name from a
root key, so adding or changing draws in one stream never shifts the values of another stream.

The generator is SHA-256 in counter mode. It does not depend on Python's `random`, `hash()`,
PYTHONHASHSEED, the platform or the interpreter version.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from decimal import Decimal
from typing import TypeVar

T = TypeVar("T")

_ROOT_DOMAIN = b"auditgym/rng/v1|"


class Stream:
    """A deterministic stream of random values."""

    __slots__ = ("_key", "_counter", "_buf")

    def __init__(self, key: bytes) -> None:
        if not isinstance(key, bytes) or len(key) != 32:
            raise ValueError("stream key must be 32 bytes")
        self._key = key
        self._counter = 0
        self._buf = b""

    # ------------------------------------------------------------------ derivation
    def child(self, name: str) -> Stream:
        """Independent stream derived from this stream's key (not from its consumed state)."""
        if not name:
            raise ValueError("child stream needs a name")
        return Stream(hashlib.sha256(self._key + b"/" + name.encode("utf-8")).digest())

    # ------------------------------------------------------------------ raw bits
    def _take(self, n: int) -> bytes:
        while len(self._buf) < n:
            block = hashlib.sha256(self._key + self._counter.to_bytes(8, "big")).digest()
            self._counter += 1
            self._buf += block
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def bits(self, k: int) -> int:
        """Uniform integer in [0, 2**k)."""
        if k <= 0:
            return 0
        nbytes = (k + 7) // 8
        value = int.from_bytes(self._take(nbytes), "big")
        return value >> (nbytes * 8 - k)

    # ------------------------------------------------------------------ integers
    def randbelow(self, n: int) -> int:
        """Uniform integer in [0, n). Rejection sampling, no modulo bias."""
        if n <= 0:
            raise ValueError("randbelow needs n > 0")
        k = n.bit_length()
        while True:
            r = self.bits(k)
            if r < n:
                return r

    def randint(self, a: int, b: int) -> int:
        """Uniform integer in [a, b], both ends included."""
        if b < a:
            raise ValueError("randint needs a <= b")
        return a + self.randbelow(b - a + 1)

    def in_range(self, bounds: Sequence[int]) -> int:
        """Uniform integer in an inclusive [min, max] pair, as used in configs."""
        lo, hi = bounds
        return self.randint(lo, hi)

    # ------------------------------------------------------------------ choices
    def choice(self, seq: Sequence[T]) -> T:
        if not seq:
            raise ValueError("choice from an empty sequence")
        return seq[self.randbelow(len(seq))]

    def weighted_choice(self, items: Sequence[T], weights: Sequence[int]) -> T:
        """Choice with non-negative integer weights."""
        if len(items) != len(weights) or not items:
            raise ValueError("items and weights must be non-empty and of equal length")
        if any(w < 0 for w in weights):
            raise ValueError("weights must be non-negative integers")
        total = sum(weights)
        if total <= 0:
            raise ValueError("weights must not all be zero")
        r = self.randbelow(total)
        acc = 0
        for item, w in zip(items, weights, strict=True):
            acc += w
            if r < acc:
                return item
        raise AssertionError("unreachable")

    def shuffled(self, seq: Sequence[T]) -> list[T]:
        """New list with the elements of `seq` in random order (Fisher-Yates)."""
        out = list(seq)
        for i in range(len(out) - 1, 0, -1):
            j = self.randbelow(i + 1)
            out[i], out[j] = out[j], out[i]
        return out

    def sample(self, seq: Sequence[T], k: int) -> list[T]:
        """k distinct elements of `seq`, in random order."""
        if k < 0 or k > len(seq):
            raise ValueError("sample size out of range")
        return self.shuffled(seq)[:k]

    def chance(self, numerator: int, denominator: int) -> bool:
        """True with probability numerator/denominator."""
        if denominator <= 0 or not 0 <= numerator <= denominator:
            raise ValueError("invalid probability")
        return self.randbelow(denominator) < numerator

    def chance_p(self, p: float | str | Decimal) -> bool:
        """True with probability p, resolved to parts per million."""
        ppm = int((Decimal(str(p)) * 1_000_000).to_integral_value())
        if not 0 <= ppm <= 1_000_000:
            raise ValueError("probability must be in [0, 1]")
        return self.randbelow(1_000_000) < ppm

    # ------------------------------------------------------------------ other values
    def decimal(self, lo: Decimal | str, hi: Decimal | str, places: int) -> Decimal:
        """Uniform decimal in [lo, hi] on a grid of 10**-places."""
        lo_d, hi_d = Decimal(str(lo)), Decimal(str(hi))
        scale = 10**places
        lo_i = int((lo_d * scale).to_integral_value())
        hi_i = int((hi_d * scale).to_integral_value())
        return Decimal(self.randint(lo_i, hi_i)).scaleb(-places)

    def token(self, nchars: int = 8, alphabet: str = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ") -> str:
        """Random string over `alphabet` (default: digits and upper case without I and O)."""
        return "".join(alphabet[self.randbelow(len(alphabet))] for _ in range(nchars))


def root_stream(seed: int, config_hash: str, version: str, attempt: int = 0) -> Stream:
    """Root stream of one generation attempt.

    The key depends on the seed, the normalized config, the generator version and the attempt
    number. Same inputs give the same stream on every platform.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    material = json.dumps(
        {"attempt": attempt, "config": config_hash, "seed": seed, "version": version},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return Stream(hashlib.sha256(_ROOT_DOMAIN + material).digest())


def keyed_token(secret: bytes, label: str, nchars: int = 10) -> str:
    """Deterministic opaque token: a keyed hash of `label`, rendered in base32-like characters.

    Used for public IDs: the output reveals nothing about `label` or about generation order
    without the secret.
    """
    alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
    digest = hashlib.sha256(secret + b"|" + label.encode("utf-8")).digest()
    value = int.from_bytes(digest, "big")
    out = []
    for _ in range(nchars):
        value, idx = divmod(value, len(alphabet))
        out.append(alphabet[idx])
    return "".join(out)
