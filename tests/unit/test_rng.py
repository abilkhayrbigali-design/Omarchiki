import subprocess
import sys

import pytest
from hypothesis import given
from hypothesis import strategies as st

from auditgym.util.rng import Stream, keyed_token, root_stream


def _draws(stream: Stream) -> list[int]:
    return [stream.randint(0, 10**6) for _ in range(20)]


def test_same_inputs_same_values():
    a = root_stream(7, "cfg", "0.1.0")
    b = root_stream(7, "cfg", "0.1.0")
    assert _draws(a) == _draws(b)


@pytest.mark.parametrize(
    "other",
    [
        dict(seed=8, config_hash="cfg", version="0.1.0", attempt=0),
        dict(seed=7, config_hash="cfg2", version="0.1.0", attempt=0),
        dict(seed=7, config_hash="cfg", version="0.2.0", attempt=0),
        dict(seed=7, config_hash="cfg", version="0.1.0", attempt=1),
    ],
)
def test_any_input_changes_stream(other):
    assert _draws(root_stream(7, "cfg", "0.1.0")) != _draws(root_stream(**other))


def test_child_streams_are_independent_of_parent_consumption():
    root1 = root_stream(1, "c", "v")
    root2 = root_stream(1, "c", "v")
    _draws(root2)  # consume the parent
    assert _draws(root1.child("world")) == _draws(root2.child("world"))
    assert _draws(root1.child("world")) != _draws(root1.child("inject"))


def test_values_pinned_across_platforms():
    # Regression pin: if this changes, every generated episode changes.
    s = root_stream(42, "pin", "0.0.0")
    assert [s.randint(0, 999) for _ in range(5)] == [430, 446, 123, 822, 948]


def test_independent_of_pythonhashseed():
    code = (
        "from auditgym.util.rng import root_stream;"
        "s=root_stream(5,'h','v');print([s.randint(0,10**9) for _ in range(5)])"
    )
    outs = {
        subprocess.run(
            [sys.executable, "-c", code],
            env={"PYTHONHASHSEED": seed, "PATH": ""},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for seed in ("0", "1", "12345")
    }
    assert len(outs) == 1


@given(st.integers(min_value=1, max_value=10**12))
def test_randbelow_in_range(n):
    s = root_stream(n % 1000, "h", "v")
    for _ in range(5):
        assert 0 <= s.randbelow(n) < n


def test_shuffle_is_permutation():
    s = root_stream(3, "h", "v")
    items = list(range(50))
    out = s.shuffled(items)
    assert sorted(out) == items
    assert out != items


def test_weighted_choice_respects_zero_weight():
    s = root_stream(3, "h", "v")
    assert {s.weighted_choice(["a", "b"], [0, 5]) for _ in range(50)} == {"b"}


def test_chance_p_bounds():
    s = root_stream(3, "h", "v")
    assert not any(s.chance_p(0) for _ in range(100))
    assert all(s.chance_p(1) for _ in range(100))
    with pytest.raises(ValueError):
        s.chance_p(1.5)


def test_keyed_token_deterministic_and_secret_dependent():
    assert keyed_token(b"k", "invoice:1") == keyed_token(b"k", "invoice:1")
    assert keyed_token(b"k", "invoice:1") != keyed_token(b"k2", "invoice:1")
    assert len(keyed_token(b"k", "x", nchars=12)) == 12
