"""Pagination: at most 50 rows, stable opaque cursors, full coverage, no duplicates (E05)."""

from __future__ import annotations

import pytest
from envkit import act, big_episode, list_page, with_episode_id

from auditgym.contracts.constants import PAGE_LIMIT_MAX, TABLES
from auditgym.fixtures.tiny import build_tiny_episode


def _ids(records):
    out = []
    for r in records:
        out.append(
            r.get("row_id")
            or r.get("doc_id")
            or f"{r['date']}/{r['source_currency']}/{r['target_currency']}"
        )
    return out


def _page_all(env, table, limit):
    pages, cursor = [], None
    while True:
        obs = env.step(list_page(table, cursor, limit))[0]
        assert obs["error"] is None
        pages.append(obs["result"])
        cursor = obs["result"]["next_cursor"]
        if cursor is None:
            return pages


def test_E05_page_is_capped_at_50(make_env):
    episode = big_episode(110)  # 137 ledger rows
    env = make_env(episode=episode)
    env.reset(0)
    pages = _page_all(env, "ledger", 50)
    assert [len(p["records"]) for p in pages] == [50, 50, 37]
    assert all(len(p["records"]) <= PAGE_LIMIT_MAX for p in pages)


def test_E05_default_limit_is_50(make_env):
    env = make_env(episode=big_episode(110))
    env.reset(0)
    obs = env.step(act("list_records", table="ledger"))[0]
    assert len(obs["result"]["records"]) == 50
    assert obs["result"]["next_cursor"] is not None


@pytest.mark.parametrize("limit", [1, 4, 7, 50])
@pytest.mark.parametrize("table", TABLES)
def test_E05_pages_cover_table_exactly_once_in_order(make_env, table, limit):
    episode = big_episode(60)
    env = make_env(episode=episode)
    env.reset(0)
    pages = _page_all(env, table, limit)
    ids = [i for p in pages for i in _ids(p["records"])]
    expected = {
        "ledger": [r.row_id for r in episode.public.ledger],
        "invoices": [r.doc_id for r in episode.public.invoices],
        "payments": [r.doc_id for r in episode.public.payments],
        "fx": [r.rate_id for r in episode.public.fx],
    }[table]
    assert ids == expected
    assert len(set(ids)) == len(ids)
    assert all(1 <= len(p["records"]) <= limit for p in pages)
    assert pages[-1]["next_cursor"] is None


def test_E05_cursor_is_stable_and_replayable(make_env):
    env = make_env(episode=big_episode(110))
    env.reset(0)
    first = env.step(list_page("ledger", None, 20))[0]["result"]
    cursor = first["next_cursor"]
    a = env.step(list_page("ledger", cursor, 20))[0]["result"]
    b = env.step(list_page("ledger", cursor, 20))[0]["result"]
    assert a == b
    # Same position after a reset of the same episode -> same token.
    env.reset(0)
    again = env.step(list_page("ledger", None, 20))[0]["result"]
    assert again["next_cursor"] == cursor
    # A cursor marks a row position: a different limit continues from the same row.
    c = env.step(list_page("ledger", cursor, 5))[0]["result"]
    assert c["records"] == a["records"][:5]


def test_E05_cursor_is_opaque(make_env):
    env = make_env(episode=big_episode(110))
    env.reset(0)
    cursor = env.step(list_page("ledger", None, 20))[0]["result"]["next_cursor"]
    assert isinstance(cursor, str) and not cursor.isdigit()
    assert "ledger" not in cursor and "R-" not in cursor and "ep_" not in cursor


def test_E05_cursor_from_other_table_is_rejected(make_env):
    env = make_env(episode=big_episode(110))
    env.reset(0)
    cursor = env.step(list_page("ledger", None, 3))[0]["result"]["next_cursor"]
    obs = env.step(list_page("fx", cursor, 3))[0]
    assert obs["error"]["code"] == "invalid_cursor"
    assert obs["result"] is None


def test_E05_cursor_from_other_episode_is_rejected(make_env):
    env = make_env(episode=build_tiny_episode())
    env.reset(0)
    cursor = env.step(list_page("ledger", None, 3))[0]["result"]["next_cursor"]
    other = make_env(episode=with_episode_id(build_tiny_episode(), "ep_OTHER000001"))
    other.reset(0)
    assert other.step(list_page("ledger", cursor, 3))[0]["error"]["code"] == "invalid_cursor"


@pytest.mark.parametrize("cursor", ["", "c_", "c_" + "0" * 24, "0", "50", "R-005", "x" * 500])
def test_E05_forged_cursor_is_rejected(make_env, cursor):
    env = make_env()
    env.reset(0)
    obs = env.step(list_page("ledger", cursor, 3))[0]
    assert obs["error"]["code"] == "invalid_cursor"


def test_E05_empty_table(make_env):
    import dataclasses

    from auditgym.contracts.types import Episode

    tiny = build_tiny_episode()
    episode = Episode(public=dataclasses.replace(tiny.public, payments=()), private=tiny.private)
    env = make_env(episode=episode)
    env.reset(0)
    obs = env.step(list_page("payments", None, 10))[0]
    assert obs["result"] == {"table": "payments", "records": [], "next_cursor": None}
