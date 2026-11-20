"""Our run of the jury's published FinTech checks through judge_adapter.py.

The jury connects to every team through the same adapter (generate_episode, make_env with
reset/step, score_episode) and runs four blocks (docs/REQUIREMENTS.md, section 8):

    Block 1  C01-C20   contract tests on fixture episodes C (clean), E (one error), L (>= 101 rows)
                       and an id U that is absent from the open data;
    Block 2  D01-D20   closed episodes from our generator through its documented config: clean
                       companies with distractors, each error type, both interaction scenarios;
                       each must have balanced books before the injection, the declared
                       discrepancy after it, and an evidence set that proves it;
    Block 3  T01-T20   matcher and reward on a hand-written fixture with one error A (one record
           + M01-M12   R1, 10000 KZT, evidence {D1, D2}, tolerance 0) and the open examples;
    Block 4  P01-P20   pairs: the jury copies an episode, permutes its records and/or replaces
                       its ids in every file, and the oracle must get the same result.

Everything here goes through the adapter and the episode files, the way the jury does it: the
fixture of Block 3 is a truth.json written by hand, the transformation of Block 4 is a plain
file rewrite written independently of auditgym.harness.transform. Expected numbers are written
out from the check definitions, not computed by our grader. This is our reading of the published
definitions; the jury's own harness may differ in details.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import shutil
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any

from auditgym.contracts.constants import (
    INTERACTION_SCENARIOS,
    ISSUE_TYPES,
    MAX_STEPS,
)
from auditgym.contracts.io import read_episode
from auditgym.contracts.types import Invoice, LedgerRow, Payment, PublicEpisode
from auditgym.judge import SCORE_LOG, generate_episode, make_env, score_episode

Check = Callable[[], tuple[bool, str]]
TOL = 1e-9


def _none_cfg(config_id: str, **over: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "config_id": config_id,
        "errors": {t: [0, 0] for t in ISSUE_TYPES},
        "interactions": {s: [0, 0] for s in INTERACTION_SCENARIOS},
    }
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key] = {**cfg[key], **value}
        else:
            cfg[key] = value
    return cfg


CLEAN_CFG = {"config_id": "judge.clean", "clean": True}
ONE_ERROR_CFG = _none_cfg("judge.one", errors={"duplicate_invoice": [1, 1]})


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _closed(d: Path) -> dict[str, Any]:
    return {"mode": "closed", "episode_dir": str(d)}


def _jsonable(*values: Any) -> bool:
    try:
        for v in values:
            if json.loads(json.dumps(v, allow_nan=False)) != v:
                return False
    except (TypeError, ValueError):
        return False
    return True


def _oracle_findings(d: Path) -> list[dict[str, Any]]:
    truth = json.loads((d / "private" / "truth.json").read_text("utf-8"))
    return [
        {
            "type": i["type"],
            "affected_ids": list(i["affected_ids"]),
            "claimed_amount_minor": i["expected_amount_minor"],
            "currency": i["currency"],
            "evidence_ids": list(i["evidence_sets"][0]),
            "explanation": "oracle",
        }
        for i in truth["issues"]
    ]


def _submit(eid: str, findings: list[dict[str, Any]]) -> dict[str, Any]:
    return {"name": "submit", "args": {"findings": {"episode_id": eid, "findings": findings}}}


def _ok(results: list[dict[str, Any]]) -> int:
    return sum(1 for r in results if r["ok"])


def _run(checks: dict[str, Check]) -> list[dict[str, Any]]:
    out = []
    for name, fn in checks.items():
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001 - a crash is a failed check, reported
            ok, detail = False, f"raised {type(exc).__name__}: {exc}"
        out.append({"id": name, "ok": bool(ok), "detail": detail})
    return out


# ---------------------------------------------------------------------------------- block 1
def block1(work: Path) -> list[dict[str, Any]]:
    c_dir, e_dir = work / "C", work / "E"
    generate_episode(101, CLEAN_CFG, str(c_dir))
    generate_episode(102, ONE_ERROR_CFG, str(e_dir))
    c_pub = read_episode(c_dir).public
    e_ep = read_episode(e_dir)
    l_dir, l_pub = c_dir, c_pub  # the ledger of C has 200+ rows: three pages and more
    r1 = c_pub.ledger[0]
    d1 = c_pub.invoices[0]
    known = c_pub.open_ids() | e_ep.public.open_ids()
    u = "U-000000"
    while u in known:
        u += "0"
    c_eid, e_eid = c_pub.episode_id, e_ep.public.episode_id

    def c01():
        obs, info = make_env().reset(None, _closed(c_dir))
        return _jsonable(obs, info) and isinstance(obs, dict) and isinstance(info, dict), "pair"

    def c02():
        obs, _ = make_env().reset(None, _closed(c_dir))
        keys = sorted(obs)
        text = json.dumps(obs)
        ok = keys == ["manifest", "policy", "remaining_steps", "task"]
        ok = ok and obs["remaining_steps"] == MAX_STEPS and isinstance(obs["task"], str)
        ok = ok and '"seed"' not in text and "truth" not in text.lower()
        return ok, f"keys {keys}"

    def c03():
        obs, _ = make_env().reset(None, _closed(c_dir))
        m = obs["manifest"]
        need = ["schema_version", "episode_id", "base_currency", "period_start", "period_end"]
        need += ["timezone", "counts", "rounding"]
        counts = m.get("counts", {})
        real = {
            "ledger": len(c_pub.ledger),
            "invoices": len(c_pub.invoices),
            "payments": len(c_pub.payments),
            "fx": len(c_pub.fx),
            "documents": len(c_pub.invoices) + len(c_pub.payments),
        }
        ok = all(k in m for k in need) and all(counts.get(k) == v for k, v in real.items())
        return ok and m["episode_id"] == c_eid, f"counts {counts} vs {real}"

    def c04():
        env = make_env()
        env.reset(None, _closed(c_dir))
        res = env.step({"name": "list_records", "args": {"table": "ledger", "cursor": None}})
        obs, reward, term, trunc, info = res
        ok = len(res) == 5 and type(term) is bool and type(trunc) is bool
        ok = ok and _jsonable(obs, reward, info) and isinstance(obs["result"]["records"], list)
        return ok and obs["remaining_steps"] == MAX_STEPS - 1, "5-tuple, bool flags"

    def c05():
        env = make_env()
        env.reset(None, _closed(l_dir))
        seen: list[str] = []
        cursor = None
        pages = 0
        while True:
            obs, *_ = env.step(
                {"name": "list_records", "args": {"table": "ledger", "cursor": cursor, "limit": 50}}
            )
            recs = obs["result"]["records"]
            if len(recs) > 50:
                return False, "page over 50"
            seen += [r["row_id"] for r in recs]
            pages += 1
            cursor = obs["result"]["next_cursor"]
            if cursor is None:
                break
        want = [r.row_id for r in l_pub.ledger]
        ok = len(want) >= 101 and seen == want and len(set(seen)) == len(seen)
        return ok, f"{pages} pages, {len(seen)} of {len(want)} rows"

    def c06():
        env = make_env()
        env.reset(None, _closed(l_dir))
        first, *_ = env.step({"name": "list_records", "args": {"table": "ledger", "limit": 50}})
        cur = first["result"]["next_cursor"]
        a, *_ = env.step({"name": "list_records", "args": {"table": "ledger", "cursor": cur}})
        b, *_ = env.step({"name": "list_records", "args": {"table": "ledger", "cursor": cur}})
        ok = a["result"] == b["result"] and a["remaining_steps"] == b["remaining_steps"] + 1
        return ok, "same page twice"

    def c07():
        env = make_env()
        env.reset(None, _closed(c_dir))
        obs, *_ = env.step({"name": "get_record", "args": {"table": "ledger", "id": r1.row_id}})
        return obs["result"] == json.loads(json.dumps(r1.to_dict())), "R1"

    def c08():
        env = make_env()
        env.reset(None, _closed(c_dir))
        obs, *_ = env.step({"name": "get_document", "args": {"doc_id": d1.doc_id}})
        return obs["result"] == json.loads(json.dumps(d1.to_dict())), "D1 with line items"

    def c09():
        env = make_env()
        env.reset(None, _closed(c_dir))
        obs, *_ = env.step({"name": "calculator", "args": {"expression": "0.1+0.2"}})
        from decimal import Decimal

        return Decimal(obs["result"]) == Decimal("0.3"), f"0.1+0.2 -> {obs['result']!r}"

    def c10():
        env = make_env()
        env.reset(None, _closed(c_dir))
        outs = []
        for expr in ("open('/etc/passwd').read()", "__import__('os').system('id')", "1+1;ls"):
            obs, *_ = env.step({"name": "calculator", "args": {"expression": expr}})
            outs.append(obs)
        ok = all(o["result"] is None and o["error"]["code"] == "calculator_error" for o in outs)
        return ok and "root:" not in json.dumps(outs), "rejected"

    def _unknown(action: dict[str, Any]) -> tuple[bool, str]:
        env = make_env()
        env.reset(None, _closed(c_dir))
        obs, _, term, trunc, _ = env.step(action)
        again, *_ = env.step({"name": "get_record", "args": {"table": "ledger", "id": r1.row_id}})
        ok = obs["result"] is None and isinstance(obs["error"], dict) and "code" in obs["error"]
        ok = ok and obs["remaining_steps"] == MAX_STEPS - 1 and not term and not trunc
        ok = ok and again["result"] is not None
        return ok, f"error {obs['error']}"

    def c11():
        return _unknown({"name": "get_record", "args": {"table": "ledger", "id": u}})

    def c12():
        return _unknown({"name": "get_document", "args": {"doc_id": u}})

    def c13():
        return _unknown({"name": "drop_tables", "args": {}})

    def c14():
        env = make_env()
        env.reset(None, _closed(c_dir))
        a, *_ = env.step({"name": "get_record", "args": {"table": "ledger", "id": r1.row_id}})
        b, *_ = env.step({"name": "get_record", "args": {"table": "ledger", "id": u}})
        c, *_ = env.step({"name": "get_record", "args": {"table": "ledger", "id": r1.row_id}})
        rem = [a["remaining_steps"], b["remaining_steps"], c["remaining_steps"]]
        return rem == [99, 98, 97] and a["result"] == c["result"], f"budget {rem}"

    def c15():
        env = make_env()
        env.reset(None, _closed(e_dir))
        _, _, term, trunc, _ = env.step(_submit(e_eid, _oracle_findings(e_dir)))
        graded = copy.deepcopy(env.grade_log)
        later = [
            env.step(_submit(e_eid, [])),
            env.step({"name": "list_records", "args": {"table": "ledger"}}),
        ]
        ok = term and not trunc and env.grade_log == graded and len(graded) == 1
        ok = ok and all(r[1] == 0.0 and r[0]["error"]["code"] == "episode_finished" for r in later)
        return ok, f"reward in log {graded[-1]['reward']}"

    def c16():
        env = make_env()
        env.reset(None, _closed(e_dir))
        for _ in range(MAX_STEPS - 1):
            env.step({"name": "calculator", "args": {"expression": "1+1"}})
        obs, _, term, trunc, _ = env.step(_submit(e_eid, _oracle_findings(e_dir)))
        log = env.grade_log[-1]
        ok = term and not trunc and obs["result"] == {"accepted": True}
        ok = ok and log["inspection_steps"] == MAX_STEPS - 1 and log["tp"] == 1
        return ok, f"submit as action 100: tp {log['tp']}, steps {log['inspection_steps']}"

    def c17():
        env = make_env()
        env.reset(None, _closed(e_dir))
        last = None
        for _ in range(MAX_STEPS):
            last = env.step({"name": "calculator", "args": {"expression": "1+1"}})
        after = env.step(_submit(e_eid, _oracle_findings(e_dir)))
        log = env.grade_log
        ok = last[3] is True and last[2] is False and len(log) == 1
        ok = ok and log[0]["tp"] == 0 and log[0]["fp"] == 0 and after[0]["error"] is not None
        return ok, f"truncated {last[3]}, findings graded: tp {log[0]['tp']} fp {log[0]['fp']}"

    def c18():
        clock = FakeClock()
        env = make_env(clock=clock)
        env.reset(None, _closed(e_dir))
        clock.now = 30.0
        mid, *_ = env.step({"name": "calculator", "args": {"expression": "1+1"}})
        clock.now = 60.5
        obs, _, term, trunc, _ = env.step(_submit(e_eid, _oracle_findings(e_dir)))
        log = env.grade_log[-1]
        ok = trunc and not term and log["end"] == "timeout" and log["tp"] == log["fp"] == 0
        ok = ok and obs["remaining_steps"] == mid["remaining_steps"]  # late submit not executed
        return ok, f"truncated {trunc}, end {log['end']}"

    def c19():
        env = make_env()
        _, info0 = env.reset(None, _closed(e_dir))
        rewards, infos = [], [info0]
        for action in (
            {"name": "list_records", "args": {"table": "ledger"}},
            {"name": "get_record", "args": {"table": "ledger", "id": u}},
            _submit(e_eid, _oracle_findings(e_dir)),
        ):
            obs, reward, _, _, info = env.step(action)
            rewards.append(reward)
            infos.append(info)
        ok = all(type(r) is float and r == 0.0 for r in rewards)
        ok = ok and obs["result"] == {"accepted": True} and env.grade_log[-1]["reward"] > 0.9
        text = json.dumps([obs, infos])
        ok = ok and "tp" not in json.loads(text)[1][-1] and '"f1"' not in text
        return ok, f"policy rewards {rewards}, grader {env.grade_log[-1]['reward']}"

    def c20():
        secrets: set[str] = set()
        for d in (c_dir, e_dir):
            ep = read_episode(d)
            secrets |= {i.issue_id for i in ep.private.truth.issues}
            secrets.add(str(d))
        texts: list[str] = []
        for d, pub in ((c_dir, c_pub), (e_dir, e_ep.public)):
            docs = [x.doc_id for x in (*pub.invoices, *pub.payments)]
            actions: list[dict[str, Any]] = []
            for table in ("ledger", "invoices", "payments", "fx"):
                actions.append({"name": "list_records", "args": {"table": table}})
            actions += [{"name": "get_document", "args": {"doc_id": x}} for x in docs]
            actions += [
                {"name": "get_document", "args": {"doc_id": "../private/truth.json"}},
                {"name": "get_record", "args": {"table": "private", "id": "truth.json"}},
                {"name": "calculator", "args": {"expression": "__import__('os')"}},
            ]
            env = None
            for k, action in enumerate(actions):
                if k % 90 == 0:
                    env = make_env()
                    obs, info = env.reset(None, _closed(d))
                    texts.append(json.dumps([obs, info]))
                out = env.step(action)
                texts.append(json.dumps(out))
                if action["name"] == "list_records":  # walk every page of the table
                    cursor = out[0]["result"]["next_cursor"]
                    while cursor is not None:
                        out = env.step(
                            {"name": "list_records", "args": {**action["args"], "cursor": cursor}}
                        )
                        texts.append(json.dumps(out))
                        cursor = out[0]["result"]["next_cursor"]
        blob = "\n".join(texts)
        leaks = sorted(s for s in secrets if s in blob)
        bad = [
            w for w in ("Traceback", "provenance", "expected_amount_minor", '"seed"') if w in blob
        ]
        return not leaks and not bad, f"{len(texts)} exchanges, leaks {leaks + bad}"

    checks = {
        "C01": c01,
        "C02": c02,
        "C03": c03,
        "C04": c04,
        "C05": c05,
        "C06": c06,
        "C07": c07,
        "C08": c08,
        "C09": c09,
        "C10": c10,
        "C11": c11,
        "C12": c12,
        "C13": c13,
        "C14": c14,
        "C15": c15,
        "C16": c16,
        "C17": c17,
        "C18": c18,
        "C19": c19,
        "C20": c20,
    }
    return _run(checks)


# ---------------------------------------------------------------------------------- block 2
def d_matrix() -> dict[str, tuple[int, dict[str, Any]]]:
    """The jury's scenario matrix as configs of our generator (seeds are ours)."""
    clean = {"config_id": "judge.d.clean", "clean": True}
    m: dict[str, tuple[int, dict[str, Any]]] = {
        "D01": (7001, clean),
        "D02": (7002, {**clean, "distractors": {"split_payment": [3, 3]}}),
        "D03": (7003, {**clean, "distractors": {"credit_note_refund": [2, 2]}}),
        "D04": (
            7004,
            {**clean, "distractors": {"same_amount_orders": [2, 2], "subscription": [3, 3]}},
        ),
        "D05": (
            7005,
            {
                **clean,
                "distractors": {
                    "boundary_offset": [3, 3],
                    "fx_spike": [1, 1],
                    "posting_lag": [4, 4],
                },
            },
        ),
    }
    # Each error episode orders one variant through the documented `variants` config section, so
    # the matrix covers what the jury asks for: a missing and a partial allocation, a wrong date
    # and a wrong pair, an invoice and a payment posted in the wrong period.
    plan = [
        ("D06", "duplicate_invoice", "same_invoice_id"),
        ("D07", "duplicate_invoice", "new_invoice_id"),
        ("D08", "duplicate_invoice", None),
        ("D09", "payment_allocation", "none"),
        ("D10", "payment_allocation", "partial"),
        ("D11", "payment_allocation", "other_invoice"),
        ("D12", "fx_conversion", "wrong_date"),
        ("D13", "fx_conversion", "wrong_pair"),
        ("D14", "period_cutoff", "invoice_posted_late"),
        ("D15", "period_cutoff", "payment_posted_late"),
    ]
    for k, (name, issue_type, variant) in enumerate(plan):
        cfg = _none_cfg(f"judge.d.{issue_type}", errors={issue_type: [1, 1]})
        if variant is not None:
            cfg["variants"] = {issue_type: [variant]}
        m[name] = (7006 + k, cfg)
    for k, name in enumerate(("D16", "D17", "D18")):
        m[name] = (
            7016 + k,
            _none_cfg("judge.d.fxb", interactions={"fx_partial_at_boundary": [1, 1]}),
        )
    for k, name in enumerate(("D19", "D20")):
        m[name] = (
            7019 + k,
            _none_cfg("judge.d.dupmis", interactions={"duplicate_misallocated": [1, 1]}),
        )
    return m


def _undo(public: PublicEpisode, changes: list[dict[str, Any]]) -> PublicEpisode:
    """The books before any injection: provenance changes undone in reverse order."""
    tables: dict[str, dict[str, Any]] = {
        "ledger": {r.row_id: r for r in public.ledger},
        "invoices": {d.doc_id: d for d in public.invoices},
        "payments": {d.doc_id: d for d in public.payments},
    }
    build = {
        "ledger": LedgerRow.from_dict,
        "invoices": Invoice.from_dict,
        "payments": Payment.from_dict,
    }
    for ch in reversed(changes):
        table = tables[ch["table"]]
        current = table.get(ch["id"])
        after = None if current is None else current.to_dict()
        if after != ch["after"]:
            raise ValueError(f"provenance 'after' does not match the data for {ch['id']}")
        if ch["before"] is None:
            del table[ch["id"]]
        else:
            table[ch["id"]] = build[ch["table"]](ch["before"])
    return PublicEpisode(
        manifest=public.manifest,
        policy=public.policy,
        ledger=tuple(tables["ledger"].values()),
        invoices=tuple(tables["invoices"].values()),
        payments=tuple(tables["payments"].values()),
        fx=public.fx,
    )


def verify_dir(d: Path) -> dict[str, Any]:
    """Block 2 conditions for one episode directory, from its files only."""
    from auditgym.harness.sweep import verify_episode

    ep = read_episode(d)
    clean = _undo(ep.public, list(ep.private.provenance.get("changes", [])))
    result = verify_episode(ep, clean)
    truth_ids = {i.issue_id for i in ep.private.truth.issues}
    changed = {i for ch in ep.private.provenance.get("changes", []) for i in ch["issue_ids"]}
    if changed != truth_ids:
        result["ok"] = False
        result["problems"].append("provenance changes do not belong to exactly the truth issues")
    return result


def block2(work: Path) -> list[dict[str, Any]]:
    out = []
    for name, (seed, cfg) in d_matrix().items():
        d = work / name
        generate_episode(seed, cfg, str(d))
        res = verify_dir(d)
        ep = read_episode(d)
        types = sorted(i.type for i in ep.private.truth.issues)
        ok = res["ok"]
        if name <= "D05":
            ok = ok and not types
        ordered = cfg.get("variants", {})
        produced = sorted(
            (inj["injector"], inj["variant"])
            for inj in ep.private.provenance.get("injections", [])
            if inj.get("scenario") is None
        )
        for issue_type, allowed in ordered.items():
            ok = ok and all(v in allowed for t, v in produced if t == issue_type)
            ok = ok and any(t == issue_type for t, _ in produced)
        variants = ", ".join(v for _, v in produced)
        detail = f"{types or 'clean'}{f' ({variants})' if variants else ''}; " + (
            "ok" if res["ok"] else "; ".join(res["problems"][:2])
        )
        out.append({"id": name, "ok": ok, "detail": detail})
    return out


# ---------------------------------------------------------------------------------- block 3
def _write_fixture(src: Path, dst: Path, issues: list[dict[str, Any]]) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    eid = json.loads((dst / "manifest.json").read_text("utf-8"))["episode_id"]
    (dst / "private" / "provenance.json").unlink(missing_ok=True)
    truth = {"episode_id": eid, "issues": issues}
    (dst / "private" / "truth.json").write_text(json.dumps(truth, indent=2), "utf-8")


def _expected(tp: int, fp: int, fn: int, steps: int) -> dict[str, Any]:
    p = Fraction(tp, tp + fp) if tp + fp else Fraction(0)
    r = Fraction(tp, tp + fn) if tp + fn else Fraction(0)
    f1 = Fraction(1) if tp + fp + fn == 0 else Fraction(2 * tp, 2 * tp + fp + fn)
    ret = min(Fraction(1), max(Fraction(-1), 2 * f1 - 1 - Fraction(2, 1000) * steps))
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1, "return": ret}


def _same(got: dict[str, Any], want: dict[str, Any]) -> bool:
    for key, value in want.items():
        g = got.get(key)
        if value is None or key in ("tp", "fp", "fn"):
            if g != value or (value is not None and type(g) is not int):
                return False
        elif g is None or abs(float(g) - float(value)) > TOL:
            return False
    return True


INVALID = {"tp": None, "fp": None, "fn": None, "precision": None, "recall": None, "f1": None}


def block3(work: Path) -> list[dict[str, Any]]:
    base = work / "B3src"
    generate_episode(103, ONE_ERROR_CFG, str(base))
    pub = read_episode(base).public
    r = [row.row_id for row in pub.ledger[:5]]
    r1, r2, r3 = r[0], r[1], r[2]
    d1, d2, d3 = pub.invoices[0].doc_id, pub.payments[0].doc_id, pub.invoices[1].doc_id
    u = "U-000000"
    while u in pub.open_ids():
        u += "0"
    t = "duplicate_invoice"
    other_t = "period_cutoff"

    def issue(iid: str, typ: str, affected: list[str], ev: list[list[str]], tol: int | None = None):
        out = {
            "issue_id": iid,
            "type": typ,
            "affected_ids": affected,
            "expected_amount_minor": 10000,
            "currency": "KZT",
            "evidence_sets": ev,
        }
        if tol is not None:
            out["amount_tolerance_minor"] = tol
        return out

    def f(typ=t, affected=None, amount=10000, cur="KZT", ev=None, text="short reason"):
        return {
            "type": typ,
            "affected_ids": [r1] if affected is None else affected,
            "claimed_amount_minor": amount,
            "currency": cur,
            "evidence_ids": [d1, d2] if ev is None else ev,
            "explanation": text,
        }

    fx_a = work / "B3_A"
    _write_fixture(base, fx_a, [issue("A", t, [r1], [[d1, d2]])])
    fx_empty = work / "B3_empty"
    _write_fixture(base, fx_empty, [])
    fx_ab = work / "B3_AB"
    _write_fixture(
        base, fx_ab, [issue("A", t, [r1], [[d1, d2]]), issue("B", other_t, [r2], [[d3]])]
    )
    fx_m = work / "B3_M"  # open examples: A = {R1, R2}, evidence {D1, D2}
    _write_fixture(base, fx_m, [issue("A", t, [r1, r2], [[d1, d2]])])
    fx_m8 = work / "B3_M8"
    _write_fixture(base, fx_m8, [issue("A", t, [r1, r2], [[d1, d2]], tol=1)])
    fx_m10 = work / "B3_M10"  # F0 fits A and B, F1 fits only A
    _write_fixture(base, fx_m10, [issue("A", t, [r1], [[d1]]), issue("B", t, [r1], [[d2]])])

    def score(d: Path, findings: list[dict[str, Any]], steps: int = 10) -> dict[str, Any]:
        eid = json.loads((d / "manifest.json").read_text("utf-8"))["episode_id"]
        return score_episode(str(d), {"episode_id": eid, "findings": findings}, steps)

    def expect(d, findings, tp, fp, fn, steps=10, value=None):
        got = score(d, findings, steps)
        want = _expected(tp, fp, fn, steps)
        if value is not None and abs(float(want["return"]) - value) > TOL:
            return False, f"definition says {value}, formula gives {float(want['return'])}"
        return _same(got, want), f"got {got}"

    def invalid(d, findings):
        got = score(d, findings)
        return _same(got, {**INVALID, "return": -1}), f"got {got}"

    m_ok = f(affected=[r1, r2])
    every = [f(typ=typ, affected=[rid]) for rid in r for typ in ISSUE_TYPES]
    checks: dict[str, Check] = {
        "T01": lambda: expect(fx_a, [f()], 1, 0, 0, value=0.98),
        "T02": lambda: expect(fx_a, [], 0, 0, 1, value=-1.0),
        "T03": lambda: expect(fx_empty, [], 0, 0, 0, value=0.98),
        "T04": lambda: expect(fx_empty, [f()], 0, 1, 0, value=-1.0),
        "T05": lambda: expect(fx_a, [f(), f()], 1, 1, 0, value=47 / 150),
        "T06": lambda: expect(fx_a, [f(typ=other_t)], 0, 1, 1, value=-1.0),
        "T07": lambda: expect(fx_a, [f(affected=[r3])], 0, 1, 1, value=-1.0),
        "T08": lambda: expect(fx_a, [f(amount=10001)], 0, 1, 1, value=-1.0),
        "T09": lambda: expect(fx_a, [f(cur="USD")], 0, 1, 1, value=-1.0),
        "T10": lambda: expect(fx_a, [f(ev=[d1])], 0, 1, 1, value=-1.0),
        "T11": lambda: expect(fx_a, [f(ev=[d1, u])], 0, 1, 1, value=-1.0),
        "T12": lambda: expect(fx_a, every, 1, 19, 0, value=-871 / 1050),
        "T13": lambda: expect(fx_ab, [f()], 1, 0, 1, value=47 / 150),
        "T14": lambda: expect(fx_ab, [f(), f(affected=[r3])], 1, 1, 1, value=-0.02),
        "T15": lambda: expect(fx_a, [f()], 1, 0, 0, steps=0, value=1.0),
        "T16": lambda: expect(fx_a, [f()], 1, 0, 0, steps=50, value=0.90),
        "T17": lambda: expect(fx_a, [f()], 1, 0, 0, steps=100, value=0.80),
        "T18": lambda: invalid(fx_a, [f()] * 31),
        "T19": lambda: invalid(fx_a, [f(ev=[d1, d2] + [f"{d1}-{k}" for k in range(9)])]),
        "T20": lambda: invalid(fx_a, [f(text="a" * 1001)]),
        "M01": lambda: expect(fx_m, [f(affected=[r2, r1, r1])], 1, 0, 0),
        "M02": lambda: expect(fx_m, [f(affected=[r1, r2, r3])], 0, 1, 1),
        "M03": lambda: expect(fx_m, [f(affected=[r1, r2], ev=[d2, d1, d3])], 1, 0, 0),
        "M04": lambda: expect(fx_m, [f(affected=[r1, r2], ev=[d1, d2, u])], 0, 1, 1),
        "M05": lambda: invalid(fx_m, [f(affected=[r1, r2], ev=[d2] + [d1] * 10)]),
        "M06": lambda: _all(
            expect(fx_m, [f(affected=[r1, r2], cur="kzt")], 0, 1, 1),
            expect(fx_m, [f(affected=[r1, r2], cur="KZT ")], 0, 1, 1),
        ),
        "M07": lambda: expect(fx_m, [f(affected=[r1, r2], amount=10001)], 0, 1, 1),
        "M08": lambda: _all(
            expect(fx_m8, [f(affected=[r1, r2], amount=10001)], 1, 0, 0),
            expect(fx_m8, [f(affected=[r1, r2], amount=10002)], 0, 1, 1),
        ),
        "M09": lambda: _all(
            invalid(fx_m, [f(affected=[r1, r2], amount="10000")]),
            invalid(fx_m, [f(affected=[r1, r2], amount=True)]),
        ),
        "M10": lambda: _pairs(score, fx_m10, [f(ev=[d1, d2]), f(ev=[d1])], [(0, "B"), (1, "A")]),
        "M11": lambda: _all(
            _pairs(score, fx_m10, [f(ev=[d1, d2]), f(ev=[d1, d2])], [(0, "A"), (1, "B")]),
            expect(fx_m10, [f(ev=[d1, d2]), f(ev=[d1, d2], text="other order")], 2, 0, 0),
        ),
        "M12": lambda: _all(
            expect(fx_m, [m_ok, copy.deepcopy(m_ok)], 1, 1, 0),
            (
                score(fx_m, [m_ok, m_ok])["return"] == score(fx_a, [f(), f()])["return"],
                "same vector as test 5",
            ),
        ),
    }
    return _run(checks)


def _all(*results: tuple[bool, str]) -> tuple[bool, str]:
    return all(ok for ok, _ in results), " | ".join(d for _, d in results)


def _pairs(score: Callable, d: Path, findings: list, want: list[tuple[int, str]]):
    got = score(d, findings)
    pairs = [tuple(p) for p in SCORE_LOG[-1]["matches"]]
    return got["tp"] == len(want) and pairs == want, f"pairs {pairs}, score {got}"


# ---------------------------------------------------------------------------------- block 4
ID_FIELDS = {
    "ledger": {"row_id": "row", "entity_id": "ent", "invoice_id": "inv", "payment_id": "pay"},
    "invoices": {"doc_id": "doc", "invoice_id": "inv", "entity_id": "ent"},
    "payments": {"doc_id": "doc", "payment_id": "pay"},
}


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    reader = csv.DictReader(io.StringIO(path.read_text("utf-8-sig")))
    return list(reader.fieldnames or []), list(reader)


def _write_csv(path: Path, header: list[str], rows: list[dict[str, str]]) -> None:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=header, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(buf.getvalue(), "utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text("utf-8-sig").splitlines() if x.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), "utf-8")


def jury_transform(
    src: Path, dst: Path, *, permute: bool, rename: bool, salt: str
) -> dict[str, str]:
    """A copy of an episode with records permuted and/or ids replaced in every file.

    Written from the jury's description only (independent of auditgym.harness.transform):
    ids of each namespace get new spellings, every reference to them (ledger links, invoice_refs,
    truth affected_ids and evidence sets, provenance) is rewritten with the same mapping; fx keys
    (date/source/target) are not ids and stay; line items keep their order. Returns the mapping.
    """
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    ledger_header, ledger = _read_csv(dst / "ledger.csv")
    fx_header, fx = _read_csv(dst / "fx.csv")
    invoices = _read_jsonl(dst / "invoices.jsonl")
    payments = _read_jsonl(dst / "payments.jsonl")
    mapping: dict[str, str] = {}

    def new(value: str, space: str) -> str:
        if value in mapping:
            return mapping[value]
        digest = hashlib.sha256(f"{salt}|{space}|{value}".encode()).hexdigest()[:12].upper()
        mapping[value] = f"{space.upper()}_{digest}"
        return mapping[value]

    if rename:
        for table, rows in (("ledger", ledger), ("invoices", invoices), ("payments", payments)):
            for row in rows:
                for field, space in ID_FIELDS[table].items():
                    if row.get(field) not in (None, ""):
                        row[field] = new(row[field], space)
        for pay in payments:
            if pay.get("invoice_refs") is not None:
                pay["invoice_refs"] = [new(x, "inv") for x in pay["invoice_refs"]]
        if len(set(mapping.values())) != len(mapping):
            raise ValueError("id collision")

    def remap(value: Any) -> Any:
        if isinstance(value, str):
            return mapping.get(value, value)
        if isinstance(value, list):
            return [remap(v) for v in value]
        if isinstance(value, dict):
            return {k: remap(v) for k, v in value.items()}
        return value

    if permute:

        def shuffled(rows: list[Any]) -> list[Any]:
            def key(row: Any) -> bytes:
                return hashlib.sha256(f"{salt}|{json.dumps(row, sort_keys=True)}".encode()).digest()

            return sorted(rows, key=key)

        ledger, fx, invoices, payments = map(shuffled, (ledger, fx, invoices, payments))
    _write_csv(dst / "ledger.csv", ledger_header, ledger)
    _write_csv(dst / "fx.csv", fx_header, fx)
    _write_jsonl(dst / "invoices.jsonl", invoices)
    _write_jsonl(dst / "payments.jsonl", payments)
    for name in ("truth.json", "provenance.json"):
        path = dst / "private" / name
        if path.exists():
            path.write_text(
                json.dumps(remap(json.loads(path.read_text("utf-8"))), indent=1), "utf-8"
            )
    return mapping


def _oracle_run(d: Path) -> tuple[list[Any], dict[str, Any], list[dict[str, Any]]]:
    """Three exploration steps, then the oracle's submit. Returns transitions, grade, findings."""
    env = make_env()
    obs, info = env.reset(None, _closed(d))
    eid = obs["manifest"]["episode_id"]
    trace: list[Any] = [obs, info]
    first, *rest = env.step({"name": "list_records", "args": {"table": "ledger", "limit": 5}})
    trace.append([first, *rest])
    rid = first["result"]["records"][0]["row_id"]
    trace.append(env.step({"name": "get_record", "args": {"table": "ledger", "id": rid}}))
    trace.append(env.step({"name": "calculator", "args": {"expression": "(1.5+2)*3"}}))
    findings = _oracle_findings(d)
    trace.append(env.step(_submit(eid, findings)))
    return trace, env.grade_log[-1], findings


GRADE_KEYS = ("tp", "fp", "fn", "precision", "recall", "f1", "reward", "inspection_steps")


def _canon_findings(findings: list[dict[str, Any]], back: dict[str, str]) -> list[tuple]:
    def ids(xs: list[str]) -> tuple[str, ...]:
        return tuple(sorted({back.get(x, x) for x in xs}))

    return sorted(
        (
            f["type"],
            ids(f["affected_ids"]),
            f["claimed_amount_minor"],
            f["currency"],
            ids(f["evidence_ids"]),
        )
        for f in findings
    )


def block4(work: Path) -> list[dict[str, Any]]:
    out = []
    for k, (name, (seed, cfg)) in enumerate(d_matrix().items()):
        pair = f"P{k + 1:02d}"
        src = work / f"{pair}_{name}"
        generate_episode(seed, cfg, str(src))
        permute, rename = (True, False) if k < 5 else (False, True) if k < 10 else (True, True)
        dst = work / f"{pair}_{name}_t"
        mapping = jury_transform(src, dst, permute=permute, rename=rename, salt=f"jury-{pair}")
        back = {v: key for key, v in mapping.items()}
        t1, g1, f1 = _oracle_run(src)
        t2, g1b, _ = _oracle_run(src)
        t3, g2, f2 = _oracle_run(dst)
        same_replay = t1 == t2 and g1 == g1b
        same_grade = all(g1[key] == g2[key] for key in GRADE_KEYS)
        same_answer = _canon_findings(f1, {}) == _canon_findings(f2, back)
        same_score = score_episode(
            str(src), {"episode_id": t1[0]["manifest"]["episode_id"], "findings": f1}, 3
        ) == score_episode(
            str(dst), {"episode_id": t3[0]["manifest"]["episode_id"], "findings": f2}, 3
        )
        ok = same_replay and same_grade and same_answer and same_score and g1["fn"] == 0
        what = "permute" if not rename else "rename" if not permute else "permute+rename"
        out.append(
            {
                "id": pair,
                "ok": ok,
                "detail": f"{name} {what}: replay {same_replay}, grade {same_grade}, "
                f"answer {same_answer}, score {same_score}, reward {g1['reward']}",
            }
        )
    return out


# ---------------------------------------------------------------------------------- all
POINTS = {"block1": 25, "block2": 25, "block3": 25, "block4": 15}


def run_judge_checks(work: str | Path) -> dict[str, Any]:
    """All four blocks in a scratch directory. Returns per-check results and a summary."""
    root = Path(work)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    report: dict[str, Any] = {
        "block1": block1(root / "block1"),
        "block2": block2(root / "block2"),
        "block3": block3(root / "block3"),
        "block4": block4(root / "block4"),
    }
    summary = {}
    for block, results in report.items():
        # Block 3 has 20 scored tests; M01-M12 are the published examples, reported alongside.
        scored = [r for r in results if not r["id"].startswith("M")]
        summary[block] = {
            "passed": _ok(scored),
            "of": len(scored),
            "points": round(POINTS[block] * _ok(scored) / len(scored), 2),
        }
    examples = [r for r in report["block3"] if r["id"].startswith("M")]
    summary["examples_M"] = {"passed": _ok(examples), "of": len(examples)}
    report["summary"] = summary
    return report
