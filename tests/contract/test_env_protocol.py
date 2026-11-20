"""JSON-lines protocol of the server, run in-process over byte streams (E01, E02, E07, Q2)."""

from __future__ import annotations

import io
import json

import pytest
from envkit import RecordingGrader, act, list_page, oracle_submission

from auditgym.contracts.config import ConfigError
from auditgym.env import AuditEnv
from auditgym.fixtures.tiny import build_tiny_episode
from auditgym.server.app import serve
from auditgym.server.protocol import MAX_LINE_BYTES, PROTOCOL_ERROR_MESSAGES, handle_line


def _env(mode="open", generate=None):
    return AuditEnv(
        mode=mode,
        generate_fn=generate or (lambda seed, config=None: build_tiny_episode()),
        grader=RecordingGrader(reward=0.5),
    )


def _serve(lines: list[bytes], env=None) -> list[dict]:
    out = io.BytesIO()
    serve(env or _env(), io.BytesIO(b"".join(lines)), out)
    return [json.loads(x) for x in out.getvalue().decode("utf-8").splitlines()]


def _line(obj) -> bytes:
    return (json.dumps(obj) + "\n").encode("utf-8")


def test_protocol_reset_step_close_round_trip():
    responses = _serve(
        [
            _line({"op": "reset", "seed": 3, "config": None}),
            _line({"op": "step", "action": list_page("ledger", None, 2)}),
            _line(
                {
                    "op": "step",
                    "action": act("submit", findings=oracle_submission(build_tiny_episode())),
                }
            ),
            _line({"op": "close"}),
            _line({"op": "reset", "seed": 3}),  # after close: never read
        ]
    )
    assert len(responses) == 4
    reset, step, submit, close = responses
    env = _env()
    obs, info = env.reset(3)
    assert reset == {"ok": True, "op": "reset", "observation": obs, "info": info}
    o, r, t, tr, i = env.step(list_page("ledger", None, 2))
    assert step == {
        "ok": True,
        "op": "step",
        "observation": o,
        "reward": r,
        "terminated": t,
        "truncated": tr,
        "info": i,
    }
    assert submit["terminated"] is True and submit["reward"] == 0.5
    assert close == {"ok": True, "op": "close"}


def test_protocol_truncate_op():
    responses = _serve(
        [
            _line({"op": "reset", "seed": 1}),
            _line({"op": "truncate"}),
            _line({"op": "truncate"}),
        ]
    )
    assert responses[1]["truncated"] is True and responses[1]["reward"] == 0.5
    assert responses[2]["observation"]["error"]["code"] == "episode_finished"


BAD_LINES = [
    b"not json\n",
    b"\n",
    b"[1, 2]\n",
    b'"reset"\n',
    b"{}\n",
    b'{"op": "explode"}\n',
    b'{"op": "reset"}\n',
    b'{"op": "reset", "seed": "7"}\n',
    b'{"op": "reset", "seed": true}\n',
    b'{"op": "reset", "seed": 1.5}\n',
    b'{"op": "reset", "seed": 1, "config": [1]}\n',
    b'{"op": "reset", "seed": 1, "path": "/etc"}\n',
    b'{"op": "step"}\n',
    b'{"op": "step", "action": {}, "debug": true}\n',
    b'{"op": "close", "now": 1}\n',
    b"\xff\xfe\n",
    b'{"op": "reset", "seed": NaN}\n',
    b"[" * 100_000 + b"\n",
]


@pytest.mark.parametrize("line", BAD_LINES, ids=range(len(BAD_LINES)))
def test_E07_malformed_request_gets_fixed_error_and_server_continues(line):
    responses = _serve([line, _line({"op": "reset", "seed": 1}), _line({"op": "close"})])
    assert responses[0] == {
        "ok": False,
        "error": {"code": "invalid_request", "message": "Malformed request line."},
    }
    assert responses[1]["ok"] is True and responses[1]["op"] == "reset"
    assert responses[2] == {"ok": True, "op": "close"}


def test_E07_over_long_line_is_rejected_and_skipped():
    big = b'{"op": "step", "action": "' + b"x" * (MAX_LINE_BYTES + 10) + b'"}\n'
    responses = _serve([big, _line({"op": "reset", "seed": 1}), _line({"op": "close"})])
    assert responses[0]["error"]["code"] == "invalid_request"
    assert responses[1]["op"] == "reset"
    assert len(responses) == 3


def test_E07_malformed_request_costs_no_step_but_bad_action_does():
    responses = _serve(
        [
            _line({"op": "reset", "seed": 1}),
            b'{"op": "step"}\n',
            _line({"op": "step", "action": "garbage"}),
            _line({"op": "step", "action": list_page("fx", None, 1)}),
        ]
    )
    assert responses[1]["ok"] is False
    assert responses[2]["observation"]["error"]["code"] == "invalid_arguments"
    assert responses[2]["observation"]["steps_remaining"] == 99
    assert responses[3]["observation"]["steps_remaining"] == 98


def test_protocol_step_before_reset_is_an_observation():
    (resp,) = _serve([_line({"op": "step", "action": list_page("ledger")})])
    assert resp["ok"] is True
    assert resp["observation"]["error"]["code"] == "episode_finished"


def test_protocol_config_error_and_internal_error_are_fixed():
    def bad_config(seed, config=None):
        raise ConfigError("unknown key 'secret' in /home/x/configs/a.json")

    def crash(seed, config=None):
        raise RuntimeError("Traceback /home/x/src/auditgym/episode/generate.py")

    resp, close = handle_line(_env(generate=bad_config), _line({"op": "reset", "seed": 1}))
    assert resp == {
        "ok": False,
        "error": {"code": "invalid_config", "message": "Invalid episode config."},
    }
    resp, close = handle_line(_env(generate=crash), _line({"op": "reset", "seed": 1}))
    assert resp == {"ok": False, "error": {"code": "internal_error", "message": "Internal error."}}
    assert set(PROTOCOL_ERROR_MESSAGES.values()) >= {"Internal error."}


def test_protocol_eof_ends_server():
    assert _serve([_line({"op": "reset", "seed": 1})])[0]["ok"] is True


def test_protocol_responses_are_canonical_json():
    out = io.BytesIO()
    serve(_env(), io.BytesIO(_line({"op": "reset", "seed": 1})), out)
    raw = out.getvalue().decode("utf-8")
    assert raw.endswith("\n") and raw.count("\n") == 1
    data = json.loads(raw)
    assert raw == json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
