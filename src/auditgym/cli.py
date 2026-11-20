"""Command-line entry point: python -m auditgym <command>.

Owned by the integrator. Packages expose functions; commands are wired here.
Heavy imports happen inside commands so `--help` works even if a package is broken.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from auditgym.contracts.versions import (
    ENV_VERSION,
    GENERATOR_VERSION,
    GRADER_VERSION,
    SCHEMA_VERSION,
)

REPO = Path(__file__).resolve().parents[2]
POLICY_NAMES = ("random", "reconcile", "flag_all", "oracle")


# ---------------------------------------------------------------------------------- helpers
def _config(path: str | None) -> dict[str, Any]:
    from auditgym.contracts.config import load_config, normalize_config

    return load_config(path) if path else normalize_config({})


def _policy(name: str, episode: Any) -> Any:
    from auditgym.policies.reconcile import Reconcile
    from auditgym.policies.simple import FlagAll, Oracle, RandomExplorer

    if name == "random":
        return RandomExplorer(seed=0)
    if name == "reconcile":
        return Reconcile()
    if name == "flag_all":
        return FlagAll()
    if name == "oracle":
        return Oracle(episode.private.truth)
    raise SystemExit(f"unknown policy {name!r}; choose from {POLICY_NAMES}")


def _splits() -> dict[str, Any]:
    return json.loads((REPO / "configs" / "splits.json").read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    from auditgym.util.canon import dumps_pretty

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_pretty(obj), encoding="utf-8")


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True))


# ---------------------------------------------------------------------------------- commands
def _cmd_version(_: argparse.Namespace) -> int:
    print(
        f"schema {SCHEMA_VERSION}  generator {GENERATOR_VERSION}  env {ENV_VERSION}  grader {GRADER_VERSION}"
    )
    return 0


def _cmd_fixture(args: argparse.Namespace) -> int:
    from auditgym.contracts.io import fingerprint, write_episode
    from auditgym.fixtures.tiny import build_tiny_episode

    episode = build_tiny_episode()
    write_episode(episode, args.out, include_private=not args.public_only)
    print(f"wrote {args.out}  fingerprint {fingerprint(episode)[:16]}")
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    from auditgym.contracts.io import fingerprint, write_episode
    from auditgym.episode.generate import generate_episode

    episode = generate_episode(args.seed, _config(args.config))
    write_episode(episode, args.out, include_private=not args.public_only)
    pub = episode.public
    print(
        f"wrote {args.out}: {pub.episode_id}, {len(pub.ledger)} ledger rows, "
        f"{len(pub.invoices) + len(pub.payments)} documents, fingerprint {fingerprint(episode)[:16]}"
    )
    return 0


def _load_or_generate(args: argparse.Namespace) -> Any:
    if getattr(args, "episode_dir", None):
        from auditgym.contracts.io import read_episode

        return read_episode(args.episode_dir)
    from auditgym.episode.generate import generate_episode

    return generate_episode(args.seed, _config(args.config))


def _cmd_run(args: argparse.Namespace) -> int:
    from auditgym.harness.run import run_episode
    from auditgym.util.canon import jsonl

    episode = _load_or_generate(args)
    run = run_episode(_policy(args.policy, episode), episode, mode=args.mode, seed=args.seed)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(jsonl(run.trajectory), encoding="utf-8")
    _print(run.summary())
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    """Readable walk-through of one episode with the reconcile policy."""
    from auditgym.env import AuditEnv

    episode = _load_or_generate(args)
    grades: list[dict[str, Any]] = []
    env = AuditEnv(mode="open", grader_log=grades.append)
    policy = _policy(args.policy, episode)
    obs, info = env.reset_with_episode(episode)
    m = obs["manifest"]
    print(
        f"Episode {m['episode_id']}: {m['company']}, base {m['base_currency']}, "
        f"period {m['period_start']} .. {m['period_end']} ({m['timezone']})"
    )
    print(f"Records: {m['counts']}")
    policy.reset(obs, info)
    done = False
    while not done:
        action = policy.act(obs)
        obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        args_short = json.dumps(action.get("args", {}), ensure_ascii=False)
        if action["action"] == "submit":
            n = len(action["args"]["findings"]["findings"])
            args_short = f"{n} findings"
        result = obs.get("result") or {}
        brief = (
            f"{len(result['records'])} records"
            if "records" in result
            else obs["error"]["code"]
            if obs.get("error")
            else "ok"
        )
        print(
            f"  step {MAX_STEPS_HINT - obs['steps_remaining']:>3}  {action['action']:<13} {args_short[:70]:<70} -> {brief}"
        )
    g = grades[-1]
    print(
        f"Reward {g['reward']}  F1 {g['f1']}  TP {g['tp']} FP {g['fp']} FN {g['fn']}  "
        f"inspection steps {g['inspection_steps']}"
    )
    print(
        f"Truth had {len(episode.private.truth.issues)} issues: "
        + ", ".join(sorted(i.type for i in episode.private.truth.issues))
    )
    return 0


MAX_STEPS_HINT = 100


def _cmd_sweep(args: argparse.Namespace) -> int:
    from auditgym.harness.sweep import sweep

    seeds = list(range(args.start, args.start + args.seeds))
    report = sweep(_config(args.config), seeds)
    if args.out:
        _write_json(Path(args.out), report)
    _print(
        {k: v for k, v in report.items() if k != "failures"}
        | {"failures_shown": len(report["failures"])}
    )
    for f in report["failures"][:5]:
        print(f"FAIL seed {f['seed']}: {f['problems'][:2]}")
    c = report["counts"]
    return 0 if c.get("ok", 0) == c.get("episodes", -1) and not c.get("generation_errors") else 1


def _cmd_export(args: argparse.Namespace) -> int:
    """Write the open train/dev episodes and the published trajectories."""
    import shutil

    from auditgym.contracts.config import load_config
    from auditgym.contracts.io import fingerprint, write_episode
    from auditgym.episode.generate import generate_episode
    from auditgym.harness.run import run_episode
    from auditgym.util.canon import jsonl

    out_root = Path(args.out)
    for split in ("train", "dev"):  # generated artifacts only: rebuilt from seeds every time
        shutil.rmtree(out_root / split, ignore_errors=True)
    index = []
    trajectories: list[dict[str, Any]] = []
    for split in ("train", "dev"):
        for spec in _splits()[split]:
            cfg = load_config(REPO / "configs" / f"{spec['config']}.json")
            for seed in spec["seeds"]:
                episode = generate_episode(seed, cfg)
                target = out_root / split / episode.public.episode_id
                write_episode(episode, target, include_private=True)
                index.append(
                    {
                        "split": split,
                        "seed": seed,
                        "config_id": cfg["config_id"],
                        "episode_id": episode.public.episode_id,
                        "fingerprint": fingerprint(episode),
                        "issues": len(episode.private.truth.issues),
                        "ledger_rows": len(episode.public.ledger),
                        "documents": len(episode.public.invoices) + len(episode.public.payments),
                    }
                )
                if split == "train":
                    for name in ("random", "reconcile", "flag_all"):
                        run = run_episode(_policy(name, episode), episode, mode="open", seed=seed)
                        for line in run.trajectory:
                            trajectories.append({**line, "policy": name})
    _write_json(
        out_root / "index.json", {"generator_version": GENERATOR_VERSION, "episodes": index}
    )
    traj_path = REPO / "trajectories" / "trajectories.jsonl"
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    traj_path.write_text(jsonl(trajectories), encoding="utf-8")
    print(
        f"wrote {len(index)} episodes to {out_root} and {len(trajectories)} trajectory lines to {traj_path}"
    )
    return 0


def _cmd_transform(args: argparse.Namespace) -> int:
    from auditgym.contracts.io import read_episode, write_episode
    from auditgym.harness.transform import transform_episode

    t = transform_episode(read_episode(args.input), args.salt)
    write_episode(t.episode, args.out)
    _write_json(Path(args.out) / "private" / "id_map.json", t.id_map)
    print(f"wrote {args.out} ({len(t.id_map)} ids renamed)")
    return 0


def _summarize_jury(report: dict[str, Any]) -> list[str]:
    lines = [f"Score estimate: {report['score_estimate']}"]
    for key in ("K1", "K2", "K3", "K4"):
        rows = report[key]
        bad = [r for r in rows if not r["ok"]]
        lines.append(f"{key}: {len(rows) - len(bad)}/{len(rows)} passed")
        lines += [f"  FAIL {r['check']}: {r['detail']}" for r in bad[:10]]
    lines.append("K5: " + ", ".join(f"{i['item']}={i['points']}" for i in report["K5"]))
    for name, row in report.get("trick_probes", {}).items():
        lines.append(
            f"  probe {name:<24} F1 {row['f1_micro']}  P {row['precision']}  R {row['recall']}  reward {row['mean_reward']}"
        )
    for policy, cats in report.get("baselines", {}).items():
        a = cats.get("all", {})
        lines.append(
            f"  {policy:<16} reward {a.get('reward')}  F1 {a.get('f1')}  P {a.get('precision')}  "
            f"R {a.get('recall')}  steps {a.get('steps')}  s {a.get('seconds')}"
        )
    return lines


def _cmd_jury(args: argparse.Namespace) -> int:
    from auditgym.harness.jury import run_jury

    report = run_jury(args.seed, with_baselines=not args.no_baselines)
    _write_json(Path(args.out), report)
    print("\n".join(_summarize_jury(report)))
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    """The single verification command: tests, generator sweep, jury simulation."""
    from auditgym.harness.jury import run_jury
    from auditgym.harness.sweep import sweep

    started = time.perf_counter()
    status: dict[str, Any] = {}
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    if not args.full:
        cmd += ["-m", "not slow"]
    print("== 1/3 tests:", " ".join(cmd[2:]))
    status["tests_exit_code"] = subprocess.call(cmd, cwd=REPO)

    print(f"== 2/3 generator sweep: {args.sweep_seeds} seeds per config")
    sweeps = {}
    for name in ("train.v1", "dev.v1", "clean.v1", "interacting.v1"):
        rep = sweep(
            _config(str(REPO / "configs" / f"{name}.json")),
            list(range(50_000, 50_000 + args.sweep_seeds)),
        )
        sweeps[name] = rep
        c = rep["counts"]
        print(
            f"  {name:<16} ok {c.get('ok', 0)}/{c.get('episodes', 0)}  retries {c.get('retries', 0)}  "
            f"generation errors {c.get('generation_errors', 0)}"
        )
    status["sweep_ok"] = all(
        r["counts"].get("ok", 0) == r["counts"].get("episodes", -1)
        and not r["counts"].get("generation_errors")
        for r in sweeps.values()
    )

    print("== 3/3 jury simulation")
    jury = run_jury(args.jury_seed, with_baselines=True)
    print("\n".join(_summarize_jury(jury)))
    status["jury_all_checks_passed"] = all(
        r["ok"] for k in ("K1", "K2", "K3", "K4") for r in jury[k]
    )
    status["seconds"] = round(time.perf_counter() - started, 1)

    out = Path(args.out)
    _write_json(out / "check_report.json", {"status": status, "sweeps": sweeps, "jury": jury})
    ok = status["tests_exit_code"] == 0 and status["sweep_ok"] and status["jury_all_checks_passed"]
    print(
        f"== RESULT: {'PASS' if ok else 'FAIL'}  ({status['seconds']} s)  report: {out / 'check_report.json'}"
    )
    return 0 if ok else 1


def _cmd_serve(args: argparse.Namespace) -> int:
    from auditgym.server.app import main as serve_main

    return int(serve_main(args.server_args) or 0)


# ---------------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="auditgym", description="AuditGym RL environment")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("version", help="print component versions")
    p.set_defaults(func=_cmd_version)

    p = sub.add_parser("fixture", help="write the tiny hand-built episode to a directory")
    p.add_argument("--out", default="runs/fixture_tiny")
    p.add_argument("--public-only", action="store_true")
    p.set_defaults(func=_cmd_fixture)

    p = sub.add_parser("generate", help="generate one episode to a directory")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--config", default=None, help="path to a config JSON (default: built-in defaults)"
    )
    p.add_argument("--out", required=True)
    p.add_argument("--public-only", action="store_true")
    p.set_defaults(func=_cmd_generate)

    for name, func, helptext in (
        ("run", _cmd_run, "run one policy on one episode and print diagnostics"),
        ("demo", _cmd_demo, "readable step-by-step run of one episode"),
    ):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--policy", choices=POLICY_NAMES, default="reconcile")
        p.add_argument("--seed", type=int, default=1000)
        p.add_argument("--config", default=str(REPO / "configs" / "train.v1.json"))
        p.add_argument("--episode-dir", default=None, help="load an episode from disk instead")
        if name == "run":
            p.add_argument("--mode", choices=("open", "closed"), default="open")
            p.add_argument("--out", default=None, help="write the trajectory as JSONL")
        p.set_defaults(func=func)

    p = sub.add_parser("sweep", help="generate and independently verify many episodes")
    p.add_argument("--config", default=str(REPO / "configs" / "train.v1.json"))
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--out", default=None)
    p.set_defaults(func=_cmd_sweep)

    p = sub.add_parser("export", help="write open train/dev episodes and trajectories.jsonl")
    p.add_argument("--out", default=str(REPO / "episodes"))
    p.set_defaults(func=_cmd_export)

    p = sub.add_parser("transform", help="permute rows and rename ids of a stored episode")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--salt", default="paired-check")
    p.set_defaults(func=_cmd_transform)

    p = sub.add_parser("jury", help="simulate the organizers' closed check and estimate the score")
    p.add_argument("--seed", type=int, default=20261122)
    p.add_argument("--out", default="runs/jury/report.json")
    p.add_argument("--no-baselines", action="store_true")
    p.set_defaults(func=_cmd_jury)

    p = sub.add_parser("check", help="the single verification command")
    p.add_argument("--full", action="store_true", help="include slow tests")
    p.add_argument("--sweep-seeds", type=int, default=100)
    p.add_argument("--jury-seed", type=int, default=20261122)
    p.add_argument("--out", default="runs/check")
    p.set_defaults(func=_cmd_check)

    p = sub.add_parser(
        "serve",
        help="grader-side server speaking JSON lines on stdin/stdout "
        "(options: --mode, --grader-log, --episode-dir)",
    )
    p.add_argument("server_args", nargs=argparse.REMAINDER)
    p.set_defaults(func=_cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))
