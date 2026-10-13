"""CLI: score, replay, and diff traces. No API key needed for any of it."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .replay import replay_trace
from .score import compare_traces, score_trace
from .trace import Trace


def _load(path: str) -> Trace:
    return Trace.load(path)


def _cmd_score(args: argparse.Namespace) -> int:
    trace = _load(args.trace)
    card = score_trace(
        trace,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        require_tools=args.require_tool or (),
        forbid_tools=args.forbid_tool or (),
    )
    print(card.summary())
    if args.json:
        print(json.dumps(card.to_dict(), indent=2, default=str))
    for v in card.violations:
        print(f"  ! {v}", file=sys.stderr)
    return 0 if card.passed else 1


def _cmd_replay(args: argparse.Namespace) -> int:
    trace = _load(args.trace)
    result = replay_trace(trace, times=args.times, verify=True)
    print(f"steps={result.steps_replayed} calls={result.calls_replayed} "
          f"replays={result.replay_count} deterministic={result.deterministic}")
    print(f"fingerprint={result.fingerprint[:24]}")
    for n in result.notes:
        print(f"  ! {n}", file=sys.stderr)
    if args.out:
        result.trace.save(args.out)
        print(f"wrote {args.out}")
    return 0 if result.deterministic else 1


def _cmd_diff(args: argparse.Namespace) -> int:
    old, new = _load(args.baseline), _load(args.candidate)
    drift = compare_traces(old, new)
    print(drift.report())
    if args.json:
        print(json.dumps(drift.to_dict(), indent=2, default=str))
    return 0 if drift.is_clean else 1


def _cmd_policy(args: argparse.Namespace) -> int:
    from .gate import Policy

    policy = Policy(args.name)
    for spec in args.allow or []:
        policy.allow(spec)
    for spec in args.deny or []:
        policy.deny(spec)
    print(json.dumps(policy.to_dict(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="replaygraph",
        description="Deterministic replay and evaluation for LangGraph agents.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="grade a trace against observable invariants")
    s.add_argument("trace")
    s.add_argument("--max-steps", type=int, default=12)
    s.add_argument("--max-tokens", type=int, default=32000)
    s.add_argument("--require-tool", action="append", default=[])
    s.add_argument("--forbid-tool", action="append", default=[])
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=_cmd_score)

    r = sub.add_parser("replay", help="re-execute a trace offline; proves determinism")
    r.add_argument("trace")
    r.add_argument("--times", type=int, default=3)
    r.add_argument("--out")
    r.set_defaults(fn=_cmd_replay)

    d = sub.add_parser("diff", help="compare two traces of the same graph")
    d.add_argument("baseline")
    d.add_argument("candidate")
    d.add_argument("--json", action="store_true")
    d.set_defaults(fn=_cmd_diff)

    po = sub.add_parser("policy", help="inspect a default-deny policy")
    po.add_argument("--name", default="default")
    po.add_argument("--allow", action="append", default=[])
    po.add_argument("--deny", action="append", default=[])
    po.set_defaults(fn=_cmd_policy)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # CLI must not traceback at users
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
