"""Command-line interface: ``generate``, ``evaluate``, ``leaderboard``, ``verify``."""

from __future__ import annotations

import argparse
import glob
import json
from datetime import date
from pathlib import Path
from typing import Any

from latticebench.contamination import make_split, verify_regeneration
from latticebench.dataset import read_jsonl, write_jsonl
from latticebench.harness import EBMModel, LLMModel, SolverModel, evaluate
from latticebench.harness.base import Model
from latticebench.leaderboard import build_leaderboard, load_results


def _build_model(spec: str) -> Model:
    parts = spec.split(":")
    kind = parts[0]
    if kind == "solver":
        return SolverModel(solver=parts[1] if len(parts) > 1 else "ortools")
    if kind == "ebm":
        return EBMModel()
    if kind == "llm":
        if len(parts) < 3:
            raise SystemExit("llm spec must be llm:<provider>:<model>")
        provider, model = parts[1], ":".join(parts[2:])
        if provider == "anthropic":
            from latticebench.harness.llm import AnthropicChat

            client: Any = AnthropicChat(model)
        elif provider == "openai":
            from latticebench.harness.llm import OpenAIChat

            client = OpenAIChat(model)
        else:
            raise SystemExit(f"unknown llm provider: {provider}")
        return LLMModel(client, name=f"llm:{model}")
    raise SystemExit(f"unknown model spec: {spec}")


def cmd_generate(args: argparse.Namespace) -> int:
    records = make_split(
        args.seed,
        n=args.n,
        m=args.m,
        count=args.count,
        created=args.created,
        split=args.split,
        include_solution=not args.challenge,
    )
    written = write_jsonl(args.out, records)
    print(f"wrote {written} puzzles to {args.out}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    model = _build_model(args.model)
    result = evaluate(model, read_jsonl(args.puzzles))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    s = result.summary
    print(
        f"{model.name}: exact={s['exact_match']:.3f} cell={s['cell_accuracy']:.3f} "
        f"over {int(s['n'])} puzzles -> {args.out}"
    )
    return 0


def cmd_leaderboard(args: argparse.Namespace) -> int:
    paths: list[str] = []
    for pattern in args.results:
        paths.extend(sorted(glob.glob(pattern)))
    if not paths:
        raise SystemExit("no result files matched")
    board = build_leaderboard(load_results(paths))
    text = board.to_json() if args.format == "json" else board.to_markdown()
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"wrote leaderboard to {args.out}")
    else:
        print(text)
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    ok = bad = 0
    for rec in read_jsonl(args.puzzles):
        if verify_regeneration(rec):
            ok += 1
        else:
            bad += 1
    print(f"verified {ok} ok, {bad} failed")
    return 1 if bad else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="latticebench")
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="mint a split of puzzles to JSONL")
    g.add_argument("--n", type=int, required=True, help="houses per row")
    g.add_argument("--m", type=int, required=True, help="attributes")
    g.add_argument("--count", type=int, default=100)
    g.add_argument("--seed", type=int, required=True, help="master seed")
    g.add_argument("--created", default=date.today().isoformat())
    g.add_argument("--split", default="default")
    g.add_argument("--out", required=True)
    g.add_argument("--challenge", action="store_true", help="omit solutions")
    g.set_defaults(func=cmd_generate)

    e = sub.add_parser("evaluate", help="run a model over a split")
    e.add_argument("--model", required=True, help="solver[:backend] | ebm | llm:<provider>:<model>")
    e.add_argument("--puzzles", required=True)
    e.add_argument("--out", required=True)
    e.set_defaults(func=cmd_evaluate)

    lb = sub.add_parser("leaderboard", help="aggregate results into a table")
    lb.add_argument("--results", nargs="+", required=True, help="result JSON paths or globs")
    lb.add_argument("--format", choices=("md", "json"), default="md")
    lb.add_argument("--out", default=None)
    lb.set_defaults(func=cmd_leaderboard)

    v = sub.add_parser("verify", help="regenerate every instance and check hashes")
    v.add_argument("--puzzles", required=True)
    v.set_defaults(func=cmd_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
