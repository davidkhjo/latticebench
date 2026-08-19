"""Run the benchmark across a range of grid sizes and produce the paper artifacts.

Generates manifest-backed splits, evaluates every model on the same instances,
and writes the results table and figures. The language model is the slow leg, so
it can be turned off (``--no-llm``) or given a smaller instance count.

Run: uv run python experiments/run.py            (full trial)
     uv run python experiments/run.py --count 3 --no-llm   (smoke)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import latticebench as lb
from latticebench import analysis, viz
from latticebench.harness import EBMModel, SolverModel, evaluate
from latticebench.harness.base import EvalResult
from latticebench.harness.gnn import GNNSolverModel
from latticebench.harness.metrics import by_tier, summarize

MASTER_SEED = 20260819
CREATED = "2026-12-31"
SWEEP = [(3, 2), (3, 3), (4, 3), (4, 4), (5, 4), (6, 4)]


def make_models(*, solver_ckpt: str, with_llm: bool, llm_model: str):
    models = [
        SolverModel("ortools"),
        SolverModel("z3"),
        EBMModel(steps=2000, restarts=12, name="ebm:annealing"),
        GNNSolverModel(solver_ckpt),
    ]
    if with_llm:
        from latticebench.harness.llm import LLMModel
        from latticebench.harness.local_llm import MLXClient, short_name, solve_template

        client = MLXClient(llm_model, max_tokens=2048)
        models.append(
            LLMModel(client, name=f"llm:{short_name(llm_model)}", template=solve_template)
        )
    return models


def combine(name: str, results: list[EvalResult]) -> EvalResult:
    scores = [s for r in results for s in r.per_puzzle]
    return EvalResult(name, scores, summarize(scores), by_tier(scores))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=15)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument(
        "--llm-max-n", type=int, default=5, help="skip the LLM on larger grids (it is slow)"
    )
    ap.add_argument("--llm-model", default="mlx-community/Qwen2.5-3B-Instruct-4bit")
    ap.add_argument("--solver-ckpt", default="checkpoints/solver.pt")
    ap.add_argument("--figdir", default="paper/figures")
    args = ap.parse_args()

    figdir = Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    Path("manifests").mkdir(exist_ok=True)

    # generate the splits and record their manifests
    splits: dict[tuple[int, int], list] = {}
    manifests = []
    for i, (n, m) in enumerate(SWEEP):
        seed = MASTER_SEED + i
        split = f"sweep-n{n}m{m}"
        recs = list(lb.make_split(seed, n=n, m=m, count=args.count, created=CREATED, split=split))
        splits[(n, m)] = recs
        lb.write_jsonl(f"data/{split}.jsonl", recs)
        manifests.append(
            lb.build_manifest(
                seed,
                n=n,
                m=m,
                count=args.count,
                created=CREATED,
                split=split,
                generator_version=lb.__version__,
            ).to_dict()
        )
    Path("manifests/sweep.json").write_text(
        json.dumps(
            {"master_seed": MASTER_SEED, "count": args.count, "manifests": manifests}, indent=2
        )
    )

    models = make_models(
        solver_ckpt=args.solver_ckpt, with_llm=not args.no_llm, llm_model=args.llm_model
    )

    per_model_size: dict[str, list[tuple[int, EvalResult]]] = {}
    joined_by_model: dict[str, list[dict]] = {}
    combined = []
    for model in models:
        by_size = []
        joined: list[dict] = []
        is_llm = model.name.startswith("llm:")
        for (n, m), recs in splits.items():
            if is_llm and n > args.llm_max_n:
                continue
            result = evaluate(model, recs)
            Path(f"results/{model.name.replace(':', '_')}-n{n}m{m}.json").write_text(
                json.dumps(result.to_dict(), indent=2)
            )
            by_size.append((n, result))
            joined += analysis.join_scores(result, recs)
            print(f"{model.name:22s} n={n} m={m}: exact={result.summary['exact_match']:.2f}")
        per_model_size[model.name] = by_size
        joined_by_model[model.name] = joined
        combined.append(combine(model.name, [r for _, r in by_size]))

    # main results table
    board = lb.build_leaderboard(combined)
    (figdir / "results_table.md").write_text(board.to_markdown() + "\n")
    print("\n" + board.to_markdown())

    # figures
    size_data = {name: analysis.size_accuracy(bs) for name, bs in per_model_size.items()}
    viz.plot_accuracy_vs_size(size_data, str(figdir / "accuracy_vs_size.png"))

    learners = [n for n in joined_by_model if n.startswith(("llm:", "gnn"))]
    if learners:
        focus = learners[0]
        viz.plot_conflict_vs_accuracy(
            joined_by_model[focus], str(figdir / "conflict_vs_accuracy.png")
        )
        for c in combined:
            if c.model == focus:
                viz.plot_energy_calibration(c, str(figdir / "energy_calibration.png"))

    agreement = analysis.difficulty_agreement(joined_by_model)
    (figdir / "difficulty_agreement.json").write_text(json.dumps(agreement, indent=2))
    print("difficulty agreement:", agreement)
    print("wrote figures + tables to", figdir)


if __name__ == "__main__":
    main()
