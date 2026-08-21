"""Train the recurrent relational solver and write a checkpoint.

Trains on grids of at most 4 houses so that 5- and 6-house grids stay a genuine
test of generalisation. Run: uv run python experiments/train_solver.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from latticebench.models.solver import RecurrentSolver
from latticebench.models.train import build_dataset, save_solver, train_solver

TRAIN_SIZES = [(3, 2), (3, 3), (4, 3), (4, 4), (5, 4), (6, 4)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-size", type=int, default=400)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--mp-steps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/solver.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    print(f"generating {args.per_size} puzzles per size for {TRAIN_SIZES} ...")
    data = build_dataset(TRAIN_SIZES, args.per_size, seed=1000)

    model = RecurrentSolver(hidden=args.hidden, steps=args.mp_steps)
    train_solver(model, data, steps=args.steps, seed=args.seed)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_solver(model, args.out, {"hidden": args.hidden, "steps": args.mp_steps, "type_dim": 32})
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
