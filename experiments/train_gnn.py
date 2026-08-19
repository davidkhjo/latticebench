"""Train the graph energy model and write a checkpoint.

Trains on grids up to 4 houses so we can measure generalisation to 5 later.
Run: uv run python experiments/train_gnn.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from latticebench.models.gnn import GNNEnergy
from latticebench.models.train import build_dataset, finetune_cd, pretrain_pl, save_checkpoint

TRAIN_SIZES = [(3, 2), (3, 3), (4, 3), (4, 4)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-size", type=int, default=400)
    ap.add_argument("--pl-steps", type=int, default=1500)
    ap.add_argument("--cd-steps", type=int, default=1500)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/gnn.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    print(f"generating {args.per_size} puzzles per size for {TRAIN_SIZES} ...")
    data = build_dataset(TRAIN_SIZES, args.per_size, seed=1000)

    model = GNNEnergy(hidden=args.hidden, rounds=args.rounds)
    print("pseudo-likelihood pretraining ...")
    pretrain_pl(model, data, steps=args.pl_steps, seed=args.seed)
    print("contrastive-divergence fine-tuning ...")
    finetune_cd(model, data, steps=args.cd_steps, seed=args.seed)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, args.out, {"hidden": args.hidden, "rounds": args.rounds, "type_dim": 32})
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
