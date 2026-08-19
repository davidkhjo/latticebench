"""Training for the graph energy model.

Two stages. Pseudo-likelihood pretraining shapes the single-position conditionals
(hold every item at the truth, sweep one item over its houses, and push energy
down on the right house) -- no sampling, so it converges quickly and gives the
later sampler a sane landscape to walk. Contrastive divergence then fine-tunes
against negatives drawn by Gibbs-with-gradients, which is the genuinely
energy-based part.
"""

from __future__ import annotations

from collections.abc import Sequence

import ebm
import torch
import torch.nn.functional as F
from torch import Tensor

from latticebench.energy import EnergyGrid
from latticebench.generate import generate_puzzle
from latticebench.models.gnn import GNNEnergy, one_hot_init, to_tensors
from latticebench.models.graph import PuzzleGraph, batch_graphs, build_graph
from latticebench.models.solver import RecurrentSolver

Bucket = tuple[list[PuzzleGraph], Tensor]  # graphs and (N, I, n) true configs
Dataset = dict[tuple[int, int], Bucket]


def build_dataset(sizes: Sequence[tuple[int, int]], per_size: int, *, seed: int = 0) -> Dataset:
    data: Dataset = {}
    counter = seed
    for n, m in sizes:
        graphs: list[PuzzleGraph] = []
        configs = []
        for _ in range(per_size):
            counter += 1
            p = generate_puzzle(n, m, seed=counter)
            assert p.solution is not None
            grid = EnergyGrid(p.domain, p.clues)
            graphs.append(build_graph(p))
            configs.append(
                torch.tensor(grid.encode(p.solution), dtype=torch.float32).reshape(m * n, n)
            )
        data[(n, m)] = (graphs, torch.stack(configs))
    return data


def _sample_batch(
    bucket: Bucket, batch: int, gen: torch.Generator
) -> tuple[list[PuzzleGraph], Tensor]:
    graphs, configs = bucket
    idx = torch.randint(len(graphs), (batch,), generator=gen)
    return [graphs[i] for i in idx.tolist()], configs[idx]


class _SweepBatch:
    """A fixed minibatch with the graph tensors for its single-item energy sweep
    precomputed, so pseudo-likelihood steps are pure tensor work."""

    def __init__(self, graphs: list[PuzzleGraph], x_pos: Tensor) -> None:
        self.x_pos = x_pos
        self.I = graphs[0].num_items
        self.n = graphs[0].n
        reps = self.I * self.n  # one graph copy per (item, house) candidate
        self.gt = to_tensors(batch_graphs([g for g in graphs for _ in range(reps)]))
        self._ar = torch.arange(self.I)
        self._eye = torch.eye(self.n)

    def loss(self, model: GNNEnergy) -> Tensor:
        B, I, n = self.x_pos.shape
        # candidate (b, j, h): the truth with item j moved to house h
        cand = self.x_pos.view(B, 1, 1, I, n).repeat(1, I, n, 1, 1)
        cand[:, self._ar, :, self._ar, :] = self._eye
        energy = model(self.gt, cand.reshape(B * I * n, I, n)).view(B * I, n)
        target = self.x_pos.argmax(-1).reshape(B * I)
        return F.cross_entropy(-energy, target)


def _fixed_minibatches(data: Dataset, batch: int, gen: torch.Generator) -> list[_SweepBatch]:
    out = []
    for graphs, configs in data.values():
        n = len(graphs)
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n - n % batch, batch):
            idx = perm[i : i + batch]
            out.append(_SweepBatch([graphs[k] for k in idx.tolist()], configs[idx]))
    return out


def pretrain_pl(
    model: GNNEnergy,
    data: Dataset,
    *,
    steps: int = 1500,
    batch: int = 32,
    lr: float = 1e-3,
    clip: float = 1.0,
    seed: int = 0,
    log_every: int = 200,
) -> list[dict]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    batches = _fixed_minibatches(data, batch, gen)
    order = torch.randperm(len(batches), generator=gen).tolist()
    history = []
    for step in range(steps):
        mb = batches[order[step % len(order)]]
        loss = mb.loss(model)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        if step % log_every == 0:
            v = float(loss.detach())
            history.append({"step": step, "pl": v})
            print(f"[pl {step:>5}] pl={v:.4f}")
    return history


def finetune_cd(
    model: GNNEnergy,
    data: Dataset,
    *,
    steps: int = 2000,
    batch: int = 64,
    lr: float = 2e-4,
    gwg_mult: int = 3,
    energy_reg: float = 0.1,
    clip: float = 1.0,
    seed: int = 0,
    log_every: int = 200,
) -> list[dict]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    buckets = list(data.keys())

    def _init_for(n: int):
        return lambda s: one_hot_init(s, n)

    buffers = {}
    for n, m in data:
        buffers[(n, m)] = ebm.ReplayBuffer(8000, (m * n, n), reinit_prob=0.05, init_fn=_init_for(n))

    history = []
    for step in range(steps):
        n, m = buckets[step % len(buckets)]
        I = m * n
        graphs, x_pos = _sample_batch(data[(n, m)], batch, gen)
        gt = to_tensors(batch_graphs(graphs))
        sampler = ebm.CategoricalGibbsWithGradients(steps=gwg_mult * I)

        x_init = buffers[(n, m)].sample(batch)
        # restart a few chains from the data to keep negatives near the manifold
        k = batch // 4
        x_init[:k] = x_pos[:k]
        x_neg = sampler.sample(model.energy_closure(gt), x_init).detach()

        e_pos = model(gt, x_pos)
        e_neg = model(gt, x_neg)
        loss = (
            e_pos.mean() - e_neg.mean() + energy_reg * (e_pos.pow(2).mean() + e_neg.pow(2).mean())
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        buffers[(n, m)].push(x_neg)

        if step % log_every == 0:
            e_pos_v = float(e_pos.mean().detach())
            gap = e_pos_v - float(e_neg.mean().detach())
            history.append({"step": step, "gap": gap, "e_pos": e_pos_v})
            print(f"[cd {step:>5}] gap={gap:+.4f} e_pos={e_pos_v:.3f}")

    return history


def train_solver(
    model: RecurrentSolver,
    data: Dataset,
    *,
    steps: int = 2000,
    batch: int = 64,
    lr: float = 1e-3,
    clip: float = 1.0,
    seed: int = 0,
    log_every: int = 200,
) -> list[dict]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    buckets = list(data.keys())
    history = []
    for step in range(steps):
        n, m = buckets[step % len(buckets)]
        graphs, x_pos = _sample_batch(data[(n, m)], batch, gen)
        gt = to_tensors(batch_graphs(graphs))
        target = x_pos.argmax(-1)  # (B, I)
        outs = model(gt, return_all=True)

        loss = x_pos.new_zeros(())
        wsum = 0.0
        for t, out in enumerate(outs):
            w = (t + 1) / len(outs)  # weight later refinement steps more
            loss = loss + w * F.cross_entropy(out.reshape(-1, n), target.reshape(-1))
            wsum += w
        loss = loss / wsum

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()

        if step % log_every == 0:
            pred = outs[-1].argmax(-1)
            exact = float((pred == target).all(-1).float().mean())
            history.append({"step": step, "loss": float(loss.detach()), "exact": exact})
            print(f"[solver {step:>5}] loss={float(loss.detach()):.4f} train_exact={exact:.2f}")
    return history


def save_solver(model: RecurrentSolver, path: str, config: dict) -> None:
    torch.save({"state_dict": model.state_dict(), "config": config}, path)


def load_solver(path: str) -> RecurrentSolver:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = blob["config"]
    model = RecurrentSolver(
        hidden=cfg.get("hidden", 96), steps=cfg.get("steps", 16), type_dim=cfg.get("type_dim", 32)
    )
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


def save_checkpoint(model: GNNEnergy, path: str, config: dict) -> None:
    torch.save({"state_dict": model.state_dict(), "config": config}, path)


def load_checkpoint(path: str) -> GNNEnergy:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = blob["config"]
    model = GNNEnergy(
        hidden=cfg.get("hidden", 96),
        rounds=cfg.get("rounds", 3),
        type_dim=cfg.get("type_dim", 32),
    )
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model
