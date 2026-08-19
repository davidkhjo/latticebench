"""A graph energy network for logic-grid puzzles.

The energy of a candidate assignment is a sum of edge and node terms produced by
a few rounds of message passing over the constraint graph. The load-bearing
feature is the displacement correlation on each edge: for endpoints with house
distributions ``p_i`` and ``p_j`` it measures how much probability sits at each
relative offset, which is exactly what the hand-written clue penalties look at --
but here the network has to learn, per clue type, which offset should be cheap.
That is what lets one set of weights run at any grid size.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from latticebench.models.graph import N_EDGE_TYPES, BatchedGraph

POS_DIM = 8
CORR_DIM = 7


@dataclass
class GraphTensors:
    edge_src: Tensor
    edge_dst: Tensor
    edge_type: Tensor
    edge_dir: Tensor
    edge_house: Tensor
    node_batch: Tensor
    edge_batch: Tensor
    batch_size: int
    num_items: int
    n: int


def to_tensors(bg: BatchedGraph, device: torch.device | str = "cpu") -> GraphTensors:
    t = lambda a: torch.as_tensor(a, device=device)  # noqa: E731
    return GraphTensors(
        edge_src=t(bg.edge_index[0]),
        edge_dst=t(bg.edge_index[1]),
        edge_type=t(bg.edge_type),
        edge_dir=t(bg.edge_dir).float(),
        edge_house=t(bg.edge_house),
        node_batch=t(bg.node_batch),
        edge_batch=t(bg.edge_batch),
        batch_size=bg.batch_size,
        num_items=bg.num_items,
        n=bg.n,
    )


def _house_pos(n: int, device: torch.device | str) -> Tensor:
    """Fixed size-agnostic features for each of the n house positions."""
    idx = torch.arange(n, device=device, dtype=torch.float32)
    u = idx / max(n - 1, 1)
    pi = torch.pi
    return torch.stack(
        [
            u,
            1 - u,
            (idx == 0).float(),
            (idx == n - 1).float(),
            torch.sin(pi * u),
            torch.cos(pi * u),
            torch.sin(2 * pi * u),
            torch.cos(2 * pi * u),
        ],
        dim=1,
    )  # (n, POS_DIM)


def _shift(vec: Tensor, k: int, n: int) -> Tensor:
    """w[h] = vec[h-k], zero-padded (no wraparound)."""
    if k == 0:
        return vec
    if abs(k) >= n:
        return torch.zeros_like(vec)
    if k > 0:
        return torch.cat([vec.new_zeros(vec.shape[0], k), vec[:, : n - k]], dim=1)
    kk = -k
    return torch.cat([vec[:, kk:], vec.new_zeros(vec.shape[0], kk)], dim=1)


def _corr(p_src: Tensor, p_dst: Tensor, n: int) -> Tensor:
    """Displacement-correlation buckets between two house distributions."""
    r0 = (p_src * p_dst).sum(1)
    rp1 = (p_src * _shift(p_dst, -1, n)).sum(1)
    rm1 = (p_src * _shift(p_dst, 1, n)).sum(1)
    r2 = (p_src * (_shift(p_dst, -2, n) + _shift(p_dst, 2, n))).sum(1)
    r3 = (p_src * (_shift(p_dst, -3, n) + _shift(p_dst, 3, n))).sum(1)
    cj = torch.cumsum(p_dst, dim=1)
    jleft = (p_src * (cj - p_dst)).sum(1)
    rev = torch.flip(torch.cumsum(torch.flip(p_dst, [1]), 1), [1])
    jright = (p_src * (rev - p_dst)).sum(1)
    return torch.stack([r0, rp1, rm1, r2, r3, jleft, jright], dim=1)  # (E, CORR_DIM)


def _mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for a, b in zip(sizes[:-1], sizes[1:], strict=False):
        layers += [nn.Linear(a, b), nn.SiLU()]
    return nn.Sequential(*layers[:-1])  # drop the trailing activation


class GNNEnergy(nn.Module):
    def __init__(
        self, hidden: int = 96, rounds: int = 3, type_dim: int = 32, n_types: int = N_EDGE_TYPES
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.rounds = rounds
        self.type_emb = nn.Embedding(n_types, type_dim)
        self.node_in = _mlp([POS_DIM + 1, hidden, hidden])
        edge_in = 2 * hidden + CORR_DIM + type_dim + 1
        self.msg = _mlp([edge_in, hidden, hidden])
        self.upd = _mlp([2 * hidden, hidden, hidden])
        self.edge_energy = _mlp([edge_in, hidden, 1])
        self.node_energy = _mlp([hidden, hidden, 1])
        self.query = nn.Linear(hidden, POS_DIM)

    def _states(self, gt: GraphTensors, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        n = gt.n
        p = x.reshape(-1, n)  # (B*I, n), node order matches node_batch
        p_src = p[gt.edge_src]
        p_dst = p[gt.edge_dst]
        unary = gt.edge_house >= 0
        if unary.any():
            oh = F.one_hot(gt.edge_house.clamp(min=0), n).float()
            p_dst = torch.where(unary.unsqueeze(1), oh, p_dst)
        return p, p_src, p_dst

    def forward(self, gt: GraphTensors, x: Tensor) -> Tensor:
        n = gt.n
        p, p_src, p_dst = self._states(gt, x)
        corr = _corr(p_src, p_dst, n)
        te = self.type_emb(gt.edge_type)
        edir = gt.edge_dir.unsqueeze(1)
        pos = _house_pos(n, x.device)

        h = self.node_in(torch.cat([p @ pos, p.max(1, keepdim=True).values], dim=1))
        n_nodes = h.shape[0]
        for _ in range(self.rounds):
            edge_in = torch.cat([h[gt.edge_src], h[gt.edge_dst], corr, te, edir], dim=1)
            msg = self.msg(edge_in)
            agg = h.new_zeros(n_nodes, self.hidden).index_add_(0, gt.edge_dst, msg)
            h = h + self.upd(torch.cat([h, agg], dim=1))

        edge_in = torch.cat([h[gt.edge_src], h[gt.edge_dst], corr, te, edir], dim=1)
        e_edge = self.edge_energy(edge_in).squeeze(1)
        e_node = self.node_energy(h).squeeze(1)

        energy = x.new_zeros(gt.batch_size)
        energy = energy.index_add(0, gt.edge_batch, e_edge)
        energy = energy.index_add(0, gt.node_batch, e_node)
        return energy  # (B,)

    def item_logits(self, gt: GraphTensors, x: Tensor) -> Tensor:
        """Per-item house logits, size-agnostic, used for greedy warm starts."""
        n = gt.n
        p, p_src, p_dst = self._states(gt, x)
        corr = _corr(p_src, p_dst, n)
        te = self.type_emb(gt.edge_type)
        edir = gt.edge_dir.unsqueeze(1)
        pos = _house_pos(n, x.device)
        h = self.node_in(torch.cat([p @ pos, p.max(1, keepdim=True).values], dim=1))
        n_nodes = h.shape[0]
        for _ in range(self.rounds):
            edge_in = torch.cat([h[gt.edge_src], h[gt.edge_dst], corr, te, edir], dim=1)
            agg = h.new_zeros(n_nodes, self.hidden).index_add_(0, gt.edge_dst, self.msg(edge_in))
            h = h + self.upd(torch.cat([h, agg], dim=1))
        logits = self.query(h) @ pos.t()  # (B*I, n)
        return logits.reshape(gt.batch_size, gt.num_items, n)

    def energy_closure(self, gt: GraphTensors):
        return lambda x: self.forward(gt, x)


def one_hot_init(shape: tuple[int, ...], n: int) -> Tensor:
    return F.one_hot(torch.randint(n, shape[:-1]), n).float()
