"""A recurrent relational solver over the constraint graph.

Each item keeps a soft distribution over houses. A few rounds of message passing
refine those distributions, and the network is supervised at every round to match
the true assignment (deep supervision, as in recurrent relational networks). At
inference we read off the argmax. This shares the graph and the displacement
correlation feature with the energy model in ``gnn.py``; the difference is that
it predicts the assignment directly instead of scoring one.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from latticebench.models.gnn import CORR_DIM, POS_DIM, GraphTensors, _corr, _house_pos, _mlp
from latticebench.models.graph import N_EDGE_TYPES


class RecurrentSolver(nn.Module):
    def __init__(
        self, hidden: int = 96, steps: int = 16, type_dim: int = 32, n_types: int = N_EDGE_TYPES
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.steps = steps
        self.type_emb = nn.Embedding(n_types, type_dim)
        self.msg = _mlp([2 * hidden + CORR_DIM + type_dim + 1, hidden, hidden])
        self.gru = nn.GRUCell(hidden, hidden)
        self.query = nn.Linear(hidden, POS_DIM)
        self.h0 = nn.Parameter(torch.zeros(hidden))

    def _edges(self, gt: GraphTensors, logits: Tensor) -> Tensor:
        n = gt.n
        p = F.softmax(logits, dim=-1)
        p_src, p_dst = p[gt.edge_src], p[gt.edge_dst]
        unary = gt.edge_house >= 0
        if unary.any():
            oh = F.one_hot(gt.edge_house.clamp(min=0), n).float()
            p_dst = torch.where(unary.unsqueeze(1), oh, p_dst)
        return _corr(p_src, p_dst, n)

    def forward(self, gt: GraphTensors, *, return_all: bool = False):
        n = gt.n
        n_nodes = gt.batch_size * gt.num_items
        pos = _house_pos(n, self.h0.device)
        te = self.type_emb(gt.edge_type)
        edir = gt.edge_dir.unsqueeze(1)

        h = self.h0.expand(n_nodes, self.hidden).contiguous()
        logits = self.query(h) @ pos.t()
        outs = []
        for _ in range(self.steps):
            corr = self._edges(gt, logits)
            edge_in = torch.cat([h[gt.edge_src], h[gt.edge_dst], corr, te, edir], dim=1)
            agg = h.new_zeros(n_nodes, self.hidden).index_add_(0, gt.edge_dst, self.msg(edge_in))
            h = self.gru(agg, h)
            logits = self.query(h) @ pos.t()
            outs.append(logits.view(gt.batch_size, gt.num_items, n))
        return outs if return_all else outs[-1]
