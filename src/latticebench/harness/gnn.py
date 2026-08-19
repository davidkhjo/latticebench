"""Harness adapter for the trained graph energy model.

Inference minimises the *learned* energy by iterated conditional modes from a
handful of random restarts, then reports the *hand* energy of the decoded
assignment, so ``ebm:gnn`` sits on the same scale as the solver, the annealing
baseline, and the LLMs.
"""

from __future__ import annotations

from time import perf_counter

from latticebench.energy import EnergyGrid
from latticebench.harness.base import Prediction
from latticebench.schema import Puzzle


class GNNSolverModel:
    """The recurrent relational solver: predicts the assignment directly."""

    def __init__(self, checkpoint: str = "checkpoints/solver.pt", *, name: str = "gnn") -> None:
        self.checkpoint = checkpoint
        self.name = name
        self._model = None

    def _load(self):
        if self._model is None:
            from latticebench.models.train import load_solver

            self._model = load_solver(self.checkpoint)
        return self._model

    def predict(self, puzzle: Puzzle) -> Prediction:
        import torch

        from latticebench.models.gnn import to_tensors
        from latticebench.models.graph import batch_graphs, build_graph

        model = self._load()
        grid = EnergyGrid(puzzle.domain, puzzle.clues)
        gt = to_tensors(batch_graphs([build_graph(puzzle)]))
        t0 = perf_counter()
        with torch.no_grad():
            logits = model(gt)  # (1, items, n)
            pred = logits.argmax(-1)[0]  # (items,)
        x = torch.nn.functional.one_hot(pred, puzzle.domain.n).float()
        elapsed = perf_counter() - t0
        assignment = grid.decode(x.reshape(-1).numpy())
        return Prediction(
            assignment=assignment,
            energy=float(grid.energy(grid.encode(assignment))),
            solve_time=elapsed,
        )


class GNNEnergyModel:
    def __init__(
        self,
        checkpoint: str = "checkpoints/gnn.pt",
        *,
        restarts: int = 24,
        steps: int = 200,
        seed: int = 0,
        name: str = "ebm:gnn",
    ) -> None:
        self.checkpoint = checkpoint
        self.restarts = restarts
        self.steps = steps
        self.seed = seed
        self.name = name
        self._model = None

    def _load(self):
        if self._model is None:
            from latticebench.models.train import load_checkpoint

            self._model = load_checkpoint(self.checkpoint)
        return self._model

    def predict(self, puzzle: Puzzle) -> Prediction:
        import torch

        from latticebench.models.gnn import one_hot_init, to_tensors
        from latticebench.models.graph import batch_graphs, build_graph

        model = self._load()
        grid = EnergyGrid(puzzle.domain, puzzle.clues)
        n = puzzle.domain.n
        items = puzzle.domain.m * n
        graph = build_graph(puzzle)

        R = self.restarts
        gt_r = to_tensors(batch_graphs([graph] * R))
        gt_rn = to_tensors(batch_graphs([graph] * (R * n)))
        eye = torch.eye(n)

        t0 = perf_counter()
        torch.manual_seed(self.seed)
        with torch.no_grad():
            x = one_hot_init((R, items, n), n)
            # iterated conditional modes: repeatedly set each item to the house that
            # minimises the learned energy given the others. the pseudo-likelihood
            # objective trains exactly these conditionals, so this is the natural
            # decoder; random restarts break out of local minima.
            for _ in range(self.steps):
                moved = False
                for j in range(items):
                    cand = x.unsqueeze(1).repeat(1, n, 1, 1)  # (R, n, items, n)
                    cand[:, :, j, :] = eye
                    energy = model(gt_rn, cand.reshape(R * n, items, n)).view(R, n)
                    pick = energy.argmin(1)
                    new_j = eye[pick]
                    moved = moved or not torch.equal(new_j, x[:, j, :])
                    x[:, j, :] = new_j
                if not moved:
                    break
            scores = model(gt_r, x)
        best = x[int(scores.argmin())]
        elapsed = perf_counter() - t0

        assignment = grid.decode(best.reshape(-1).numpy())
        return Prediction(
            assignment=assignment,
            energy=float(grid.energy(grid.encode(assignment))),
            solve_time=elapsed,
        )
