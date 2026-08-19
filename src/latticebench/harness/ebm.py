"""Energy-based-model adapter.

The default baseline minimizes the puzzle's own energy directly by simulated
annealing over permutation assignments — an honest, dependency-light energy
solver that needs no training. Its output is scored like any other model, and its
residual energy is reported, so it lands on the same leaderboard as the LLMs and
solvers. A trained EBM can be plugged in through the ``[ebm]`` extra: build the
torch energy with :meth:`EnergyGrid.as_torch_energy` and run an ebmkit sampler
over it (see :func:`solve_with_sampler`).
"""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any

import numpy as np

from latticebench.energy import EnergyGrid
from latticebench.harness.base import Prediction
from latticebench.schema import Assignment, Domain, Puzzle


def _assignment_from_state(domain: Domain, state: dict[str, list[int]]) -> Assignment:
    return {
        a.name: {v: int(state[a.name][i]) for i, v in enumerate(a.values)}
        for a in domain.attributes
    }


def solve_by_annealing(
    grid: EnergyGrid,
    domain: Domain,
    *,
    steps: int,
    restarts: int,
    rng: np.random.Generator,
    t_start: float = 1.0,
    t_end: float = 0.01,
) -> tuple[Assignment | None, float]:
    """Minimize the puzzle energy by simulated annealing over per-attribute permutations."""
    best_assignment: Assignment | None = None
    best_energy = math.inf

    for _ in range(restarts):
        state = {a.name: list(rng.permutation(domain.n)) for a in domain.attributes}
        cur = _assignment_from_state(domain, state)
        cur_e = float(grid.energy(grid.encode(cur)))

        for step in range(steps):
            frac = step / max(steps - 1, 1)
            temp = t_start * (t_end / t_start) ** frac
            attr = domain.attributes[int(rng.integers(domain.m))]
            i, j = (int(x) for x in rng.integers(domain.n, size=2))
            if i == j:
                continue
            row = state[attr.name]
            row[i], row[j] = row[j], row[i]
            cand = _assignment_from_state(domain, state)
            cand_e = float(grid.energy(grid.encode(cand)))
            delta = cand_e - cur_e
            if delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-9)):
                cur, cur_e = cand, cand_e
            else:
                row[i], row[j] = row[j], row[i]  # revert
            if cur_e < best_energy:
                best_energy, best_assignment = cur_e, cur
            if best_energy == 0.0:
                return best_assignment, best_energy

    return best_assignment, best_energy


def solve_with_sampler(
    grid: EnergyGrid, sampler: Any, *, steps: int = 500, restarts: int = 10
) -> tuple[Assignment, float]:
    """Minimize the torch energy with an ebmkit sampler (requires the ``[ebm]`` extra)."""
    import torch

    energy_fn = grid.as_torch_energy()
    best_x: Any = None
    best_e = math.inf
    for _ in range(restarts):
        x = torch.rand(grid.D, requires_grad=True)
        x = sampler.sample(energy_fn, x, steps=steps)  # ebmkit sampler contract
        e = float(energy_fn(x.detach()[None]).item())
        if e < best_e:
            best_e, best_x = e, x.detach()
    assignment = grid.decode(best_x.numpy())
    return assignment, float(grid.energy(grid.encode(assignment)))


class EBMModel:
    """An energy-based baseline that minimizes each puzzle's energy directly."""

    def __init__(
        self, *, steps: int = 3000, restarts: int = 20, seed: int = 0, name: str = "ebm:annealing"
    ) -> None:
        self.steps = steps
        self.restarts = restarts
        self.seed = seed
        self.name = name

    def predict(self, puzzle: Puzzle) -> Prediction:
        grid = EnergyGrid(puzzle.domain, puzzle.clues)
        rng = np.random.default_rng(self.seed)
        t0 = perf_counter()
        assignment, energy = solve_by_annealing(
            grid, puzzle.domain, steps=self.steps, restarts=self.restarts, rng=rng
        )
        return Prediction(assignment=assignment, energy=energy, solve_time=perf_counter() - t0)
