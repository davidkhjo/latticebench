"""The constraint-solver baseline. Clears every well-formed puzzle."""

from __future__ import annotations

from time import perf_counter

from latticebench import oracle
from latticebench.harness.base import Prediction
from latticebench.schema import Puzzle


class SolverModel:
    """Solve a puzzle with CPMpy's backend (OR-Tools CP-SAT by default, or Z3)."""

    def __init__(self, solver: str = "ortools", time_limit: float | None = 10.0) -> None:
        self.solver = solver
        self.time_limit = time_limit
        self.name = f"solver:{solver}"

    def predict(self, puzzle: Puzzle) -> Prediction:
        t0 = perf_counter()
        assignment = oracle.solve(
            puzzle.domain, puzzle.clues, solver=self.solver, time_limit=self.time_limit
        )
        elapsed = perf_counter() - t0
        energy = 0.0 if assignment is not None else None
        return Prediction(assignment=assignment, energy=energy, solve_time=elapsed)
