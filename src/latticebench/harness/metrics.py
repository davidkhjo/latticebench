"""Scoring metrics.

Exact match is the headline. Cell accuracy gives partial credit. Final energy is
the model-class-neutral graded metric the energy formulation makes possible: any
model's output is encoded and scored on the same energy scale, and a correct
answer is exactly the zero-energy point.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import median
from typing import TYPE_CHECKING, Any

from latticebench.schema import Assignment, Solution

if TYPE_CHECKING:
    from latticebench.energy import EnergyGrid
    from latticebench.harness.base import Prediction


@dataclass
class PuzzleScore:
    id: str
    exact: bool
    cell_acc: float
    energy: float
    solve_time: float
    tier: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def exact_match(pred: Assignment | None, sol: Solution) -> bool:
    if pred is None:
        return False
    truth = sol.to_assignment()
    for attr, row in truth.items():
        got = pred.get(attr)
        if got is None:
            return False
        for value, house in row.items():
            if got.get(value) != house:
                return False
    return True


def cell_accuracy(pred: Assignment | None, sol: Solution) -> float:
    truth = sol.to_assignment()
    total = sum(len(row) for row in truth.values())
    if total == 0:
        return 1.0
    if pred is None:
        return 0.0
    correct = 0
    for attr, row in truth.items():
        got = pred.get(attr, {})
        for value, house in row.items():
            if got.get(value) == house:
                correct += 1
    return correct / total


def final_energy(pred: Assignment | None, grid: EnergyGrid) -> float:
    if pred is None:
        return math.inf
    return float(grid.energy(grid.encode(pred)))


def score_prediction(
    puzzle_id: str,
    pred: Prediction,
    sol: Solution,
    grid: EnergyGrid,
    solve_time: float,
    tier: str,
) -> PuzzleScore:
    return PuzzleScore(
        id=puzzle_id,
        exact=exact_match(pred.assignment, sol),
        cell_acc=cell_accuracy(pred.assignment, sol),
        energy=final_energy(pred.assignment, grid),
        solve_time=solve_time,
        tier=tier,
    )


def summarize(scores: list[PuzzleScore]) -> dict[str, float]:
    if not scores:
        return {
            "n": 0,
            "exact_match": 0.0,
            "cell_accuracy": 0.0,
            "mean_energy": 0.0,
            "median_solve_time": 0.0,
            "parse_rate": 0.0,
        }
    finite = [s.energy for s in scores if math.isfinite(s.energy)]
    return {
        "n": len(scores),
        "exact_match": sum(s.exact for s in scores) / len(scores),
        "cell_accuracy": sum(s.cell_acc for s in scores) / len(scores),
        "mean_energy": (sum(finite) / len(finite)) if finite else math.inf,
        "median_solve_time": median(s.solve_time for s in scores),
        "parse_rate": len(finite) / len(scores),
    }


def by_tier(scores: list[PuzzleScore]) -> dict[str, dict[str, float]]:
    tiers: dict[str, list[PuzzleScore]] = {}
    for s in scores:
        tiers.setdefault(s.tier, []).append(s)
    return {tier: summarize(group) for tier, group in tiers.items()}
