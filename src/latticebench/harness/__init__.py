"""Evaluation harness: one protocol and one leaderboard for every model class."""

from __future__ import annotations

from latticebench.harness.base import EvalResult, Model, Prediction, evaluate
from latticebench.harness.ebm import EBMModel, solve_by_annealing
from latticebench.harness.llm import LLMModel, parse_grid_response
from latticebench.harness.metrics import (
    PuzzleScore,
    cell_accuracy,
    exact_match,
    final_energy,
    summarize,
)
from latticebench.harness.solver import SolverModel

__all__ = [
    "EBMModel",
    "EvalResult",
    "LLMModel",
    "Model",
    "Prediction",
    "PuzzleScore",
    "SolverModel",
    "cell_accuracy",
    "evaluate",
    "exact_match",
    "final_energy",
    "parse_grid_response",
    "solve_by_annealing",
    "summarize",
]
