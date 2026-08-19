"""The evaluation harness.

Every model class — solver, language model, energy-based model — implements the
same :class:`Model` protocol, so one :func:`evaluate` call and one leaderboard
cover all of them on identical instances. The model is shown a puzzle with the
solution stripped; grading uses a solved copy re-derived from the record.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol, runtime_checkable

from latticebench.dataset import PuzzleRecord, from_record
from latticebench.energy import EnergyGrid
from latticebench.generate import generate_puzzle
from latticebench.harness.metrics import PuzzleScore, by_tier, score_prediction, summarize
from latticebench.schema import Assignment, Puzzle


@dataclass
class Prediction:
    """A model's answer: a decoded grid (``None`` if it refused or failed to parse)."""

    assignment: Assignment | None
    energy: float | None = None
    solve_time: float | None = None
    raw: Any = None


@runtime_checkable
class Model(Protocol):
    name: str

    def predict(self, puzzle: Puzzle) -> Prediction: ...


@dataclass
class EvalResult:
    model: str
    per_puzzle: list[PuzzleScore]
    summary: dict[str, float]
    by_difficulty: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "summary": self.summary,
            "by_difficulty": self.by_difficulty,
            "per_puzzle": [s.to_dict() for s in self.per_puzzle],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalResult:
        return cls(
            model=d["model"],
            per_puzzle=[PuzzleScore(**s) for s in d.get("per_puzzle", [])],
            summary=d["summary"],
            by_difficulty=d.get("by_difficulty", {}),
        )


def _solved_puzzle(rec: PuzzleRecord) -> Puzzle:
    if rec.solution is not None:
        return from_record(rec)
    return generate_puzzle(rec.n, rec.m, seed=rec.seed, generator_version=rec.generator_version)


def evaluate(model: Model, records: Iterable[PuzzleRecord]) -> EvalResult:
    """Run ``model`` over ``records`` and score every prediction."""
    scores: list[PuzzleScore] = []
    for rec in records:
        solved = _solved_puzzle(rec)
        assert solved.solution is not None
        challenge = Puzzle(
            domain=solved.domain, clues=solved.clues, solution=None, meta=solved.meta
        )
        grid = EnergyGrid(solved.domain, solved.clues)

        t0 = perf_counter()
        pred = model.predict(challenge)
        elapsed = perf_counter() - t0
        solve_time = pred.solve_time if pred.solve_time is not None else elapsed

        tier = str(rec.difficulty.get("tier", "unknown"))
        scores.append(score_prediction(rec.id, pred, solved.solution, grid, solve_time, tier))

    return EvalResult(
        model=model.name,
        per_puzzle=scores,
        summary=summarize(scores),
        by_difficulty=by_tier(scores),
    )
