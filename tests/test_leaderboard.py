"""Tests for leaderboard aggregation."""

from __future__ import annotations

import json

from latticebench.harness.base import EvalResult
from latticebench.leaderboard import build_leaderboard


def _result(name: str, exact: float) -> EvalResult:
    return EvalResult(
        model=name,
        per_puzzle=[],
        summary={
            "n": 2,
            "exact_match": exact,
            "cell_accuracy": exact,
            "mean_energy": 0.0,
            "median_solve_time": 0.01,
            "parse_rate": 1.0,
        },
        by_difficulty={"easy": {"exact_match": exact}},
    )


def test_sorted_by_exact_match_desc() -> None:
    weak = _result("llm:weak", 0.25)
    strong = _result("solver:ortools", 1.0)
    board = build_leaderboard([weak, strong])
    assert [r.model for r in board.rows] == ["solver:ortools", "llm:weak"]
    assert board.rows[0].model_class == "solver"


def test_to_markdown_contains_model_names() -> None:
    board = build_leaderboard([_result("solver:ortools", 1.0), _result("ebm:annealing", 0.5)])
    md = board.to_markdown()
    assert "solver:ortools" in md
    assert "ebm:annealing" in md
    assert md.startswith("|")


def test_to_json_round_trips() -> None:
    board = build_leaderboard([_result("solver:ortools", 1.0), _result("ebm:annealing", 0.5)])
    data = json.loads(board.to_json())
    assert isinstance(data, list)
    assert data[0]["model"] == "solver:ortools"
    assert data[0]["exact_match"] == 1.0
    assert data[0]["by_tier"] == {"easy": 1.0}
