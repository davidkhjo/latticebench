"""Tests for the model adapters and the evaluation harness."""

from __future__ import annotations

import math

import pytest

from latticebench.contamination import make_split
from latticebench.harness import (
    EBMModel,
    EvalResult,
    LLMModel,
    Prediction,
    SolverModel,
    evaluate,
)
from latticebench.harness.base import Model


@pytest.fixture
def small_split_3x3():
    return list(make_split(101, n=3, m=3, count=3, created="2027-01-01", split="t"))


@pytest.fixture
def small_split_4x4():
    return list(make_split(202, n=4, m=4, count=2, created="2027-01-01", split="t"))


def test_solver_model_clears_every_puzzle(small_split_3x3) -> None:
    model = SolverModel(solver="ortools")
    assert isinstance(model, Model)
    result = evaluate(model, small_split_3x3)
    assert result.summary["exact_match"] == 1.0
    assert result.summary["mean_energy"] == 0.0
    assert result.summary["parse_rate"] == 1.0
    assert result.summary["n"] == 3


def test_ebm_model_solves_4x4(small_split_4x4) -> None:
    model = EBMModel(steps=1500, restarts=8, seed=0)
    result = evaluate(model, small_split_4x4)
    assert result.summary["exact_match"] == 1.0
    assert result.summary["mean_energy"] == 0.0


def test_llm_stub_that_fails_to_parse(small_split_3x3) -> None:
    def dumb_client(prompt: str) -> str:
        return "sorry, I cannot solve this puzzle"

    model = LLMModel(dumb_client, name="llm:stub")
    result = evaluate(model, small_split_3x3)
    assert result.summary["parse_rate"] == 0.0
    assert result.summary["exact_match"] == 0.0
    assert all(math.isinf(s.energy) for s in result.per_puzzle)


def test_llm_stub_that_solves() -> None:
    # A client that echoes the correct grid, read from the puzzle it is shown.
    split = list(make_split(303, n=3, m=3, count=1, created="2027-01-01", split="t"))

    def oracle_client_factory():
        from latticebench.dataset import from_record

        solved = from_record(split[0])

        def client(prompt: str) -> str:
            import json

            dom = solved.domain
            sol = solved.solution
            grid = {a.name: [sol.value_at(a.name, h) for h in range(dom.n)] for a in dom.attributes}
            return json.dumps(grid)

        return client

    model = LLMModel(oracle_client_factory(), name="llm:oracle")
    result = evaluate(model, split)
    assert result.summary["exact_match"] == 1.0


def test_prediction_defaults() -> None:
    p = Prediction(assignment=None)
    assert p.energy is None
    assert p.solve_time is None
    assert p.raw is None


def test_eval_result_round_trip(small_split_3x3) -> None:
    result = evaluate(SolverModel(), small_split_3x3)
    d = result.to_dict()
    restored = EvalResult.from_dict(d)
    assert restored.model == result.model
    assert restored.summary == result.summary
    assert len(restored.per_puzzle) == len(result.per_puzzle)


def test_by_difficulty_keyed_by_tier(small_split_3x3) -> None:
    result = evaluate(SolverModel(), small_split_3x3)
    assert result.by_difficulty
    for tier, summary in result.by_difficulty.items():
        assert isinstance(tier, str)
        assert "exact_match" in summary
