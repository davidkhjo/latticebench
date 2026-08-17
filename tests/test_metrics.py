"""Tests for scoring metrics and grid-response parsing."""

from __future__ import annotations

import json
import math

from latticebench.energy import EnergyGrid
from latticebench.harness.llm import parse_grid_response
from latticebench.harness.metrics import (
    PuzzleScore,
    cell_accuracy,
    exact_match,
    final_energy,
    summarize,
)
from latticebench.schema import Domain, Solution


def test_exact_match(tiny_solution: Solution) -> None:
    truth = tiny_solution.to_assignment()
    assert exact_match(truth, tiny_solution) is True
    assert exact_match(None, tiny_solution) is False
    # One cell swapped -> not exact.
    perturbed = {k: dict(v) for k, v in truth.items()}
    perturbed["color"]["red"], perturbed["color"]["green"] = 1, 0
    assert exact_match(perturbed, tiny_solution) is False
    # Missing an attribute -> not exact.
    missing = {"color": dict(truth["color"])}
    assert exact_match(missing, tiny_solution) is False


def test_cell_accuracy(tiny_solution: Solution) -> None:
    truth = tiny_solution.to_assignment()
    assert cell_accuracy(truth, tiny_solution) == 1.0
    assert cell_accuracy(None, tiny_solution) == 0.0
    # Swap two of the three color cells -> 4 of 6 correct.
    perturbed = {k: dict(v) for k, v in truth.items()}
    perturbed["color"]["red"], perturbed["color"]["green"] = 1, 0
    assert math.isclose(cell_accuracy(perturbed, tiny_solution), 4 / 6)


def test_final_energy(tiny_domain: Domain, tiny_solution: Solution) -> None:
    grid = EnergyGrid(tiny_domain, [])
    assert final_energy(tiny_solution.to_assignment(), grid) == 0.0
    assert final_energy(None, grid) == math.inf


def test_parse_grid_response_well_formed(tiny_domain: Domain) -> None:
    text = json.dumps({"color": ["red", "green", "blue"], "pet": ["cat", "dog", "bird"]})
    parsed = parse_grid_response(text, tiny_domain)
    assert parsed == {
        "color": {"red": 0, "green": 1, "blue": 2},
        "pet": {"cat": 0, "dog": 1, "bird": 2},
    }


def test_parse_grid_response_embedded_in_prose(tiny_domain: Domain) -> None:
    text = 'Here is my answer: {"color": ["blue", "red", "green"], "pet": ["dog", "cat", "bird"]}'
    parsed = parse_grid_response(text, tiny_domain)
    assert parsed == {
        "color": {"blue": 0, "red": 1, "green": 2},
        "pet": {"dog": 0, "cat": 1, "bird": 2},
    }


def test_parse_grid_response_non_json(tiny_domain: Domain) -> None:
    assert parse_grid_response("I have no idea", tiny_domain) is None


def test_parse_grid_response_wrong_length(tiny_domain: Domain) -> None:
    text = json.dumps({"color": ["red", "green"], "pet": ["cat", "dog", "bird"]})
    parsed = parse_grid_response(text, tiny_domain)
    # color row is malformed and dropped, but pet parses.
    assert parsed == {"pet": {"cat": 0, "dog": 1, "bird": 2}}


def test_summarize_empty() -> None:
    s = summarize([])
    assert s["n"] == 0
    assert s["parse_rate"] == 0.0
    assert s["exact_match"] == 0.0


def test_summarize_mixed() -> None:
    scores = [
        PuzzleScore(id="a", exact=True, cell_acc=1.0, energy=0.0, solve_time=0.1, tier="easy"),
        PuzzleScore(
            id="b", exact=False, cell_acc=0.5, energy=math.inf, solve_time=0.3, tier="easy"
        ),
    ]
    s = summarize(scores)
    assert s["n"] == 2
    assert s["exact_match"] == 0.5
    assert s["cell_accuracy"] == 0.75
    assert s["parse_rate"] == 0.5  # one finite energy of two
    assert s["mean_energy"] == 0.0  # only the finite energy counts
    assert s["median_solve_time"] == 0.2
