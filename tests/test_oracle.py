"""Tests for the CPMpy constraint model and uniqueness oracle."""

from __future__ import annotations

import pytest

from latticebench.clues import FoundAt, NotAt, SameHouse
from latticebench.generate import generate_puzzle
from latticebench.oracle import (
    conflict_count,
    count_solutions,
    is_unique,
    solve,
)
from latticebench.schema import Domain

# A clue set that pins tiny_domain to exactly the tiny_solution:
#   color: red=0, green=1, (blue=2 forced); pet: cat=0, dog=1, (bird=2 forced).
UNIQUE_CLUES = (
    FoundAt("color", "red", 0),
    FoundAt("color", "green", 1),
    SameHouse("color", "red", "pet", "cat"),
    SameHouse("color", "green", "pet", "dog"),
)


def test_unique_clue_set_has_one_solution(tiny_domain: Domain) -> None:
    assert count_solutions(tiny_domain, UNIQUE_CLUES) == 1
    assert is_unique(tiny_domain, UNIQUE_CLUES) is True
    sol = solve(tiny_domain, UNIQUE_CLUES)
    assert sol == {
        "color": {"red": 0, "green": 1, "blue": 2},
        "pet": {"cat": 0, "dog": 1, "bird": 2},
    }


def test_contradiction_is_unsatisfiable(tiny_domain: Domain) -> None:
    clues = (FoundAt("color", "red", 0), NotAt("color", "red", 0))
    assert count_solutions(tiny_domain, clues) == 0
    assert is_unique(tiny_domain, clues) is False
    assert solve(tiny_domain, clues) is None


def test_loose_clue_has_many_solutions(tiny_domain: Domain) -> None:
    clues = (FoundAt("color", "red", 0),)
    assert count_solutions(tiny_domain, clues, limit=5) > 1
    assert is_unique(tiny_domain, clues) is False


def test_ortools_and_z3_agree_on_uniqueness() -> None:
    puzzle = generate_puzzle(4, 3, seed=7)
    assert is_unique(puzzle.domain, puzzle.clues, solver="ortools") is True
    assert is_unique(puzzle.domain, puzzle.clues, solver="z3") is True


def test_conflict_count_is_nonnegative_int(tiny_domain: Domain) -> None:
    c = conflict_count(tiny_domain, UNIQUE_CLUES, solver="ortools")
    assert isinstance(c, int)
    assert c >= 0


@pytest.mark.parametrize("solver", ["ortools", "z3"])
def test_solve_returns_a_valid_assignment(tiny_domain: Domain, solver: str) -> None:
    sol = solve(tiny_domain, UNIQUE_CLUES, solver=solver)
    assert sol is not None
    # Each attribute row is a bijection value -> house.
    for row in sol.values():
        assert sorted(row.values()) == [0, 1, 2]
