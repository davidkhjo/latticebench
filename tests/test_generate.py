"""Tests for procedural, deterministic puzzle generation."""

from __future__ import annotations

import numpy as np

from latticebench.clues import DEFAULT_INVENTORY
from latticebench.energy import EnergyGrid
from latticebench.generate import (
    all_true_clues,
    build_domain,
    generate_puzzle,
    sample_solution,
)
from latticebench.oracle import is_unique


def test_build_domain_shape() -> None:
    domain = build_domain(4, 3)
    assert domain.n == 4
    assert domain.m == 3
    for a in domain.attributes:
        assert len(a.values) == 4


def test_generated_puzzle_is_unique_minimal_and_zero_energy() -> None:
    puzzle = generate_puzzle(4, 4, seed=11)
    assert is_unique(puzzle.domain, puzzle.clues) is True

    grid = EnergyGrid(puzzle.domain, puzzle.clues)
    assert abs(float(grid.energy(grid.encode(puzzle.solution)))) < 1e-9

    # Minimal: removing any single clue destroys uniqueness.
    clues = list(puzzle.clues)
    assert len(clues) >= 1
    for i in range(len(clues)):
        trimmed = clues[:i] + clues[i + 1 :]
        assert is_unique(puzzle.domain, trimmed) is False


def test_generation_is_deterministic() -> None:
    a = generate_puzzle(4, 4, seed=21)
    b = generate_puzzle(4, 4, seed=21)
    assert [c.to_dict() for c in a.clues] == [c.to_dict() for c in b.clues]
    assert a.solution.pos == b.solution.pos


def test_different_seeds_differ() -> None:
    a = generate_puzzle(4, 4, seed=21)
    b = generate_puzzle(4, 4, seed=22)
    same_clues = [c.to_dict() for c in a.clues] == [c.to_dict() for c in b.clues]
    same_solution = a.solution.pos == b.solution.pos
    assert not (same_clues and same_solution)


def test_meta_records_generation_parameters() -> None:
    puzzle = generate_puzzle(4, 3, seed=1)
    assert puzzle.meta["seed"] == 1
    assert puzzle.meta["n"] == 4
    assert puzzle.meta["m"] == 3
    assert "generator_version" in puzzle.meta


def test_all_true_clues_all_hold() -> None:
    domain = build_domain(3, 2)
    rng = np.random.default_rng(0)
    sol = sample_solution(domain, rng)
    clues = all_true_clues(domain, sol, DEFAULT_INVENTORY)
    assert clues
    assert all(c.holds(sol) for c in clues)
