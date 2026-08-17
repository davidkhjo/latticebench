"""Tests for difficulty grading."""

from __future__ import annotations

import math

from latticebench.difficulty import (
    Difficulty,
    grade,
    log_search_space,
    search_space_size,
)
from latticebench.generate import build_domain, generate_puzzle

_TIER_ORDER = {"easy": 0, "medium": 1, "hard": 2, "expert": 3}


def test_search_space_size_formula() -> None:
    for n, m in [(3, 3), (4, 4), (4, 2), (5, 3)]:
        domain = build_domain(n, m)
        assert search_space_size(domain) == math.factorial(n) ** (m - 1)


def test_search_space_size_single_attribute() -> None:
    domain = build_domain(3, 1)
    assert search_space_size(domain) == 1.0


def test_log_search_space_matches_size() -> None:
    domain = build_domain(4, 3)
    assert math.isclose(log_search_space(domain), math.log(search_space_size(domain)), rel_tol=1e-9)


def test_grade_returns_valid_tier() -> None:
    puzzle = generate_puzzle(3, 3, seed=1)
    diff = grade(puzzle)
    assert isinstance(diff, Difficulty)
    assert diff.tier in _TIER_ORDER
    assert diff.n == 3
    assert diff.m == 3
    assert diff.n_clues == len(puzzle.clues)
    assert diff.z3_conflicts >= 0


def test_larger_grid_is_harder() -> None:
    small = generate_puzzle(3, 3, seed=1)
    large = generate_puzzle(5, 5, seed=1)  # one slow puzzle, kept minimal
    d_small = grade(small)
    d_large = grade(large)
    assert d_small.log_search_space < d_large.log_search_space
    assert _TIER_ORDER[d_small.tier] <= _TIER_ORDER[d_large.tier]
    assert d_small.tier != d_large.tier
