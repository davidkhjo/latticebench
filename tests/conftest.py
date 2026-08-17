"""Shared fixtures. Every test runs with the RNG seeded for reproducibility."""

import random

import numpy as np
import pytest

from latticebench.schema import Attribute, Domain, Solution


@pytest.fixture(autouse=True)
def _seed():
    """Seed numpy and stdlib RNGs before each test."""
    np.random.seed(0)
    random.seed(0)


@pytest.fixture
def tiny_domain() -> Domain:
    """A 3x2 grid: three houses, two attributes (color, pet)."""
    return Domain(
        n=3,
        attributes=(
            Attribute("color", ("red", "green", "blue")),
            Attribute("pet", ("cat", "dog", "bird")),
        ),
    )


@pytest.fixture
def tiny_solution(tiny_domain: Domain) -> Solution:
    """A fixed solution over ``tiny_domain``."""
    return Solution(
        pos={
            "color": {"red": 0, "green": 1, "blue": 2},
            "pet": {"cat": 0, "dog": 1, "bird": 2},
        }
    )
