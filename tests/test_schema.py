"""Tests for the puzzle data model."""

from __future__ import annotations

import pytest

from latticebench.schema import Attribute, Domain, Puzzle, Solution


def test_domain_exposes_shape(tiny_domain: Domain) -> None:
    assert tiny_domain.n == 3
    assert tiny_domain.m == 2
    assert tiny_domain.attribute_names == ("color", "pet")
    assert tiny_domain.attribute("color").values == ("red", "green", "blue")
    assert tiny_domain.values("pet") == ("cat", "dog", "bird")


def test_domain_validates_value_counts() -> None:
    # An attribute with the wrong number of values for n must be rejected.
    with pytest.raises(ValueError):
        Domain(n=3, attributes=(Attribute("color", ("red", "green")),))


def test_domain_rejects_duplicate_attribute_names() -> None:
    with pytest.raises(ValueError):
        Domain(
            n=2,
            attributes=(
                Attribute("color", ("red", "green")),
                Attribute("color", ("blue", "yellow")),
            ),
        )


def test_domain_rejects_nonpositive_n() -> None:
    with pytest.raises(ValueError):
        Domain(n=0, attributes=())


def test_attribute_rejects_duplicate_values() -> None:
    with pytest.raises(ValueError):
        Attribute("color", ("red", "red"))


def test_attribute_unknown_name_raises(tiny_domain: Domain) -> None:
    with pytest.raises(KeyError):
        tiny_domain.attribute("nope")


def test_solution_round_trip(tiny_solution: Solution) -> None:
    assert tiny_solution.house_of("color", "green") == 1
    assert tiny_solution.value_at("color", 1) == "green"
    # value_at and house_of are inverses across every cell.
    for attr in ("color", "pet"):
        for value in tiny_solution.pos[attr]:
            house = tiny_solution.house_of(attr, value)
            assert tiny_solution.value_at(attr, house) == value


def test_solution_value_at_unknown_house(tiny_solution: Solution) -> None:
    with pytest.raises(KeyError):
        tiny_solution.value_at("color", 99)


def test_to_assignment_is_a_plain_copy(tiny_solution: Solution) -> None:
    a = tiny_solution.to_assignment()
    assert a == {
        "color": {"red": 0, "green": 1, "blue": 2},
        "pet": {"cat": 0, "dog": 1, "bird": 2},
    }
    # A fresh dict per attribute, not an alias of the solution's internal mapping.
    a["color"]["red"] = 2
    assert tiny_solution.house_of("color", "red") == 0


def test_puzzle_defaults(tiny_domain: Domain) -> None:
    puzzle = Puzzle(domain=tiny_domain, clues=())
    assert puzzle.solution is None
    assert puzzle.meta == {}
