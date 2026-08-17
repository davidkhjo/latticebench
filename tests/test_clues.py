"""Tests for the ten clue types and their self-describing contract."""

from __future__ import annotations

import pytest

from latticebench.clues import (
    DEFAULT_INVENTORY,
    DirectLeft,
    DirectRight,
    FoundAt,
    LeftOf,
    NotAt,
    OneBetween,
    RightOf,
    SameHouse,
    SideBySide,
    TwoBetween,
)
from latticebench.clues.base import CLUE_REGISTRY, Clue
from latticebench.schema import Attribute, Domain, Solution


@pytest.fixture
def domain4() -> Domain:
    """A 4-house, 2-attribute grid (needed so TwoBetween can hold)."""
    return Domain(
        n=4,
        attributes=(
            Attribute("color", ("red", "green", "blue", "yellow")),
            Attribute("pet", ("cat", "dog", "bird", "fish")),
        ),
    )


@pytest.fixture
def sol4() -> Solution:
    """color and pet both in ascending house order 0,1,2,3."""
    return Solution(
        pos={
            "color": {"red": 0, "green": 1, "blue": 2, "yellow": 3},
            "pet": {"cat": 0, "dog": 1, "bird": 2, "fish": 3},
        }
    )


# (clue that holds under sol4, clue of the same type that does not)
HOLDS_AND_FAILS = [
    (FoundAt("color", "red", 0), FoundAt("color", "red", 1)),
    (NotAt("color", "red", 1), NotAt("color", "red", 0)),
    (
        SameHouse("color", "red", "pet", "cat"),
        SameHouse("color", "red", "pet", "dog"),
    ),
    (
        DirectLeft("color", "red", "pet", "dog"),
        DirectLeft("color", "red", "pet", "cat"),
    ),
    (
        DirectRight("color", "green", "pet", "cat"),
        DirectRight("color", "red", "pet", "cat"),
    ),
    (
        SideBySide("color", "red", "pet", "dog"),
        SideBySide("color", "red", "pet", "bird"),
    ),
    (
        LeftOf("color", "red", "pet", "fish"),
        LeftOf("color", "yellow", "pet", "cat"),
    ),
    (
        RightOf("color", "yellow", "pet", "cat"),
        RightOf("color", "red", "pet", "fish"),
    ),
    (
        OneBetween("color", "red", "pet", "bird"),
        OneBetween("color", "red", "pet", "dog"),
    ),
    (
        TwoBetween("color", "red", "pet", "fish"),
        TwoBetween("color", "red", "pet", "bird"),
    ),
]


def test_inventory_has_ten_distinct_types() -> None:
    assert len(DEFAULT_INVENTORY) == 10
    assert len(set(DEFAULT_INVENTORY)) == 10


def test_every_inventory_type_registered() -> None:
    for cls in DEFAULT_INVENTORY:
        assert CLUE_REGISTRY[cls.kind] is cls


@pytest.mark.parametrize("good,bad", HOLDS_AND_FAILS, ids=lambda c: type(c).__name__)
def test_holds_true_and_false(good: Clue, bad: Clue, sol4: Solution) -> None:
    assert good.holds(sol4) is True
    assert bad.holds(sol4) is False


@pytest.mark.parametrize("good,_bad", HOLDS_AND_FAILS, ids=lambda c: type(c).__name__)
def test_to_dict_from_dict_round_trip(good: Clue, _bad: Clue) -> None:
    d = good.to_dict()
    assert d["kind"] == type(good).kind
    restored = Clue.from_dict(d)
    assert restored == good  # frozen dataclasses compare by value
    assert type(restored) is type(good)


@pytest.mark.parametrize("good,_bad", HOLDS_AND_FAILS, ids=lambda c: type(c).__name__)
def test_to_text_mentions_values(good: Clue, _bad: Clue, domain4: Domain) -> None:
    text = good.to_text(domain4)
    assert isinstance(text, str)
    assert text
    for field in ("value", "value1", "value2"):
        val = getattr(good, field, None)
        if val is not None:
            assert val in text


def test_all_ten_types_covered() -> None:
    covered = {type(good) for good, _ in HOLDS_AND_FAILS}
    assert covered == set(DEFAULT_INVENTORY)
