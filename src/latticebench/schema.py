"""Puzzle data model.

A puzzle places ``n`` houses in a row (positions ``0 .. n-1``) and assigns each
house one value from every attribute. Each attribute has exactly ``n`` values,
one per house, so a solution is a bijection ``value -> house`` per attribute.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from latticebench.clues.base import Clue

# value -> house index, per attribute name. A partial or invalid guess.
Assignment = dict[str, dict[str, int]]


@dataclass(frozen=True)
class Attribute:
    """A named attribute and its ``n`` distinct values."""

    name: str
    values: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(set(self.values)) != len(self.values):
            raise ValueError(f"attribute {self.name!r} has duplicate values")


@dataclass(frozen=True)
class Domain:
    """The shape of a puzzle: ``n`` houses and ``m`` attributes of ``n`` values."""

    n: int
    attributes: tuple[Attribute, ...]

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("n must be positive")
        names = [a.name for a in self.attributes]
        if len(set(names)) != len(names):
            raise ValueError("duplicate attribute names")
        for a in self.attributes:
            if len(a.values) != self.n:
                raise ValueError(
                    f"attribute {a.name!r} has {len(a.values)} values, expected n={self.n}"
                )

    @property
    def m(self) -> int:
        return len(self.attributes)

    @property
    def attribute_names(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.attributes)

    def attribute(self, name: str) -> Attribute:
        for a in self.attributes:
            if a.name == name:
                return a
        raise KeyError(name)

    def values(self, name: str) -> tuple[str, ...]:
        return self.attribute(name).values


@dataclass(frozen=True)
class Solution:
    """A complete assignment: ``pos[attr][value]`` is the house of that value."""

    pos: Mapping[str, Mapping[str, int]]

    def house_of(self, attr: str, value: str) -> int:
        return self.pos[attr][value]

    def value_at(self, attr: str, house: int) -> str:
        for value, h in self.pos[attr].items():
            if h == house:
                return value
        raise KeyError((attr, house))

    def to_assignment(self) -> Assignment:
        return {attr: dict(vals) for attr, vals in self.pos.items()}


@dataclass(frozen=True)
class Puzzle:
    """A domain, its clues, and (optionally) the solution and generation metadata."""

    domain: Domain
    clues: tuple[Clue, ...]
    solution: Solution | None = None
    meta: Mapping[str, object] = field(default_factory=dict)
