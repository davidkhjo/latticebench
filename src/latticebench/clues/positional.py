"""Positional clues: constraints on the distance or order of two values in the row."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from latticebench.clues.base import BinaryClue, Indicators, _ref
from latticebench.schema import Domain, Solution


@dataclass(frozen=True)
class DirectLeft(BinaryClue):
    """value1 is immediately to the left of value2 (pos1 + 1 == pos2)."""

    kind = "direct_left"

    def holds(self, sol: Solution) -> bool:
        p1, p2 = self._p(sol)
        return p1 + 1 == p2

    def to_text(self, domain: Domain) -> str:
        return (
            f"{_ref(self.attr1, self.value1).capitalize()} is immediately to the "
            f"left of {_ref(self.attr2, self.value2)}."
        )

    def to_constraints(self, pos: dict[str, dict[str, Any]]) -> list[Any]:
        return [pos[self.attr1][self.value1] + 1 == pos[self.attr2][self.value2]]

    def penalty(self, ind: Indicators) -> Any:
        p1 = ind.vec(self.attr1, self.value1)
        p2 = ind.vec(self.attr2, self.value2)
        return 1 - ind.dot(p1, ind.shift(p2, -1))


@dataclass(frozen=True)
class DirectRight(BinaryClue):
    """value1 is immediately to the right of value2 (pos1 == pos2 + 1)."""

    kind = "direct_right"

    def holds(self, sol: Solution) -> bool:
        p1, p2 = self._p(sol)
        return p1 == p2 + 1

    def to_text(self, domain: Domain) -> str:
        return (
            f"{_ref(self.attr1, self.value1).capitalize()} is immediately to the "
            f"right of {_ref(self.attr2, self.value2)}."
        )

    def to_constraints(self, pos: dict[str, dict[str, Any]]) -> list[Any]:
        return [pos[self.attr1][self.value1] == pos[self.attr2][self.value2] + 1]

    def penalty(self, ind: Indicators) -> Any:
        p1 = ind.vec(self.attr1, self.value1)
        p2 = ind.vec(self.attr2, self.value2)
        return 1 - ind.dot(p1, ind.shift(p2, 1))


@dataclass(frozen=True)
class SideBySide(BinaryClue):
    """value1 and value2 are next to each other (|pos1 - pos2| == 1)."""

    kind = "side_by_side"

    def holds(self, sol: Solution) -> bool:
        p1, p2 = self._p(sol)
        return abs(p1 - p2) == 1

    def to_text(self, domain: Domain) -> str:
        return (
            f"{_ref(self.attr1, self.value1).capitalize()} is next to "
            f"{_ref(self.attr2, self.value2)}."
        )

    def to_constraints(self, pos: dict[str, dict[str, Any]]) -> list[Any]:
        a, b = pos[self.attr1][self.value1], pos[self.attr2][self.value2]
        return [(a - b == 1) | (b - a == 1)]

    def penalty(self, ind: Indicators) -> Any:
        p1 = ind.vec(self.attr1, self.value1)
        p2 = ind.vec(self.attr2, self.value2)
        neighbours = ind.shift(p2, 1) + ind.shift(p2, -1)
        return 1 - ind.dot(p1, neighbours)


@dataclass(frozen=True)
class LeftOf(BinaryClue):
    """value1 is somewhere to the left of value2 (pos1 < pos2)."""

    kind = "left_of"

    def holds(self, sol: Solution) -> bool:
        p1, p2 = self._p(sol)
        return p1 < p2

    def to_text(self, domain: Domain) -> str:
        return (
            f"{_ref(self.attr1, self.value1).capitalize()} is somewhere to the "
            f"left of {_ref(self.attr2, self.value2)}."
        )

    def to_constraints(self, pos: dict[str, dict[str, Any]]) -> list[Any]:
        return [pos[self.attr1][self.value1] < pos[self.attr2][self.value2]]

    def penalty(self, ind: Indicators) -> Any:
        # violation (pos1 >= pos2) counted as p1 . cumsum(p2)
        p1 = ind.vec(self.attr1, self.value1)
        p2 = ind.vec(self.attr2, self.value2)
        return ind.dot(p1, ind.cumsum(p2))


@dataclass(frozen=True)
class RightOf(BinaryClue):
    """value1 is somewhere to the right of value2 (pos1 > pos2)."""

    kind = "right_of"

    def holds(self, sol: Solution) -> bool:
        p1, p2 = self._p(sol)
        return p1 > p2

    def to_text(self, domain: Domain) -> str:
        return (
            f"{_ref(self.attr1, self.value1).capitalize()} is somewhere to the "
            f"right of {_ref(self.attr2, self.value2)}."
        )

    def to_constraints(self, pos: dict[str, dict[str, Any]]) -> list[Any]:
        return [pos[self.attr1][self.value1] > pos[self.attr2][self.value2]]

    def penalty(self, ind: Indicators) -> Any:
        # violation (pos1 <= pos2) counted as p1 . reverse-cumsum(p2)
        p1 = ind.vec(self.attr1, self.value1)
        p2 = ind.vec(self.attr2, self.value2)
        revcumsum = ind.flip(ind.cumsum(ind.flip(p2)))
        return ind.dot(p1, revcumsum)


@dataclass(frozen=True)
class OneBetween(BinaryClue):
    """Exactly one house sits between value1 and value2 (|pos1 - pos2| == 2)."""

    kind = "one_between"

    def holds(self, sol: Solution) -> bool:
        p1, p2 = self._p(sol)
        return abs(p1 - p2) == 2

    def to_text(self, domain: Domain) -> str:
        return (
            f"There is exactly one house between {_ref(self.attr1, self.value1)} "
            f"and {_ref(self.attr2, self.value2)}."
        )

    def to_constraints(self, pos: dict[str, dict[str, Any]]) -> list[Any]:
        a, b = pos[self.attr1][self.value1], pos[self.attr2][self.value2]
        return [(a - b == 2) | (b - a == 2)]

    def penalty(self, ind: Indicators) -> Any:
        p1 = ind.vec(self.attr1, self.value1)
        p2 = ind.vec(self.attr2, self.value2)
        return 1 - ind.dot(p1, ind.shift(p2, 2) + ind.shift(p2, -2))


@dataclass(frozen=True)
class TwoBetween(BinaryClue):
    """Exactly two houses sit between value1 and value2 (|pos1 - pos2| == 3)."""

    kind = "two_between"

    def holds(self, sol: Solution) -> bool:
        p1, p2 = self._p(sol)
        return abs(p1 - p2) == 3

    def to_text(self, domain: Domain) -> str:
        return (
            f"There are exactly two houses between {_ref(self.attr1, self.value1)} "
            f"and {_ref(self.attr2, self.value2)}."
        )

    def to_constraints(self, pos: dict[str, dict[str, Any]]) -> list[Any]:
        a, b = pos[self.attr1][self.value1], pos[self.attr2][self.value2]
        return [(a - b == 3) | (b - a == 3)]

    def penalty(self, ind: Indicators) -> Any:
        p1 = ind.vec(self.attr1, self.value1)
        p2 = ind.vec(self.attr2, self.value2)
        return 1 - ind.dot(p1, ind.shift(p2, 3) + ind.shift(p2, -3))
