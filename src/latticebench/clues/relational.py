"""Relational clues: two (attribute, value) pairs sharing a house."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from latticebench.clues.base import BinaryClue, Indicators, _ref
from latticebench.schema import Domain, Solution


@dataclass(frozen=True)
class SameHouse(BinaryClue):
    kind = "same_house"

    def holds(self, sol: Solution) -> bool:
        p1, p2 = self._p(sol)
        return p1 == p2

    def to_text(self, domain: Domain) -> str:
        return f"{_ref(self.attr1, self.value1).capitalize()} is {_ref(self.attr2, self.value2)}."

    def to_constraints(self, pos: dict[str, dict[str, Any]]) -> list[Any]:
        return [pos[self.attr1][self.value1] == pos[self.attr2][self.value2]]

    def penalty(self, ind: Indicators) -> Any:
        p1 = ind.vec(self.attr1, self.value1)
        p2 = ind.vec(self.attr2, self.value2)
        return 1 - ind.dot(p1, p2)
