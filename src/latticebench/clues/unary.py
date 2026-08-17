"""Unary clues: an (attribute, value) pinned to, or barred from, a house."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from latticebench.clues.base import Indicators, UnaryClue, _ref
from latticebench.schema import Domain, Solution


@dataclass(frozen=True)
class FoundAt(UnaryClue):
    kind = "found_at"

    def holds(self, sol: Solution) -> bool:
        return sol.house_of(self.attr, self.value) == self.house

    def to_text(self, domain: Domain) -> str:
        return f"{_ref(self.attr, self.value).capitalize()} is house {self.house + 1}."

    def to_constraints(self, pos: dict[str, dict[str, Any]]) -> list[Any]:
        return [pos[self.attr][self.value] == self.house]

    def penalty(self, ind: Indicators) -> Any:
        return 1 - ind.house(self.attr, self.value, self.house)


@dataclass(frozen=True)
class NotAt(UnaryClue):
    kind = "not_at"

    def holds(self, sol: Solution) -> bool:
        return sol.house_of(self.attr, self.value) != self.house

    def to_text(self, domain: Domain) -> str:
        return f"{_ref(self.attr, self.value).capitalize()} is not house {self.house + 1}."

    def to_constraints(self, pos: dict[str, dict[str, Any]]) -> list[Any]:
        return [pos[self.attr][self.value] != self.house]

    def penalty(self, ind: Indicators) -> Any:
        return ind.house(self.attr, self.value, self.house)
