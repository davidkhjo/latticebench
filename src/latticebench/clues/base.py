"""The clue abstraction.

A clue is self-describing along three axes that must always agree:

* ``holds`` — is it true under a complete solution (used during generation),
* ``to_constraints`` — its CPMpy form (used by the solver and uniqueness oracle),
* ``penalty`` — its energy term, ``>= 0`` and ``0`` iff satisfied, at most
  quadratic in the one-hot assignment variables (used by the energy formulation).

``penalty`` is written against the :class:`Indicators` accessor so the same code
runs on a numpy array or a torch tensor without change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

from latticebench.schema import Domain, Solution

CLUE_REGISTRY: dict[str, type[Clue]] = {}


@runtime_checkable
class Indicators(Protocol):
    """Accessor over the position indicators of a (possibly batched) assignment.

    ``vec(attr, value)`` returns the length-``n`` indicator vector whose ``h``-th
    entry is 1 when ``value`` sits in house ``h``. All operations preserve any
    leading batch axes and never wrap around the ends of the row.
    """

    n: int

    def vec(self, attr: str, value: str) -> Any: ...
    def house(self, attr: str, value: str, h: int) -> Any: ...
    def shift(self, vec: Any, k: int) -> Any: ...
    def cumsum(self, vec: Any) -> Any: ...
    def flip(self, vec: Any) -> Any: ...
    def dot(self, a: Any, b: Any) -> Any: ...


class Clue(ABC):
    """Base class for all clue types. Concrete subclasses set a ``kind`` string."""

    kind: ClassVar[str]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        kind = cls.__dict__.get("kind")
        if kind:
            CLUE_REGISTRY[kind] = cls

    @abstractmethod
    def holds(self, sol: Solution) -> bool: ...

    @abstractmethod
    def to_text(self, domain: Domain) -> str: ...

    @abstractmethod
    def to_constraints(self, pos: dict[str, dict[str, Any]]) -> list[Any]: ...

    @abstractmethod
    def penalty(self, ind: Indicators) -> Any: ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Clue:
        return CLUE_REGISTRY[d["kind"]]._from_fields(d)

    @classmethod
    @abstractmethod
    def _from_fields(cls, d: dict[str, Any]) -> Clue: ...


def _ref(attr: str, value: str) -> str:
    return f"the house where {attr} is {value}"


@dataclass(frozen=True)
class UnaryClue(Clue):
    """A clue about a single (attribute, value) and an absolute house."""

    attr: str
    value: str
    house: int

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "attr": self.attr, "value": self.value, "house": self.house}

    @classmethod
    def _from_fields(cls, d: dict[str, Any]) -> UnaryClue:
        return cls(attr=d["attr"], value=d["value"], house=int(d["house"]))


@dataclass(frozen=True)
class BinaryClue(Clue):
    """A clue relating two (attribute, value) pairs by their houses."""

    attr1: str
    value1: str
    attr2: str
    value2: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "attr1": self.attr1,
            "value1": self.value1,
            "attr2": self.attr2,
            "value2": self.value2,
        }

    @classmethod
    def _from_fields(cls, d: dict[str, Any]) -> BinaryClue:
        return cls(attr1=d["attr1"], value1=d["value1"], attr2=d["attr2"], value2=d["value2"])

    def _p(self, sol: Solution) -> tuple[int, int]:
        return sol.house_of(self.attr1, self.value1), sol.house_of(self.attr2, self.value2)
