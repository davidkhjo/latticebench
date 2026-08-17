"""Clue inventory. Import a clue by name, or use ``DEFAULT_INVENTORY``."""

from __future__ import annotations

from latticebench.clues.base import CLUE_REGISTRY, BinaryClue, Clue, Indicators, UnaryClue
from latticebench.clues.positional import (
    DirectLeft,
    DirectRight,
    LeftOf,
    OneBetween,
    RightOf,
    SideBySide,
    TwoBetween,
)
from latticebench.clues.relational import SameHouse
from latticebench.clues.unary import FoundAt, NotAt

# The clue types the generator draws from, in a stable order.
DEFAULT_INVENTORY: tuple[type[Clue], ...] = (
    FoundAt,
    NotAt,
    SameHouse,
    DirectLeft,
    DirectRight,
    SideBySide,
    LeftOf,
    RightOf,
    OneBetween,
    TwoBetween,
)

__all__ = [
    "CLUE_REGISTRY",
    "DEFAULT_INVENTORY",
    "BinaryClue",
    "Clue",
    "DirectLeft",
    "DirectRight",
    "FoundAt",
    "Indicators",
    "LeftOf",
    "NotAt",
    "OneBetween",
    "RightOf",
    "SameHouse",
    "SideBySide",
    "TwoBetween",
    "UnaryClue",
]
