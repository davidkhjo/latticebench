"""Difficulty grading.

Difficulty has a structural part that depends only on the grid shape (the number
of joint arrangements the solver must rule out) and a search part measured from
the solver itself (conflict count). The tier is assigned from the structural
part, which is deterministic; the conflict count is recorded alongside as an
empirical hardness signal that correlates with model failure.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from latticebench.oracle import conflict_count
from latticebench.schema import Domain, Puzzle

_TIERS = (("easy", 5.0), ("medium", 12.0), ("hard", 22.0))


@dataclass(frozen=True)
class Difficulty:
    n: int
    m: int
    n_clues: int
    z3_conflicts: int
    log_search_space: float
    tier: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def search_space_size(domain: Domain) -> float:
    """Number of joint arrangements up to a fixed reference attribute: ``(n!)^(m-1)``."""
    return float(math.factorial(domain.n) ** max(domain.m - 1, 0))


def log_search_space(domain: Domain) -> float:
    return max(domain.m - 1, 0) * math.lgamma(domain.n + 1)


def _tier(log_ss: float) -> str:
    for name, threshold in _TIERS:
        if log_ss < threshold:
            return name
    return "expert"


def grade(puzzle: Puzzle, *, solver: str = "z3") -> Difficulty:
    """Grade a puzzle by grid shape, clue count, and solver conflict count."""
    log_ss = log_search_space(puzzle.domain)
    conflicts = conflict_count(puzzle.domain, puzzle.clues, solver=solver)
    return Difficulty(
        n=puzzle.domain.n,
        m=puzzle.domain.m,
        n_clues=len(puzzle.clues),
        z3_conflicts=int(conflicts),
        log_search_space=log_ss,
        tier=_tier(log_ss),
    )
