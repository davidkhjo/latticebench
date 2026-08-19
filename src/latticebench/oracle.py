"""Constraint model and uniqueness oracle.

CPMpy is the single glue layer: it compiles a puzzle to an ``AllDifferent`` model
plus the clue constraints and dispatches to OR-Tools CP-SAT (the default and the
uniqueness oracle) or Z3. ``is_unique`` is the source of truth for "solvable":
the generator only ever emits puzzles with exactly one solution.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cpmpy as cp

from latticebench.clues.base import Clue
from latticebench.schema import Assignment, Domain


@dataclass
class PosVars:
    """A CPMpy model with an integer position variable per (attribute, value)."""

    domain: Domain
    pos: dict[str, dict[str, Any]]
    model: cp.Model


def build_model(domain: Domain, clues: Sequence[Clue]) -> PosVars:
    """Build ``pos[attr][value] in 0..n-1`` with all-different rows and clue constraints."""
    pos: dict[str, dict[str, Any]] = {}
    model = cp.Model()
    for attr in domain.attributes:
        row = {v: cp.intvar(0, domain.n - 1, name=f"{attr.name}::{v}") for v in attr.values}
        pos[attr.name] = row
        model += cp.AllDifferent(list(row.values()))
    for clue in clues:
        model += clue.to_constraints(pos)
    return PosVars(domain=domain, pos=pos, model=model)


def _extract(pv: PosVars) -> Assignment:
    return {attr: {v: int(var.value()) for v, var in row.items()} for attr, row in pv.pos.items()}


def solve(
    domain: Domain,
    clues: Sequence[Clue],
    *,
    solver: str = "ortools",
    time_limit: float | None = None,
) -> Assignment | None:
    """Return one solution, or ``None`` if the constraints are unsatisfiable."""
    pv = build_model(domain, clues)
    ok = pv.model.solve(solver=solver, time_limit=time_limit)
    if not ok:
        return None
    return _extract(pv)


def count_solutions(
    domain: Domain,
    clues: Sequence[Clue],
    *,
    limit: int = 2,
    solver: str = "ortools",
) -> int:
    """Count solutions, stopping once ``limit`` have been found."""
    pv = build_model(domain, clues)
    return int(pv.model.solveAll(solver=solver, solution_limit=limit, display=None))


def is_unique(domain: Domain, clues: Sequence[Clue], *, solver: str = "ortools") -> bool:
    """True iff the puzzle has exactly one solution."""
    return count_solutions(domain, clues, limit=2, solver=solver) == 1


def conflict_count(domain: Domain, clues: Sequence[Clue], *, solver: str = "ortools") -> int:
    """A best-effort search-hardness signal (solver conflicts); 0 if unavailable."""
    pv = build_model(domain, clues)
    s = cp.SolverLookup.get(solver, pv.model)
    s.solve()

    ort = getattr(s, "ort_solver", None)
    if ort is not None:
        fn = getattr(ort, "NumConflicts", None)
        if callable(fn):
            try:
                return int(fn())
            except Exception:
                pass
        val = getattr(ort, "num_conflicts", None)
        if val is not None:
            try:
                return int(val)
            except Exception:
                pass

    z3s = getattr(s, "z3_solver", None) or getattr(s, "z3solver", None)
    if z3s is not None:
        try:
            stats = z3s.statistics()
            for key in stats:
                if "conflict" in key:
                    return int(stats.get_key_value(key))
        except Exception:
            pass
    return 0
