"""Procedural puzzle generation.

The pipeline follows ZebraLogic: sample a full solution, enumerate every clue
that is true under it, add clues in random order until the solution is unique,
then greedily drop clues until the set is minimal. Generation is a pure function
of ``(n, m, seed, inventory, value_pool, generator_version)`` — all randomness
flows through one seeded numpy Generator and there is no wall-clock — so a puzzle
regenerates bit-for-bit from its seed. That purity is what the contamination-free
protocol relies on.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from latticebench import __version__
from latticebench.clues import DEFAULT_INVENTORY
from latticebench.clues.base import BinaryClue, Clue, UnaryClue
from latticebench.oracle import is_unique
from latticebench.schema import Attribute, Domain, Puzzle, Solution

# Attribute names paired with a pool of values. A domain of size (n, m) uses the
# first m attributes and the first n values of each, so it supports n, m <= 8, 6.
DEFAULT_VALUE_POOL: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "nationality",
        ("english", "spanish", "japanese", "italian", "norwegian", "german", "french", "swedish"),
    ),
    ("color", ("red", "green", "blue", "yellow", "white", "purple", "orange", "black")),
    ("drink", ("tea", "coffee", "milk", "water", "juice", "cola", "wine", "cocoa")),
    ("pet", ("cat", "dog", "bird", "fish", "horse", "snake", "rabbit", "hamster")),
    (
        "hobby",
        ("chess", "painting", "reading", "dancing", "cooking", "hiking", "gaming", "sailing"),
    ),
    ("food", ("apple", "bread", "cheese", "grapes", "mango", "olives", "pasta", "quiche")),
)


def build_domain(
    n: int, m: int, *, value_pool: Sequence[tuple[str, tuple[str, ...]]] = DEFAULT_VALUE_POOL
) -> Domain:
    """Build a domain from the first ``m`` attributes and ``n`` values of the pool."""
    if m > len(value_pool):
        raise ValueError(f"m={m} exceeds the {len(value_pool)} attributes in the pool")
    attrs = []
    for name, values in list(value_pool)[:m]:
        if n > len(values):
            raise ValueError(f"n={n} exceeds the {len(values)} values for {name!r}")
        attrs.append(Attribute(name, tuple(values[:n])))
    return Domain(n=n, attributes=tuple(attrs))


def sample_solution(domain: Domain, rng: np.random.Generator) -> Solution:
    """Assign each attribute's values to houses by an independent random permutation."""
    pos: dict[str, dict[str, int]] = {}
    for attr in domain.attributes:
        perm = rng.permutation(domain.n)
        pos[attr.name] = {v: int(perm[i]) for i, v in enumerate(attr.values)}
    return Solution(pos=pos)


def all_true_clues(
    domain: Domain,
    solution: Solution,
    inventory: Sequence[type[Clue]] = DEFAULT_INVENTORY,
) -> list[Clue]:
    """Enumerate every clue of every inventory type that holds under ``solution``."""
    items = [(a.name, v) for a in domain.attributes for v in a.values]
    out: list[Clue] = []
    for clue_type in inventory:
        if issubclass(clue_type, UnaryClue):
            for attr, value in items:
                for house in range(domain.n):
                    unary: Clue = clue_type(attr, value, house)
                    if unary.holds(solution):
                        out.append(unary)
        elif issubclass(clue_type, BinaryClue):
            for a1, v1 in items:
                for a2, v2 in items:
                    if (a1, v1) == (a2, v2):
                        continue
                    binary: Clue = clue_type(a1, v1, a2, v2)
                    if binary.holds(solution):
                        out.append(binary)
    return out


def minimize_clues(domain: Domain, clues: Sequence[Clue], *, solver: str = "ortools") -> list[Clue]:
    """Greedily drop clues while the solution stays unique, until the set is minimal."""
    kept = list(clues)
    changed = True
    while changed:
        changed = False
        for clue in list(kept):
            trial = [c for c in kept if c is not clue]
            if trial and is_unique(domain, trial, solver=solver):
                kept = trial
                changed = True
    return kept


def generate_puzzle(
    n: int,
    m: int,
    *,
    seed: int,
    inventory: Sequence[type[Clue]] = DEFAULT_INVENTORY,
    value_pool: Sequence[tuple[str, tuple[str, ...]]] = DEFAULT_VALUE_POOL,
    solver: str = "ortools",
    generator_version: str = __version__,
) -> Puzzle:
    """Generate a puzzle with a certified unique solution and a minimal clue set."""
    rng = np.random.default_rng(seed)
    domain = build_domain(n, m, value_pool=value_pool)
    solution = sample_solution(domain, rng)

    candidates = all_true_clues(domain, solution, inventory)
    order = rng.permutation(len(candidates))
    shuffled = [candidates[i] for i in order]

    chosen: list[Clue] = []
    for clue in shuffled:
        chosen.append(clue)
        if is_unique(domain, chosen, solver=solver):
            break
    else:
        raise RuntimeError("clue superset failed to pin a unique solution")

    chosen = minimize_clues(domain, chosen, solver=solver)
    meta = {
        "seed": int(seed),
        "generator_version": generator_version,
        "n": n,
        "m": m,
    }
    return Puzzle(domain=domain, clues=tuple(chosen), solution=solution, meta=meta)
