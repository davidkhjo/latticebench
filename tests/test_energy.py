"""Tests for the QUBO/Potts energy formulation (the linchpin).

The invariant that ties the whole benchmark together: on the valid permutation
assignments, ``energy(x) == 0`` exactly when every clue holds.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from latticebench.energy import EnergyGrid
from latticebench.generate import build_domain, generate_puzzle
from latticebench.schema import Domain, Solution


def _all_permutation_assignments(domain: Domain):
    """Yield every Solution that is a per-attribute bijection value -> house."""
    per_attr = []
    for a in domain.attributes:
        opts = [
            dict(zip(a.values, perm, strict=True))
            for perm in itertools.permutations(range(domain.n))
        ]
        per_attr.append(opts)
    for combo in itertools.product(*per_attr):
        pos = {a.name: combo[i] for i, a in enumerate(domain.attributes)}
        yield Solution(pos=pos)


def test_solution_has_zero_energy() -> None:
    puzzle = generate_puzzle(3, 3, seed=3)
    grid = EnergyGrid(puzzle.domain, puzzle.clues)
    assert abs(float(grid.energy(grid.encode(puzzle.solution)))) < 1e-9


def test_energy_zero_iff_all_clues_hold() -> None:
    puzzle = generate_puzzle(3, 3, seed=3)
    grid = EnergyGrid(puzzle.domain, puzzle.clues)
    seen_zero = 0
    for sol in _all_permutation_assignments(puzzle.domain):
        e = float(grid.energy(grid.encode(sol)))
        all_hold = all(c.holds(sol) for c in puzzle.clues)
        assert (abs(e) < 1e-9) == all_hold
        if abs(e) < 1e-9:
            seen_zero += 1
    # The generated puzzle is unique, so exactly one permutation has zero energy.
    assert seen_zero == 1


def test_structural_energy_zero_on_valid_permutation(
    tiny_domain: Domain, tiny_solution: Solution
) -> None:
    grid = EnergyGrid(tiny_domain, [])
    x = grid.encode(tiny_solution)
    assert abs(float(grid.structural_energy(x))) < 1e-9


def test_structural_energy_positive_on_invalid() -> None:
    domain = build_domain(3, 2)
    grid = EnergyGrid(domain, [])
    x = np.zeros(grid.D)  # nothing placed: every value/house count is off
    assert float(grid.structural_energy(x)) > 0.0


def test_clue_energy_zero_on_solution() -> None:
    puzzle = generate_puzzle(3, 3, seed=5)
    grid = EnergyGrid(puzzle.domain, puzzle.clues)
    assert abs(float(grid.clue_energy(grid.encode(puzzle.solution)))) < 1e-9


def test_qubo_reproduces_energy_on_random_binaries() -> None:
    puzzle = generate_puzzle(3, 3, seed=2)
    grid = EnergyGrid(puzzle.domain, puzzle.clues)
    Q, offset = grid.qubo()
    rng = np.random.default_rng(0)
    for _ in range(200):
        x = rng.integers(0, 2, size=grid.D).astype(float)
        direct = float(grid.energy(x))
        via_qubo = float(x @ Q @ x + offset)
        assert abs(direct - via_qubo) < 1e-9


def test_decode_encode_round_trip(tiny_domain: Domain, tiny_solution: Solution) -> None:
    grid = EnergyGrid(tiny_domain, [])
    x = grid.encode(tiny_solution)
    assert grid.decode(x) == tiny_solution.to_assignment()


def test_encode_accepts_assignment_and_solution(
    tiny_domain: Domain, tiny_solution: Solution
) -> None:
    grid = EnergyGrid(tiny_domain, [])
    from_sol = grid.encode(tiny_solution)
    from_assignment = grid.encode(tiny_solution.to_assignment())
    assert np.array_equal(from_sol, from_assignment)


def test_batched_energy_shape() -> None:
    puzzle = generate_puzzle(3, 3, seed=3)
    grid = EnergyGrid(puzzle.domain, puzzle.clues)
    x = grid.encode(puzzle.solution)
    batch = np.stack([x, x])
    out = grid.energy(batch)
    assert out.shape == (2,)
    assert np.allclose(out, 0.0)


def test_var_indices_are_distinct_and_in_range() -> None:
    domain = build_domain(3, 3)
    grid = EnergyGrid(domain, [])
    idx = [
        grid.var(a.name, v, h) for a in domain.attributes for v in a.values for h in range(domain.n)
    ]
    assert len(set(idx)) == grid.D
    assert min(idx) == 0
    assert max(idx) == grid.D - 1


def test_as_torch_energy_matches_numpy() -> None:
    torch = pytest.importorskip("torch")
    puzzle = generate_puzzle(3, 3, seed=3)
    grid = EnergyGrid(puzzle.domain, puzzle.clues)
    x = grid.encode(puzzle.solution)
    fn = grid.as_torch_energy()
    xt = torch.tensor(x[None], dtype=torch.float64)
    e = float(fn(xt).item())
    assert abs(e - float(grid.energy(x))) < 1e-9
