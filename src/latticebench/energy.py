"""Energy formulation.

A puzzle compiles to an energy over binary variables ``x`` indexed by
``(attribute, value, house)``. Two families of terms make it up:

* structural terms that force each value into one house and each house to hold
  one value per attribute (a permutation per attribute), and
* one clue term per clue, ``>= 0`` and ``0`` iff the clue is satisfied.

With ``lam_struct`` large enough that a structural violation outweighs any clue
saving, ``energy(x) == 0`` holds exactly when ``x`` encodes a valid solution that
satisfies every clue. Since the generator guarantees a unique solution, that is
the single global minimum. Every term is at most quadratic in ``x``, so the whole
puzzle is a Potts/QUBO model, and ``as_torch_energy`` exposes it with ebmkit's
``p(x) proportional to exp(-E(x))`` convention.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from latticebench.clues.base import Clue
from latticebench.schema import Assignment, Domain, Solution


class _NumpyBackend:
    @staticmethod
    def reshape(x: Any, shape: tuple[int, ...]) -> Any:
        return x.reshape(shape)

    @staticmethod
    def sum_axes(x: Any, axes: tuple[int, ...]) -> Any:
        return x.sum(axis=axes)

    @staticmethod
    def sum_last(x: Any) -> Any:
        return x.sum(axis=-1)

    @staticmethod
    def cumsum_last(x: Any) -> Any:
        return np.cumsum(x, axis=-1)

    @staticmethod
    def flip_last(x: Any) -> Any:
        return np.flip(x, axis=-1)

    @staticmethod
    def cat_last(parts: list[Any]) -> Any:
        return np.concatenate(parts, axis=-1)

    @staticmethod
    def zeros_tail(vec: Any, k: int) -> Any:
        return np.zeros((*vec.shape[:-1], k), dtype=vec.dtype)


class _TorchBackend:
    def __init__(self, torch: Any) -> None:
        self.t = torch

    def reshape(self, x: Any, shape: tuple[int, ...]) -> Any:
        return x.reshape(shape)

    def sum_axes(self, x: Any, axes: tuple[int, ...]) -> Any:
        return x.sum(dim=axes)

    def sum_last(self, x: Any) -> Any:
        return x.sum(dim=-1)

    def cumsum_last(self, x: Any) -> Any:
        return self.t.cumsum(x, dim=-1)

    def flip_last(self, x: Any) -> Any:
        return self.t.flip(x, dims=(-1,))

    def cat_last(self, parts: list[Any]) -> Any:
        return self.t.cat(parts, dim=-1)

    @staticmethod
    def zeros_tail(vec: Any, k: int) -> Any:
        return vec.new_zeros((*vec.shape[:-1], k))


class _Indicators:
    """Implements the :class:`~latticebench.clues.base.Indicators` protocol."""

    def __init__(self, x: Any, grid: EnergyGrid, backend: Any) -> None:
        self.x = x
        self.g = grid
        self.b = backend
        self.n = grid.n

    def vec(self, attr: str, value: str) -> Any:
        return self.x[..., self.g._vec_idx[attr][value]]

    def house(self, attr: str, value: str, h: int) -> Any:
        return self.x[..., self.g.var(attr, value, h)]

    def shift(self, vec: Any, k: int) -> Any:
        n = self.n
        if k == 0:
            return vec
        if abs(k) >= n:
            return self.b.zeros_tail(vec, n)
        if k > 0:
            return self.b.cat_last([self.b.zeros_tail(vec, k), vec[..., : n - k]])
        kk = -k
        return self.b.cat_last([vec[..., kk:], self.b.zeros_tail(vec, kk)])

    def cumsum(self, vec: Any) -> Any:
        return self.b.cumsum_last(vec)

    def flip(self, vec: Any) -> Any:
        return self.b.flip_last(vec)

    def dot(self, a: Any, b: Any) -> Any:
        return self.b.sum_last(a * b)


class EnergyGrid:
    """Compiles a puzzle into an energy over ``x`` of length ``D = m * n * n``."""

    def __init__(
        self,
        domain: Domain,
        clues: Sequence[Clue],
        *,
        lam_struct: float | None = None,
    ) -> None:
        self.domain = domain
        self.clues = tuple(clues)
        self.n = domain.n
        self.m = domain.m
        self.D = self.m * self.n * self.n
        self._attr_index = {a.name: i for i, a in enumerate(domain.attributes)}
        self._val_index = {
            a.name: {v: j for j, v in enumerate(a.values)} for a in domain.attributes
        }
        self._vec_idx = {
            a.name: {v: [self.var(a.name, v, h) for h in range(self.n)] for v in a.values}
            for a in domain.attributes
        }
        if lam_struct is None:
            # larger than the worst clue saving on a non-permutation assignment
            lam_struct = float(len(self.clues) * self.n + 1)
        self.lam_struct = float(lam_struct)

    def var(self, attr: str, value: str, house: int) -> int:
        return (self._attr_index[attr] * self.n + self._val_index[attr][value]) * self.n + house

    # -- energy ------------------------------------------------------------

    def _structural(self, x: Any, b: Any) -> Any:
        lead = tuple(x.shape[:-1])
        xr = b.reshape(x, (*lead, self.m, self.n, self.n))
        row = b.sum_axes((b.sum_last(xr) - 1) ** 2, (-2, -1))  # each value one house
        col = b.sum_axes((b.sum_axes(xr, (-2,)) - 1) ** 2, (-2, -1))  # each house one value
        return row + col

    def _energy(self, x: Any, b: Any) -> Any:
        struct = self._structural(x, b)
        ind = _Indicators(x, self, b)
        clue = struct * 0
        for c in self.clues:
            clue = clue + c.penalty(ind)
        return self.lam_struct * struct + clue

    def structural_energy(self, x: Any) -> Any:
        return self._structural(np.asarray(x, dtype=float), _NumpyBackend())

    def clue_energy(self, x: Any) -> Any:
        xa = np.asarray(x, dtype=float)
        ind = _Indicators(xa, self, _NumpyBackend())
        total = np.zeros(xa.shape[:-1])
        for c in self.clues:
            total = total + c.penalty(ind)
        return total

    def energy(self, x: Any) -> Any:
        return self._energy(np.asarray(x, dtype=float), _NumpyBackend())

    # -- encode / decode ---------------------------------------------------

    def encode(self, a: Assignment | Solution) -> np.ndarray:
        if isinstance(a, Solution):
            a = a.to_assignment()
        x = np.zeros(self.D, dtype=float)
        for attr, row in a.items():
            for value, house in row.items():
                x[self.var(attr, value, int(house))] = 1.0
        return x

    def decode(self, x: Any) -> Assignment:
        xa = np.asarray(x, dtype=float)
        out: Assignment = {}
        for attr in self.domain.attributes:
            out[attr.name] = {
                v: int(np.argmax(xa[self._vec_idx[attr.name][v]])) for v in attr.values
            }
        return out

    # -- alternative representations --------------------------------------

    def qubo(self) -> tuple[np.ndarray, float]:
        """Return ``(Q, offset)`` with ``energy(x) == x @ Q @ x + offset`` for binary ``x``.

        Recovered from the energy itself by finite differences, so it always
        matches :meth:`energy` without a second hand-written clue representation.
        """
        d = self.D
        eye = np.eye(d, dtype=float)
        c = float(self.energy(np.zeros(d)))
        e1 = np.asarray(self.energy(eye), dtype=float)  # energy(e_i)
        diag = e1 - c
        Q = np.zeros((d, d), dtype=float)
        np.fill_diagonal(Q, diag)
        iu, ju = np.triu_indices(d, k=1)
        pairs = eye[iu] + eye[ju]
        epair = np.asarray(self.energy(pairs), dtype=float)
        cross = epair - e1[iu] - e1[ju] + c
        Q[iu, ju] = cross / 2.0
        Q[ju, iu] = cross / 2.0
        return Q, c

    def as_torch_energy(self) -> Any:
        """Return ``fn(x: Tensor[..., D]) -> Tensor[...]`` for use with ebmkit samplers."""
        import torch  # noqa: F401  (optional dependency, guarded at call time)

        backend = _TorchBackend(torch)

        def energy_fn(x: Any) -> Any:
            return self._energy(x, backend)

        return energy_fn


def energy_grid(puzzle: Any, *, lam_struct: float | None = None) -> EnergyGrid:
    return EnergyGrid(puzzle.domain, puzzle.clues, lam_struct=lam_struct)
