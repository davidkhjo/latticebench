"""The contamination-free protocol.

A split is distributed as a :class:`Manifest`: a master seed, the generator
version, and the git commit that produced it. Anyone re-derives the exact same
puzzles bit-for-bit from that manifest, so no solved instances are ever
published. Two checks guard against leakage. Each instance carries a creation
date, so a split can be held out to instances minted after a model's training
cutoff. And each instance hashes back to its seed, so a submitted result cannot
be gamed by editing the puzzle: ``verify_regeneration`` regenerates from the seed
and compares hashes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from latticebench.generate import generate_puzzle
from latticebench.schema import Puzzle

if TYPE_CHECKING:
    from latticebench.dataset import PuzzleRecord

_MASK64 = (1 << 64) - 1


def canonical_bytes(puzzle: Puzzle) -> bytes:
    """A serialization invariant to clue order, for stable hashing."""
    obj: dict[str, Any] = {
        "n": puzzle.domain.n,
        "m": puzzle.domain.m,
        "domain": {a.name: list(a.values) for a in puzzle.domain.attributes},
        "clues": sorted(json.dumps(c.to_dict(), sort_keys=True) for c in puzzle.clues),
    }
    if puzzle.solution is not None:
        obj["solution"] = {
            attr: dict(sorted(vals.items())) for attr, vals in puzzle.solution.pos.items()
        }
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def puzzle_hash(puzzle: Puzzle) -> str:
    return hashlib.sha256(canonical_bytes(puzzle)).hexdigest()


def derive_seed(master_seed: int, index: int) -> int:
    """Deterministic per-instance seed (SplitMix64)."""
    z = (master_seed + index * 0x9E3779B97F4A7C15) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (z ^ (z >> 31)) & _MASK64


def verify_regeneration(rec: PuzzleRecord) -> bool:
    """Regenerate the instance from its seed and confirm the hash matches."""
    puzzle = generate_puzzle(rec.n, rec.m, seed=rec.seed, generator_version=rec.generator_version)
    return puzzle_hash(puzzle) == rec.hash


def held_out_after(records: Iterable[PuzzleRecord], date: str) -> list[PuzzleRecord]:
    """Keep only instances created strictly after ``date`` (ISO ``YYYY-MM-DD``)."""
    return [r for r in records if r.created > date]


def make_split(
    master_seed: int,
    *,
    n: int,
    m: int,
    count: int,
    created: str,
    split: str,
    include_solution: bool = True,
) -> Iterator[PuzzleRecord]:
    """Yield ``count`` puzzle records derived from ``master_seed``."""
    from latticebench.dataset import to_record  # deferred to avoid an import cycle

    for i in range(count):
        seed = derive_seed(master_seed, i)
        puzzle = generate_puzzle(n, m, seed=seed)
        yield to_record(puzzle, created=created, split=split, include_solution=include_solution)


@dataclass(frozen=True)
class Manifest:
    """The distributable description of a split. Re-derives the puzzles exactly."""

    master_seed: int
    generator_version: str
    git_commit: str
    n: int
    m: int
    count: int
    created: str
    split: str
    hash_of_hashes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Manifest:
        return cls(**d)


def build_manifest(
    master_seed: int,
    *,
    n: int,
    m: int,
    count: int,
    created: str,
    split: str,
    generator_version: str,
    git_commit: str = "",
) -> Manifest:
    """Generate a split and fingerprint it into a manifest."""
    hashes = [
        rec.hash
        for rec in make_split(master_seed, n=n, m=m, count=count, created=created, split=split)
    ]
    hoh = hashlib.sha256("".join(hashes).encode("utf-8")).hexdigest()
    return Manifest(
        master_seed=master_seed,
        generator_version=generator_version,
        git_commit=git_commit,
        n=n,
        m=m,
        count=count,
        created=created,
        split=split,
        hash_of_hashes=hoh,
    )
