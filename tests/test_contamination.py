"""Tests for the contamination-free protocol."""

from __future__ import annotations

import dataclasses

from latticebench.contamination import (
    build_manifest,
    derive_seed,
    held_out_after,
    make_split,
    puzzle_hash,
    verify_regeneration,
)
from latticebench.dataset import to_record
from latticebench.generate import generate_puzzle
from latticebench.schema import Puzzle


def test_puzzle_hash_invariant_to_clue_order() -> None:
    puzzle = generate_puzzle(4, 3, seed=4)
    reversed_puzzle = Puzzle(
        domain=puzzle.domain,
        clues=tuple(reversed(puzzle.clues)),
        solution=puzzle.solution,
        meta=puzzle.meta,
    )
    assert puzzle_hash(puzzle) == puzzle_hash(reversed_puzzle)


def test_derive_seed_is_deterministic() -> None:
    assert derive_seed(123, 0) == derive_seed(123, 0)
    assert derive_seed(123, 1) != derive_seed(123, 0)
    assert derive_seed(999, 5) != derive_seed(123, 5)


def test_verify_regeneration_true_then_false() -> None:
    rec = next(make_split(555, n=4, m=3, count=1, created="2027-01-01", split="t"))
    assert verify_regeneration(rec) is True
    tampered = dataclasses.replace(rec, seed=rec.seed + 1)
    assert verify_regeneration(tampered) is False


def test_make_split_is_reproducible() -> None:
    a = list(make_split(777, n=4, m=3, count=2, created="2027-01-01", split="t"))
    b = list(make_split(777, n=4, m=3, count=2, created="2027-01-01", split="t"))
    assert [r.hash for r in a] == [r.hash for r in b]


def test_held_out_after_filters_strictly() -> None:
    puzzle = generate_puzzle(4, 3, seed=8)
    before = to_record(puzzle, created="2025-01-01", split="t")
    on = to_record(puzzle, created="2026-06-01", split="t")
    after = to_record(puzzle, created="2027-01-01", split="t")
    kept = held_out_after([before, on, after], "2026-06-01")
    # Strictly after: the boundary date itself is excluded.
    assert kept == [after]


def test_build_manifest_is_stable() -> None:
    kwargs = dict(n=4, m=3, count=2, created="2027-01-01", split="t", generator_version="0.1.0")
    m1 = build_manifest(2024, **kwargs)
    m2 = build_manifest(2024, **kwargs)
    assert m1.hash_of_hashes == m2.hash_of_hashes
    assert m1 == m2
    # A different master seed yields a different fingerprint.
    m3 = build_manifest(2025, **kwargs)
    assert m3.hash_of_hashes != m1.hash_of_hashes
