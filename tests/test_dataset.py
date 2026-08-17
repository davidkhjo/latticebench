"""Tests for JSONL records and natural-language rendering."""

from __future__ import annotations

from latticebench.contamination import puzzle_hash
from latticebench.dataset import (
    SCHEMA_VERSION,
    from_record,
    read_jsonl,
    render_prompt,
    to_record,
    write_jsonl,
)
from latticebench.generate import generate_puzzle


def test_to_record_from_record_preserves_hash() -> None:
    puzzle = generate_puzzle(4, 3, seed=8)
    rec = to_record(puzzle, created="2027-01-01", split="test")
    assert rec.schema_version == SCHEMA_VERSION
    assert rec.hash == puzzle_hash(puzzle)
    rebuilt = from_record(rec)
    assert puzzle_hash(rebuilt) == rec.hash


def test_jsonl_round_trip(tmp_path) -> None:
    puzzles = [generate_puzzle(4, 3, seed=s) for s in (1, 2, 3)]
    records = [to_record(p, created="2027-01-01", split="test") for p in puzzles]
    path = tmp_path / "puzzles.jsonl"
    n = write_jsonl(path, records)
    assert n == 3
    loaded = list(read_jsonl(path))
    assert loaded == records


def test_challenge_record_omits_solution() -> None:
    puzzle = generate_puzzle(4, 3, seed=8)
    rec = to_record(puzzle, created="2027-01-01", split="test", include_solution=False)
    assert rec.solution is None


def test_render_prompt_is_deterministic_and_lists_values() -> None:
    puzzle = generate_puzzle(4, 3, seed=8)
    p1 = render_prompt(puzzle)
    p2 = render_prompt(puzzle)
    assert p1 == p2
    for a in puzzle.domain.attributes:
        assert a.name in p1
        for v in a.values:
            assert v in p1
    # Every clue is stated.
    assert p1.count("\n") > 0
    assert "JSON" in p1
