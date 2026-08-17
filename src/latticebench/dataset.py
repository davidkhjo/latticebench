"""JSONL dataset records and natural-language rendering.

A record carries everything needed to present a puzzle to a model and to grade
it, plus the seed and version that regenerate it. Challenge records omit the
solution; the harness scores against a solved copy re-derived from the seed.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from latticebench.clues.base import Clue
from latticebench.contamination import puzzle_hash
from latticebench.difficulty import grade
from latticebench.schema import Attribute, Domain, Puzzle, Solution

SCHEMA_VERSION = 1


@dataclass
class PuzzleRecord:
    id: str
    schema_version: int
    seed: int
    generator_version: str
    n: int
    m: int
    domain: dict[str, list[str]]
    clues: list[dict[str, Any]]
    clue_texts: list[str]
    nl_prompt: str
    solution: dict[str, dict[str, int]] | None
    difficulty: dict[str, Any]
    hash: str
    created: str
    split: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PuzzleRecord:
        return cls(**d)


def render_prompt(puzzle: Puzzle) -> str:
    """A deterministic natural-language statement plus the required answer format."""
    dom = puzzle.domain
    n = dom.n
    attr_names = ", ".join(dom.attribute_names)
    lines = [
        f"There are {n} houses in a row, numbered 1 to {n} from left to right.",
        f"Each house has exactly one of each attribute: {attr_names}.",
        "The possible values are:",
    ]
    for a in dom.attributes:
        lines.append(f"- {a.name}: {', '.join(a.values)}")
    lines.append("")
    lines.append("Clues:")
    for i, clue in enumerate(puzzle.clues, start=1):
        lines.append(f"{i}. {clue.to_text(dom)}")
    lines.append("")
    lines.append(
        "Work out which value of each attribute belongs to each house. There is "
        "exactly one arrangement consistent with the clues."
    )
    example = "{" + ", ".join(f'"{a.name}": [...]' for a in dom.attributes) + "}"
    lines.append(
        f"Return only a JSON object mapping each attribute to a list of {n} values, "
        f"where the i-th entry is that attribute's value in house i, like {example}."
    )
    return "\n".join(lines)


def to_record(
    puzzle: Puzzle, *, created: str, split: str, include_solution: bool = True
) -> PuzzleRecord:
    """Build a JSONL record from a generated puzzle."""
    h = puzzle_hash(puzzle)
    solution = None
    if include_solution and puzzle.solution is not None:
        solution = {attr: dict(vals) for attr, vals in puzzle.solution.pos.items()}
    return PuzzleRecord(
        id=h[:12],
        schema_version=SCHEMA_VERSION,
        seed=int(cast(int, puzzle.meta["seed"])),
        generator_version=str(puzzle.meta["generator_version"]),
        n=puzzle.domain.n,
        m=puzzle.domain.m,
        domain={a.name: list(a.values) for a in puzzle.domain.attributes},
        clues=[c.to_dict() for c in puzzle.clues],
        clue_texts=[c.to_text(puzzle.domain) for c in puzzle.clues],
        nl_prompt=render_prompt(puzzle),
        solution=solution,
        difficulty=grade(puzzle).to_dict(),
        hash=h,
        created=created,
        split=split,
    )


def from_record(rec: PuzzleRecord) -> Puzzle:
    """Reconstruct a puzzle from a record (without regenerating it)."""
    attrs = tuple(Attribute(name, tuple(vals)) for name, vals in rec.domain.items())
    domain = Domain(n=rec.n, attributes=attrs)
    clues = tuple(Clue.from_dict(d) for d in rec.clues)
    solution = None
    if rec.solution is not None:
        solution = Solution(
            pos={a: {v: int(h) for v, h in row.items()} for a, row in rec.solution.items()}
        )
    meta = {"seed": rec.seed, "generator_version": rec.generator_version, "n": rec.n, "m": rec.m}
    return Puzzle(domain=domain, clues=clues, solution=solution, meta=meta)


def write_jsonl(path: str | Path, records: Iterable[PuzzleRecord]) -> int:
    """Write records as JSON lines; return the count written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict()) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> Iterator[PuzzleRecord]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield PuzzleRecord.from_dict(json.loads(line))
