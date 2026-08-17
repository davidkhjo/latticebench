"""End-to-end smoke test of the command-line interface."""

from __future__ import annotations

import json

from latticebench.cli import build_parser, main


def test_build_parser_requires_subcommand() -> None:
    parser = build_parser()
    assert parser.prog == "latticebench"


def test_full_pipeline(tmp_path) -> None:
    puzzles = tmp_path / "p.jsonl"
    result = tmp_path / "solver.json"
    board = tmp_path / "board.md"

    rc = main(
        [
            "generate",
            "--n",
            "4",
            "--m",
            "4",
            "--count",
            "5",
            "--seed",
            "1",
            "--created",
            "2027-01-01",
            "--split",
            "t",
            "--out",
            str(puzzles),
        ]
    )
    assert rc == 0
    assert puzzles.exists()
    assert sum(1 for _ in puzzles.open()) == 5

    rc = main(
        [
            "evaluate",
            "--model",
            "solver",
            "--puzzles",
            str(puzzles),
            "--out",
            str(result),
        ]
    )
    assert rc == 0
    assert result.exists()
    data = json.loads(result.read_text())
    assert data["summary"]["exact_match"] == 1.0

    rc = main(
        [
            "leaderboard",
            "--results",
            str(result),
            "--out",
            str(board),
        ]
    )
    assert rc == 0
    assert board.exists()
    assert "solver:ortools" in board.read_text()

    rc = main(["verify", "--puzzles", str(puzzles)])
    assert rc == 0
