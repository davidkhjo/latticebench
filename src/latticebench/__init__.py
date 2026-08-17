"""LatticeBench: a contamination-free logic-grid benchmark.

Logic-grid ("zebra") puzzles are procedurally generated with a certified unique
solution, compiled to a Potts/QUBO energy whose unique zero-energy configuration
is that solution, and solved by energy-based models, language models, and
constraint solvers through one evaluation harness.
"""

__version__ = "0.1.0"

from latticebench.clues import DEFAULT_INVENTORY
from latticebench.contamination import (
    Manifest,
    build_manifest,
    make_split,
    puzzle_hash,
    verify_regeneration,
)
from latticebench.dataset import (
    PuzzleRecord,
    from_record,
    read_jsonl,
    render_prompt,
    to_record,
    write_jsonl,
)
from latticebench.difficulty import Difficulty, grade
from latticebench.energy import EnergyGrid
from latticebench.generate import build_domain, generate_puzzle, sample_solution
from latticebench.harness import (
    EBMModel,
    EvalResult,
    LLMModel,
    Prediction,
    SolverModel,
    evaluate,
)
from latticebench.leaderboard import Leaderboard, build_leaderboard, load_results
from latticebench.oracle import count_solutions, is_unique, solve
from latticebench.schema import Assignment, Attribute, Domain, Puzzle, Solution

__all__ = [
    "DEFAULT_INVENTORY",
    "Assignment",
    "Attribute",
    "Difficulty",
    "Domain",
    "EBMModel",
    "EnergyGrid",
    "EvalResult",
    "LLMModel",
    "Leaderboard",
    "Manifest",
    "Prediction",
    "Puzzle",
    "PuzzleRecord",
    "Solution",
    "SolverModel",
    "__version__",
    "build_domain",
    "build_leaderboard",
    "build_manifest",
    "count_solutions",
    "evaluate",
    "from_record",
    "generate_puzzle",
    "grade",
    "is_unique",
    "load_results",
    "make_split",
    "puzzle_hash",
    "read_jsonl",
    "render_prompt",
    "sample_solution",
    "solve",
    "to_record",
    "verify_regeneration",
    "write_jsonl",
]
