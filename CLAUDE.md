# latticebench

Python library and CLI (`import latticebench`, PyPI name `latticebench`) that
generates contamination-free logic-grid puzzles and benchmarks energy-based
models, language models, and constraint solvers on them. src/ layout, hatchling
build.

## Commands

- `uv run pytest` — full test suite (CPU-only; RNG seeded in `tests/conftest.py`)
- `uv run ruff check .` and `uv run ruff format .`
- `uv run mypy`

## Conventions

- Energy sign convention `p(x) ∝ exp(-E(x))`: the unique solution is the unique
  configuration with `E(x) = 0`; every clue penalty is `>= 0` and `0` iff
  satisfied. Never flip this.
- Every clue penalty is at most quadratic in the one-hot assignment variables,
  so a whole puzzle is a Potts/QUBO energy. New clue types must keep this.
- Generation is a pure function of `(n, m, seed, inventory, value_pool,
  generator_version)`. No wall-clock, no global RNG — all randomness flows
  through a seeded `numpy` Generator. This is what makes the benchmark
  regenerable and contamination-checkable, so do not break it.
- A clue is self-describing along three axes: `to_text` (natural language),
  `to_constraints` (CPMpy), and `penalty` (energy). All three must agree.
- The uniqueness oracle is the source of truth for "solvable": a generated
  puzzle always has exactly one solution and a minimal (irreducible) clue set.
- Adapters (solver / LLM / EBM) all satisfy the `Model` protocol so one
  `evaluate()` and one leaderboard cover every model class.

## Writing and commits

- Commit messages are terse, lowercase, one line, no trailing period, no
  co-author trailers.
- Prose in the paper, README, and blog avoids AI-writing tells: plain academic
  voice, concrete numbers, no filler. See `paper/STYLE.md`.

## Research context

`research/` holds the design record and the prior-art / contamination-protocol
notes the benchmark is based on.
