# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- A recurrent relational graph network (`gnn`) that learns to solve puzzles from
  the constraint graph, with training scripts and a saved checkpoint.
- An energy-based graph network and contrastive-divergence training built on
  ebmkit (reported as a negative result: it fits local conditionals but does not
  solve from scratch at trial scale).
- A local open language-model baseline via mlx-lm (`llm:local:<model>`), behind
  an optional `[llm-local]` extra so Linux CI is unaffected.
- An analysis module (score/record joins, Wilson intervals, rank correlations)
  and three figures: accuracy vs grid size, conflict vs accuracy, and
  energy-vs-correctness calibration.
- An experiment runner (`experiments/run.py`) and a `make reproduce` target that
  regenerate the manifest-backed splits, results table, and figures.

## [0.1.0]

Initial release.

### Added

- Puzzle schema (`Domain`, `Attribute`, `Solution`, `Puzzle`) and a clue
  inventory covering positional, relational, and negation constraints.
- Procedural generator with certified unique solutions and minimal clue sets,
  reproducible from a single seed.
- CP/SMT uniqueness oracle and solver baseline (CPMpy, OR-Tools CP-SAT, Z3).
- Energy formulation compiling each puzzle to a Potts/QUBO energy whose unique
  zero-energy configuration is the solution.
- Difficulty grading from grid size, clue count, and solver conflict counts.
- JSONL dataset format and a seed-manifest contamination-free protocol.
- Evaluation harness with solver, language-model, and energy-based-model
  adapters, plus exact-match, cell-accuracy, and final-energy metrics.
- Command-line interface: `generate`, `evaluate`, `leaderboard`, `verify`.
