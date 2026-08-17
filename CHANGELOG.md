# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
