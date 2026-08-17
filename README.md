# LatticeBench

[![CI](https://github.com/davidkhjo/latticebench/actions/workflows/ci.yml/badge.svg)](https://github.com/davidkhjo/latticebench/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/latticebench.svg)](https://pypi.org/project/latticebench/)
[![Python](https://img.shields.io/pypi/pyversions/latticebench.svg)](https://pypi.org/project/latticebench/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A logic-grid reasoning benchmark that generates its own test set on demand, so
it never leaks into training data, and that scores energy-based models, language
models, and constraint solvers on the same puzzles.

Every instance is an Einstein-style ("zebra") puzzle: a set of houses, a few
attributes, and a handful of natural-language clues with exactly one consistent
assignment. Puzzles are minted from a seed, checked for a unique solution by a
CP/SMT solver, and reduced to a minimal clue set. Because generation is a pure
function of the seed, the benchmark is effectively infinite and ships a seed
manifest rather than a fixed corpus of answers.

Each puzzle also compiles to an energy function over binary assignment variables
whose only zero-energy configuration is the solution. That gives energy-based
models a native target and lets any model's output be scored on a graded energy
scale, not just right-or-wrong.

## Why another logic-grid benchmark

| Benchmark | Procedural | Contamination-free | EBMs as a model class | Solvers as baselines | Graded energy metric |
|---|:---:|:---:|:---:|:---:|:---:|
| PutnamBench | no | no (fixed) | no | oracle only | no |
| MiniF2F | no | no | no | no | no |
| ZebraLogic | yes | no (ships instances) | no | difficulty oracle | no |
| GridPuzzle | no | no | no | no | no |
| **LatticeBench** | **yes** | **yes** | **yes** | **yes** | **yes** |

Constraint solvers clear these puzzles at every size. Language models do well on
small grids and fall off sharply as the grid grows. LatticeBench puts both, plus
energy-based models, on one leaderboard.

## Install

```bash
pip install latticebench            # generator, solvers, energy encoding
pip install "latticebench[llm]"     # + language-model adapters
pip install "latticebench[ebm]"     # + energy-based-model adapter (via ebmkit)
```

## Quickstart

```python
import latticebench as lb

puzzle = lb.generate_puzzle(n=4, m=4, seed=20261231)
print(lb.render_prompt(puzzle))

solver = lb.SolverModel()
pred = solver.predict(puzzle)
assert pred.assignment == {a: dict(v) for a, v in puzzle.solution.pos.items()}
```

Generate a held-out split, run a baseline, and print the leaderboard:

```bash
latticebench generate --n 4 --m 4 --count 100 --seed 20261231 \
    --created 2026-12-31 --split held-out-2026Q4 --out data/puzzles.jsonl
latticebench evaluate --model solver --puzzles data/puzzles.jsonl --out results/solver.json
latticebench leaderboard --results results/*.json
```

## Contamination-free protocol

A LatticeBench split is distributed as a manifest: a master seed, the generator
version, and the git commit that produced it. Anyone regenerates the exact same
puzzles bit-for-bit; nothing but the generator and the seed is public. Two checks
guard against leakage: instances carry a creation date so they can be held out
after a model's training cutoff, and every instance hashes back to its seed, so a
result cannot be gamed by editing the puzzle. See `research/02-contamination-protocol.md`.

## Citation

```bibtex
@misc{latticebench,
  title  = {LatticeBench: A Contamination-Free Logic-Grid Benchmark for Energy-Based Models, Language Models, and Constraint Solvers},
  author = {David},
  year   = {2026},
  url    = {https://github.com/davidkhjo/latticebench}
}
```

## License

MIT
