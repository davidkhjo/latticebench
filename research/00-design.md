# LatticeBench design record

This is the maintainer-facing description of what LatticeBench is, why it is
built the way it is, and how the pieces fit together. It is deliberately plain.
If you are extending the generator, the oracle, or an evaluation adapter, read
this first.

## What LatticeBench is

LatticeBench is a procedurally-generated, contamination-free benchmark of
logic-grid puzzles, the family also called "zebra" or "Einstein" puzzles. Each
instance places `n` houses in a row and assigns `m` attributes, where every
attribute has `n` distinct values and each value occupies exactly one house. A
set of natural-language clues constrains the arrangement, and the clue set is
constructed so that exactly one assignment is consistent with all of them.

Every instance is minted from a seed rather than authored by hand. A puzzle is
generated, checked for a unique solution by a constraint solver, and then reduced
to a minimal clue set. The same instance compiles to three representations: the
natural-language clues a language model reads, the constraint program a solver
reads, and an energy function over binary assignment variables that an
energy-based model minimizes. Because all three come from one source object, a
result on one representation is directly comparable to a result on another.

## The four contributions

LatticeBench is not the first procedural logic-grid benchmark. Its claim is that
it is the first to occupy all four of the following positions at once.

1. **Procedural, on-demand generation.** Instances are minted from a seed at
   evaluation time, not drawn from a shipped fixed pool. Grid size and clue
   budget are parameters, so difficulty is a dial rather than a fixed
   distribution.

2. **A rigorous contamination-free protocol.** We distribute a seed *manifest*
   (master seed, generator version, git commit), not solved puzzles. Anyone
   regenerates the exact instances bit-for-bit. Each instance carries a creation
   date so that a held-out split can be forward-dated past a model's training
   cutoff, and each instance hashes back to its seed so a submitted result cannot
   be quietly swapped for an easier one. The protocol is specified in
   `research/02-contamination-protocol.md`.

3. **An energy formulation that makes EBMs a first-class evaluated class.** Every
   puzzle compiles to an energy function whose unique zero-energy configuration
   is the solution. This gives EBMs a native way to be scored, and it gives a
   graded "final energy" metric that awards partial credit for near-solutions
   rather than only pass/fail.

4. **Classical CP/SMT solvers as real baselines.** Constraint solvers are run as
   an evaluated model class and reported in the same tables as LLMs and EBMs, not
   used only as a difficulty oracle behind the scenes. They set the "solved"
   ceiling the learned methods are measured against.

## Module architecture

The codebase is organized around a single instance object that every consumer
reads. The modules below sit under `src/` and are written concurrently with this
document; this section describes their responsibilities, not their current line
count.

**Schema.** Defines the instance object: grid size `n`, the attribute/value
tables, the solution assignment, the ordered clue list, the creation date, and
the seed provenance (master seed, index within the batch, generator version, git
commit). Serialization is canonical so that two regenerations hash identically.

**Clues.** Defines the clue vocabulary and its two faces: a structured predicate
(machine-readable, used by the oracle and the energy compiler) and a
natural-language rendering (used by the LLM adapter). Clue types cover the
standard logic-grid repertoire: equality ("the doctor drinks tea"), negation,
positional relations ("immediately left of", "somewhere left of"), adjacency,
ends-of-row, and ordinal position. Every clue type must expose both a solver
constraint and an energy penalty term, or it is not admissible.

**Oracle.** Wraps a constraint solver through CPMpy, backed by OR-Tools CP-SAT
with a Z3 fallback. The oracle answers three questions the generator needs: is
this clue set satisfiable, is the solution unique, and (for minimization) is a
given clue redundant. Uniqueness is checked by finding a solution, adding a
constraint that forbids exactly that assignment, and re-solving; a second
solution means the set is not yet unique.

**Energy.** Compiles an instance to an energy function over binary variables
`x[a, v, h]` (attribute `a` takes value `v` in house `h`). See "Energy
formulation" below. This module is the bridge between the puzzle and the EBM
adapter.

**Generator.** Runs the generation pipeline (next section) from a seed to a
finished, minimized, unique instance. It owns the RNG discipline that makes
regeneration deterministic.

**Contamination protocol.** Turns a master seed plus configuration into a
manifest, expands a manifest into a batch of instances, and verifies that a
claimed instance actually derives from its manifest. This is the code path that
makes results ungameable.

**Evaluation harness.** Loads a batch, dispatches each instance to an adapter,
collects predictions, and grades them. Three adapters implement one interface:
the solver adapter (runs the oracle as a competitor and records conflict count
and solve time), the LLM adapter (renders clues to a prompt, parses a returned
grid), and the EBM adapter (minimizes the energy function and reads off the
argmin configuration). Grading is factored: the final answer grid is scored
separately from the justification, following PutnamBench's answer-vs-proof split.

**CLI.** The entry point a submitter uses: fetch a manifest, regenerate its
batch, run a chosen adapter, and emit a results JSON. Documented in
`SUBMISSION.md`.

## Generation pipeline

The generator produces a unique, minimal puzzle from a seed in four stages. The
lineage is ZebraLogic-style: sample a solution, then find clues that hold on it,
then keep adding clues until the solution is pinned down, then trim.

1. **Sample a solution.** Seed the RNG, pick `n` and `m`, and draw a random
   assignment: for each attribute, a random permutation of its values across the
   houses. This assignment is the intended unique answer.

2. **Enumerate true clues.** Generate the set of candidate clues from the
   vocabulary that are true of the sampled solution. This is the pool the puzzle
   will draw from; every clue in it is consistent by construction.

3. **Add until unique.** Shuffle the pool and add clues one at a time, asking the
   oracle after each addition whether the accumulated set now has a unique
   solution. Stop as soon as uniqueness is reached. This yields a sufficient but
   usually redundant clue set.

4. **Greedily minimize.** Walk the accepted clues in a fixed order and try
   removing each one; keep the removal if the oracle confirms the solution is
   still unique without it, otherwise put it back. The result is a clue set from
   which no single clue can be dropped without admitting a second solution.

The order of operations in stages 3 and 4 is driven entirely by the seeded RNG,
so the finished instance is a deterministic function of the seed and the
generator version. Change either and the batch changes; that is why both are
pinned in the manifest.
