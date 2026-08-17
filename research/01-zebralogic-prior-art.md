# Prior-art scan

This is the positioning document. For each related benchmark or line of work it
records the citation and the specific, checkable way LatticeBench differs. The
goal is that a reader can verify each difference against the cited source, not
take our word for it.

## Reasoning benchmarks we position against

**PutnamBench** (arXiv:2407.11214, NeurIPS 2024 Datasets & Benchmarks). This is
the template we emulate: multi-representation formalization (Lean, Isabelle,
Coq), factored grading that separates the final answer from its justification,
heterogeneous baselines, dramatic unsaturation, and a public leaderboard. The
checkable difference is that PutnamBench is a fixed set of 657 hand-formalized
competition problems. It cannot be regenerated from a seed, so as solutions and
formalizations circulate the set leaks, and its size is bounded by human
formalization effort. LatticeBench keeps the multi-representation and factored-
grading ideas but mints instances procedurally so the pool is unbounded and
re-derivable.

**MiniF2F** (arXiv:2109.00110). A cross-system formal-math benchmark of 488
Olympiad-level problems, used to measure autoformalization and proving across
Lean, Metamath, and Isabelle. We cite it as the cautionary case: a static
benchmark that reported single-digit or low pass rates at release now sits near
80 percent on its stronger splits, so the headroom that made it useful has been
consumed. The checkable difference is the same as with PutnamBench, and the
motivation is sharper: LatticeBench is designed so that saturation of any single
batch is irrelevant, because a fresh batch is one seed away.

**ZebraLogic** (arXiv:2502.01100, ICML 2025). This is the closest competitor and
the one to read carefully. It is procedural and difficulty-graded over logic-grid
CSPs, and it demonstrated a "curse of complexity" where LLM accuracy collapses as
grid size grows. There are four checkable differences. First, ZebraLogic ships a
fixed set of generated instances, so despite being procedurally *produced* it is
distributed as a static pool that can leak like any other. Second, it uses Z3
only as an oracle to label difficulty, not as an evaluated model class. Third, it
has no energy formulation. Fourth, it does not evaluate energy-based models.
LatticeBench distributes seed manifests instead of instances, runs the solver as
a reported baseline, compiles every puzzle to an energy function, and evaluates
EBMs as a first-class class.

**GridPuzzle** (arXiv:2407.14790, EMNLP 2024). A dataset of 274 logic-grid
puzzles scraped from Puzzle Baron with an emphasis on evaluating and diagnosing
multi-step reasoning chains. The checkable difference is provenance and control:
it is a fixed, human-authored, web-scraped set with no contamination protocol and
no regeneration path, which is exactly the leakage surface LatticeBench removes.

**PuzzleBench** (arXiv:2402.02611). A benchmark of hard combinatorial puzzles
aimed at LLM-plus-solver pipelines, where the model writes a formulation and a
solver runs it. We share the belief that solvers belong in the loop, but
PuzzleBench is a fixed problem set and uses the solver as an execution backend
for model-written programs rather than reporting a native solver baseline on the
same instances. LatticeBench reports the solver as its own model class on the
identical instance.

**BIG-Bench logic_grid_puzzle.** The logic-grid task inside BIG-Bench, widely
used as an off-the-shelf reasoning probe. It is a small fixed task with no
difficulty dial and no contamination control, and it is old enough that it is
plausibly in many training corpora. We cite it as the low-control baseline that
motivated procedural generation in the first place.

**ProofWriter** (arXiv:2012.13048) and **PrOntoQA** (arXiv:2210.01240). Two
synthetic natural-language deductive-reasoning suites. They are procedural and
controllable, which we admire, but they target multi-hop entailment over
generated rule sets rather than constraint satisfaction with a unique
assignment, and neither ships a solver baseline or an energy view. We cite them
as evidence that procedural generation is an accepted methodology, and position
LatticeBench as the CSP-and-energy point in that design space.

**PuzzleClone** (arXiv:2508.15180). A recent framework for scaling puzzle
benchmarks by cloning and mutating seed problems under formal specifications. It
is close in spirit to our procedural stance. The checkable difference is that
PuzzleClone grows a distributable dataset by mutation, whereas LatticeBench
distributes the generator and a seed rather than the resulting instances, and
adds the energy representation and EBM evaluation on top.

**CombiBench** (arXiv:2505.03171). A benchmark of combinatorics problems
formalized in Lean. We cite it as adjacent evidence that combinatorial reasoning
is under-measured and that formal verification of answers is valued; it is fixed
and formal-math-flavored rather than procedural logic-grid, so it does not
overlap with our four positions.

## Generation-pipeline lineage

The generation pipeline in `research/00-design.md` follows the ZebraLogic-style
recipe: sample a ground-truth assignment, enumerate the clues true of it, add
clues until a solver confirms uniqueness, then greedily minimize. We adopt this
lineage on purpose so that our difficulty behavior is comparable to ZebraLogic's
reported curve, and our contribution is what we wrap around it (manifests,
hashing, forward-dating, the energy compiler, and the solver-and-EBM adapters)
rather than a novel puzzle-generation trick.

## CSP-to-energy-landscape references

The energy formulation rests on the standard mapping from constraint
satisfaction to a low-energy configuration search. We cite the CSP-as-energy-
landscape framing (arXiv:2501.00227) and the self-supervised CSP-solver line
(arXiv:2502.15794) as the basis for compiling one-hot and all-different structure
plus per-clue penalties into a Potts/QUBO energy whose global minimum is the
unique solution. On the EBM-reasoning side we position against Energy-Based
Transformers (arXiv:2507.02092), which cast prediction itself as energy
minimization; against Logical Intelligence's Kona EBRM and Aleph work, which we
cite carefully as a non-peer-reviewed blog source; and against Enso
(github.com/MVPandey/Enso) as an open EBM-reasoning implementation. LatticeBench
does not propose a new EBM; it provides the substrate on which these models can
be measured against LLMs and solvers on identical instances.
