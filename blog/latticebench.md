# LatticeBench: a reasoning benchmark you can't memorize

Reasoning benchmarks rot. A suite that cleanly separates models on the day it
ships loses that power as its problems and solutions spread across the internet
and into the next training run. MiniF2F was a hard target at release; its stronger
splits now sit near 80 percent solved. PutnamBench is a beautiful benchmark, with
multiple formal representations and grading that separates the final answer from
its justification, but it is a fixed set of hand-formalized problems, so it can
leak and it can never be regenerated. The pattern is the same every time: a static
set of problems has a shelf life, and the better it is, the faster people consume
it.

I built LatticeBench to have no shelf life. It is a benchmark of logic-grid
puzzles, the "zebra" or "Einstein" kind, where you place people, drinks, and pets
in a row of houses from a handful of clues. The twist is that LatticeBench does
not ship puzzles at all. It ships the generator and a seed. From a published
manifest, anyone reconstructs the exact puzzles bit-for-bit, and the grader
re-derives them server-side, so a leaked or edited instance is worth nothing. When
a batch gets stale, you publish a new seed and you have a fresh benchmark in one
line.

## One puzzle, three ways to read it

Every LatticeBench puzzle exists in three aligned forms. Here is a small one with
three houses in a row and two attributes, nationality and drink.

The natural-language clues, which is what a language model sees:

1. The Brit lives in the first house.
2. The Swede drinks milk.
3. The Dane lives immediately to the right of the Brit.
4. Water is drunk in the house immediately to the left of the milk.

The constraint form, which is what a solver sees. Positions are house indices 1,
2, 3, and each attribute is a bijection from its values to houses:

```
pos(Brit) = 1
drink(Swede) = milk
pos(Dane) = pos(Brit) + 1
pos(water) = pos(milk) - 1
all-different over nationalities and over drinks
```

The energy form, which is what an energy-based model sees. We use binary variables
`x[attr, val, house]`, where the variable is 1 when that attribute takes that value
in that house. The energy is

```
E(x) = lambda * structural(x) + sum over clues of penalty_c(x)
```

The structural term is zero only when each attribute takes exactly one value per
house and each value lands in exactly one house. Each clue contributes a
nonnegative penalty that is zero exactly when the clue holds. Every term is at most
quadratic, so this is a Potts/QUBO energy, and it has one global property that
makes it useful: `E(x) = 0` if and only if `x` is the puzzle's unique solution, and
`E(x) > 0` everywhere else. All three forms come from the same source object, so a
language model, a solver, and an energy-based model can be scored on the identical
instance.

The solution here is house 1 Brit drinking tea, house 2 Dane drinking water, house
3 Swede drinking milk. Clue 3 puts the Dane in house 2, which forces the Swede into
house 3; clue 2 gives that house milk; clue 4 puts water in house 2 and tea in
house 1. Four clues, one answer, and the generator has already checked with a
solver that no second answer exists and that no clue can be dropped.

## The four-way difference

There are good procedural logic-grid benchmarks already. ZebraLogic is the closest,
and it showed something important: language-model accuracy on these puzzles
collapses as the grid grows. What LatticeBench adds is four things at once, and no
existing benchmark has all four.

| Benchmark    | Procedural | Contamination-free | EBMs as a class | Solvers as baselines | Graded energy |
|--------------|:----------:|:------------------:|:---------------:|:--------------------:|:-------------:|
| PutnamBench  | no         | no                 | no              | no                   | no            |
| MiniF2F      | no         | no                 | no              | no                   | no            |
| ZebraLogic   | yes        | no                 | no              | no                   | no            |
| GridPuzzle   | no         | no                 | no              | no                   | no            |
| LatticeBench | yes        | yes                | yes             | yes                  | yes           |

The contamination column is the one I care most about. ZebraLogic is generated,
but it is distributed as a fixed pool of instances, so it can leak like any static
set. LatticeBench distributes a manifest (master seed, generator version, git
commit), regenerates deterministically, hashes each instance back to its seed so
substitution or editing is caught, and dates each manifest so a held-out split can
be forward-dated past a model's training cutoff. ZebraLogic also uses Z3 only as a
difficulty oracle; LatticeBench runs the solver as a reported baseline and adds the
energy view so energy-based models are a first-class evaluated class rather than an
afterthought.

## The chart that matters

The headline result is a single plot: accuracy against instance difficulty, one
line per model class (`paper/figures/accuracy_vs_difficulty.png`). The constraint
solvers stay flat at 100 percent across the whole range, from three houses to six.
On the same axes and the same instances, everything else falls away from that
ceiling. The puzzles are not intrinsically hard, since a solver dispatches every one
of them, so the gaps below the ceiling measure how each method scales rather than
anything about the puzzles.

In a first trial run those gaps are large. A recurrent graph network I trained on
the full range of sizes solves every three-by-two puzzle and almost every
three-by-three, holds around two thirds on the four-house grids, and then falls to
near zero on five and six houses even though it trained on them: it learns the small
grids but cannot propagate enough constraint information to assemble a six-house
solution. An energy-based baseline that is handed the exact energy and minimizes it
by annealing is the strongest non-solver, near 100 percent until the largest grid,
where it drops to 0.42. A small open language model run locally (Qwen2.5-3B) solves
almost none of them, two percent, and a 7B model does no better on exact answers.

## The honest part

I will say the uncomfortable thing directly, because pretending otherwise would
make the benchmark worse. My own learned model is not close to the solvers, and it
fails on the largest grids even after training on them. That is a measurement, not
a defect in the benchmark, and it is one of the few clean measurements of where a
graph-based reasoning model actually stands against solvers and language models on
identical problems. I also tried a purely energy-based version of the graph model,
minimizing a learned energy at test time; it learned the local structure but could
not assemble whole solutions in the budget I gave it, and I am reporting that as a
negative result rather than quietly dropping it. The graded energy metric earns its
place here: for the graph network, low final energy lines up with correct grids and
rises smoothly as more cells are misplaced. I would rather report a disappointing
number honestly than tune the benchmark until my preferred method wins.

## Run it

If you have a model, you can put it on the leaderboard. Fetch a published seed
manifest, run your model through the CLI, and open a pull request with the results
JSON. The manifest is re-derived and the puzzles are re-graded server-side, so you
cannot game the score by editing instances; the only thing that moves your number
is your model solving more puzzles. Grid size is a dial, so you can report where
your model's accuracy starts to fall, and a forward-dated manifest lets you show a
score on puzzles that provably postdate your training cutoff.

Repository: https://github.com/davidkhjo/latticebench

How to submit: see `SUBMISSION.md` in the repository.
