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

## The chart I expect to matter

The headline result is a single plot: accuracy against grid size, one line per
model class. I expect the language-model line to fall steadily as the grid grows
from small puzzles to large ones, tracking the collapse ZebraLogic documented,
while the constraint solvers stay flat near 100 percent across the whole range. On
the same axes and the same instances, that gap is the story: the puzzles are not
intrinsically hard, since a solver dispatches them, so the falling line is
measuring something specific about how language models scale on constraint
satisfaction rather than something about the puzzles themselves.

## The honest part

I will say the uncomfortable thing directly, because pretending otherwise would
make the benchmark worse. Energy-based models may well underperform language models
on LatticeBench. That is not a defect in the benchmark; it is a measurement, and
right now it is one of the few clean measurements of where energy-based reasoning
actually stands against language models and solvers on identical problems. If the
EBM line sits below the LLM line, that is a result worth publishing, and the graded
energy metric will at least show whether the energy models are landing near the
solution or nowhere close. I would rather report a disappointing number honestly
than tune the benchmark until my preferred method wins.

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
