# Submitting a model to the LatticeBench leaderboard

A submission is a results JSON produced by running your model against a published
seed manifest through the LatticeBench CLI, plus a pull request that adds it to the
leaderboard. You never submit puzzles or answers directly; you submit your model's
predictions against a manifest, and the grader re-derives and re-grades everything
on its side.

## Why this can't be gamed

The leaderboard does not trust the instances in your submission. When your pull
request is graded, the manifest you ran against is re-derived from its master seed,
generator version, and git commit, the puzzles are regenerated bit-for-bit, and
your predictions are re-graded against those regenerated puzzles. Editing an
instance to make your model look correct changes its hash and is rejected;
swapping in easier puzzles fails the same check. The only way to move your score
is to solve more of the puzzles the manifest actually specifies.

## Steps

1. **Install the package and fetch a manifest.** Pick a published manifest from
   the leaderboard directory. A manifest names a master seed, a generator version,
   and a git commit; that is all the grader needs to reconstruct the batch.

   ```
   pip install latticebench
   latticebench fetch-manifest <manifest-id> --out manifest.json
   ```

2. **Regenerate the batch locally (optional but recommended).** This lets you
   inspect the puzzles you are about to run against and confirms your environment
   reproduces the released batch bit-for-bit.

   ```
   latticebench regenerate manifest.json --out batch/
   ```

3. **Run your model.** Point the CLI at your adapter. Adapters exist for the three
   model classes (solver, LLM, EBM); implement the adapter interface for a new
   model. The run emits a results JSON with per-instance predictions and the
   metrics (exact-match, cell-accuracy, mean energy, median time).

   ```
   latticebench run manifest.json --adapter <your-adapter> --out results.json
   ```

4. **Open a pull request.** Add `results.json` under the leaderboard directory
   with your model name and class, and open a PR against
   https://github.com/davidkhjo/latticebench. Continuous integration re-derives the
   manifest, re-grades your predictions server-side, and regenerates the
   leaderboard table. If the re-graded numbers match your submission, the PR is
   eligible to merge.

## What to include

- The manifest id you ran against, and its creation date (so readers can compare
  it against your model's training cutoff).
- Your model name and class (solver, LLM, or EBM).
- The results JSON emitted by the CLI, unedited.

Do not hand-edit the results JSON. The re-grading step will overwrite any metric
you change, and a mismatch between your claimed numbers and the re-graded numbers
is treated as a failed submission.
