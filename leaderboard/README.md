# LatticeBench leaderboard

The leaderboard is generated from submitted results, not edited by hand. Each
submission is one results JSON produced by the CLI (see `../SUBMISSION.md`); a
build step re-grades every submission server-side and regenerates the table below.

## Format

- **One JSON per model.** A submission lives at `results/<model>.json` and holds
  the manifest id it ran against, the model name and class, per-instance
  predictions, and the aggregate metrics. Predictions are kept so the grader can
  re-verify them against the regenerated puzzles.
- **A generated markdown table.** The table below is rebuilt from the JSON files on
  every merge. Do not edit it by hand; edits are overwritten on the next build.

The columns are the four reported metrics plus the model and its class. Exact-match
is the fraction of fully correct grids, cell-accuracy is the fraction of correct
attribute-value-house cells, mean-energy is the average final energy at the model's
chosen configuration (zero for a solved instance), and median-time is the median
wall-clock latency per instance.

## Standings

| Model | Class | Exact-match | Cell-acc | Mean energy | Median time (s) |
|-------|-------|-------------|----------|-------------|-----------------|
| _no submissions yet_ | | | | | |
