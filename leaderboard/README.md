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

Reference numbers from the initial trial run (`experiments/run.py`), aggregated
over grid configurations 3x2 to 6x4, twenty-four instances each (ten for the
language model). Open for submissions.

| Model | Class | Exact-match | Cell-acc | Mean energy | Median time (s) |
|-------|-------|-------------|----------|-------------|-----------------|
| CP-SAT (OR-Tools) | solver | 1.000 | 1.000 | 0.00 | 0.002 |
| Z3 | solver | 1.000 | 1.000 | 0.00 | 0.003 |
| Energy annealing | ebm | 0.889 | 0.922 | 0.11 | 0.019 |
| Graph network (ours) | gnn | 0.562 | 0.774 | 175.8 | 0.004 |
| Qwen2.5-3B-Instruct | llm | 0.020 | 0.171 | 94.9 | 6.87 |
