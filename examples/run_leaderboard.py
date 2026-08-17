"""Run the solver and the energy-based baseline on one split and print the board."""

import latticebench as lb

records = list(lb.make_split(7, n=4, m=4, count=20, created="2027-01-01", split="demo"))

results = [
    lb.evaluate(lb.SolverModel(), records),
    lb.evaluate(lb.EBMModel(), records),
]
print(lb.build_leaderboard(results).to_markdown())
