"""Generate one puzzle, show it, solve it, and confirm the solution has zero energy."""

import latticebench as lb

puzzle = lb.generate_puzzle(n=4, m=4, seed=20261231)

print(lb.render_prompt(puzzle))
print()

solver = lb.SolverModel()
pred = solver.predict(puzzle)
print("solver answer:", pred.assignment)

grid = lb.EnergyGrid(puzzle.domain, puzzle.clues)
print("energy of the true solution:", grid.energy(grid.encode(puzzle.solution)))
