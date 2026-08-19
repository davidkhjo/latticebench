"""Fast tests for the graph/GNN/solver model stack and its harness adapters.

The graph module is torch-free; every other test guards on torch via
``pytest.importorskip``. Everything is kept tiny (n, m <= 4, per_size <= 6,
<= 60 training steps) so the whole file runs in a few seconds on CPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from latticebench.energy import EnergyGrid
from latticebench.generate import generate_puzzle
from latticebench.models.graph import (
    CLUE_TYPE_IDS,
    batch_graphs,
    build_graph,
)


def _sd_equal(a: dict, b: dict) -> None:
    import torch

    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), f"tensor mismatch at {k}"


# ---------------------------------------------------------------------------
# 1. graph ordering (torch-free)
# ---------------------------------------------------------------------------
def test_graph_ordering_matches_energy_var() -> None:
    puzzle = generate_puzzle(4, 3, seed=2)
    grid = EnergyGrid(puzzle.domain, puzzle.clues)
    graph = build_graph(puzzle)

    assert graph.num_items == puzzle.domain.m * puzzle.domain.n
    for (attr, value), idx in graph.item_of.items():
        assert idx * graph.n == grid.var(attr, value, 0)

    assert "mutex" in CLUE_TYPE_IDS
    assert CLUE_TYPE_IDS["mutex"] in set(graph.edge_type.tolist())

    batched = batch_graphs([graph, graph])
    assert int(batched.edge_index.max()) < 2 * graph.num_items


# ---------------------------------------------------------------------------
# 2. GNNEnergy forward + gradient
# ---------------------------------------------------------------------------
def test_gnn_energy_forward_and_grad() -> None:
    torch = pytest.importorskip("torch")
    from latticebench.models.gnn import GNNEnergy, to_tensors

    puzzle = generate_puzzle(3, 2, seed=3)
    grid = EnergyGrid(puzzle.domain, puzzle.clues)
    graph = build_graph(puzzle)
    n, items = graph.n, graph.num_items

    gt = to_tensors(batch_graphs([graph, graph, graph]))
    config = torch.tensor(grid.encode(puzzle.solution), dtype=torch.float32).reshape(items, n)
    x = torch.stack([config, config, config])  # (3, I, n)

    model = GNNEnergy(hidden=16, rounds=2)
    energy = model(gt, x)
    assert energy.shape == (3,)
    assert torch.isfinite(energy).all()

    x_req = x.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(model(gt, x_req).sum(), x_req)
    assert torch.isfinite(grad).all()


# ---------------------------------------------------------------------------
# 3. RecurrentSolver forward
# ---------------------------------------------------------------------------
def test_recurrent_solver_forward() -> None:
    torch = pytest.importorskip("torch")
    from latticebench.models.gnn import to_tensors
    from latticebench.models.solver import RecurrentSolver

    puzzle = generate_puzzle(3, 2, seed=4)
    graph = build_graph(puzzle)
    n, items = graph.n, graph.num_items
    gt = to_tensors(batch_graphs([graph, graph]))

    model = RecurrentSolver(hidden=16, steps=4)
    out = model(gt)
    assert out.shape == (2, items, n)
    assert torch.isfinite(out).all()

    outs = model(gt, return_all=True)
    assert isinstance(outs, list)
    assert outs[-1].shape == out.shape
    assert torch.isfinite(outs[-1]).all()


# ---------------------------------------------------------------------------
# 4. train_solver produces a learning signal
# ---------------------------------------------------------------------------
def test_train_solver_reduces_loss() -> None:
    pytest.importorskip("torch")
    from latticebench.models.solver import RecurrentSolver
    from latticebench.models.train import build_dataset, train_solver

    data = build_dataset([(3, 2), (3, 3)], per_size=6, seed=0)
    model = RecurrentSolver(hidden=32, steps=8)
    history = train_solver(model, data, steps=60, batch=6, seed=0, log_every=5)

    assert len(history) >= 2
    assert history[-1]["loss"] < history[0]["loss"]


# ---------------------------------------------------------------------------
# 5. save / load solver round-trip
# ---------------------------------------------------------------------------
def test_save_load_solver_roundtrip(tmp_path) -> None:
    pytest.importorskip("torch")
    from latticebench.models.solver import RecurrentSolver
    from latticebench.models.train import load_solver, save_solver

    model = RecurrentSolver(hidden=32, steps=8)
    path = str(tmp_path / "solver.pt")
    config = {"hidden": 32, "steps": 8, "type_dim": 32}
    save_solver(model, path, config)

    loaded = load_solver(path)
    _sd_equal(model.state_dict(), loaded.state_dict())


# ---------------------------------------------------------------------------
# 6. GNNSolverModel.predict end-to-end
# ---------------------------------------------------------------------------
def test_gnn_solver_model_predict(tmp_path) -> None:
    pytest.importorskip("torch")
    from latticebench.harness.gnn import GNNSolverModel
    from latticebench.models.solver import RecurrentSolver
    from latticebench.models.train import save_solver

    model = RecurrentSolver(hidden=32, steps=8)
    path = str(tmp_path / "solver.pt")
    save_solver(model, path, {"hidden": 32, "steps": 8, "type_dim": 32})

    adapter = GNNSolverModel(path)
    puzzle = generate_puzzle(3, 2, seed=9)
    pred = adapter.predict(puzzle)

    assert pred.assignment is not None
    for attr in puzzle.domain.attributes:
        assert attr.name in pred.assignment
        for value in attr.values:
            house = pred.assignment[attr.name][value]
            assert isinstance(house, int)
            assert 0 <= house < puzzle.domain.n

    assert isinstance(pred.energy, float)
    assert np.isfinite(pred.energy)
    assert pred.energy >= 0.0
    assert isinstance(pred.solve_time, float)


# ---------------------------------------------------------------------------
# 7. GNNEnergy checkpoint round-trip
# ---------------------------------------------------------------------------
def test_energy_checkpoint_roundtrip(tmp_path) -> None:
    pytest.importorskip("torch")
    from latticebench.models.gnn import GNNEnergy
    from latticebench.models.train import load_checkpoint, save_checkpoint

    model = GNNEnergy(hidden=16, rounds=2)
    path = str(tmp_path / "gnn.pt")
    save_checkpoint(model, path, {"hidden": 16, "rounds": 2, "type_dim": 32})

    loaded = load_checkpoint(path)
    _sd_equal(model.state_dict(), loaded.state_dict())


# ---------------------------------------------------------------------------
# 8. lazy adapters / cli / pure helpers
# ---------------------------------------------------------------------------
def test_lazy_adapters_and_pure_helpers() -> None:
    from latticebench.cli import _build_model
    from latticebench.harness.gnn import GNNSolverModel
    from latticebench.harness.local_llm import DEFAULT_MODEL, short_name, solve_template

    # construction must not touch (or need) the checkpoint file
    assert GNNSolverModel("nonexistent.pt").name == "gnn"
    assert _build_model("gnn").name == "gnn"

    assert isinstance(DEFAULT_MODEL, str)
    assert short_name("a/b/c") == "c"

    puzzle = generate_puzzle(3, 2, seed=5)
    prompt = solve_template(puzzle)
    assert "JSON" in prompt
    assert puzzle.clues[0].to_text(puzzle.domain) in prompt
