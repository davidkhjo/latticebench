"""Learned models for LatticeBench. Graph construction is always available; the
torch-based energy network and training loop require the ``[ebm]`` extra."""

from latticebench.models.graph import (
    CLUE_TYPE_IDS,
    N_EDGE_TYPES,
    BatchedGraph,
    PuzzleGraph,
    batch_graphs,
    build_graph,
)

__all__ = [
    "CLUE_TYPE_IDS",
    "N_EDGE_TYPES",
    "BatchedGraph",
    "PuzzleGraph",
    "batch_graphs",
    "build_graph",
]

try:  # torch is optional
    from latticebench.models.gnn import GNNEnergy, one_hot_init, to_tensors

    __all__ += ["GNNEnergy", "one_hot_init", "to_tensors"]
except ImportError:
    pass
