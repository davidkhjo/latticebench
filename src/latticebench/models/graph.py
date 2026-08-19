"""Constraint graph for a puzzle.

Nodes are (attribute, value) items; a node's state is its candidate distribution
over houses. Edges are typed: mutual-exclusion edges tie together the values of
one attribute, and each clue becomes a typed edge (or a self-edge, for unary
clues) between the items it names. Nothing here identifies attributes or values
by name -- only the graph shape and the clue kind -- so a model over this graph
has to learn what each clue type means instead of memorising a domain.

Item indices follow ``EnergyGrid.var`` exactly (``attr_idx * n + val_idx``), so a
state tensor of shape ``(num_items, n)`` flattens straight into the ``D = m*n*n``
layout that ``EnergyGrid.encode``/``decode`` use.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from latticebench.clues import DEFAULT_INVENTORY
from latticebench.clues.base import BinaryClue, UnaryClue
from latticebench.schema import Puzzle

# clue kinds get ids 0..9 in inventory order; mutex edges get the next id.
CLUE_TYPE_IDS: dict[str, int] = {cls.kind: i for i, cls in enumerate(DEFAULT_INVENTORY)}
CLUE_TYPE_IDS["mutex"] = len(CLUE_TYPE_IDS)
N_EDGE_TYPES = len(CLUE_TYPE_IDS)


@dataclass(frozen=True)
class PuzzleGraph:
    n: int
    m: int
    num_items: int
    item_attr: np.ndarray  # (I,) attribute index per item, only used to build mutex edges
    edge_index: np.ndarray  # (2, E)
    edge_type: np.ndarray  # (E,)
    edge_dir: np.ndarray  # (E,) +1 forward, -1 reverse, 0 symmetric
    edge_house: np.ndarray  # (E,) referenced house for unary self-edges, else -1
    item_of: dict[tuple[str, str], int]


@dataclass(frozen=True)
class BatchedGraph:
    """Disjoint union of graphs that share the same (n, m)."""

    batch_size: int
    n: int
    m: int
    num_items: int  # per graph
    edge_index: np.ndarray  # (2, sumE) into the flattened batch_size * num_items nodes
    edge_type: np.ndarray
    edge_dir: np.ndarray
    edge_house: np.ndarray
    node_batch: np.ndarray  # (batch_size * num_items,) graph id per node
    edge_batch: np.ndarray  # (sumE,) graph id per edge


def build_graph(puzzle: Puzzle) -> PuzzleGraph:
    domain = puzzle.domain
    n, m = domain.n, domain.m
    num_items = m * n

    item_of: dict[tuple[str, str], int] = {}
    item_attr = np.empty(num_items, dtype=np.int64)
    attr_items: list[list[int]] = []
    for a_i, attr in enumerate(domain.attributes):
        row = []
        for v_i, value in enumerate(attr.values):
            idx = a_i * n + v_i
            item_of[(attr.name, value)] = idx
            item_attr[idx] = a_i
            row.append(idx)
        attr_items.append(row)

    src: list[int] = []
    dst: list[int] = []
    etype: list[int] = []
    edir: list[int] = []
    ehouse: list[int] = []

    def add(i: int, j: int, kind: str, direction: int, house: int) -> None:
        src.append(i)
        dst.append(j)
        etype.append(CLUE_TYPE_IDS[kind])
        edir.append(direction)
        ehouse.append(house)

    mutex = "mutex"
    for row in attr_items:
        for a in range(len(row)):
            for b in range(a + 1, len(row)):
                add(row[a], row[b], mutex, 0, -1)
                add(row[b], row[a], mutex, 0, -1)

    for clue in puzzle.clues:
        if isinstance(clue, UnaryClue):
            i = item_of[(clue.attr, clue.value)]
            add(i, i, clue.kind, 0, clue.house)
        elif isinstance(clue, BinaryClue):
            i = item_of[(clue.attr1, clue.value1)]
            j = item_of[(clue.attr2, clue.value2)]
            add(i, j, clue.kind, 1, -1)
            add(j, i, clue.kind, -1, -1)

    return PuzzleGraph(
        n=n,
        m=m,
        num_items=num_items,
        item_attr=item_attr,
        edge_index=np.array([src, dst], dtype=np.int64) if src else np.zeros((2, 0), np.int64),
        edge_type=np.array(etype, dtype=np.int64),
        edge_dir=np.array(edir, dtype=np.int64),
        edge_house=np.array(ehouse, dtype=np.int64),
        item_of=item_of,
    )


def batch_graphs(graphs: Sequence[PuzzleGraph]) -> BatchedGraph:
    if not graphs:
        raise ValueError("cannot batch an empty sequence")
    n, m, I = graphs[0].n, graphs[0].m, graphs[0].num_items
    if any((g.n, g.m) != (n, m) for g in graphs):
        raise ValueError("batch_graphs requires all graphs to share (n, m)")

    edge_index_parts = []
    node_batch = []
    edge_batch = []
    for b, g in enumerate(graphs):
        edge_index_parts.append(g.edge_index + b * I)
        node_batch.append(np.full(I, b, dtype=np.int64))
        edge_batch.append(np.full(g.edge_index.shape[1], b, dtype=np.int64))

    return BatchedGraph(
        batch_size=len(graphs),
        n=n,
        m=m,
        num_items=I,
        edge_index=np.concatenate(edge_index_parts, axis=1),
        edge_type=np.concatenate([g.edge_type for g in graphs]),
        edge_dir=np.concatenate([g.edge_dir for g in graphs]),
        edge_house=np.concatenate([g.edge_house for g in graphs]),
        node_batch=np.concatenate(node_batch),
        edge_batch=np.concatenate(edge_batch),
    )
