"""Post-hoc analysis of evaluation results (numpy only; no torch/matplotlib).

These helpers turn the harness's per-puzzle scores into the aggregates the paper
reports: exact-match accuracy with a Wilson confidence interval, accuracy sliced
by grid size, and rank correlations between the solver's difficulty signal, the
language model's errors, and the energy-based model's final energy.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

from latticebench.dataset import PuzzleRecord
from latticebench.harness.base import EvalResult


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Returns ``(lo, hi)`` clamped to ``[0, 1]``. With no trials the interval is
    ``(0.0, 0.0)``. The default ``z = 1.96`` gives a ~95% interval.
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (lo, hi)


def join_scores(result: EvalResult, records: Iterable[PuzzleRecord]) -> list[dict]:
    """Join per-puzzle scores to their source records by ``id``.

    Each returned row carries the graded metrics alongside the instance's
    structural descriptors, so downstream analyses can correlate accuracy and
    energy against size and solver difficulty. Records absent from ``result``
    are skipped; ids are assumed unique.
    """
    by_id = {s.id: s for s in result.per_puzzle}
    rows: list[dict] = []
    for rec in records:
        score = by_id.get(rec.id)
        if score is None:
            continue
        rows.append(
            {
                "id": rec.id,
                "exact": int(score.exact),
                "cell_acc": score.cell_acc,
                "energy": score.energy,
                "solve_time": score.solve_time,
                "tier": score.tier,
                "n": rec.n,
                "m": rec.m,
                "z3_conflicts": int(rec.difficulty.get("z3_conflicts", 0)),
            }
        )
    return rows


def accuracy_with_ci(result: EvalResult) -> dict:
    """Exact-match accuracy with a Wilson 95% interval over the per-puzzle flags."""
    n = len(result.per_puzzle)
    successes = sum(1 for s in result.per_puzzle if s.exact)
    p = successes / n if n else 0.0
    lo, hi = wilson_interval(successes, n)
    return {"exact": p, "lo": lo, "hi": hi, "n": n}


def size_accuracy(model_results: Sequence[tuple[int, EvalResult]]) -> list[dict]:
    """Exact-match accuracy with a Wilson interval per grid size ``n``.

    Takes ``(n, EvalResult)`` pairs (one result per size for a single model) and
    returns one row ``{"n", "exact", "lo", "hi"}`` per pair, sorted by ``n``.
    """
    rows: list[dict] = []
    for n, result in model_results:
        acc = accuracy_with_ci(result)
        rows.append({"n": n, "exact": acc["exact"], "lo": acc["lo"], "hi": acc["hi"]})
    return sorted(rows, key=lambda r: r["n"])


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Assign ranks to ``a``, averaging ranks over ties (like scipy's default)."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # Average ranks within each group of equal values.
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0  # mean of ranks (1-based) in [i, j]
            ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation: Pearson correlation of the average ranks.

    Returns ``0.0`` for degenerate input (fewer than two points, or either side
    constant so its ranks have no variance).
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.size < 2 or y.size < 2 or x.size != y.size:
        return 0.0
    rx = _rankdata(x)
    ry = _rankdata(y)
    if rx.std() == 0.0 or ry.std() == 0.0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def difficulty_agreement(joined_by_model: dict[str, list[dict]]) -> dict[str, float]:
    """Rank-correlate the difficulty signals across models on shared instances.

    Given ``{model_name: joined_rows}`` (rows from :func:`join_scores`), pick the
    ``llm:``-prefixed model and, if present, the ``ebm:``-prefixed model, then
    compute Spearman correlations over the instance ids they share:

    - ``conflict_vs_llm_error``: solver ``z3_conflicts`` vs the LLM's error
      indicator ``1 - exact``. Positive means harder instances (more solver
      conflicts) trip up the language model more often.
    - ``conflict_vs_ebm_energy``: solver ``z3_conflicts`` vs the EBM's final
      ``energy`` (only if an ``ebm:`` model is present).
    - ``llm_error_vs_ebm_energy``: LLM error vs EBM energy on shared instances.

    Correlations are computed only over ids present for both relevant models.
    Missing models simply drop their entries from the result.
    """

    def _pick(prefix: str) -> str | None:
        for name in joined_by_model:
            if name.startswith(prefix):
                return name
        return None

    out: dict[str, float] = {}
    llm = _pick("llm:")
    ebm = _pick("ebm:")

    llm_rows = {r["id"]: r for r in joined_by_model[llm]} if llm else {}
    ebm_rows = {r["id"]: r for r in joined_by_model[ebm]} if ebm else {}

    if llm:
        ids = sorted(llm_rows)
        conflicts = [llm_rows[i]["z3_conflicts"] for i in ids]
        llm_err = [1 - llm_rows[i]["exact"] for i in ids]
        out["conflict_vs_llm_error"] = spearman(conflicts, llm_err)

    if ebm:
        ids = sorted(ebm_rows)
        conflicts = [ebm_rows[i]["z3_conflicts"] for i in ids]
        energy = [ebm_rows[i]["energy"] for i in ids]
        out["conflict_vs_ebm_energy"] = spearman(conflicts, energy)

    if llm and ebm:
        shared = sorted(set(llm_rows) & set(ebm_rows))
        llm_err = [1 - llm_rows[i]["exact"] for i in shared]
        energy = [ebm_rows[i]["energy"] for i in shared]
        out["llm_error_vs_ebm_energy"] = spearman(llm_err, energy)

    return out
