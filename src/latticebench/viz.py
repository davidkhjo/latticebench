"""Optional plots (requires the ``[viz]`` extra)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from latticebench.harness.base import EvalResult

_TIER_ORDER = ("easy", "medium", "hard", "expert")


def plot_accuracy_by_tier(results: Sequence[EvalResult], path: str | None = None) -> Any:
    """Plot exact-match accuracy per difficulty tier, one line per model."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for res in results:
        tiers = [t for t in _TIER_ORDER if t in res.by_difficulty]
        ys = [res.by_difficulty[t].get("exact_match", 0.0) for t in tiers]
        ax.plot(tiers, ys, marker="o", label=res.model)
    ax.set_xlabel("difficulty tier")
    ax.set_ylabel("exact-match accuracy")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_accuracy_vs_size(data: dict[str, list[dict]], path: str | None = None) -> Any:
    """Exact-match accuracy vs grid size ``n``, one line per model.

    This is the headline plot. ``data`` maps each model name to a list of
    ``{"n", "exact", "lo", "hi"}`` rows (as produced by
    :func:`latticebench.analysis.size_accuracy`); the Wilson interval is drawn as
    an error band around each line.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for model, rows in data.items():
        rows = sorted(rows, key=lambda r: r["n"])
        xs = [r["n"] for r in rows]
        ys = [r["exact"] for r in rows]
        lo = [r["lo"] for r in rows]
        hi = [r["hi"] for r in rows]
        (line,) = ax.plot(xs, ys, marker="o", label=model)
        ax.fill_between(xs, lo, hi, alpha=0.15, color=line.get_color())
    ax.set_xlabel("grid size (houses n)")
    ax.set_ylabel("exact-match accuracy")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_accuracy_vs_difficulty(data: dict[str, list[dict]], path: str | None = None) -> Any:
    """Exact-match accuracy against instance difficulty (log search space).

    ``data`` maps each model to a list of ``{"x", "exact", "lo", "hi"}`` rows.
    Using a difficulty scalar on the x-axis keeps the ordering clean when the
    sweep varies both the number of houses and the number of attributes.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for model, rows in data.items():
        rows = sorted(rows, key=lambda r: r["x"])
        xs = [r["x"] for r in rows]
        ys = [r["exact"] for r in rows]
        (line,) = ax.plot(xs, ys, marker="o", label=model)
        ax.fill_between(
            xs, [r["lo"] for r in rows], [r["hi"] for r in rows], alpha=0.15, color=line.get_color()
        )
    ax.set_xlabel("instance difficulty (log search space)")
    ax.set_ylabel("exact-match accuracy")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_conflict_vs_accuracy(joined: list[dict], path: str | None = None, bins: int = 8) -> Any:
    """Accuracy against solver difficulty for a single model's joined rows.

    The solver's ``z3_conflicts`` count is grouped into ``bins`` quantile bins;
    each bin's mean exact-match accuracy is plotted at the bin's median conflict
    count. The raw per-instance points are overlaid as a light jittered scatter.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 4))
    conflicts = np.array([r["z3_conflicts"] for r in joined], dtype=float)
    exact = np.array([r["exact"] for r in joined], dtype=float)

    if conflicts.size:
        rng = np.random.default_rng(0)
        jitter = (rng.random(exact.size) - 0.5) * 0.05
        ax.scatter(conflicts, exact + jitter, s=12, alpha=0.2, color="gray")

        # Quantile bins over the conflict counts; unique edges guard against ties.
        edges = np.unique(np.quantile(conflicts, np.linspace(0, 1, bins + 1)))
        if edges.size >= 2:
            idx = np.clip(np.digitize(conflicts, edges[1:-1]), 0, edges.size - 2)
            centers, means = [], []
            for b in range(edges.size - 1):
                sel = idx == b
                if sel.any():
                    centers.append(float(np.median(conflicts[sel])))
                    means.append(float(exact[sel].mean()))
            ax.plot(centers, means, marker="o", color="C0", label="binned accuracy")
            ax.legend()

    ax.set_xlabel("solver conflict count")
    ax.set_ylabel("accuracy")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_energy_calibration(result: EvalResult, path: str | None = None) -> Any:
    """Final energy vs correctness for a single model.

    Scatters each puzzle's final energy (x) against its cell accuracy (y),
    coloured by exact-match. Infinite energies (unparsed predictions) carry no
    position on the energy axis and are dropped.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 4))
    finite = [s for s in result.per_puzzle if math.isfinite(s.energy)]
    if finite:
        # Clip the long right tail so a few high-energy points don't flatten the rest.
        energies = [s.energy for s in finite]
        cap = float(np.percentile(energies, 99)) if len(energies) > 1 else energies[0]
        cap = max(cap, 1e-9)
        xs = [min(s.energy, cap) for s in finite]
        ys = [s.cell_acc for s in finite]
        colors = ["C2" if s.exact else "C3" for s in finite]
        ax.scatter(xs, ys, c=colors, s=18, alpha=0.6)
    ax.set_xlabel("final energy (clipped; inf dropped)")
    ax.set_ylabel("cell accuracy")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig
