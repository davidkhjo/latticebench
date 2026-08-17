"""Optional plots (requires the ``[viz]`` extra)."""

from __future__ import annotations

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
