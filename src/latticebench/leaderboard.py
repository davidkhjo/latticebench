"""Leaderboard aggregation: turn evaluation results into a sorted table."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from latticebench.harness.base import EvalResult


def _model_class(name: str) -> str:
    return name.split(":", 1)[0]


@dataclass
class LeaderboardRow:
    model: str
    model_class: str
    exact_match: float
    cell_accuracy: float
    mean_energy: float
    median_solve_time: float
    by_tier: dict[str, float]


@dataclass
class Leaderboard:
    rows: list[LeaderboardRow]

    def to_markdown(self) -> str:
        header = "| model | class | exact | cell | mean energy | median time (s) |"
        sep = "|---|---|---:|---:|---:|---:|"
        lines = [header, sep]
        for r in self.rows:
            energy = "inf" if math.isinf(r.mean_energy) else f"{r.mean_energy:.3f}"
            lines.append(
                f"| {r.model} | {r.model_class} | {r.exact_match:.3f} | "
                f"{r.cell_accuracy:.3f} | {energy} | {r.median_solve_time:.4f} |"
            )
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(
            [
                {
                    "model": r.model,
                    "model_class": r.model_class,
                    "exact_match": r.exact_match,
                    "cell_accuracy": r.cell_accuracy,
                    "mean_energy": r.mean_energy,
                    "median_solve_time": r.median_solve_time,
                    "by_tier": r.by_tier,
                }
                for r in self.rows
            ],
            indent=2,
        )


def build_leaderboard(results: Iterable[EvalResult]) -> Leaderboard:
    rows = []
    for res in results:
        s = res.summary
        rows.append(
            LeaderboardRow(
                model=res.model,
                model_class=_model_class(res.model),
                exact_match=s.get("exact_match", 0.0),
                cell_accuracy=s.get("cell_accuracy", 0.0),
                mean_energy=s.get("mean_energy", math.inf),
                median_solve_time=s.get("median_solve_time", 0.0),
                by_tier={t: g.get("exact_match", 0.0) for t, g in res.by_difficulty.items()},
            )
        )
    rows.sort(key=lambda r: r.exact_match, reverse=True)
    return Leaderboard(rows=rows)


def load_results(paths: Sequence[str | Path]) -> list[EvalResult]:
    out = []
    for p in paths:
        with Path(p).open(encoding="utf-8") as f:
            out.append(EvalResult.from_dict(json.load(f)))
    return out
