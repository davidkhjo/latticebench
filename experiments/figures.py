"""Rebuild the paper figures and tables from saved per-config results.

Reads the JSON files written by ``run.py`` (one per model and grid size) and
produces the headline accuracy-vs-difficulty plot and a per-config table with
Wilson intervals. Kept separate from the evaluation so the figures can be redrawn
without rerunning any model.

Run: uv run python experiments/figures.py
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from latticebench import analysis, viz
from latticebench.harness.base import EvalResult

_SUFFIX = re.compile(r"-n(\d+)m(\d+)\.json$")


def load_results(results_dir: str = "results") -> dict[str, list[tuple[int, int, EvalResult]]]:
    out: dict[str, list[tuple[int, int, EvalResult]]] = {}
    for path in sorted(Path(results_dir).glob("*.json")):
        match = _SUFFIX.search(path.name)
        if not match:
            continue
        n, m = int(match.group(1)), int(match.group(2))
        result = EvalResult.from_dict(json.loads(path.read_text()))
        out.setdefault(result.model, []).append((n, m, result))
    return out


def log_search_space(n: int, m: int) -> float:
    return max(m - 1, 0) * math.lgamma(n + 1)


def main() -> None:
    figdir = Path("paper/figures")
    figdir.mkdir(parents=True, exist_ok=True)
    per_model = load_results()

    data = {}
    table = ["| model | grid (n×m) | exact-match | 95% CI |", "|---|---|--:|--:|"]
    for name, configs in per_model.items():
        configs = sorted(configs, key=lambda c: log_search_space(c[0], c[1]))
        rows = []
        for n, m, result in configs:
            ci = analysis.accuracy_with_ci(result)
            rows.append({"x": log_search_space(n, m), **ci})
            table.append(
                f"| {name} | {n}×{m} | {ci['exact']:.2f} | [{ci['lo']:.2f}, {ci['hi']:.2f}] |"
            )
        data[name] = rows

    viz.plot_accuracy_vs_difficulty(data, str(figdir / "accuracy_vs_difficulty.png"))
    (figdir / "accuracy_by_config.md").write_text("\n".join(table) + "\n")
    print(f"redrew figures for {len(per_model)} models -> {figdir}")


if __name__ == "__main__":
    main()
