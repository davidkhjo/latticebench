"""Tests for the post-hoc analysis helpers and their plots."""

from __future__ import annotations

import pytest

from latticebench import SolverModel, evaluate, make_split
from latticebench.analysis import (
    accuracy_with_ci,
    difficulty_agreement,
    join_scores,
    size_accuracy,
    spearman,
    wilson_interval,
)
from latticebench.harness.base import EvalResult
from latticebench.harness.metrics import PuzzleScore


def _fake_result(
    model: str, flags: list[int], *, energies: list[float] | None = None
) -> EvalResult:
    energies = energies or [0.0] * len(flags)
    per_puzzle = [
        PuzzleScore(
            id=f"p{i}",
            exact=bool(f),
            cell_acc=1.0 if f else 0.5,
            energy=e,
            solve_time=0.01,
            tier="easy",
        )
        for i, (f, e) in enumerate(zip(flags, energies, strict=True))
    ]
    return EvalResult(model=model, per_puzzle=per_puzzle, summary={}, by_difficulty={})


# --- wilson_interval ---------------------------------------------------------


def test_wilson_known_value() -> None:
    lo, hi = wilson_interval(8, 10)
    assert 0.49 < lo < 0.56
    assert 0.90 < hi < 0.98


def test_wilson_n_zero() -> None:
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_wilson_bounds_full_and_zero() -> None:
    for lo, hi in (wilson_interval(10, 10), wilson_interval(0, 10)):
        assert 0.0 <= lo <= hi <= 1.0


# --- join_scores -------------------------------------------------------------


def test_join_scores_solver_is_exact() -> None:
    records = list(make_split(101, n=3, m=3, count=3, created="2027-01-01", split="t"))
    result = evaluate(SolverModel(), records)
    rows = join_scores(result, records)

    assert len(rows) == len(records)
    for row in rows:
        assert "z3_conflicts" in row
        assert "n" in row
        assert row["n"] == 3
        assert row["exact"] == 1  # the solver clears every well-formed puzzle


def test_join_scores_skips_missing_records() -> None:
    records = list(make_split(101, n=3, m=3, count=3, created="2027-01-01", split="t"))
    result = evaluate(SolverModel(), records[:2])
    rows = join_scores(result, records)
    assert len(rows) == 2


# --- accuracy_with_ci --------------------------------------------------------


def test_accuracy_with_ci() -> None:
    acc = accuracy_with_ci(_fake_result("m", [1, 1, 1, 0]))
    assert acc["n"] == 4
    assert acc["exact"] == 0.75
    assert 0.0 <= acc["lo"] <= acc["exact"] <= acc["hi"] <= 1.0


# --- spearman ----------------------------------------------------------------


def test_spearman_perfect_positive() -> None:
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def test_spearman_perfect_negative() -> None:
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_constant_is_zero() -> None:
    assert spearman([1, 1, 1, 1], [1, 2, 3, 4]) == 0.0


def test_spearman_monotone_with_ties() -> None:
    # Monotone but nonlinear, with a tie: rank correlation is still 1.
    r = spearman([1, 2, 2, 3], [1, 4, 4, 9])
    assert r == pytest.approx(1.0)


# --- size_accuracy -----------------------------------------------------------


def test_size_accuracy_sorted_and_bounded() -> None:
    pairs = [
        (5, _fake_result("m", [1, 1, 0, 0, 0])),
        (3, _fake_result("m", [1, 1, 1, 0])),
        (4, _fake_result("m", [1, 1, 1, 0, 0])),
    ]
    rows = size_accuracy(pairs)
    assert [r["n"] for r in rows] == [3, 4, 5]
    for r in rows:
        assert r["lo"] <= r["exact"] <= r["hi"]
    # Accuracy falls as n grows here.
    assert rows[0]["exact"] > rows[-1]["exact"]


# --- difficulty_agreement ----------------------------------------------------


def test_difficulty_agreement_detects_prefixes() -> None:
    llm_rows = [
        {"id": "a", "exact": 1, "energy": 0.0, "z3_conflicts": 1},
        {"id": "b", "exact": 0, "energy": 0.0, "z3_conflicts": 5},
        {"id": "c", "exact": 0, "energy": 0.0, "z3_conflicts": 9},
    ]
    ebm_rows = [
        {"id": "a", "exact": 1, "energy": 0.0, "z3_conflicts": 1},
        {"id": "b", "exact": 0, "energy": 2.0, "z3_conflicts": 5},
        {"id": "c", "exact": 0, "energy": 4.0, "z3_conflicts": 9},
    ]
    out = difficulty_agreement({"llm:x": llm_rows, "ebm:y": ebm_rows})
    assert set(out) == {
        "conflict_vs_llm_error",
        "conflict_vs_ebm_energy",
        "llm_error_vs_ebm_energy",
    }
    # More conflicts track more LLM errors and higher EBM energy.
    assert out["conflict_vs_llm_error"] > 0
    assert out["conflict_vs_ebm_energy"] == pytest.approx(1.0)


def test_difficulty_agreement_llm_only() -> None:
    llm_rows = [
        {"id": "a", "exact": 1, "energy": 0.0, "z3_conflicts": 1},
        {"id": "b", "exact": 0, "energy": 0.0, "z3_conflicts": 9},
    ]
    out = difficulty_agreement({"llm:x": llm_rows})
    assert set(out) == {"conflict_vs_llm_error"}


# --- plots -------------------------------------------------------------------


def test_plot_accuracy_vs_size(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from latticebench.viz import plot_accuracy_vs_size

    data = {
        "solver": [
            {"n": 3, "exact": 1.0, "lo": 0.9, "hi": 1.0},
            {"n": 4, "exact": 0.8, "lo": 0.6, "hi": 0.95},
        ],
        "llm": [
            {"n": 3, "exact": 0.5, "lo": 0.3, "hi": 0.7},
            {"n": 4, "exact": 0.2, "lo": 0.1, "hi": 0.4},
        ],
    }
    out = tmp_path / "size.png"
    fig = plot_accuracy_vs_size(data, path=str(out))
    assert out.exists()
    assert fig is not None


def test_plot_conflict_vs_accuracy(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from latticebench.viz import plot_conflict_vs_accuracy

    joined = [{"z3_conflicts": i, "exact": int(i < 5)} for i in range(10)]
    out = tmp_path / "conflict.png"
    fig = plot_conflict_vs_accuracy(joined, path=str(out), bins=4)
    assert out.exists()
    assert fig is not None


def test_plot_energy_calibration(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    import math

    import matplotlib

    matplotlib.use("Agg")
    from latticebench.viz import plot_energy_calibration

    result = _fake_result(
        "m",
        [1, 0, 0],
        energies=[0.0, 3.0, math.inf],  # inf must be dropped, not plotted
    )
    out = tmp_path / "energy.png"
    fig = plot_energy_calibration(result, path=str(out))
    assert out.exists()
    assert fig is not None
