"""Tests for scripts/reproduce_cross_benchmark.py on small synthetic matrices (no records needed)."""
from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import stats

from scripts import reproduce_cross_benchmark as cb
from scripts import cross_benchmark_outputs as render
from scripts import matched_arms as pt
from scripts.aggregate_5seed import compute_pairwise_rho, compute_mv_accuracy, k_eff
from scripts.correlation_estimators import c_eq3
from scripts.analyze_mv_vs_weighted import slot_cv, weighted_mv_accuracy


@pytest.fixture
def mat() -> np.ndarray:
    return np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ], dtype=float)


def test_imported_estimators_and_ties(mat):
    assert cb.compute_pairwise_rho is compute_pairwise_rho
    assert cb.compute_mv_accuracy is compute_mv_accuracy
    assert cb.c_eq3 is c_eq3
    assert cb.slot_cv is slot_cv
    assert cb.weighted_mv_accuracy is weighted_mv_accuracy
    result = cb.arm_statistics(mat)
    assert result["MV8"] == pytest.approx((0 + 0 + .5 + 1 + 1 + 1) / len(mat))
    assert result["rho_hat"] == compute_pairwise_rho(mat)
    assert result["c_hat"] == c_eq3(mat)
    assert result["rho_hat"] != pytest.approx(result["c_hat"])
    assert result["K_eff_vote"] == k_eff(pt.K, result["rho_hat"])
    assert result["K_eff_from_c"] == k_eff(pt.K, result["c_hat"])
    assert result["weighted_gain_pp"] == pytest.approx(
        (weighted_mv_accuracy(mat) - compute_mv_accuracy(mat)) * 100)


def test_paired_bootstrap_identical_arms_and_reproducibility(mat):
    same = cb.paired_bootstrap(mat, mat, 40, np.random.default_rng(1))
    assert np.isfinite(same).any()
    assert (same[np.isfinite(same)] == 0).all()  # fails if arms are resampled independently
    left = cb.paired_bootstrap(mat, mat[:, ::-1], 40, np.random.default_rng(1))
    right = cb.paired_bootstrap(mat, mat[:, ::-1], 40, np.random.default_rng(1))
    np.testing.assert_array_equal(left, right)
    with pytest.raises(pt.CellMismatchError, match="matching matrices"):
        cb.paired_bootstrap(mat, mat[:-1], 40, np.random.default_rng(1))


@pytest.mark.parametrize("bad", [np.zeros((2, 7)), np.full((2, 8), .5), np.zeros((1, 8))])
def test_shape_and_binary_checks(bad):
    with pytest.raises(pt.CellMismatchError):
        cb.arm_statistics(bad)
    with pytest.raises(pt.CellMismatchError):
        cb.correctness_matrix([(["1"] * 7, "1")], "gsm8k")


def test_v2_scoring_ignores_string_exactness():
    matrix = cb.correctness_matrix([(["1,200", "1200.", None, "0", "0", "0", "0", "0"], "1200")], "gsm8k")
    np.testing.assert_array_equal(matrix, [[1, 1, 0, 0, 0, 0, 0, 0]])


def test_exclusion_is_strictly_sc_path_accuracy():
    assert cb.excluded_base_accuracy(np.nextafter(cb.BASE_ACC_THRESHOLD, 0))
    assert cb.excluded_base_accuracy(float("nan"))
    assert not cb.excluded_base_accuracy(cb.BASE_ACC_THRESHOLD)
    assert not cb.excluded_base_accuracy(np.nextafter(cb.BASE_ACC_THRESHOLD, 1))


def test_block_formula_and_eligibility(mat):
    block = cb.block_statistic(np.tile(mat, (4, 1)))
    stacked = np.tile(mat, (4, 1))
    expected = stacked.mean() * (1 - stacked.mean()) / np.var(stacked.mean(axis=1), ddof=1)
    assert block["K_eff_block"] == pytest.approx(expected)
    assert block["ratio"] == pytest.approx(expected / k_eff(pt.K, compute_pairwise_rho(stacked)))
    assert "reason" in cb.block_statistic(np.zeros((24, 8)))
    # Historical nanmean skips constant slots; they do not automatically exclude a cell.
    stacked[:, -1] = 0
    assert "ratio" in cb.block_statistic(stacked)
    assert "reason" in cb.block_statistic(np.ones((24, 8)))


def test_source_grid_rejects_duplicates_and_bad_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, "INSTANCES_PER_SEED", 1)
    monkeypatch.setattr(pt, "EXPECTED_SEEDS", frozenset({42}))
    record = {"task": "math", "seed": 42, "instance_id": 0, "gold_answer": "1", "all_answers": ["1"] * 8}
    path = tmp_path / "arm.jsonl"
    path.write_text(json.dumps(record) + "\n")
    assert cb.source_grid(path, "math", "PT") == [(42, 0)]
    with pytest.raises(pt.CellMismatchError, match="expected 32"):
        cb.source_grid(path, "math", "SC")
    path.write_text((json.dumps(record) + "\n") * 2)
    with pytest.raises(pt.CellMismatchError, match="duplicate keys"):
        cb.source_grid(path, "math", "PT")


def test_load_cell_pairs_sort_together_and_missing_cell_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, "INSTANCES_PER_SEED", 1)
    monkeypatch.setattr(pt, "EXPECTED_SEEDS", frozenset({42, 123}))
    pp, sp = tmp_path / "pt.jsonl", tmp_path / "sc.jsonl"
    def rec(s):
        return {"seed": s, "instance_id": 0, "task": "gsm8k", "gold_answer": str(s)}
    pp.write_text("".join(json.dumps({**rec(s), "all_traces": ["x"] * 8}) + "\n" for s in (123, 42)))
    sp.write_text("".join(json.dumps({**rec(s), "all_answers": ["x"] * 32}) + "\n" for s in (42, 123)))
    cells = [(["x"] * 8, str(s)) for s in (123, 42)]
    monkeypatch.setattr(cb, "fingerprint", lambda p: {"path": p.name, "md5": "test"})
    monkeypatch.setattr(pt, "load_arms", lambda _: (cells, cells, pp, sp))
    left, right, source = cb.load_cell("qwen7b", "gsm8k")
    assert [g for _, g in left] == ["42", "123"]
    assert left == right
    assert source["seeds"] == [42, 123]
    def missing(*_):
        raise FileNotFoundError("required cell missing")
    monkeypatch.setattr(pt, "load_arcwg_arms", missing)
    with pytest.raises(FileNotFoundError, match="required cell missing"):
        cb.load_cell("qwen7b", "arc_challenge")


def test_build_glue_exclusion_outputs_and_cli_order(mat, tmp_path, monkeypatch):
    def as_cells(matrix):
        return [(["1" if x else "0" for x in row], "1") for row in matrix]
    def load(model, task):
        # Low SC / accurate PT remains excluded from primary, eligible for weighted analysis.
        sc = np.zeros_like(mat) if model == pt.T12_MODELS[0] else mat
        return as_cells(mat), as_cells(sc), {
            "seeds": [42], "SC": {"path": f"synthetic/{model}/{task}/SC"},
            "PT": {"path": f"synthetic/{model}/{task}/PT"},
        }
    monkeypatch.setattr(cb, "load_cell", load)
    result = cb.build(["gsm8k", "math", "triviaqa"], n_boot=20)
    reverse = cb.build(["triviaqa", "math", "gsm8k"], n_boot=20)
    assert render.compact(result) == render.compact(reverse)
    d = result["derived"]
    assert d["valid_cells_count"] == 3 * (len(pt.T12_MODELS) - 1)
    assert d["excluded_cells_count"] == 3
    assert d["weighted_mv_stats"]["n_cells"] == 3 * len(pt.T12_MODELS)
    assert d["weighted_mv_stats"]["cv_threshold"] == cb.CV_THRESHOLD
    assert d["mann_whitney"]["alternative"] == "two-sided"
    assert d["mann_whitney"]["p_value"] == stats.mannwhitneyu([0] * 4, [0] * 4, alternative="two-sided").pvalue
    assert d["mann_whitney"]["bootstrap_fraction_p_lt_0001"] == 0
    assert all(c["delta_rho_ci95"] == [0, 0] for c in result["cells"] if not c["excluded_base_acc"])
    render.write_outputs(result, tmp_path)
    text = (tmp_path / "numbers.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    assert len(json.loads(text)["cells"]) == 3 * len(pt.T12_MODELS)
    rows = (tmp_path / "table_full_matrix_rows.tex").read_text()
    assert "open text" in rows and "numeric" in rows and "\\multirow" in rows
    with pytest.raises(ValueError, match="unique"):
        cb.build(["math", "math"], n_boot=20)
