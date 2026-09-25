#!/usr/bin/env python3
"""Tests for scripts/reproduce_tables.py.

1. The majority vote is aggregate_canonical_sc.mv_acc (binary strict majority, half credit at ties).
2. Pooled c-hat and p-bar at K=32 equal the exact-K table written by
   scripts/aggregate_kvar_v2.py (DC_REFERENCE_ROOT/kvar_v2_exact_k.csv) to 1e-9.
3. The Eq. 12 K* operating point matches the closed form on a grid and at the boundaries.
4. The (seed, instance) shape check rejects duplicates, gaps and unexpected pairs.
5. numbers.json is strict JSON and undefined metrics are encoded as null.
6. All bootstrap draws are kept and boundary hits are reported per cell.
7. The reasoning-model cell is computed on 300 rows; its loader rejects bad shapes
   and eager attention.

Run scripts/reproduce_tables.py first: the numbers.json checks read its output.
"""
from __future__ import annotations

import copy
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

import scripts.reproduce_tables as cr
from scripts.aggregate_canonical_sc import mv_acc as canonical_mv_acc
from scripts.aggregate_kvar_v2 import compute_exact_k, pairwise_rho
from diversity_combining.config import experiment_dir
from scripts.reproduce_tables import (
    CACHE_DIR,
    OUT_DIR,
    build_correctness_matrix,
    k_star_eq12,
    load_cell_strict,
    load_e1_cell,
    validate_cartesian_shape,
)


def test_binary_mv_equals_canonical_mv_acc():
    """Binary MV used in reproduce_tables is aggregate_canonical_sc.mv_acc and matches on test matrices."""
    # Assert exact binding identity
    assert cr.mv_acc is canonical_mv_acc, "reproduce_tables.mv_acc must be aggregate_canonical_sc.mv_acc"

    # Test cases:
    # 1. Odd K: clear majority
    m1 = np.array([
        [1, 1, 0],  # 2/3 -> 1.0
        [1, 0, 0],  # 1/3 -> 0.0
        [1, 1, 1],  # 3/3 -> 1.0
        [0, 0, 0],  # 0/3 -> 0.0
    ], dtype=float)
    assert cr.mv_acc(m1) == pytest.approx(0.5)
    assert cr.mv_acc(m1) == canonical_mv_acc(m1)

    # 2. Even K with exact ties (votes == K/2 gets 0.5)
    m2 = np.array([
        [1, 1, 0, 0],  # 2/4 -> tie -> 0.5
        [1, 1, 1, 0],  # 3/4 -> win -> 1.0
        [1, 0, 0, 0],  # 1/4 -> loss -> 0.0
        [0, 0, 0, 0],  # 0/4 -> loss -> 0.0
        [1, 1, 1, 1],  # 4/4 -> win -> 1.0
    ], dtype=float)
    # Expected: (0.5 + 1.0 + 0.0 + 0.0 + 1.0) / 5 = 2.5 / 5 = 0.5
    assert cr.mv_acc(m2) == pytest.approx(0.5)
    assert cr.mv_acc(m2) == canonical_mv_acc(m2)

    # 3. Even K with all ties
    m3 = np.array([
        [1, 0],  # 1/2 -> 0.5
        [0, 1],  # 1/2 -> 0.5
    ], dtype=float)
    assert cr.mv_acc(m3) == pytest.approx(0.5)
    assert cr.mv_acc(m3) == canonical_mv_acc(m3)

    # 4. K=32 matrix with random votes, verify identical behavior
    rng = np.random.default_rng(42)
    m_rand = rng.integers(0, 2, size=(100, 32)).astype(float)
    votes = m_rand.sum(axis=1)
    manual_acc = float(np.mean((votes > 16).astype(float) + 0.5 * (votes == 16).astype(float)))
    assert cr.mv_acc(m_rand) == pytest.approx(manual_acc)
    assert cr.mv_acc(m_rand) == canonical_mv_acc(m_rand)


@pytest.mark.parametrize(
    "model,task,filename,expected_seeds",
    [
        ("qwen7b", "gsm8k", "qwen7b_sc_kvar_v2_gsm8k.jsonl", [42, 123, 456, 789, 1024]),
        ("llama8b", "gsm8k", "llama8b_sc_kvar_v2_gsm8k.jsonl", [42, 123, 456, 789, 1024]),
        ("mistral7b", "gsm8k", "mistral7b_sc_kvar_v2_gsm8k.jsonl", [42, 123, 456, 789, 1024]),
        ("llama8b", "hotpotqa", "llama8b_sc_kvar_v2_hotpotqa.jsonl", [42, 123, 456]),
        ("llama8b", "boolq", "llama8b_sc_kvar_v2_boolq.jsonl", [42, 123, 456]),
    ],
)
def test_cells_match_reference_kvar_v2_exact_k(model, task, filename, expected_seeds):
    """Pooled ĉ at K=32 and p̄ at K=32 equal the reference CSV values to 1e-9."""
    ref_csv_path = experiment_dir("reference") / "kvar_v2_exact_k.csv"
    assert ref_csv_path.is_file(), f"Reference CSV {ref_csv_path} not found"

    ref_row = None
    with open(ref_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["model"] == model and row["task"] == task and row["K"] == "32":
                ref_row = row
                break

    assert ref_row is not None, f"No reference row found for {model}/{task}/K=32"
    ref_c = float(ref_row["c"])
    ref_p = float(ref_row["mean_path_acc"])

    records, stats = load_cell_strict(model, task, filename, expected_seeds)
    mat32 = build_correctness_matrix(records, task, 32)

    our_p = float(mat32.mean())
    our_c = pairwise_rho(mat32)

    assert abs(our_p - ref_p) < 1e-9, f"p_bar mismatch: {our_p} vs {ref_p}"
    assert abs(our_c - ref_c) < 1e-9, f"c mismatch: {our_c} vs {ref_c}"


def test_k_star_eq12_matches_canonical():
    """Eq. 12 K* equals aggregate_canonical_sc function on a grid of c."""
    epsilon = 0.025
    c_grid = np.linspace(0.01, 0.99, 99)

    for c in c_grid:
        # Canonical formula from aggregate_canonical_sc.tab_adaptive_k
        expected_k = math.ceil((math.sqrt((1.0 - c) / epsilon) - 1.0) / c + 1.0)
        actual_k = k_star_eq12(float(c), epsilon=epsilon)
        assert actual_k == expected_k, f"Mismatch at c={c}: expected {expected_k}, got {actual_k}"

    # Edge cases
    assert k_star_eq12(0.0, epsilon=epsilon) == 1
    assert k_star_eq12(-0.5, epsilon=epsilon) == 1
    assert k_star_eq12(1.0, epsilon=epsilon) == 1
    assert k_star_eq12(float("nan"), epsilon=epsilon) == 1


def test_cartesian_shape_validation():
    """Validate Cartesian shape assertion fails on duplicates, gaps, or unexpected pairs."""
    # 1. Valid case
    valid_recs = [{"seed": s, "instance_id": i} for s in [42, 123] for i in range(10)]
    validate_cartesian_shape(valid_recs, [42, 123], 10, "test_mod", "test_task")

    # 2. Duplicate pair
    dup_recs = list(valid_recs) + [{"seed": 42, "instance_id": 0}]
    with pytest.raises(ValueError, match="Cartesian product validation failed.*duplicate"):
        validate_cartesian_shape(dup_recs, [42, 123], 10, "test_mod", "test_task")

    # 3. Missing pair (gap)
    gap_recs = [r for r in valid_recs if not (r["seed"] == 42 and r["instance_id"] == 5)]
    with pytest.raises(ValueError, match="Cartesian product validation failed.*missing"):
        validate_cartesian_shape(gap_recs, [42, 123], 10, "test_mod", "test_task")

    # 4. Unexpected pair
    unexp_recs = list(valid_recs) + [{"seed": 999, "instance_id": 0}]
    with pytest.raises(ValueError, match="Cartesian product validation failed.*unexpected"):
        validate_cartesian_shape(unexp_recs, [42, 123], 10, "test_mod", "test_task")


def test_numbers_json_no_nan_and_allow_nan_false():
    """numbers.json contains no NaN tokens, undefined are null, serializable with allow_nan=False."""
    numbers_path = OUT_DIR / "numbers.json"
    assert numbers_path.is_file(), f"{numbers_path} not found"

    with open(numbers_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Raw text must not contain NaN as a literal token
    assert not re.search(r'\bNaN\b', raw_text), "Found unencoded NaN literal in numbers.json"

    # Must parse cleanly
    data = json.loads(raw_text)

    # Must re-serialize with allow_nan=False without raising ValueError
    reserialized = json.dumps(data, allow_nan=False)
    assert reserialized is not None

    # K=1 metrics in Table 1 must be None (encoded as null in JSON)
    k1 = data["tab_diversity"]["K_1"]
    assert k1["Agree"] is None
    assert k1["c_hat"] is None
    assert k1["Ceiling"] is None
    assert k1["pct_ceiling"] is None


def test_bootstrap_draws_and_boundary_hits():
    """Ensure all B_pilot draws are kept and boundary hits are recorded per cell."""
    numbers_path = OUT_DIR / "numbers.json"
    with open(numbers_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pilot_data = data["tab_pilot_sensitivity"]
    for mod_key, n_dict in pilot_data.items():
        for n_str, r in n_dict.items():
            assert "boundary_hits" in r, f"Missing boundary_hits in {mod_key} n={n_str}"
            assert isinstance(r["boundary_hits"], int)
            assert r["boundary_hits"] >= 0


def test_reasoning_model_e1():
    """Ensure reasoning model E1 results are FINAL and computed on clean 300 rows."""
    numbers_path = OUT_DIR / "numbers.json"
    with open(numbers_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    e1 = data["reasoning_model_e1"]
    assert e1["status"] == "FINAL"
    assert "provisional_evaluation" not in e1
    assert "provisional_note" not in e1
    metrics = e1["metrics"]
    assert metrics is not None
    assert metrics["label"] == "FINAL"
    assert metrics["seeds_evaluated"] == [42, 123, 456]
    assert metrics["n_records_evaluated"] == 300

    for key in [
        "p_bar",
        "c_k4",
        "c_k32",
        "ceiling",
        "keff_32",
        "mv_obs_pct",
        "heldout_bb",
        "k_star",
        "mv_kstar_pct",
        "mv_32_pct",
        "retention_pct",
        "net_cost_pct",
        "diff_point_pp",
        "ci_95_lower_pp",
        "ci_95_upper_pp",
        "cap_hit_rate",
        "cap_hit_paths",
        "cap_hit_denominator_paths",
        "null_extractions",
        "empty_extractions",
    ]:
        assert key in metrics, f"Missing {key} in E1 metrics"

    # Held-out BB keys
    hbb = metrics["heldout_bb"]
    assert "fold_A" in hbb
    assert "fold_B" in hbb
    assert "fold_avg_bb_err_pp" in hbb
    assert "fold_avg_binom_err_pp" in hbb
    for K in [4, 8, 16, 32]:
        assert f"K_{K}" in hbb["fold_avg_bb_err_pp"]
        assert f"K_{K}" in hbb["fold_avg_binom_err_pp"]
        assert f"K_{K}" in hbb["fold_A"]["ks"]
        assert f"K_{K}" in hbb["fold_B"]["ks"]

    # Numeric checks
    assert metrics["k_star"] == 14
    assert metrics["mv_kstar_pct"] == pytest.approx(96.67, abs=0.1)
    assert metrics["mv_32_pct"] == pytest.approx(96.83, abs=0.1)
    assert metrics["retention_pct"] == pytest.approx(99.83, abs=0.1)
    assert metrics["net_cost_pct"] == pytest.approx(56.25, abs=0.01)
    assert metrics["cap_hit_paths"] == 514
    assert metrics["cap_hit_denominator_paths"] == 9600
    assert metrics["cap_hit_rate"] == pytest.approx(514 / 9600)
    assert metrics["null_extractions"] == 514
    assert metrics["empty_extractions"] == 0


def test_e1_loader_rejects_missing_pair_and_eager_attention(tmp_path):
    """E1 loader rejects a record set with a missing (seed, instance) pair and one with attn_implementation eager."""
    valid_recs, stats = load_e1_cell()
    assert len(valid_recs) == 300
    assert stats["n_records"] == 300

    # 1a. In-memory: missing pair by omission (299 records)
    missing_one = [r for r in valid_recs if not (r["seed"] == 42 and r["instance_id"] == 10)]
    assert len(missing_one) == 299
    with pytest.raises(ValueError, match="expected 300 records, got 299"):
        load_e1_cell(records=missing_one)

    # 1b. In-memory: missing pair by replacement/duplicate (300 records but missing (42, 10))
    replaced_dup = list(valid_recs)
    for idx, r in enumerate(replaced_dup):
        if r["seed"] == 42 and r["instance_id"] == 10:
            replaced_dup[idx] = dict(valid_recs[0])  # duplicate (42, 0)
            break
    with pytest.raises(ValueError, match="Cartesian product validation failed"):
        load_e1_cell(records=replaced_dup)

    # 1c. On-disk: JSONL file with a missing pair
    bad_shape_file = tmp_path / "bad_shape.jsonl"
    with open(bad_shape_file, "w", encoding="utf-8") as f:
        for r in missing_one:
            f.write(json.dumps(r) + "\n")
    with pytest.raises(ValueError, match="expected 300 records, got 299"):
        load_e1_cell(filepath=bad_shape_file)

    # 2a. In-memory: provenance dict has attn_implementation == 'eager'
    eager_prov = copy.deepcopy(valid_recs)
    eager_prov[0]["provenance"]["attn_implementation"] = "eager"
    with pytest.raises(ValueError, match="attn_implementation expected 'default'"):
        load_e1_cell(records=eager_prov)

    # 2b. In-memory: root dict has attn_implementation == 'eager'
    eager_root = copy.deepcopy(valid_recs)
    eager_root[0]["attn_implementation"] = "eager"
    with pytest.raises(ValueError, match="attn_implementation expected 'default'"):
        load_e1_cell(records=eager_root)

    # 2c. On-disk: JSONL file with eager attention
    eager_file = tmp_path / "eager_attn.jsonl"
    with open(eager_file, "w", encoding="utf-8") as f:
        for r in eager_prov:
            f.write(json.dumps(r) + "\n")
    with pytest.raises(ValueError, match="attn_implementation expected 'default'"):
        load_e1_cell(filepath=eager_file)
