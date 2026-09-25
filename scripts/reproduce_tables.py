#!/usr/bin/env python3
"""Reproduce Tables 1, 2 and 4, Figure 2, the pilot-size and epsilon tables, and the
reasoning-model appendix from the v2 self-consistency records.

Inputs: the five primary cells (K=32) in experiment_dir("sc_records") and the
Qwen3.5-9B GSM8K cell in experiment_dir("reasoning_records"). Outputs: numbers.json,
LaTeX table rows and bb_calibration_cr.pdf in experiment_dir("tables").

- Estimators are imported from scripts.aggregate_canonical_sc and scripts.aggregate_kvar_v2.
- Majority vote is the binary strict majority on 0/1 correctness with half credit at
  exact ties (aggregate_canonical_sc.mv_acc).
- Malformed or unexpectedly shaped inputs raise immediately.
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
_SRC = PROJECT_ROOT
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ── Import estimators (Hard Rule: Never copy) ──────────────────────────────────
# From aggregate_canonical_sc:
#   mv_acc: binary strict majority vote with half credit at exact ties
#   bb_mv_acc: beta-binomial majority vote accuracy prediction
#   binom_mv_acc: binomial (independence) majority vote prediction
from scripts.aggregate_canonical_sc import (
    bb_mv_acc,
    binom_mv_acc,
    mv_acc,
)

# From aggregate_kvar_v2:
#   load_and_validate_file: cache loader and provenance checker
#   mean_pairwise_agree: pairwise agreement mean
#   pairwise_rho: Eq. 3 ĉ estimator (mean pairwise Pearson correlation of binary correctness)
#   keff: K_eff = K / (1 + (K - 1) * c)
from scripts.aggregate_kvar_v2 import (
    keff,
    load_and_validate_file,
    mean_pairwise_agree,
    pairwise_rho,
)

# From scoring_v2:
#   check_task_correct: task-specific v2 correctness evaluator
from scripts.scoring_v2 import check_task_correct


# aggregate_canonical_sc.tab_adaptive_k implements Eq. 12 K* inline:
#   if c_hat <= 0 or c_hat >= 1:
#       k_star = 1
#   else:
#       k_star = math.ceil((math.sqrt((1 - c_hat) / epsilon) - 1) / c_hat + 1)
# We provide a clean shim function wrapping this exact canonical logic.
def k_star_eq12(c_hat: float, epsilon: float = 0.025) -> int:
    """Eq. (12) Adaptive-K operating point estimator, matching aggregate_canonical_sc.tab_adaptive_k."""
    if c_hat is None or not math.isfinite(c_hat) or c_hat <= 0 or c_hat >= 1:
        return 1
    val = (math.sqrt((1.0 - c_hat) / epsilon) - 1.0) / c_hat + 1.0
    return math.ceil(val)


def fmt_range(val_min: float, val_max: float, decimals: int = 0, sep: str = "–") -> str:
    """Shared range formatting helper with uniform rounding."""
    if decimals == 0:
        return f"{val_min:.0f}{sep}{val_max:.0f}"
    return f"{val_min:.{decimals}f}{sep}{val_max:.{decimals}f}"


def validate_cartesian_shape(
    records: list[dict],
    expected_seeds: list[int],
    expected_instances: int,
    model: str,
    task: str,
) -> None:
    """Assert exactly one record per expected (seed, instance_id) pair; fail on duplicates or gaps."""
    expected_pairs = {(s, i) for s in expected_seeds for i in range(expected_instances)}
    actual_pairs = [(r.get("seed"), r.get("instance_id")) for r in records]
    pair_counts = Counter(actual_pairs)

    duplicates = [pair for pair, count in pair_counts.items() if count > 1]
    missing = sorted(list(expected_pairs - set(pair_counts.keys())))
    unexpected = sorted(list(set(pair_counts.keys()) - expected_pairs))

    if duplicates or missing or unexpected:
        raise ValueError(
            f"FATAL: Cell {model}/{task} Cartesian product validation failed: "
            f"{len(duplicates)} duplicate pairs (e.g. {duplicates[:5]}), "
            f"{len(missing)} missing pairs (e.g. {missing[:5]}), "
            f"{len(unexpected)} unexpected pairs (e.g. {unexpected[:5]})."
        )


# ── Configuration & Paths ─────────────────────────────────────────────────────
from diversity_combining.config import experiment_dir
CACHE_DIR = experiment_dir("sc_records")
RSN_CACHE_DIR = experiment_dir("reasoning_records")
E1_CACHE_FILENAME = "qwen3_5_9b_sc_kvar_v2_gsm8k.jsonl"
E1_CACHE_PATH = RSN_CACHE_DIR / E1_CACHE_FILENAME
OUT_DIR = experiment_dir("tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {"Qwen-7B": "#D55E00", "Llama-8B": "#0072B2", "Mistral-7B": "#009E73"}
MARKERS = {"Qwen-7B": "o", "Llama-8B": "s", "Mistral-7B": "D"}


# ── Cell Loader & Fail-Loud Validator ─────────────────────────────────────────
def load_cell_strict(
    model: str,
    task: str,
    filename: str,
    expected_seeds: list[int],
    expected_instances: int = 100,
) -> tuple[list[dict], dict]:
    """Load cell JSONL and fail loudly if any assertion fails."""
    path = CACHE_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"FATAL: Cache file {path} not found.")

    records = load_and_validate_file(path)
    n_expected = len(expected_seeds) * expected_instances
    if len(records) != n_expected:
        raise ValueError(
            f"FATAL: Cell {model}/{task} expected {n_expected} records, got {len(records)} in {filename}."
        )

    seeds = sorted(list({r.get("seed") for r in records}))
    if seeds != sorted(expected_seeds):
        raise ValueError(
            f"FATAL: Cell {model}/{task} expected seeds {expected_seeds}, got {seeds}."
        )

    inst_ids = sorted(list({r.get("instance_id") for r in records}))
    if inst_ids != list(range(expected_instances)):
        raise ValueError(
            f"FATAL: Cell {model}/{task} expected contiguous instance IDs 0..{expected_instances - 1}, "
            f"got {len(inst_ids)} unique instances (min={min(inst_ids)}, max={max(inst_ids)})."
        )

    # Cartesian shape assertion: exactly one record per expected (seed, instance_id) pair
    validate_cartesian_shape(records, expected_seeds, expected_instances, model, task)

    # Validate 32 answers and 32 boolean truncation flags per record
    for idx, r in enumerate(records):
        answers = r.get("all_answers")
        trunc = r.get("truncated")
        if not isinstance(answers, list) or len(answers) != 32:
            raise ValueError(
                f"FATAL: Cell {model}/{task} record {idx} does not have exactly 32 answers."
            )
        if not isinstance(trunc, list) or len(trunc) != 32 or not all(isinstance(t, bool) for t in trunc):
            raise ValueError(
                f"FATAL: Cell {model}/{task} record {idx} does not have 32 boolean truncation flags."
            )

    total_paths = len(records) * 32
    null_answers = sum(1 for r in records for a in r.get("all_answers", []) if a is None)
    empty_answers = sum(1 for r in records for a in r.get("all_answers", []) if a == "")
    truncated_paths = sum(1 for r in records for t in r.get("truncated", []) if bool(t))

    stats = {
        "model": model,
        "task": task,
        "filename": filename,
        "n_records": len(records),
        "n_seeds": len(seeds),
        "seeds": seeds,
        "n_instances": len(inst_ids),
        "total_paths": total_paths,
        "null_extractions": null_answers,
        "null_extraction_rate": null_answers / total_paths,
        "empty_extractions": empty_answers,
        "empty_extraction_rate": empty_answers / total_paths,
        "truncated_paths": truncated_paths,
        "truncated_rate": truncated_paths / total_paths,
    }
    return records, stats


def load_e1_cell(
    filepath: Path | str | None = None,
    expected_seeds: list[int] | None = None,
    expected_instances: int = 100,
    records: list[dict] | None = None,
) -> tuple[list[dict], dict]:
    """Load and strictly validate E1 reasoning model cell JSONL.

    Enforces clean provenance (attn_implementation == 'default', max_new_tokens == 16384,
    scorer == 'gsm8k_numeric'), exact Cartesian product shape over expected seeds
    and instance IDs 0..expected_instances-1, and 32 answers/truncations per record.
    Fails loudly on any discrepancy.
    """
    if expected_seeds is None:
        expected_seeds = [42, 123, 456]
    expected_seeds = sorted(expected_seeds)

    if records is None:
        target_path = Path(filepath) if filepath is not None else E1_CACHE_PATH
        if not target_path.is_file():
            raise FileNotFoundError(f"FATAL: E1 cache file {target_path} not found.")
        loaded_records = load_and_validate_file(target_path)
        filename_str = target_path.name
        filepath_str = str(target_path)
    else:
        loaded_records = list(records)
        filename_str = Path(filepath).name if filepath is not None else E1_CACHE_FILENAME
        filepath_str = str(filepath) if filepath is not None else str(E1_CACHE_PATH)

    n_expected = len(expected_seeds) * expected_instances
    if len(loaded_records) != n_expected:
        raise ValueError(
            f"FATAL: E1 reasoning cell expected {n_expected} records, got {len(loaded_records)} in {filename_str}."
        )

    seeds = sorted(list({r.get("seed") for r in loaded_records}))
    if seeds != expected_seeds:
        raise ValueError(
            f"FATAL: E1 reasoning cell expected seeds {expected_seeds}, got {seeds}."
        )

    inst_ids = sorted(list({r.get("instance_id") for r in loaded_records}))
    if inst_ids != list(range(expected_instances)):
        raise ValueError(
            f"FATAL: E1 reasoning cell expected contiguous instance IDs 0..{expected_instances - 1}, "
            f"got {len(inst_ids)} unique instances (min={min(inst_ids) if inst_ids else 'None'}, max={max(inst_ids) if inst_ids else 'None'})."
        )

    # Cartesian shape assertion: exactly one record per expected (seed, instance_id) pair
    validate_cartesian_shape(loaded_records, expected_seeds, expected_instances, "qwen3_5_9b", "gsm8k")

    # Assert provenance fields and 32 answers/truncations
    for idx, r in enumerate(loaded_records):
        prov = r.get("provenance") if isinstance(r.get("provenance"), dict) else {}

        top_attn = r.get("attn_implementation")
        prov_attn = prov.get("attn_implementation")
        if (top_attn is not None and top_attn != "default") or (prov_attn is not None and prov_attn != "default"):
            raise ValueError(
                f"FATAL: E1 provenance attn_implementation expected 'default', got top='{top_attn}', prov='{prov_attn}' in record {idx}."
            )
        if top_attn is None and prov_attn is None:
            raise ValueError(
                f"FATAL: E1 record {idx} missing required provenance field 'attn_implementation'."
            )

        top_tokens = r.get("max_new_tokens")
        prov_tokens = prov.get("max_new_tokens")
        if (top_tokens is not None and top_tokens != 16384) or (prov_tokens is not None and prov_tokens != 16384):
            raise ValueError(
                f"FATAL: E1 provenance max_new_tokens expected 16384, got top='{top_tokens}', prov='{prov_tokens}' in record {idx}."
            )
        if top_tokens is None and prov_tokens is None:
            raise ValueError(
                f"FATAL: E1 record {idx} missing required provenance field 'max_new_tokens'."
            )

        scorer = prov.get("scorer") or r.get("scorer")
        if scorer != "gsm8k_numeric":
            raise ValueError(
                f"FATAL: E1 provenance scorer expected 'gsm8k_numeric', got '{scorer}' in record {idx}."
            )

        answers = r.get("all_answers")
        trunc = r.get("truncated")
        if not isinstance(answers, list) or len(answers) != 32:
            raise ValueError(
                f"FATAL: E1 record {idx} does not have exactly 32 answers."
            )
        if not isinstance(trunc, list) or len(trunc) != 32 or not all(isinstance(t, bool) for t in trunc):
            raise ValueError(
                f"FATAL: E1 record {idx} does not have 32 boolean truncation flags."
            )

    total_paths = len(loaded_records) * 32
    null_answers = sum(1 for r in loaded_records for a in r.get("all_answers", []) if a is None)
    empty_answers = sum(1 for r in loaded_records for a in r.get("all_answers", []) if a == "")
    truncated_paths = sum(1 for r in loaded_records for t in r.get("truncated", []) if bool(t))

    stats = {
        "model": "qwen3_5_9b",
        "task": "gsm8k",
        "filename": filename_str,
        "filepath": filepath_str,
        "n_records": len(loaded_records),
        "n_seeds": len(seeds),
        "seeds": seeds,
        "n_instances": len(inst_ids),
        "total_paths": total_paths,
        "null_extractions": null_answers,
        "null_extraction_rate": null_answers / total_paths,
        "empty_extractions": empty_answers,
        "empty_extraction_rate": empty_answers / total_paths,
        "truncated_paths": truncated_paths,
        "truncated_rate": truncated_paths / total_paths,
    }
    return loaded_records, stats


def build_correctness_matrix(records: list[dict], task: str, max_k: int = 32) -> np.ndarray:
    """Build binary (N, max_k) correctness matrix using v2 scorers."""
    n = len(records)
    mat = np.zeros((n, max_k), dtype=float)
    for i, r in enumerate(records):
        gold = r.get("gold_answer", "")
        for j in range(max_k):
            ans = r["all_answers"][j]
            mat[i, j] = 1.0 if check_task_correct(ans, gold, task) else 0.0
    return mat


# ── Main Computation ──────────────────────────────────────────────────────────
def main():
    print("=== Diversity Combining: Tables 1, 2, 4, 9-11 and Figure 2 ===")
    numbers: dict = {}
    cell_reports: dict = {}

    # ─────────────────────────────────────────────────────────────────────────
    # 0. Load and validate all 5 canonical cells
    # ─────────────────────────────────────────────────────────────────────────
    cells_def = [
        ("qwen7b", "gsm8k", "qwen7b_sc_kvar_v2_gsm8k.jsonl", [42, 123, 456, 789, 1024]),
        ("llama8b", "gsm8k", "llama8b_sc_kvar_v2_gsm8k.jsonl", [42, 123, 456, 789, 1024]),
        ("mistral7b", "gsm8k", "mistral7b_sc_kvar_v2_gsm8k.jsonl", [42, 123, 456, 789, 1024]),
        ("llama8b", "hotpotqa", "llama8b_sc_kvar_v2_hotpotqa.jsonl", [42, 123, 456]),
        ("llama8b", "boolq", "llama8b_sc_kvar_v2_boolq.jsonl", [42, 123, 456]),
    ]

    loaded_cells = {}
    for mod, tsk, fn, sds in cells_def:
        recs, stats = load_cell_strict(mod, tsk, fn, sds)
        loaded_cells[(mod, tsk)] = (recs, stats)
        cell_reports[f"{mod}_{tsk}"] = stats
        print(f"Loaded {mod}/{tsk}: {len(recs)} records, null={stats['null_extractions']}, empty={stats['empty_extractions']}, trunc={stats['truncated_paths']}")

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Table 1: Diversity (tab:diversity, Qwen-7B GSM8K)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 1. tab:diversity (Qwen-7B GSM8K) ---")
    qwen_recs, _ = loaded_cells[("qwen7b", "gsm8k")]
    qwen_mat32 = build_correctness_matrix(qwen_recs, "gsm8k", 32)

    by_seed_qwen = defaultdict(list)
    for idx, r in enumerate(qwen_recs):
        by_seed_qwen[r["seed"]].append(idx)

    tab_diversity_rows = []
    tab_div_json = {}

    for K in [1, 4, 8, 16, 32]:
        sub_mat = qwen_mat32[:, :K]
        agr = mean_pairwise_agree(sub_mat) if K > 1 else None
        c = pairwise_rho(sub_mat) if K > 1 else None
        kef = keff(K, c) if (K > 1 and c is not None) else 1.0
        ceil = (1.0 / c) if (c is not None and c > 0) else None
        pct_ceil = (kef / ceil * 100.0) if (ceil is not None and ceil > 0) else None

        # Per-seed path accuracy mean +- std over 5 seeds
        per_seed_accs = []
        for s in sorted(by_seed_qwen.keys()):
            seed_row_indices = by_seed_qwen[s]
            s_mat = sub_mat[seed_row_indices]
            per_seed_accs.append(float(s_mat.mean()))
        p_mean = float(np.mean(per_seed_accs)) * 100.0
        p_std = float(np.std(per_seed_accs, ddof=1)) * 100.0 if len(per_seed_accs) > 1 else 0.0

        # Mean total generated tokens per instance (sum of gen_lens over K paths)
        tokens_mean = float(np.mean([sum(r["gen_lens"][:K]) for r in qwen_recs]))

        # Binary MV accuracy
        mv = mv_acc(sub_mat) * 100.0

        row_data = {
            "K": K,
            "Agree": agr,
            "c_hat": c,
            "Keff_vote": kef,
            "Ceiling": ceil,
            "pct_ceiling": pct_ceil,
            "p_bar_mean_pct": p_mean,
            "p_bar_std_pct": p_std,
            "mv_acc_pct": mv,
            "tokens": round(tokens_mean),
        }
        tab_diversity_rows.append(row_data)
        tab_div_json[f"K_{K}"] = row_data
        agr_str = f"{agr:.3f}" if agr is not None else "--"
        c_str = f"{c:.3f}" if c is not None else "--"
        ceil_str = f"{ceil:.2f}" if ceil is not None else "--"
        pct_str = f"{pct_ceil:.0f}%" if pct_ceil is not None else "--"
        print(f"K={K:2d}: Agree={agr_str}, c={c_str}, K_eff={kef:.2f}, Ceil={ceil_str}, %Ceil={pct_str}, p={p_mean:.1f}±{p_std:.1f}%, tokens={round(tokens_mean)}")

    numbers["tab_diversity"] = tab_div_json

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Table 2: Cross-Architecture (tab:cross_arch, 3 GSM8K models)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 2. tab:cross_arch (3 GSM8K models) ---")
    tab_cross_arch_json = {}
    tab_cross_arch_rows = []

    gsm8k_models = [
        ("Qwen2.5-7B", "qwen7b"),
        ("Llama-3.1-8B", "llama8b"),
        ("Mistral-7B", "mistral7b"),
    ]

    for label, mod_key in gsm8k_models:
        recs, _ = loaded_cells[(mod_key, "gsm8k")]
        mat32 = build_correctness_matrix(recs, "gsm8k", 32)

        c_per_k = {}
        keff_per_k = {}
        for K in [4, 8, 16, 32]:
            sub = mat32[:, :K]
            c_k = pairwise_rho(sub)
            c_per_k[K] = c_k
            keff_per_k[K] = keff(K, c_k)

        # In-sample predictions at K=32 (matching aggregate_canonical_sc.tab_cross_arch)
        p32 = float(mat32.mean())
        c32 = c_per_k[32]
        mv32_obs = mv_acc(mat32)
        bb32_pred = bb_mv_acc(32, p32, c32)
        bn32_pred = binom_mv_acc(32, p32)

        ceil32 = (1.0 / c32) if (c32 is not None and c32 > 0) else None
        pct_ceil32 = (keff_per_k[32] / ceil32 * 100.0) if (ceil32 is not None and ceil32 > 0) else None

        cell_data = {
            "model": label,
            "c_hat_k4": c_per_k[4],
            "c_hat_k8": c_per_k[8],
            "c_hat_k16": c_per_k[16],
            "c_hat_k32": c_per_k[32],
            "keff_k4": keff_per_k[4],
            "keff_k8": keff_per_k[8],
            "keff_k16": keff_per_k[16],
            "keff_k32": keff_per_k[32],
            "ceiling_k32": ceil32,
            "pct_ceiling_k32": pct_ceil32,
            "p_bar_32": p32,
            "mv_obs_32_pct": mv32_obs * 100.0,
            "bb_pred_32_pct": bb32_pred * 100.0,
            "binom_pred_32_pct": bn32_pred * 100.0,
            "bb_err_32_pp": abs(mv32_obs - bb32_pred) * 100.0,
            "binom_err_32_pp": abs(mv32_obs - bn32_pred) * 100.0,
        }
        tab_cross_arch_json[mod_key] = cell_data
        tab_cross_arch_rows.append(cell_data)
        print(f"{label}: c_hat_k4={c_per_k[4]:.3f}, c_hat_k32={c_per_k[32]:.3f}, K_eff@32={keff_per_k[32]:.2f}, MV@32={cell_data['mv_obs_32_pct']:.1f}%, BB={cell_data['bb_pred_32_pct']:.1f}%, Binom={cell_data['binom_pred_32_pct']:.1f}%")

    numbers["tab_cross_arch"] = tab_cross_arch_json

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Held-out BB Calibration (fig:bb_calibration + text)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 3. Held-out BB Calibration ---")
    heldout_bb_json = {}
    plot_data_per_model = {}

    for label, mod_key in gsm8k_models:
        recs, _ = loaded_cells[(mod_key, "gsm8k")]
        unique_insts = sorted(list({r["instance_id"] for r in recs}))
        m_insts = len(unique_insts)

        # Disjoint split matching aggregate_kvar_v2.py:539-551 (seed=2026, 50/50 instances)
        rng = np.random.default_rng(2026)
        shuffled_insts = list(unique_insts)
        rng.shuffle(shuffled_insts)
        split_idx = int(m_insts * 0.5)
        half_a = set(shuffled_insts[:split_idx])
        half_b = set(shuffled_insts[split_idx:])

        folds = [("A", half_a, half_b), ("B", half_b, half_a)]
        fold_results = {}

        # Accumulators for fold-average curve
        curve_obs = {K: [] for K in [4, 8, 16, 32]}
        curve_bb = {K: [] for K in [4, 8, 16, 32]}
        curve_bn = {K: [] for K in [4, 8, 16, 32]}

        for fold_label, pilot_insts, eval_insts in folds:
            pilot_recs = [r for r in recs if r["instance_id"] in pilot_insts]
            eval_recs = [r for r in recs if r["instance_id"] in eval_insts]

            # Fit (p, c) from K=4 prefix on pilot half
            pilot_mat4 = np.array([
                [1.0 if check_task_correct(a, r.get("gold_answer", ""), "gsm8k") else 0.0 for a in r["all_answers"][:4]]
                for r in pilot_recs
            ])
            p_pilot = float(pilot_mat4.mean())
            c_pilot = pairwise_rho(pilot_mat4)

            k_results = {}
            for K in [4, 8, 16, 32]:
                eval_matK = np.array([
                    [1.0 if check_task_correct(a, r.get("gold_answer", ""), "gsm8k") else 0.0 for a in r["all_answers"][:K]]
                    for r in eval_recs
                ])
                obs_mv = mv_acc(eval_matK)
                bb_pred = bb_mv_acc(K, p_pilot, c_pilot)
                bn_pred = binom_mv_acc(K, p_pilot)

                bb_err = abs(obs_mv - bb_pred) * 100.0
                bn_err = abs(obs_mv - bn_pred) * 100.0

                k_results[f"K_{K}"] = {
                    "obs_mv_pct": obs_mv * 100.0,
                    "bb_pred_pct": bb_pred * 100.0,
                    "binom_pred_pct": bn_pred * 100.0,
                    "bb_err_pp": bb_err,
                    "binom_err_pp": bn_err,
                }
                curve_obs[K].append(obs_mv)
                curve_bb[K].append(bb_pred)
                curve_bn[K].append(bn_pred)

            fold_results[f"fold_{fold_label}"] = {
                "pilot_p_bar": p_pilot,
                "pilot_c_hat": c_pilot,
                "ks": k_results,
            }

        # Fold-averaged errors at K=32
        fold_a_bb32 = fold_results["fold_A"]["ks"]["K_32"]["bb_err_pp"]
        fold_b_bb32 = fold_results["fold_B"]["ks"]["K_32"]["bb_err_pp"]
        fold_avg_bb32 = (fold_a_bb32 + fold_b_bb32) / 2.0

        fold_a_bn32 = fold_results["fold_A"]["ks"]["K_32"]["binom_err_pp"]
        fold_b_bn32 = fold_results["fold_B"]["ks"]["K_32"]["binom_err_pp"]
        fold_avg_bn32 = (fold_a_bn32 + fold_b_bn32) / 2.0

        heldout_bb_json[mod_key] = {
            "model": label,
            "fold_A": fold_results["fold_A"],
            "fold_B": fold_results["fold_B"],
            "fold_avg_bb_err_32_pp": fold_avg_bb32,
            "fold_avg_binom_err_32_pp": fold_avg_bn32,
        }

        # Store mean curve for plotting
        plot_data_per_model[label] = {
            "ks": [4, 8, 16, 32],
            "obs": [float(np.mean(curve_obs[K])) for K in [4, 8, 16, 32]],
            "bb": [float(np.mean(curve_bb[K])) for K in [4, 8, 16, 32]],
            "bn": [float(np.mean(curve_bn[K])) for K in [4, 8, 16, 32]],
        }
        print(f"{label}: Fold A BB@32 err={fold_a_bb32:.2f}pp, Fold B BB@32 err={fold_b_bb32:.2f}pp -> Fold Avg={fold_avg_bb32:.2f}pp")
        print(f"         Fold A Binom@32 err={fold_a_bn32:.2f}pp, Fold B Binom@32 err={fold_b_bn32:.2f}pp -> Fold Avg={fold_avg_bn32:.2f}pp")

    numbers["heldout_bb"] = heldout_bb_json

    # Plot bb_calibration_cr.pdf reproducing plot_bb_calibration.py visual style
    print("Generating bb_calibration_cr.pdf...")
    plt.rcParams.update({
        "font.size": 20, "axes.labelsize": 18, "xtick.labelsize": 16,
        "ytick.labelsize": 16, "legend.fontsize": 16, "lines.linewidth": 2.5,
        "lines.markersize": 10, "axes.linewidth": 1.5, "font.family": "sans-serif",
    })

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    for label in ["Qwen-7B", "Llama-8B", "Mistral-7B"]:
        m_key = "Qwen2.5-7B" if label == "Qwen-7B" else ("Llama-3.1-8B" if label == "Llama-8B" else "Mistral-7B")
        d = plot_data_per_model[m_key]
        ax.plot(d["ks"], d["bb"], color=COLORS[label], linestyle="-",
                marker=MARKERS[label], markersize=7, markeredgecolor="black",
                markeredgewidth=1.0, label=f"{label} BB")
        ax.plot(d["ks"], d["obs"], color=COLORS[label], linestyle="--",
                marker=MARKERS[label], markersize=7, markeredgecolor="black",
                markeredgewidth=1.0, markerfacecolor="white",
                label=f"{label} Obs")
        ax.plot(d["ks"], d["bn"], color=COLORS[label], linestyle=":",
                alpha=0.5, linewidth=2.0, label=f"{label} Binom")

    ax.set_xlabel("$K$ (number of paths)")
    ax.set_ylabel("MV Accuracy")
    ax.set_xticks([4, 8, 16, 32])
    ax.set_xlim(2, 34)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    style_handles = [
        Line2D([], [], color="#555555", linestyle="-", linewidth=3.0, label="BB pred."),
        Line2D([], [], color="#555555", linestyle="--", linewidth=3.0, label="Observed"),
        Line2D([], [], color="#555555", linestyle=":", linewidth=2.0, alpha=0.6, label="Binomial"),
    ]
    ax.legend(handles=style_handles, loc="lower right",
              framealpha=0.9, edgecolor="#cccccc", fontsize=11)

    fig.tight_layout()
    plot_pdf_path = OUT_DIR / "bb_calibration_cr.pdf"
    fig.savefig(plot_pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_pdf_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Table 4: Adaptive-K & 5. Amortized Accounting (all 5 cells)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 4. tab:adaptive_k & 5. Amortized Accounting ---")
    cells_order = [
        ("GSM8K (Math)", "Qwen-7B", "qwen7b", "gsm8k"),
        ("GSM8K (Math)", "Llama-8B", "llama8b", "gsm8k"),
        ("GSM8K (Math)", "Mistral-7B", "mistral7b", "gsm8k"),
        ("HotpotQA (QA)", "Llama-8B", "llama8b", "hotpotqa"),
        ("BoolQ (NLU)", "Llama-8B", "llama8b", "boolq"),
    ]

    adaptive_k_json = {}
    amortized_json = {}
    adaptive_k_rows = []

    rng_boot = np.random.default_rng(2026)

    for domain_task, model_name, mod_key, tsk_key in cells_order:
        recs, _ = loaded_cells[(mod_key, tsk_key)]
        mat32 = build_correctness_matrix(recs, tsk_key, 32)
        n_recs = len(recs)

        # Pilot = K=4 prefix on all pooled rows
        c_hat_k4 = pairwise_rho(mat32[:, :4])
        k_star = k_star_eq12(c_hat_k4, epsilon=0.025)

        mv_kstar = mv_acc(mat32[:, :k_star])
        mv_32 = mv_acc(mat32[:, :32])
        retained = (mv_kstar / mv_32) if mv_32 > 0 else None
        net_cost = (k_star + 4) / 32.0

        # Paired cluster bootstrap 95% CI for MV@K* - MV@32 (B=10000)
        by_inst = defaultdict(list)
        for idx, r in enumerate(recs):
            by_inst[r["instance_id"]].append(idx)
        unique_insts = sorted(by_inst.keys())
        m_clusters = len(unique_insts)

        B = 10000
        boot_diffs = []
        for _ in range(B):
            sampled_insts = rng_boot.choice(unique_insts, size=m_clusters, replace=True)
            row_idx = []
            for inst in sampled_insts:
                row_idx.extend(by_inst[inst])
            b_mat = mat32[row_idx]
            b_mv_kstar = mv_acc(b_mat[:, :k_star])
            b_mv_32 = mv_acc(b_mat[:, :32])
            boot_diffs.append(b_mv_kstar - b_mv_32)

        diff_pt = mv_kstar - mv_32
        ci_lower = float(np.percentile(boot_diffs, 2.5))
        ci_upper = float(np.percentile(boot_diffs, 97.5))

        # 5. Amortized accounting:
        # pilot = 4 paths * 100 instances = 400 paths
        # queries until repaid = ceil(400 / (32 - K*))
        # total cost at N=1000 queries = (400 + K* * N) / (32 * N)
        repaid_queries = math.ceil(400.0 / (32 - k_star)) if k_star < 32 else None
        cost_at_1000 = (400.0 + k_star * 1000.0) / 32000.0

        cell_res = {
            "task_label": domain_task,
            "model": model_name,
            "c_hat_k4": c_hat_k4,
            "k_star": k_star,
            "mv_kstar_pct": mv_kstar * 100.0,
            "mv_32_pct": mv_32 * 100.0,
            "retained_pct": (retained * 100.0) if retained is not None else None,
            "net_cost_pct": net_cost * 100.0,
            "diff_point_pp": diff_pt * 100.0,
            "ci_95_lower_pp": ci_lower * 100.0,
            "ci_95_upper_pp": ci_upper * 100.0,
        }
        adaptive_k_json[f"{mod_key}_{tsk_key}"] = cell_res
        adaptive_k_rows.append(cell_res)

        amort_res = {
            "task_label": domain_task,
            "model": model_name,
            "k_star": k_star,
            "repaid_queries": repaid_queries,
            "cost_at_1000_pct": cost_at_1000 * 100.0,
        }
        amortized_json[f"{mod_key}_{tsk_key}"] = amort_res

        ret_str = f"{retained*100:.1f}%" if retained is not None else "--"
        repaid_str = f"{repaid_queries} queries" if repaid_queries is not None else "never"
        print(f"{domain_task} {model_name}: c_hat={c_hat_k4:.3f}, K*={k_star:2d}, MV@K*={mv_kstar*100:.1f}%, MV@32={mv_32*100:.1f}%, Ret={ret_str}, NetCost={net_cost*100:.1f}%")
        print(f"   CI: diff={diff_pt*100:+.2f}pp, [{ci_lower*100:+.2f}pp, {ci_upper*100:+.2f}pp]")
        print(f"   Amortized: repaid in {repaid_str}, cost@1000={cost_at_1000*100:.2f}%")

    numbers["tab_adaptive_k"] = adaptive_k_json
    numbers["amortized_accounting"] = amortized_json

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Table 10: Epsilon Sensitivity (tab:eps_sensitivity)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 6. tab:eps_sensitivity ---")
    eps_grid = [0.01, 0.025, 0.05, 0.1]
    eps_json = {}
    eps_rows = []

    for label, mod_key in gsm8k_models:
        recs, _ = loaded_cells[(mod_key, "gsm8k")]
        mat32 = build_correctness_matrix(recs, "gsm8k", 32)
        c_hat_k4 = pairwise_rho(mat32[:, :4])
        mv_32 = mv_acc(mat32)

        eps_json[mod_key] = {"c_hat_k4": c_hat_k4, "mv_32_pct": mv_32 * 100.0, "grid": {}}
        for eps in eps_grid:
            k_star = k_star_eq12(c_hat_k4, epsilon=eps)
            mv_kstar = mv_acc(mat32[:, :k_star])
            retained = (mv_kstar / mv_32) if mv_32 > 0 else None
            net_cost = (k_star + 4) / 32.0

            r_data = {
                "model": label,
                "epsilon": eps,
                "k_star": k_star,
                "mv_kstar_pct": mv_kstar * 100.0,
                "retained_pct": (retained * 100.0) if retained is not None else None,
                "net_cost_pct": net_cost * 100.0,
            }
            eps_json[mod_key]["grid"][str(eps)] = r_data
            eps_rows.append(r_data)
            ret_str = f"{retained*100:.1f}%" if retained is not None else "--"
            print(f"{label} eps={eps:<5}: K*={k_star:2d}, MV@K*={mv_kstar*100:.1f}%, Ret={ret_str}")

    numbers["tab_eps_sensitivity"] = eps_json

    # ─────────────────────────────────────────────────────────────────────────
    # 7. Table 9: Pilot-Size Sensitivity (tab:pilot_sensitivity, clustered bootstrap)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 7. tab:pilot_sensitivity (clustered bootstrap) ---")
    n_grid = [25, 50, 100, 200]
    pilot_json = {}
    pilot_rows = []

    rng_pilot = np.random.default_rng(2026)
    B_pilot = 500

    boundary_hits_per_cell = defaultdict(dict)
    for label, mod_key in gsm8k_models:
        recs, _ = loaded_cells[(mod_key, "gsm8k")]
        by_inst = defaultdict(list)
        for r in recs:
            by_inst[r["instance_id"]].append(r)
        unique_insts = sorted(by_inst.keys())

        inst_mats4 = {}
        for inst, i_recs in by_inst.items():
            inst_mats4[inst] = np.array([
                [1.0 if check_task_correct(a, r.get("gold_answer", ""), "gsm8k") else 0.0 for a in r["all_answers"][:4]]
                for r in i_recs
            ])

        pilot_json[mod_key] = {}
        print(f"Model: {label}")
        for n in n_grid:
            c_boots = []
            kstar_boots = []
            boundary_hits = 0
            for b_idx in range(B_pilot):
                sampled_insts = rng_pilot.choice(unique_insts, size=n, replace=True)
                sampled_mat = np.concatenate([inst_mats4[inst] for inst in sampled_insts], axis=0)
                c_b = pairwise_rho(sampled_mat)
                if not math.isfinite(c_b):
                    raise ValueError(
                        f"FATAL: Bootstrap replicate produced non-finite c_hat={c_b} "
                        f"in cell {label}, pilot size n={n}, draw {b_idx + 1}/{B_pilot}."
                    )
                if c_b <= 0 or c_b >= 1:
                    boundary_hits += 1
                c_boots.append(c_b)
                k_star_b = k_star_eq12(c_b, epsilon=0.025)
                kstar_boots.append(k_star_b)

            assert len(c_boots) == B_pilot, f"FATAL: Expected {B_pilot} c draws, got {len(c_boots)}"
            assert len(kstar_boots) == B_pilot, f"FATAL: Expected {B_pilot} K* draws, got {len(kstar_boots)}"

            c_mean = float(np.mean(c_boots))
            c_std = float(np.std(c_boots, ddof=1))
            c_cv = (c_std / c_mean) * 100.0
            kstar_mean = float(np.mean(kstar_boots))
            kstar_std = float(np.std(kstar_boots, ddof=1))

            r_data = {
                "model": label,
                "n_pilot": n,
                "c_mean": c_mean,
                "c_std": c_std,
                "c_cv_pct": c_cv,
                "kstar_mean": kstar_mean,
                "kstar_std": kstar_std,
                "boundary_hits": boundary_hits,
            }
            pilot_json[mod_key][str(n)] = r_data
            pilot_rows.append(r_data)
            boundary_hits_per_cell[label][n] = boundary_hits
            print(f"  n={n:3d}: c_mean={c_mean:.3f}, c_std={c_std:.3f} (CV={c_cv:.1f}%), K*_mean={kstar_mean:.1f}, K*_std={kstar_std:.2f} (boundary={boundary_hits}/{B_pilot})")

    numbers["tab_pilot_sensitivity"] = pilot_json

    # ─────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 8. Reasoning model (Qwen3.5-9B, GSM8K) ---")
    e1_recs, e1_stats = load_e1_cell(E1_CACHE_PATH)

    e1_seed_counts = Counter(r["seed"] for r in e1_recs)
    e1_coverage = {}
    for s in sorted(e1_seed_counts.keys()):
        s_insts = sorted([r["instance_id"] for r in e1_recs if r["seed"] == s])
        missing = sorted(list(set(range(100)) - set(s_insts)))
        e1_coverage[str(s)] = {
            "count": len(s_insts),
            "expected": 100,
            "missing_instances": missing,
        }

    total_paths_e1 = len(e1_recs) * 32
    e1_null = e1_stats["null_extractions"]
    e1_empty = e1_stats["empty_extractions"]
    e1_trunc = e1_stats["truncated_paths"]

    # Build correctness matrix using scoring_v2 GSM8K scorer
    e1_mat32 = build_correctness_matrix(e1_recs, "gsm8k", 32)
    p_e1 = float(e1_mat32.mean())
    c4_e1 = pairwise_rho(e1_mat32[:, :4])
    c32_e1 = pairwise_rho(e1_mat32[:, :32])
    ceil_e1 = (1.0 / c32_e1) if (c32_e1 is not None and c32_e1 > 0) else None
    keff32_e1 = keff(32, c32_e1) if c32_e1 is not None else 1.0
    kstar_e1 = k_star_eq12(c4_e1, epsilon=0.025)
    mv_kstar_e1 = mv_acc(e1_mat32[:, :kstar_e1])
    mv_32_e1 = mv_acc(e1_mat32)
    ret_e1 = (mv_kstar_e1 / mv_32_e1) if mv_32_e1 > 0 else None
    cost_e1 = (kstar_e1 + 4) / 32.0

    mv_per_k_e1 = {
        f"K_{K}": mv_acc(e1_mat32[:, :K]) * 100.0
        for K in [4, 8, 16, 32]
    }

    # Held-out BB split (50/50 disjoint instance split, seed 2026, matching Section 3)
    unique_insts_e1 = sorted(list({r["instance_id"] for r in e1_recs}))
    rng_e1 = np.random.default_rng(2026)
    shuffled_e1 = list(unique_insts_e1)
    rng_e1.shuffle(shuffled_e1)
    split_idx_e1 = int(len(shuffled_e1) * 0.5)
    half_a_e1 = set(shuffled_e1[:split_idx_e1])
    half_b_e1 = set(shuffled_e1[split_idx_e1:])

    folds_e1 = [("fold_A", half_a_e1, half_b_e1), ("fold_B", half_b_e1, half_a_e1)]
    heldout_res_e1 = {}
    curve_bb_err_e1 = {K: [] for K in [4, 8, 16, 32]}
    curve_bn_err_e1 = {K: [] for K in [4, 8, 16, 32]}

    for f_label, pilot_insts, eval_insts in folds_e1:
        p_recs = [r for r in e1_recs if r["instance_id"] in pilot_insts]
        e_recs = [r for r in e1_recs if r["instance_id"] in eval_insts]

        p_mat4 = build_correctness_matrix(p_recs, "gsm8k", 4)
        p_pilot = float(p_mat4.mean())
        c_pilot = pairwise_rho(p_mat4)

        k_res = {}
        for K in [4, 8, 16, 32]:
            e_matK = build_correctness_matrix(e_recs, "gsm8k", K)
            obs_mv = mv_acc(e_matK)
            bb_pred = bb_mv_acc(K, p_pilot, c_pilot)
            bn_pred = binom_mv_acc(K, p_pilot)
            bb_err = abs(obs_mv - bb_pred) * 100.0
            bn_err = abs(obs_mv - bn_pred) * 100.0
            k_res[f"K_{K}"] = {
                "obs_mv_pct": obs_mv * 100.0,
                "bb_pred_pct": bb_pred * 100.0,
                "binom_pred_pct": bn_pred * 100.0,
                "bb_err_pp": bb_err,
                "binom_err_pp": bn_err,
            }
            curve_bb_err_e1[K].append(bb_err)
            curve_bn_err_e1[K].append(bn_err)

        heldout_res_e1[f_label] = {
            "pilot_p_bar": p_pilot,
            "pilot_c_hat": c_pilot,
            "ks": k_res,
        }

    fold_avg_bb_e1 = {f"K_{K}": float(np.mean(curve_bb_err_e1[K])) for K in [4, 8, 16, 32]}
    fold_avg_bn_e1 = {f"K_{K}": float(np.mean(curve_bn_err_e1[K])) for K in [4, 8, 16, 32]}

    # Paired cluster bootstrap 95% CI for MV@K* - MV@32 (B=10000, clustered by instance_id, matching Section 4)
    by_inst_e1 = defaultdict(list)
    for idx, r in enumerate(e1_recs):
        by_inst_e1[r["instance_id"]].append(idx)
    unique_insts_boot_e1 = sorted(by_inst_e1.keys())
    m_clusters_e1 = len(unique_insts_boot_e1)

    rng_boot_e1 = np.random.default_rng(2026)
    B_boot = 10000
    boot_diffs_e1 = []
    for _ in range(B_boot):
        sampled_insts = rng_boot_e1.choice(unique_insts_boot_e1, size=m_clusters_e1, replace=True)
        row_idx = []
        for inst in sampled_insts:
            row_idx.extend(by_inst_e1[inst])
        b_mat = e1_mat32[row_idx]
        b_mv_kstar = mv_acc(b_mat[:, :kstar_e1])
        b_mv_32 = mv_acc(b_mat[:, :32])
        boot_diffs_e1.append(b_mv_kstar - b_mv_32)

    diff_pt_e1 = mv_kstar_e1 - mv_32_e1
    ci_lower_e1 = float(np.percentile(boot_diffs_e1, 2.5))
    ci_upper_e1 = float(np.percentile(boot_diffs_e1, 97.5))

    e1_metrics = {
        "label": "FINAL",
        "seeds_evaluated": sorted(list({r["seed"] for r in e1_recs})),
        "n_records_evaluated": len(e1_recs),
        "p_bar": p_e1,
        "c_k4": c4_e1,
        "c_k32": c32_e1,
        "ceiling": ceil_e1,
        "keff_32": keff32_e1,
        "mv_obs_pct": mv_per_k_e1,
        "heldout_bb": {
            "fold_A": heldout_res_e1["fold_A"],
            "fold_B": heldout_res_e1["fold_B"],
            "fold_avg_bb_err_pp": fold_avg_bb_e1,
            "fold_avg_binom_err_pp": fold_avg_bn_e1,
        },
        "k_star": kstar_e1,
        "mv_kstar_pct": mv_kstar_e1 * 100.0,
        "mv_32_pct": mv_32_e1 * 100.0,
        "retention_pct": (ret_e1 * 100.0) if ret_e1 is not None else None,
        "net_cost_pct": cost_e1 * 100.0,
        "diff_point_pp": diff_pt_e1 * 100.0,
        "ci_95_lower_pp": ci_lower_e1 * 100.0,
        "ci_95_upper_pp": ci_upper_e1 * 100.0,
        "cap_hit_rate": e1_trunc / total_paths_e1,
        "cap_hit_paths": e1_trunc,
        "cap_hit_denominator_paths": total_paths_e1,
        "null_extractions": e1_null,
        "empty_extractions": e1_empty,
    }

    try:
        e1_rel_path = str(E1_CACHE_PATH.relative_to(PROJECT_ROOT))
    except ValueError:
        e1_rel_path = str(E1_CACHE_PATH)

    e1_json = {
        "filename": E1_CACHE_FILENAME,
        "cache_path": e1_rel_path,
        "total_records": len(e1_recs),
        "expected_records": len(e1_recs),
        "coverage_per_seed": e1_coverage,
        "null_extractions": e1_null,
        "empty_extractions": e1_empty,
        "truncated_paths": e1_trunc,
        "cap_hit_rate": e1_trunc / total_paths_e1,
        "status": "FINAL",
        "metrics": e1_metrics,
    }
    e1_json.update(e1_metrics)

    numbers["reasoning_model_e1"] = e1_json
    numbers["cell_summary"] = cell_reports

    print(f"FINAL E1: p_bar={p_e1:.3f}, c_k4={c4_e1:.3f}, c_k32={c32_e1:.3f}, K_eff@32={keff32_e1:.2f}, K*={kstar_e1}, MV@K*={mv_kstar_e1*100:.1f}%, Ret={ret_e1*100:.1f}%, NetCost={cost_e1*100:.1f}%")
    print(f"   CI: diff={diff_pt_e1*100:+.2f}pp, [{ci_lower_e1*100:+.2f}pp, {ci_upper_e1*100:+.2f}pp]")
    print(f"   Cap-hit: {e1_trunc}/{total_paths_e1} ({e1_trunc/total_paths_e1*100:.2f}%), Null: {e1_null}, Empty: {e1_empty}")

    # ─────────────────────────────────────────────────────────────────────────
    # Write numbers.json
    # ─────────────────────────────────────────────────────────────────────────
    numbers_path = OUT_DIR / "numbers.json"
    with open(numbers_path, "w", encoding="utf-8") as f:
        json.dump(numbers, f, indent=2, allow_nan=False)
    print(f"\nSaved numbers: {numbers_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    print("\nGenerating drop-in LaTeX rows...")

    # rows_diversity.tex (Table 1, 8 columns)
    # $K$ & Agree & $\hat{c}$ & $\Keff^{\text{vote}}$ & Ceiling & \% Ceil. & SC Acc (\%) $\uparrow$ & Tokens $\downarrow$ \\
    div_lines = []
    for r in tab_diversity_rows:
        k = r["K"]
        if k == 1:
            line = f"{k} & -- & -- & {r['Keff_vote']:.1f} & -- & -- & {r['p_bar_mean_pct']:.1f} $\\pm$ {r['p_bar_std_pct']:.1f} & {r['tokens']} \\\\"
        else:
            line = f"{k} & {r['Agree']:.3f} & {r['c_hat']:.3f} & {r['Keff_vote']:.2f} & {r['Ceiling']:.2f} & {r['pct_ceiling']:.0f}\\% & {r['p_bar_mean_pct']:.1f} $\\pm$ {r['p_bar_std_pct']:.1f} & {r['tokens']} \\\\"
        div_lines.append(line)
    with open(OUT_DIR / "rows_diversity.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(div_lines) + "\n")
    print(f"Saved: {OUT_DIR / 'rows_diversity.tex'}")

    # rows_cross_arch.tex (Table 2, 9 columns)
    # Model & $\hat{c}$ & 4 & 8 & 16 & 32 & MV@32 & BB & Binom \\
    ca_lines = [
        "% Rows for Table 2 (tab:cross_arch, 9 columns).",
        "% Column 2 shows c_hat at K=4 (0.60, 0.53, 0.45).",
        "% Per-K c_hat values:",
        f"%   Qwen2.5-7B:   c_hat(K=4)={tab_cross_arch_json['qwen7b']['c_hat_k4']:.3f}, c_hat(8)={tab_cross_arch_json['qwen7b']['c_hat_k8']:.3f}, c_hat(16)={tab_cross_arch_json['qwen7b']['c_hat_k16']:.3f}, c_hat(32)={tab_cross_arch_json['qwen7b']['c_hat_k32']:.3f}",
        f"%   Llama-3.1-8B: c_hat(K=4)={tab_cross_arch_json['llama8b']['c_hat_k4']:.3f}, c_hat(8)={tab_cross_arch_json['llama8b']['c_hat_k8']:.3f}, c_hat(16)={tab_cross_arch_json['llama8b']['c_hat_k16']:.3f}, c_hat(32)={tab_cross_arch_json['llama8b']['c_hat_k32']:.3f}",
        f"%   Mistral-7B:   c_hat(K=4)={tab_cross_arch_json['mistral7b']['c_hat_k4']:.3f}, c_hat(8)={tab_cross_arch_json['mistral7b']['c_hat_k8']:.3f}, c_hat(16)={tab_cross_arch_json['mistral7b']['c_hat_k16']:.3f}, c_hat(32)={tab_cross_arch_json['mistral7b']['c_hat_k32']:.3f}",
    ]
    for r in tab_cross_arch_rows:
        ca_lines.append(
            f"{r['model']:<12} & {r['c_hat_k4']:.2f} & {r['keff_k4']:.2f} & {r['keff_k8']:.2f} & {r['keff_k16']:.2f} & {r['keff_k32']:.2f} & {r['mv_obs_32_pct']:.1f}\\% & {r['bb_pred_32_pct']:.1f}\\% & {r['binom_pred_32_pct']:5.1f}\\% \\\\"
        )
    with open(OUT_DIR / "rows_cross_arch.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(ca_lines) + "\n")
    print(f"Saved: {OUT_DIR / 'rows_cross_arch.tex'}")

    # rows_adaptive_k.tex (Table 4, 8 columns)
    # Task & Model & $\hat{c}_{K{=}4}$ & $K^*$ & MV@$K^*$ & MV@32 & Retained & Net cost \\
    ak_lines = []
    for idx, r in enumerate(adaptive_k_rows):
        dag = "$^\\dagger$" if r["model"] == "Mistral-7B" and r["task_label"].startswith("GSM8K") else ""
        ak_lines.append(
            f"{r['task_label']:<16} & {r['model']:<10} & {r['c_hat_k4']:.2f} & {r['k_star']:2d} & {r['mv_kstar_pct']:.1f}\\% & {r['mv_32_pct']:.1f}\\% & {r['retained_pct']:.0f}\\%{dag:<10} & {r['net_cost_pct']:.0f}\\% \\\\"
        )
        if idx == 2:
            ak_lines.append("\\midrule")
    with open(OUT_DIR / "rows_adaptive_k.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(ak_lines) + "\n")
    print(f"Saved: {OUT_DIR / 'rows_adaptive_k.tex'}")

    # rows_eps_sensitivity.tex (Table 10, 5 columns)
    # Model & $\varepsilon$ & $K^*$ & MV@$K^*$ & Retained \\
    eps_lines = []
    for mod_idx, (label, mod_key) in enumerate(gsm8k_models):
        grid = eps_json[mod_key]["grid"]
        for g_idx, eps in enumerate(eps_grid):
            r = grid[str(eps)]
            mod_prefix = f"\\multirow{{4}}{{*}}{{{label}}}" if g_idx == 0 else ""
            eps_lines.append(
                f"{mod_prefix:<22} & {eps:<5} & {r['k_star']:2d} & {r['mv_kstar_pct']:.1f}\\% & {r['retained_pct']:.0f}\\% \\\\"
            )
        if mod_idx < len(gsm8k_models) - 1:
            eps_lines.append("\\midrule")
    with open(OUT_DIR / "rows_eps_sensitivity.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(eps_lines) + "\n")
    print(f"Saved: {OUT_DIR / 'rows_eps_sensitivity.tex'}")

    # rows_pilot_sensitivity.tex (Table 9, 6 columns)
    # Model & $n$ & $\hat{c}$ mean & $\hat{c}$ std & $K^*$ mean & $K^*$ std \\
    pilot_lines = []
    for mod_idx, (label, mod_key) in enumerate(gsm8k_models):
        grid = pilot_json[mod_key]
        for g_idx, n in enumerate(n_grid):
            r = grid[str(n)]
            mod_prefix = f"\\multirow{{4}}{{*}}{{{label}}}" if g_idx == 0 else ""
            pilot_lines.append(
                f"{mod_prefix:<22} & {n:<3} & {r['c_mean']:.3f} & {r['c_std']:.3f} & {r['kstar_mean']:4.1f} & {r['kstar_std']:4.2f} \\\\"
            )
        if mod_idx < len(gsm8k_models) - 1:
            pilot_lines.append("\\midrule")
    with open(OUT_DIR / "rows_pilot_sensitivity.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(pilot_lines) + "\n")
    print(f"Saved: {OUT_DIR / 'rows_pilot_sensitivity.tex'}")

    print("\n=== Recompute Complete. All files successfully written. ===")


if __name__ == "__main__":
    main()
