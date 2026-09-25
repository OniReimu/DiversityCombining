#!/usr/bin/env python3
"""Reproduce the cross-benchmark prompt-template matrix (Table 3) and its derived numbers.

Reads the capability-matrix records (5 models x 12 benchmarks, K=8, self-consistency
and prompt-template arms) from experiment_dir("cache"), scores GSM8K with the numeric
scorer and MATH with the MATH-500 canonical scorer from scripts/scoring_v2.py, and
writes numbers.json and rows_table3.tex to experiment_dir("cross_benchmark").

- Estimators are imported from scripts.aggregate_5seed and scripts.analyze_mv_vs_weighted.
- Scorers are imported from scripts.scoring_v2.
- Data shape anomalies raise immediately.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
_SRC = PROJECT_ROOT
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ── Import estimators from scripts.aggregate_5seed (Hard rule: never copy) ─────
from scripts.aggregate_5seed import (
    BASE_ACC_THRESHOLD,
    BENCHMARK_FILES,
    DOMAIN_ORDER,
    MODEL_LABELS,
    MODELS,
    bootstrap_pooled_drho_ci,
    compute_mv_accuracy,
    compute_pairwise_rho,
    k_eff,
    load_records,
)

# ── Import estimators from scripts.analyze_mv_vs_weighted ───────────────────────
from scripts.analyze_mv_vs_weighted import (
    mv_accuracy as wmv_mv_acc,
    slot_cv,
    weighted_mv_accuracy,
)

# ── Import scorers from scripts.scoring_v2 (Hard rule: never copy) ─────────────
from scripts.scoring_v2 import GSM8KScorer, MathScorer

from diversity_combining.config import experiment_dir
OUTPUT_DIR = experiment_dir("cross_benchmark")

# Task answer space labels as printed in Table 3
TASK_ANSWER_SPACE = {
    "triviaqa": "open text",
    "hotpotqa": "open text",
    "arc_challenge": "bounded (4)",
    "mmlu": "bounded (4)",
    "hellaswag": "bounded (4)",
    "winogrande": "binary (2)",
    "drop": "open text",
    "boolq": "binary (y/n)",
    "mbpp": "code pass/fail",
    "cruxeval": "code pass/fail",
    "gsm8k": "numeric",
    "math": "numeric",
}

TASK_DISPLAY_NAMES = {
    "triviaqa": "TriviaQA",
    "hotpotqa": "HotpotQA",
    "arc_challenge": "ARC-C",
    "mmlu": "MMLU",
    "hellaswag": "HellaSwag",
    "winogrande": "WinoGrande",
    "drop": "DROP$^\\dagger$",
    "boolq": "BoolQ",
    "mbpp": "MBPP",
    "cruxeval": "CruxEval",
    "gsm8k": "GSM8K",
    "math": "MATH",
}

BENCHMARK_ORDER_TABLE3 = [
    ("QA", ["triviaqa", "hotpotqa"]),
    ("Science/MC", ["arc_challenge", "mmlu"]),
    ("Commonsense", ["hellaswag", "winogrande"]),
    ("NLU", ["drop", "boolq"]),
    ("Code", ["mbpp", "cruxeval"]),
    ("Math", ["gsm8k", "math"]),
]

EXPECTED_SEEDS = [42, 123, 456, 789, 1024]
INSTANCES_PER_ARM = 50

# Exact production shape per arm: {seed: record count}. A full arm is 5 seeds x 50
# and pooled as-is by aggregate_5seed.py (5seed_summary.csv N_pooled = SC total).
FULL_ARM_SEED_COUNTS = {s: INSTANCES_PER_ARM for s in EXPECTED_SEEDS}
PARTIAL_ARM_SEED_COUNTS = {
    ("qwen7b", "math", "sc"): {42: 50, 123: 50, 456: 50, 789: 50, 1024: 26},   # N=226
    ("qwen7b", "math", "pt"): {42: 50, 123: 50},                               # N=100
    ("qwen32b", "math", "sc"): {42: 50, 123: 50, 456: 50, 789: 50, 1024: 35},  # N=235
    ("qwen32b", "math", "pt"): {42: 50, 123: 50},                              # N=100
    ("qwen7b", "mmlu", "pt"): {42: 50, 123: 50, 456: 24},                      # N=124
    ("qwen32b", "mmlu", "pt"): {42: 50, 123: 50, 456: 50, 789: 8},             # N=158
    ("qwen7b", "mbpp", "pt"): {42: 50, 123: 50, 456: 50, 789: 50, 1024: 23},   # N=223
    ("llama8b", "math", "pt"): {42: 50, 123: 50, 456: 50, 789: 50, 1024: 1},   # N=201
}


def expected_arm_seed_counts(model: str, task: str, arm: str) -> Dict[int, int]:
    """Expected {seed: record count} for one arm ("sc" or "pt") of one cell."""
    return PARTIAL_ARM_SEED_COUNTS.get((model, task, arm), FULL_ARM_SEED_COUNTS)


def assert_arm_shape(model: str, task: str, arm: str, recs: List[Dict[str, Any]]) -> None:
    """Fail unless the arm has exactly the production seeds, per-seed N and instances."""
    counts: Dict[int, int] = {}
    for r in recs:
        counts[r["seed"]] = counts.get(r["seed"], 0) + 1
    counts = dict(sorted(counts.items()))
    expected = expected_arm_seed_counts(model, task, arm)
    assert counts == expected, (
        f"{model}/{task}/{arm}: seed counts {counts} != production shape {expected}"
    )
    n_inst = len({r["instance_id"] for r in recs})
    assert n_inst == INSTANCES_PER_ARM, (
        f"{model}/{task}/{arm}: {n_inst} distinct instances != {INSTANCES_PER_ARM}"
    )
    n_keys = len({(r["seed"], r["instance_id"]) for r in recs})
    assert n_keys == len(recs), (
        f"{model}/{task}/{arm}: {len(recs) - n_keys} duplicate (seed, instance_id) records"
    )


def fmt_num(
    val: Any,
    digits: int,
    *,
    sign: bool = False,
    scale: float = 1.0,
    sci: bool = False,
    below: Optional[float] = None,
) -> str:
    """The single rounding helper for printed numbers.

    below=t renders the inequality "<t" or "≥t" (t in 1e-NN form) instead of a value.
    Undefined values are an error: nothing undefined may be printed as a number.
    """
    if val is None or isinstance(val, bool):
        raise ValueError(f"fmt_num got non-numeric value {val!r}")
    v = float(val) * scale
    if not math.isfinite(v):
        raise ValueError(f"fmt_num got non-finite value {v!r}")
    if below is not None:
        return f"<{below:.0e}" if v < below else f"≥{below:.0e}"
    spec = ("+" if sign else "") + f".{digits}" + ("e" if sci else "f")
    return format(v, spec)


def to_json_safe(obj: Any) -> Any:
    """Recursively convert NaN/Inf to None and numpy scalars to Python scalars."""
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    return obj


def get_scorers() -> Tuple[GSM8KScorer, MathScorer]:
    """Instantiate and validate the canonical scoring_v2 scorers."""
    gsm = GSM8KScorer()
    math_sc = MathScorer()
    assert isinstance(gsm, GSM8KScorer), "gsm must be instance of GSM8KScorer"
    assert isinstance(math_sc, MathScorer), "math_sc must be instance of MathScorer"
    assert gsm.name == "gsm8k_numeric", f"Unexpected GSM8K scorer name: {gsm.name}"
    assert math_sc.name == "math500_canonical_exact", f"Unexpected MATH scorer name: {math_sc.name}"
    return gsm, math_sc


def extract_correctness_vector(
    record: Dict[str, Any],
    task: str,
    gsm_scorer: GSM8KScorer,
    math_scorer: MathScorer,
) -> List[int]:
    """Extract binary correctness vector from record.

    GSM8K answers are scored with gsm8k_numeric and MATH answers with
    math500_canonical_exact. Other tasks use the stored correctness labels or
    F1 values when present, otherwise normalized exact text equality.
    """
    if "all_correct" in record:
        return [int(c) for c in record["all_correct"]]
    if "all_f1s" in record:
        return [1 if f >= 0.5 else 0 for f in record["all_f1s"]]
    if "all_answers" in record and "gold_answer" in record:
        gold = record["gold_answer"]
        answers = record["all_answers"]
        if task == "gsm8k":
            return [int(bool(gsm_scorer(a, gold))) for a in answers]
        elif task == "math":
            return [int(bool(math_scorer(a, gold))) for a in answers]
        else:
            gold_str = str(gold).strip().lower()
            return [1 if str(a).strip().lower() == gold_str else 0 for a in answers]

    raise ValueError(f"Record contains no usable answers or labels: {record}")


def build_pooled_matrix(
    records: List[Dict[str, Any]],
    task: str,
    gsm_scorer: GSM8KScorer,
    math_scorer: MathScorer,
) -> np.ndarray:
    """Build pooled (N, K) correctness matrix sorted by instance_id."""
    records_sorted = sorted(records, key=lambda r: r.get("instance_id", 0))
    rows = []
    for r in records_sorted:
        v = extract_correctness_vector(r, task, gsm_scorer, math_scorer)
        rows.append(v)
    mat = np.array(rows, dtype=float)
    assert mat.ndim == 2 and mat.shape[1] == 8, f"Malformed matrix shape: {mat.shape}"
    return mat


def build_block_covariance_matrix(
    records: List[Dict[str, Any]],
    task: str,
    gsm_scorer: GSM8KScorer,
    math_scorer: MathScorer,
) -> Optional[np.ndarray]:
    """Build PT matrix matching analyze_block_covariance.py, with scoring_v2 for GSM8K and MATH."""
    rows = []
    for r in records:
        if "all_correct" in r and r["all_correct"]:
            rows.append(r["all_correct"])
        elif "all_answers" in r and "gold_answer" in r:
            if task == "gsm8k":
                rows.append([int(bool(gsm_scorer(a, r["gold_answer"]))) for a in r["all_answers"]])
            elif task == "math":
                rows.append([int(bool(math_scorer(a, r["gold_answer"]))) for a in r["all_answers"]])
            else:
                g = str(r["gold_answer"]).strip().lower()
                rows.append([1 if str(a).strip().lower() == g else 0 for a in r["all_answers"]])
    if not rows:
        return None
    mat = np.array(rows, dtype=float)
    return mat if mat.shape[0] >= 20 and mat.mean() >= BASE_ACC_THRESHOLD else None


def run_table3_pipeline(
    n_boot: int = 10_000,
    rng_seed: int = 2026,
) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Dict[Tuple[str, str], np.ndarray]]:
    """Run the 5-seed pooled matrix pipeline across all 60 cells.

    Returns:
      cells_data: (model, task) -> metrics dict
      pt_matrices: (model, task) -> PT correctness matrix (for secondary analyses)
    """
    gsm_scorer, math_scorer = get_scorers()
    rng = np.random.default_rng(rng_seed)

    cells_data = {}
    pt_matrices = {}

    for model in MODELS:
        for suffix, task_map in BENCHMARK_FILES.items():
            records = load_records(model, suffix)
            assert records, f"Failed to load records for {model}{suffix}"
            for task, domain in task_map.items():
                t_recs = [r for r in records if r.get("task") == task]
                assert t_recs, f"No records found for {model} on {task}"

                sc_recs = [r for r in t_recs if r.get("method") == "sc"]
                pt_recs = [r for r in t_recs if r.get("method") == "sc_prompttpl"]

                assert sc_recs, f"No SC records for {model} on {task}"
                assert pt_recs, f"No PT records for {model} on {task}"

                # Fail loudly on seed / per-seed N / instance counts
                assert_arm_shape(model, task, "sc", sc_recs)
                assert_arm_shape(model, task, "pt", pt_recs)
                sc_seeds = sorted(set(r["seed"] for r in sc_recs))
                pt_seeds = sorted(set(r["seed"] for r in pt_recs))

                mat_sc = build_pooled_matrix(sc_recs, task, gsm_scorer, math_scorer)
                mat_pt = build_pooled_matrix(pt_recs, task, gsm_scorer, math_scorer)

                p_bar_sc = float(mat_sc.mean())
                p_bar_pt = float(mat_pt.mean())
                rho_sc = compute_pairwise_rho(mat_sc)
                rho_pt = compute_pairwise_rho(mat_pt)
                K = mat_sc.shape[1]
                keff_sc = k_eff(K, rho_sc)
                keff_pt = k_eff(K, rho_pt)
                mv_acc_sc = compute_mv_accuracy(mat_sc)
                mv_acc_pt = compute_mv_accuracy(mat_pt)
                n_pooled = int(mat_sc.shape[0])

                if not np.isnan(rho_sc) and abs(rho_sc) > 0.01:
                    drho_pct = (rho_pt - rho_sc) / abs(rho_sc) * 100.0
                else:
                    drho_pct = float("nan")
                dkeff = keff_pt - keff_sc

                # Base accuracy exclusion rule
                excluded = bool(np.isnan(p_bar_sc) or p_bar_sc < BASE_ACC_THRESHOLD)

                # Bootstrap CI
                if n_boot > 0 and not excluded:
                    ci_lo, ci_hi = bootstrap_pooled_drho_ci(mat_sc, mat_pt, n_boot, rng)
                else:
                    ci_lo, ci_hi = float("nan"), float("nan")

                cells_data[(model, task)] = {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "task": task,
                    "domain": domain,
                    "n_seeds_sc": len(sc_seeds),
                    "n_seeds_pt": len(pt_seeds),
                    "N_pooled": n_pooled,
                    "N_pt": int(mat_pt.shape[0]),
                    "K": K,
                    "p_bar_sc_pooled": p_bar_sc,
                    "p_bar_pt_pooled": p_bar_pt,
                    "rho_sc_pooled": rho_sc,
                    "rho_pt_pooled": rho_pt,
                    "keff_sc_pooled": keff_sc,
                    "keff_pt_pooled": keff_pt,
                    "mv_acc_sc_pooled": mv_acc_sc,
                    "mv_acc_pt_pooled": mv_acc_pt,
                    "drho_pct_pooled": drho_pct,
                    "dkeff_pooled": dkeff,
                    "drho_ci_low": ci_lo,
                    "drho_ci_high": ci_hi,
                    "excluded_base_acc": excluded,
                }
                pt_matrices[(model, task)] = mat_pt

    assert len(cells_data) == 60, f"Expected 60 cells, got {len(cells_data)}"
    return cells_data, pt_matrices


def compute_derived_numbers(
    cells: Dict[Tuple[str, str], Dict[str, Any]],
    pt_matrices: Dict[Tuple[str, str], np.ndarray],
) -> Dict[str, Any]:
    """Compute every derived number in the paper from the matrix."""
    valid_cells = [c for c in cells.values() if not c["excluded_base_acc"]]
    assert len(valid_cells) == 57, f"Expected 57 valid cells, got {len(valid_cells)}"

    drho_vals = np.array([c["drho_pct_pooled"] for c in valid_cells])
    dkeff_vals = np.array([c["dkeff_pooled"] for c in valid_cells])

    # 1. Headline valid counts and means
    decorrelates_count = int((drho_vals < 0).sum())
    mean_drho = float(drho_vals.mean())
    mean_dkeff = float(dkeff_vals.mean())

    # Exceptions (Delta rho >= 0)
    exceptions = [c for c in valid_cells if c["drho_pct_pooled"] >= 0]
    exceptions_summary = [
        {
            "model": c["model"],
            "model_label": c["model_label"],
            "task": c["task"],
            "drho_pct": c["drho_pct_pooled"],
            "p_bar_sc": c["p_bar_sc_pooled"],
        }
        for c in exceptions
    ]

    # 2. Table 3 benchmark rows
    table3_rows = {}
    for dom, tasks in BENCHMARK_ORDER_TABLE3:
        for task in tasks:
            task_cells = [c for c in valid_cells if c["task"] == task]
            n_models = len(task_cells)
            mean_drho_task = float(np.mean([c["drho_pct_pooled"] for c in task_cells]))
            mean_dkeff_task = float(np.mean([c["dkeff_pooled"] for c in task_cells]))
            mean_rho_sc_task = float(np.mean([c["rho_sc_pooled"] for c in task_cells]))
            mean_rho_pt_task = float(np.mean([c["rho_pt_pooled"] for c in task_cells]))

            table3_rows[task] = {
                "domain": dom,
                "benchmark": task,
                "display_name": TASK_DISPLAY_NAMES[task],
                "answer_space": TASK_ANSWER_SPACE[task],
                "n_models": n_models,
                "mean_drho_pct": mean_drho_task,
                "mean_dkeff": mean_dkeff_task,
                "mean_rho_sc": mean_rho_sc_task,
                "mean_rho_pt": mean_rho_pt_task,
            }

    # 3. Domain means
    domain_means = {}
    for dom in ["QA", "Science/MC", "Commonsense", "NLU/Reasoning", "Code", "Math"]:
        dom_cells = [c for c in valid_cells if c["domain"] == dom]
        drhos = [c["drho_pct_pooled"] for c in dom_cells]
        domain_means[dom] = {
            "n_cells": len(dom_cells),
            "mean_drho_pct": float(np.mean(drhos)),
            "min_drho_pct": float(np.min(drhos)),
            "max_drho_pct": float(np.max(drhos)),
        }

    # 4. Mann-Whitney test: Math vs QA
    math_drhos = [c["drho_pct_pooled"] for c in valid_cells if c["domain"] == "Math"]
    qa_drhos = [c["drho_pct_pooled"] for c in valid_cells if c["domain"] == "QA"]
    assert len(math_drhos) == 10 and len(qa_drhos) == 10
    mw_res = stats.mannwhitneyu(math_drhos, qa_drhos, alternative="greater")
    mann_whitney = {
        "n_math": len(math_drhos),
        "n_qa": len(qa_drhos),
        "u_statistic": float(mw_res.statistic),
        "p_value": float(mw_res.pvalue),
    }

    # 5. Bootstrap CI analysis (Appendix app:statistical_precision)
    ci_exclude_zero = [
        c for c in valid_cells if (c["drho_ci_high"] < 0 or c["drho_ci_low"] > 0)
    ]
    ci_widths = [c["drho_ci_high"] - c["drho_ci_low"] for c in valid_cells]
    qwen32b_math = cells[("qwen32b", "math")]

    bootstrap_stats = {
        "cells_excluding_zero_count": len(ci_exclude_zero),
        "ci_width_median": float(np.median(ci_widths)),
        "ci_width_iqr_25": float(np.percentile(ci_widths, 25)),
        "ci_width_iqr_75": float(np.percentile(ci_widths, 75)),
        "drho_min": float(np.min(drho_vals)),
        "drho_max": float(np.max(drho_vals)),
        "qwen32b_math_drho": qwen32b_math["drho_pct_pooled"],
        "qwen32b_math_ci_low": qwen32b_math["drho_ci_low"],
        "qwen32b_math_ci_high": qwen32b_math["drho_ci_high"],
    }

    # 6. Table 8: Per-domain MV accuracy deltas (tab:pt_acc_delta)
    table8_rows = {}
    for dom in ["QA", "Science/MC", "Commonsense", "NLU/Reasoning", "Code", "Math"]:
        dom_cells = [c for c in valid_cells if c["domain"] == dom]
        mv_sc = [c["mv_acc_sc_pooled"] * 100.0 for c in dom_cells]
        mv_pt = [c["mv_acc_pt_pooled"] * 100.0 for c in dom_cells]
        delta_mv = [(c["mv_acc_pt_pooled"] - c["mv_acc_sc_pooled"]) * 100.0 for c in dom_cells]
        table8_rows[dom] = {
            "n_cells": len(dom_cells),
            "mv_sc_mean": float(np.mean(mv_sc)),
            "mv_pt_mean": float(np.mean(mv_pt)),
            "delta_acc_mean": float(np.mean(delta_mv)),
            "delta_acc_std": float(np.std(delta_mv, ddof=1)),
        }

    # 7. Entropy-predictor figure correlation (Fig. 4, line 563)
    bench_diversity = {}
    for model in MODELS:
        for suffix, task_map in BENCHMARK_FILES.items():
            records = load_records(model, suffix)
            for task_name in task_map:
                sc_recs = [r for r in records if r.get("task") == task_name and r["method"] == "sc"]
                for r in sc_recs:
                    answers = r.get("all_answers", [])
                    if not answers:
                        correct = r.get("all_correct", [])
                        if correct:
                            answers = [str(c) for c in correct]
                    if not answers:
                        continue
                    K = len(answers)
                    n_unique = len(set(str(a).strip().lower() for a in answers))
                    bench_diversity.setdefault(task_name, []).append(n_unique / K)

    tasks_sorted = sorted(bench_diversity.keys())
    assert len(tasks_sorted) == 12
    xs = [float(np.mean(bench_diversity[t])) for t in tasks_sorted]
    ys = [table3_rows[t]["mean_drho_pct"] for t in tasks_sorted]
    r_entropy, p_entropy = stats.pearsonr(xs, ys)
    entropy_predictor = {
        "pearson_r": float(r_entropy),
        "p_value": float(p_entropy),
        "n_points": len(tasks_sorted),
    }

    # 8. Accuracy-weighted voting across all 60 PT cells (Appendix app:hetero_slot)
    wmv_results = []
    for (model, task), mat in pt_matrices.items():
        if mat is None or mat.shape[0] < 20 or mat.mean() < BASE_ACC_THRESHOLD:
            continue
        mv = wmv_mv_acc(mat)
        wmv = weighted_mv_accuracy(mat)
        cv = slot_cv(mat)
        wmv_results.append({
            "model": model,
            "task": task,
            "mv": mv,
            "wmv": wmv,
            "cv": cv,
            "gap": wmv - mv,
        })

    assert len(wmv_results) == 60, f"Expected 60 PT cells, got {len(wmv_results)}"
    cvs = [r["cv"] for r in wmv_results]
    gaps = [r["gap"] for r in wmv_results]
    r_wmv, p_wmv = stats.pearsonr(cvs, gaps)

    high_cv = [r for r in wmv_results if r["cv"] > 0.225]
    low_cv = [r for r in wmv_results if r["cv"] <= 0.225]
    high_mv = [r for r in wmv_results if r["mv"] >= 0.30]
    low_mv = [r for r in wmv_results if r["mv"] < 0.30]

    weighted_mv_stats = {
        "n_cells": len(wmv_results),
        "pearson_r": float(r_wmv),
        "p_value": float(p_wmv),
        "high_cv_mean_gap_pp": float(np.mean([r["gap"] for r in high_cv]) * 100.0),
        "low_cv_mean_gap_pp": float(np.mean([r["gap"] for r in low_cv]) * 100.0),
        "high_mv_mean_gap_pp": float(np.mean([r["gap"] for r in high_mv]) * 100.0),
        "high_mv_mean_cv": float(np.mean([r["cv"] for r in high_mv])),
        "low_mv_mean_gap_pp": float(np.mean([r["gap"] for r in low_mv]) * 100.0),
        "low_mv_mean_cv": float(np.mean([r["cv"] for r in low_mv])),
        "low_mv_with_high_cv_pct": float(sum(1 for r in low_mv if r["cv"] > 0.225) / len(low_mv) * 100.0),
        "high_mv_with_high_cv_pct": float(sum(1 for r in high_mv if r["cv"] > 0.225) / len(high_mv) * 100.0),
    }

    # 9. Block covariance analysis over valid PT cells (Appendix J, line 882)
    # Uses build_block_covariance_matrix matching analyze_block_covariance.py
    ratios = []
    t_qwen7b_ratio = None
    gsm_sc, math_sc = get_scorers()

    for model in MODELS:
        for suffix, task_map in BENCHMARK_FILES.items():
            records = load_records(model, suffix)
            for task_name in task_map:
                pt_recs = [r for r in records if r.get("task") == task_name and r["method"] == "sc_prompttpl"]
                mat = build_block_covariance_matrix(pt_recs, task_name, gsm_sc, math_sc)
                if mat is None:
                    continue
                N, K = mat.shape
                corr_matrix = np.corrcoef(mat.T)
                np.fill_diagonal(corr_matrix, 0)
                valid_pairs = ~np.isnan(corr_matrix)
                if valid_pairs.sum() == 0:
                    continue
                c_mean = float(np.nanmean(corr_matrix[np.triu_indices(K, k=1)]))
                keff_eq = K / (1.0 + (K - 1.0) * c_mean) if c_mean > -1.0 / (K - 1.0) else float("nan")
                p_bar = float(mat.mean())
                S_K = mat.sum(axis=1)
                var_sk_over_k = float(np.var(S_K / K, ddof=1))
                var_indep = p_bar * (1.0 - p_bar) / K
                keff_blk = var_indep / var_sk_over_k * K if var_sk_over_k > 0 else float("nan")

                if not np.isnan(keff_eq) and not np.isnan(keff_blk) and keff_eq > 0:
                    ratio = keff_blk / keff_eq
                    ratios.append(ratio)
                    if task_name == "triviaqa" and model == "qwen7b":
                        t_qwen7b_ratio = ratio

    assert len(ratios) == 58, f"Expected 58 valid PT cells for block cov, got {len(ratios)}"
    block_covariance_stats = {
        "n_cells": len(ratios),
        "mean_ratio": float(np.mean(ratios)),
        "std_ratio": float(np.std(ratios, ddof=0)),  # population std matching numpy/paper
        "triviaqa_qwen7b_ratio": float(t_qwen7b_ratio) if t_qwen7b_ratio is not None else float("nan"),
    }

    return {
        "valid_cells_count": len(valid_cells),
        "decorrelates_count": decorrelates_count,
        "mean_drho_pct": mean_drho,
        "mean_dkeff": mean_dkeff,
        "exceptions": exceptions_summary,
        "table3_rows": table3_rows,
        "domain_means": domain_means,
        "mann_whitney": mann_whitney,
        "bootstrap_stats": bootstrap_stats,
        "table8_rows": table8_rows,
        "entropy_predictor": entropy_predictor,
        "weighted_mv_stats": weighted_mv_stats,
        "block_covariance_stats": block_covariance_stats,
    }


def format_table3_latex(table3_rows: Dict[str, Any]) -> str:
    """Format LaTeX rows for Table 3."""
    lines = []
    for i, (domain, tasks) in enumerate(BENCHMARK_ORDER_TABLE3):
        if i > 0:
            lines.append(r"\midrule")
        lines.append(f"\\multirow{{{len(tasks)}}}{{*}}{{{domain}}}")
        for task in tasks:
            row = table3_rows[task]
            name = row["display_name"]
            ans_space = row["answer_space"]
            n = row["n_models"]
            drho = row["mean_drho_pct"]
            dkeff = row["mean_dkeff"]
            r_sc = row["mean_rho_sc"]
            r_pt = row["mean_rho_pt"]

            # Match paper's sign and phantom spacing
            if drho < 0:
                if abs(drho) < 10.0:
                    drho_str = f"$\\hphantom{{-}}{{-}}{abs(drho):.1f}$"
                else:
                    drho_str = f"$-${abs(drho):.1f}"
            else:
                drho_str = f"$+${drho:.1f}"

            dkeff_str = f"$+${dkeff:.1f}" if dkeff >= 0 else f"$-${abs(dkeff):.1f}"

            line = f"  & {name:<10} & {ans_space:<15} & {n} & {drho_str} & {dkeff_str} & {r_sc:.2f} & {r_pt:.2f} \\\\"
            lines.append(line)

    return "\n".join(lines)


def main():
    print("=" * 80)
    print("Cross-benchmark prompt-template matrix (Table 3)")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Run the pipeline (numeric scorer for GSM8K, MATH-500 scorer for MATH)
    print("\n[1/3] Running the 60-cell pipeline...")
    cells, pt_mats = run_table3_pipeline(n_boot=10_000, rng_seed=2026)
    derived = compute_derived_numbers(cells, pt_mats)
    print("      Completed: {} valid, {} decorrelate, mean Delta rho = {}%".format(
        derived["valid_cells_count"], derived["decorrelates_count"], fmt_num(derived["mean_drho_pct"], 2)
    ))

    # 2. Assemble numbers.json
    print("\n[2/3] Writing numbers.json...")
    cells_json = {f"{model}_{task}": cell for (model, task), cell in cells.items()}
    deliverable_numbers = {
        "metadata": {
            "output_dir": str(OUTPUT_DIR),
            "n_cells": len(cells_json),
            "valid_cells_count": derived["valid_cells_count"],
        },
        "cells": cells_json,
        "derived_numbers": derived,
    }
    numbers_path = OUTPUT_DIR / "numbers.json"
    with open(numbers_path, "w") as f:
        json.dump(to_json_safe(deliverable_numbers), f, indent=2, allow_nan=False)
    print(f"      Wrote {numbers_path}")

    # 3. Assemble rows_table3.tex
    print("\n[3/3] Writing rows_table3.tex...")
    tex_content = format_table3_latex(derived["table3_rows"])
    tex_path = OUTPUT_DIR / "rows_table3.tex"
    with open(tex_path, "w") as f:
        f.write(tex_content + "\n")
    print(f"      Wrote {tex_path}")


if __name__ == "__main__":
    main()
