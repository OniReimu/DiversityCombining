#!/usr/bin/env python3
"""Analyze when MV and weighted aggregation diverge (P6).

For each PT cell (model × benchmark), computes:
  - MV accuracy (uniform voting)
  - Accuracy-weighted voting (weight slot k by its observed accuracy)
  - Slot-pruned voting (drop worst 2 slots, MV on remaining 6)
  - Slot accuracy CV (coefficient of variation)

The prediction: higher slot CV → larger gap between weighted/pruned and MV.
"""
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import sys
_SRC = Path(__file__).resolve().parent.parent
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from diversity_combining.config import experiment_dir

CACHE_DIR = experiment_dir("cache")
MODELS = ["qwen05b", "qwen7b", "qwen32b", "llama8b", "mistral7b"]
MODEL_LABELS = {
    "qwen05b": "Qwen-0.5B", "qwen7b": "Qwen-7B", "qwen32b": "Qwen-32B",
    "llama8b": "Llama-8B", "mistral7b": "Mistral-7B",
}
BENCHMARK_FILES = {
    "": {"gsm8k": "Math", "math": "Math"},
    "_qa": {"hotpotqa": "QA"},
    "_triviaqa": {"triviaqa": "QA"},
    "_arc": {"arc_challenge": "Science/MC"},
    "_mmlu": {"mmlu": "Science/MC"},
    "_mbpp": {"mbpp": "Code"},
    "_cruxeval": {"cruxeval": "Code"},
    "_hellaswag": {"hellaswag": "Commonsense"},
    "_winogrande": {"winogrande": "Commonsense"},
    "_boolq": {"boolq": "NLU"},
    "_drop": {"drop": "NLU"},
}
MIN_ACCURACY = 0.02


def build_pt_matrix(records):
    """Build (N, K) correctness matrix for PT method."""
    rows = []
    for r in records:
        if "all_correct" in r and r["all_correct"]:
            rows.append(r["all_correct"])
        elif "all_f1s" in r and r.get("all_f1s"):
            rows.append([1 if f >= 0.5 else 0 for f in r["all_f1s"]])
        elif "all_answers" in r and "gold_answer" in r:
            g = str(r["gold_answer"]).strip().lower()
            rows.append([1 if str(a).strip().lower() == g else 0 for a in r["all_answers"]])
    if not rows:
        return None
    mat = np.array(rows, dtype=float)
    return mat if mat.shape[0] >= 20 and mat.mean() >= MIN_ACCURACY else None


def mv_accuracy(mat):
    """Uniform majority vote."""
    N, K = mat.shape
    votes = mat.sum(axis=1)
    correct = (votes > K / 2).astype(float)
    if K % 2 == 0:
        correct += 0.5 * (votes == K / 2).astype(float)
    return correct.mean()


def weighted_mv_accuracy(mat):
    """Accuracy-weighted voting: weight each slot by its mean accuracy."""
    N, K = mat.shape
    slot_acc = mat.mean(axis=0)  # per-slot accuracy
    # For each instance, weighted vote = sum(w_k * Y_ik) > 0.5 * sum(w_k)
    weighted_votes = mat @ slot_acc
    threshold = slot_acc.sum() / 2
    correct = (weighted_votes > threshold).astype(float)
    correct += 0.5 * (weighted_votes == threshold).astype(float)
    return correct.mean()


def pruned_mv_accuracy(mat, n_drop=2):
    """Slot-pruned MV: drop n_drop worst-accuracy slots, MV on rest."""
    N, K = mat.shape
    slot_acc = mat.mean(axis=0)
    keep = np.argsort(slot_acc)[n_drop:]  # drop worst n_drop
    mat_pruned = mat[:, keep]
    return mv_accuracy(mat_pruned)


def slot_cv(mat):
    """Coefficient of variation of per-slot accuracy."""
    slot_acc = mat.mean(axis=0)
    return slot_acc.std() / slot_acc.mean() if slot_acc.mean() > 0 else 0


def main():
    results = []

    for model in MODELS:
        for suffix, task_map in BENCHMARK_FILES.items():
            path = CACHE_DIR / f"{model}_capgated{suffix}.jsonl"
            if not path.exists():
                continue
            records = [json.loads(l) for l in open(path) if l.strip()]

            for task_name, domain in task_map.items():
                pt_recs = [r for r in records
                           if r.get("task") == task_name and r["method"] == "sc_prompttpl"]
                if not pt_recs:
                    continue
                mat = build_pt_matrix(pt_recs)
                if mat is None:
                    continue

                mv = mv_accuracy(mat)
                wmv = weighted_mv_accuracy(mat)
                pmv = pruned_mv_accuracy(mat, n_drop=2)
                cv = slot_cv(mat)

                results.append({
                    "model": model, "task": task_name, "domain": domain,
                    "mv": mv, "wmv": wmv, "pmv": pmv, "cv": cv,
                    "gap_wmv": wmv - mv, "gap_pmv": pmv - mv,
                    "slot_accs": mat.mean(axis=0).tolist(),
                })

    # Print results
    print(f"{'Task':<16} {'Model':<12} {'MV':>6} {'WMV':>6} {'Pruned':>6} "
          f"{'Δ_WMV':>7} {'Δ_Prune':>7} {'CV':>6}")
    print("-" * 80)

    for r in sorted(results, key=lambda x: x["cv"], reverse=True):
        print(f"{r['task']:<16} {MODEL_LABELS[r['model']]:<12} "
              f"{r['mv']:>5.1%} {r['wmv']:>5.1%} {r['pmv']:>5.1%} "
              f"{r['gap_wmv']:>+6.1%} {r['gap_pmv']:>+6.1%} {r['cv']:>6.3f}")

    # Summary statistics
    gaps_wmv = [r["gap_wmv"] for r in results]
    gaps_pmv = [r["gap_pmv"] for r in results]
    cvs = [r["cv"] for r in results]

    print(f"\n{'='*80}")
    print(f"N = {len(results)} cells")
    print(f"WMV gain: mean={np.mean(gaps_wmv):+.1%}, median={np.median(gaps_wmv):+.1%}, "
          f"range=[{min(gaps_wmv):+.1%}, {max(gaps_wmv):+.1%}]")
    print(f"Pruned gain: mean={np.mean(gaps_pmv):+.1%}, median={np.median(gaps_pmv):+.1%}, "
          f"range=[{min(gaps_pmv):+.1%}, {max(gaps_pmv):+.1%}]")

    # Correlation: CV vs gap
    from scipy import stats
    r_wmv, p_wmv = stats.pearsonr(cvs, gaps_wmv)
    r_pmv, p_pmv = stats.pearsonr(cvs, gaps_pmv)
    print(f"\nCorrelation(CV, Δ_WMV): r={r_wmv:.3f}, p={p_wmv:.4f}")
    print(f"Correlation(CV, Δ_Prune): r={r_pmv:.3f}, p={p_pmv:.4f}")

    # High CV vs low CV comparison
    median_cv = np.median(cvs)
    high_cv = [r for r in results if r["cv"] > median_cv]
    low_cv = [r for r in results if r["cv"] <= median_cv]
    print(f"\nHigh CV (>{median_cv:.3f}): mean Δ_WMV={np.mean([r['gap_wmv'] for r in high_cv]):+.1%}, "
          f"mean Δ_Prune={np.mean([r['gap_pmv'] for r in high_cv]):+.1%}")
    print(f"Low CV  (<={median_cv:.3f}): mean Δ_WMV={np.mean([r['gap_wmv'] for r in low_cv]):+.1%}, "
          f"mean Δ_Prune={np.mean([r['gap_pmv'] for r in low_cv]):+.1%}")


if __name__ == "__main__":
    main()
