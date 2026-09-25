#!/usr/bin/env python3
"""Block covariance analysis.

Computes full K×K correlation matrix from PT data, then compares:
  - Keff_equicorr = K/(1+(K-1)*c_mean)  (equicorrelated assumption)
  - Keff_block = K * var(p_bar) / var(S_K/K)  (exact from full covariance)

If they're close, the equicorrelated approximation is justified.
"""
import json
from pathlib import Path
from itertools import combinations

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


def build_pt_matrix(records):
    rows = []
    for r in records:
        if "all_correct" in r and r["all_correct"]:
            rows.append(r["all_correct"])
        elif "all_answers" in r and "gold_answer" in r:
            g = str(r["gold_answer"]).strip().lower()
            answers = r.get("all_answers", [])
            rows.append([1 if str(a).strip().lower() == g else 0 for a in answers])
    if not rows:
        return None
    mat = np.array(rows, dtype=float)
    return mat if mat.shape[0] >= 20 and mat.mean() >= 0.02 else None


def main():
    print(f"{'Task':<14} {'Model':<12} {'K':>3} {'ĉ_equi':>8} {'Keff_eq':>8} {'Keff_blk':>9} {'Ratio':>7}")
    print("-" * 70)

    ratios = []

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

                N, K = mat.shape

                # Full K×K correlation matrix
                corr_matrix = np.corrcoef(mat.T)
                # Handle NaN diagonal or off-diagonal
                np.fill_diagonal(corr_matrix, 0)
                valid_pairs = ~np.isnan(corr_matrix)
                if valid_pairs.sum() == 0:
                    continue

                c_mean = np.nanmean(corr_matrix[np.triu_indices(K, k=1)])

                # Keff_equicorr
                keff_eq = K / (1 + (K - 1) * c_mean) if c_mean > -1/(K-1) else float("nan")

                # Keff_block = K * Var(p_bar) / Var(S_K/K)
                # = K * p(1-p) / Var(S_K/K)
                # where Var(S_K/K) = (1/K^2) * sum_ij Cov(Y_i, Y_j)
                p_bar = mat.mean()
                S_K = mat.sum(axis=1)
                var_sk_over_k = np.var(S_K / K, ddof=1)
                var_indep = p_bar * (1 - p_bar) / K
                keff_blk = var_indep / var_sk_over_k * K if var_sk_over_k > 0 else float("nan")

                if not np.isnan(keff_eq) and not np.isnan(keff_blk) and keff_eq > 0:
                    ratio = keff_blk / keff_eq
                    ratios.append(ratio)
                    print(f"{task_name:<14} {MODEL_LABELS[model]:<12} {K:>3} "
                          f"{c_mean:>8.3f} {keff_eq:>8.2f} {keff_blk:>9.2f} {ratio:>7.2f}")

    if ratios:
        print(f"\n{'='*70}")
        print(f"Mean Keff_block / Keff_equicorr ratio: {np.mean(ratios):.3f} "
              f"(std={np.std(ratios):.3f}, range=[{min(ratios):.2f}, {max(ratios):.2f}])")
        print(f"N cells: {len(ratios)}")
        print("\nIf ratio ≈ 1.0, equicorrelated approximation is accurate.")


if __name__ == "__main__":
    main()
