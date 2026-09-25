#!/usr/bin/env python3
"""FLAG-2 5-seed aggregation for the capgated main matrix.

Produces `results/5seed_summary.csv` with per-cell (model x benchmark) mean+-std
across 5 seeds, plus bootstrap CIs on Delta_rho (per app:statistical_precision).

Metrics per (model, task) cell:
  p_bar_mean/std       single-path accuracy over seeds
  rho_sc_mean/std      pairwise Pearson correlation under SC
  rho_pt_mean/std      pairwise Pearson correlation under PT (sc_prompttpl)
  drho_pct_mean/std    (rho_pt - rho_sc) / |rho_sc| * 100 percent
  keff_sc_mean/std     K / (1 + (K-1) c_sc)
  keff_pt_mean/std     K / (1 + (K-1) c_pt)
  mv_acc_sc_mean/std   majority-vote accuracy under SC
  mv_acc_pt_mean/std   majority-vote accuracy under PT
  drho_ci_low/high     95 percent bootstrap CI on mean drho_pct (instance resampling, 10k replicates)

Usage:
    uv run python scripts/aggregate_5seed.py
    uv run python scripts/aggregate_5seed.py --bootstrap 0     # skip bootstrap for speed
    uv run python scripts/aggregate_5seed.py --out results/custom.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
_SRC = Path(__file__).resolve().parent.parent
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from diversity_combining.config import experiment_dir

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = experiment_dir("cache")
RESULTS_DIR = experiment_dir("results")

MODELS = ["qwen05b", "qwen7b", "qwen32b", "llama8b", "mistral7b"]
MODEL_LABELS = {
    "qwen05b": "Qwen-0.5B",
    "qwen7b": "Qwen-7B",
    "qwen32b": "Qwen-32B",
    "llama8b": "Llama-8B",
    "mistral7b": "Mistral-7B",
}

# File suffix -> {task_name: domain}
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
    "_boolq": {"boolq": "NLU/Reasoning"},
    "_drop": {"drop": "NLU/Reasoning"},
}

DOMAIN_ORDER = ["Math", "QA", "Science/MC", "Code", "Commonsense", "NLU/Reasoning"]
BASE_ACC_THRESHOLD = 0.02  # matches analyze_all_benchmarks.py and paper exclusion rule


# ---------- primitives (kept identical to analyze_all_benchmarks.py) ----------

def compute_pairwise_rho(correctness_matrix: np.ndarray) -> float:
    N, K = correctness_matrix.shape
    if K < 2 or N < 2:
        return float("nan")
    rhos = []
    for i, j in combinations(range(K), 2):
        ci = correctness_matrix[:, i]
        cj = correctness_matrix[:, j]
        if ci.std() < 1e-10 or cj.std() < 1e-10:
            continue
        r = np.corrcoef(ci, cj)[0, 1]
        rhos.append(r)
    if not rhos:
        return float("nan")
    return float(np.mean(rhos))


def compute_mv_accuracy(correctness_matrix: np.ndarray) -> float:
    N, K = correctness_matrix.shape
    vote_counts = correctness_matrix.sum(axis=1)
    mv_correct = (vote_counts > K / 2).astype(float)
    if K % 2 == 0:
        mv_correct += 0.5 * (vote_counts == K / 2).astype(float)
    return float(mv_correct.mean())


def k_eff(K: int, rho: float) -> float:
    if np.isnan(rho):
        return float("nan")
    return K / (1 + (K - 1) * rho)


# ---------- loading ----------

def load_records(model: str, suffix: str):
    path = CACHE_DIR / f"{model}_capgated{suffix}.jsonl"
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def correctness_from_record(r: dict) -> list[int] | None:
    """Return per-path binary correctness list, or None if record unusable."""
    if "all_correct" in r:
        return list(r["all_correct"])
    if "all_f1s" in r:
        return [1 if f >= 0.5 else 0 for f in r["all_f1s"]]
    if "all_answers" in r and "gold_answer" in r:
        gold = str(r["gold_answer"]).strip().lower()
        return [1 if str(a).strip().lower() == gold else 0 for a in r["all_answers"]]
    return None


# ---------- per-seed stats ----------

def build_seed_matrix(records: list[dict]) -> np.ndarray | None:
    """Stack per-instance correctness vectors in instance_id order."""
    records = sorted(records, key=lambda r: r.get("instance_id", 0))
    rows = []
    for r in records:
        v = correctness_from_record(r)
        if v is None:
            continue
        rows.append(v)
    if not rows:
        return None
    return np.array(rows, dtype=float)


def analyze_cell(model: str, suffix: str, task: str):
    """Compute per-seed stats for one (model, task) cell, both methods.

    Returns dict with per-seed lists: {method: {stat: [val_per_seed]}, meta}.
    """
    records = [r for r in load_records(model, suffix) if r.get("task") == task]
    if not records:
        return None

    by_seed = defaultdict(lambda: {"sc": [], "sc_prompttpl": []})
    for r in records:
        if r.get("method") not in ("sc", "sc_prompttpl"):
            continue
        by_seed[r["seed"]][r["method"]].append(r)

    seeds = sorted(by_seed.keys())
    if len(seeds) < 2:
        return None  # need at least 2 seeds to report std

    per_seed = {
        "sc":           {"p_bar": [], "rho": [], "K_eff": [], "mv_acc": [], "N": [], "K": []},
        "sc_prompttpl": {"p_bar": [], "rho": [], "K_eff": [], "mv_acc": [], "N": [], "K": []},
    }
    matrices = {"sc": {}, "sc_prompttpl": {}}  # method -> {seed: matrix} for bootstrap

    for seed in seeds:
        for method in ("sc", "sc_prompttpl"):
            method_records = by_seed[seed][method]
            mat = build_seed_matrix(method_records)
            if mat is None or mat.shape[0] < 20:
                continue
            matrices[method][seed] = mat
            N, K = mat.shape
            p_bar = float(mat.mean())
            rho = compute_pairwise_rho(mat)
            per_seed[method]["p_bar"].append(p_bar)
            per_seed[method]["rho"].append(rho)
            per_seed[method]["K_eff"].append(k_eff(K, rho))
            per_seed[method]["mv_acc"].append(compute_mv_accuracy(mat))
            per_seed[method]["N"].append(N)
            per_seed[method]["K"].append(K)

    return {
        "seeds": seeds,
        "per_seed": per_seed,
        "matrices": matrices,
    }


def mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return (float("nan"), float("nan"))
    xs = [x for x in xs if not np.isnan(x)]
    if not xs:
        return (float("nan"), float("nan"))
    return (float(np.mean(xs)), float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0)


def bootstrap_delta_rho_ci(sc_mats: dict, pt_mats: dict, n_boot: int, rng: np.random.Generator):
    """Bootstrap 95 percent CI on mean Delta_rho_pct across seeds.

    Instance-resamples within each seed (paired), recomputes per-seed Delta_rho,
    averages over seeds, repeats n_boot times.
    """
    common_seeds = sorted(set(sc_mats) & set(pt_mats))
    if not common_seeds or n_boot <= 0:
        return (float("nan"), float("nan"))

    boot_means = []
    for _ in range(n_boot):
        seed_drhos = []
        for s in common_seeds:
            sc_m = sc_mats[s]
            pt_m = pt_mats[s]
            n_sc = sc_m.shape[0]
            n_pt = pt_m.shape[0]
            idx_sc = rng.integers(0, n_sc, size=n_sc)
            idx_pt = rng.integers(0, n_pt, size=n_pt)
            rho_sc = compute_pairwise_rho(sc_m[idx_sc])
            rho_pt = compute_pairwise_rho(pt_m[idx_pt])
            if np.isnan(rho_sc) or abs(rho_sc) < 0.01:
                continue
            seed_drhos.append((rho_pt - rho_sc) / abs(rho_sc) * 100)
        if seed_drhos:
            boot_means.append(np.mean(seed_drhos))
    if not boot_means:
        return (float("nan"), float("nan"))
    lo = float(np.percentile(boot_means, 2.5))
    hi = float(np.percentile(boot_means, 97.5))
    return (lo, hi)


# ---------- main ----------

def analyze_cell_pooled(model: str, suffix: str, task: str):
    """Pool all seeds into a single (N*n_seeds, K) matrix per method.

    Matches paper's original estimator (same as 2-seed pipeline, just with
    more data). Returns per-method matrices for bootstrap on the pooled n.
    """
    records = [r for r in load_records(model, suffix) if r.get("task") == task]
    if not records:
        return None
    by_method = {"sc": [], "sc_prompttpl": []}
    for r in records:
        if r.get("method") not in by_method:
            continue
        by_method[r["method"]].append(r)
    mats = {}
    for m, recs in by_method.items():
        mat = build_seed_matrix(recs)  # sorts by instance_id then stacks
        if mat is None or mat.shape[0] < 20:
            continue
        mats[m] = mat
    if "sc" not in mats or "sc_prompttpl" not in mats:
        return None
    return mats


def _mean_pairwise_rho_vec(mat: np.ndarray) -> float:
    """Vectorized mean pairwise Pearson correlation across K columns.

    Equivalent to compute_pairwise_rho but ~100x faster by computing the
    full K x K correlation matrix at once via column-standardization.
    """
    N, K = mat.shape
    if K < 2 or N < 2:
        return float("nan")
    # column standardize
    mu = mat.mean(axis=0, keepdims=True)
    sd = mat.std(axis=0, keepdims=True, ddof=0)
    # handle near-constant columns: they contribute NaN rows/cols we'll mask
    valid = (sd > 1e-10).flatten()
    if valid.sum() < 2:
        return float("nan")
    X = (mat[:, valid] - mu[:, valid]) / sd[:, valid]
    R = (X.T @ X) / N  # K_valid x K_valid
    # mean of off-diagonal upper triangle
    Kv = R.shape[0]
    idx = np.triu_indices(Kv, k=1)
    return float(R[idx].mean())


def bootstrap_pooled_drho_ci(sc_mat: np.ndarray, pt_mat: np.ndarray, n_boot: int,
                             rng: np.random.Generator):
    """Vectorized bootstrap CI on Delta_rho (pct) for pooled cell."""
    if n_boot <= 0:
        return (float("nan"), float("nan"))
    n_sc = sc_mat.shape[0]
    n_pt = pt_mat.shape[0]
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        rho_sc = _mean_pairwise_rho_vec(sc_mat[rng.integers(0, n_sc, size=n_sc)])
        rho_pt = _mean_pairwise_rho_vec(pt_mat[rng.integers(0, n_pt, size=n_pt)])
        if np.isnan(rho_sc) or abs(rho_sc) < 0.01:
            draws[b] = np.nan
            continue
        draws[b] = (rho_pt - rho_sc) / abs(rho_sc) * 100
    draws = draws[~np.isnan(draws)]
    if draws.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=10_000,
                    help="bootstrap replicates for Delta_rho CI (0 to skip)")
    ap.add_argument("--out", default=str(RESULTS_DIR / "5seed_summary.csv"))
    ap.add_argument("--mode", choices=["pooled", "per_seed", "both"], default="both",
                    help="pooled = paper-original estimator (pools seeds into n=250 matrix); "
                         "per_seed = mean/std across per-seed estimates; "
                         "both (default) = emit pooled columns + per-seed columns")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)

    rows = []
    all_cells = []

    for model in MODELS:
        for suffix, task_map in BENCHMARK_FILES.items():
            for task, domain in task_map.items():
                cell = analyze_cell(model, suffix, task)
                if cell is None:
                    continue

                seeds = cell["seeds"]
                per = cell["per_seed"]
                mats = cell["matrices"]

                # ----- pooled estimator (paper-original) -----
                pooled = analyze_cell_pooled(model, suffix, task)
                pooled_vals = {}
                if pooled is not None:
                    sc_mat = pooled["sc"]
                    pt_mat = pooled["sc_prompttpl"]
                    pooled_vals["p_bar_sc_pooled"] = float(sc_mat.mean())
                    pooled_vals["p_bar_pt_pooled"] = float(pt_mat.mean())
                    pooled_vals["rho_sc_pooled"] = compute_pairwise_rho(sc_mat)
                    pooled_vals["rho_pt_pooled"] = compute_pairwise_rho(pt_mat)
                    K = sc_mat.shape[1]
                    pooled_vals["keff_sc_pooled"] = k_eff(K, pooled_vals["rho_sc_pooled"])
                    pooled_vals["keff_pt_pooled"] = k_eff(K, pooled_vals["rho_pt_pooled"])
                    pooled_vals["mv_acc_sc_pooled"] = compute_mv_accuracy(sc_mat)
                    pooled_vals["mv_acc_pt_pooled"] = compute_mv_accuracy(pt_mat)
                    pooled_vals["N_pooled"] = int(sc_mat.shape[0])
                    rho_sc_p = pooled_vals["rho_sc_pooled"]
                    if not np.isnan(rho_sc_p) and abs(rho_sc_p) > 0.01:
                        pooled_vals["drho_pct_pooled"] = (pooled_vals["rho_pt_pooled"] - rho_sc_p) / abs(rho_sc_p) * 100
                    else:
                        pooled_vals["drho_pct_pooled"] = float("nan")
                    pooled_vals["dkeff_pooled"] = pooled_vals["keff_pt_pooled"] - pooled_vals["keff_sc_pooled"]
                else:
                    for k in ("p_bar_sc_pooled","p_bar_pt_pooled","rho_sc_pooled","rho_pt_pooled",
                              "keff_sc_pooled","keff_pt_pooled","mv_acc_sc_pooled","mv_acc_pt_pooled",
                              "drho_pct_pooled","dkeff_pooled"):
                        pooled_vals[k] = float("nan")
                    pooled_vals["N_pooled"] = 0

                # base accuracy threshold on POOLED p_bar_sc (paper-consistent)
                excluded = bool(np.isnan(pooled_vals["p_bar_sc_pooled"]) or
                                pooled_vals["p_bar_sc_pooled"] < BASE_ACC_THRESHOLD)

                # pooled bootstrap CI
                if args.bootstrap > 0 and not excluded and pooled is not None:
                    ci_lo, ci_hi = bootstrap_pooled_drho_ci(pooled["sc"], pooled["sc_prompttpl"],
                                                             args.bootstrap, rng)
                else:
                    ci_lo, ci_hi = (float("nan"), float("nan"))
                pooled_vals["drho_ci_low"] = ci_lo
                pooled_vals["drho_ci_high"] = ci_hi

                # ----- per-seed estimator (robustness) -----
                drho_pct_per_seed = []
                dkeff_per_seed = []
                common_seeds = sorted(set(mats["sc"]) & set(mats["sc_prompttpl"]))
                for s in common_seeds:
                    idx = seeds.index(s)
                    if idx >= len(per["sc"]["rho"]) or idx >= len(per["sc_prompttpl"]["rho"]):
                        continue
                    rho_sc = per["sc"]["rho"][idx]
                    rho_pt = per["sc_prompttpl"]["rho"][idx]
                    if np.isnan(rho_sc) or abs(rho_sc) < 0.01:
                        continue
                    drho_pct_per_seed.append((rho_pt - rho_sc) / abs(rho_sc) * 100)
                    dkeff_per_seed.append(per["sc_prompttpl"]["K_eff"][idx] - per["sc"]["K_eff"][idx])

                row = {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "task": task,
                    "domain": domain,
                    "n_seeds": len(seeds),
                    "excluded_base_acc": excluded,
                    "N_per_seed": int(np.median(per["sc"]["N"])) if per["sc"]["N"] else 0,
                    "K": int(per["sc"]["K"][0]) if per["sc"]["K"] else 0,
                }
                # pooled columns first (paper-primary)
                row.update(pooled_vals)
                # per-seed mean/std per metric (robustness)
                for stat in ("p_bar", "rho", "K_eff", "mv_acc"):
                    for method, mlabel in (("sc", "sc"), ("sc_prompttpl", "pt")):
                        m, s = mean_std(per[method][stat])
                        row[f"{stat}_{mlabel}_mean"] = m
                        row[f"{stat}_{mlabel}_std"] = s
                m_drho, s_drho = mean_std(drho_pct_per_seed)
                row["drho_pct_mean"] = m_drho
                row["drho_pct_std"] = s_drho
                m_dkeff, s_dkeff = mean_std(dkeff_per_seed)
                row["dkeff_mean"] = m_dkeff
                row["dkeff_std"] = s_dkeff

                rows.append(row)
                if not excluded and not np.isnan(row["drho_pct_pooled"]):
                    all_cells.append(row)

    # write CSV
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} rows to {out}")

    # summary stats (echo for a quick check) — pooled primary, per-seed robustness
    if all_cells:
        drho_p = np.array([r["drho_pct_pooled"] for r in all_cells])
        drho_s = np.array([r["drho_pct_mean"] for r in all_cells])
        dkeff_p = np.array([r["dkeff_pooled"] for r in all_cells])
        print()
        print(f"Valid cells (p_bar_sc_pooled >= {BASE_ACC_THRESHOLD}): {len(all_cells)} / {len(rows)}")
        print("----- POOLED estimator (paper-primary, n=250/cell) -----")
        print(f"Mean Delta_rho pct: {drho_p.mean():.1f}")
        print(f"Cells with Delta_rho < 0 (PT decorrelates): {(drho_p < 0).sum()}/{len(all_cells)}")
        print(f"Mean Delta_K_eff: {dkeff_p.mean():+.2f}")
        print("----- PER-SEED estimator (robustness, mean-of-5-seed) -----")
        print(f"Mean Delta_rho pct: {drho_s.mean():.1f} (+- {drho_s.std(ddof=1):.1f})")
        print(f"Cells with Delta_rho < 0: {(drho_s < 0).sum()}/{len(all_cells)}")

        # domain-level aggregate (Math vs QA Mann-Whitney) on pooled
        try:
            from scipy import stats
            math_drho = [r["drho_pct_pooled"] for r in all_cells if r["domain"] == "Math"]
            qa_drho   = [r["drho_pct_pooled"] for r in all_cells if r["domain"] == "QA"]
            if math_drho and qa_drho:
                u, pval = stats.mannwhitneyu(math_drho, qa_drho, alternative="greater")
                print(f"\nMann-Whitney (Math vs QA on POOLED Delta_rho): U={u:.0f} p={pval:.2e}")
                print(f"  Math mean {np.mean(math_drho):+.1f} pct ; QA mean {np.mean(qa_drho):+.1f} pct")
        except ImportError:
            pass


if __name__ == "__main__":
    main()
