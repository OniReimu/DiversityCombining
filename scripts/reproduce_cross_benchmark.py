#!/usr/bin/env python3
"""Reproduce the cross-benchmark prompt-template analysis (5 models x 12 benchmarks, K=8).

For every model-benchmark cell, loads the prompt-template (PT) arm and the matched
self-consistency (SC) arm (first 8 of the 32 SC paths) on the same (seed, instance)
pairs through scripts/matched_arms.py, scores both arms with
scripts/scoring_v2.check_task_correct, and computes the mean pairwise correlation,
majority-vote accuracy, K_eff, accuracy-weighted voting and a paired bootstrap of
Delta rho. Writes numbers.json and the LaTeX rows of Tables 7 and 9 to
experiment_dir("cross_benchmark").

- Estimators are imported from scripts.aggregate_5seed, scripts.correlation_estimators
  and scripts.analyze_mv_vs_weighted.
- Every requested cell is mandatory; shape or matching anomalies raise immediately.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Sequence

import numpy as np
from scipy import stats
import scipy

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# ruff: noqa: E402 -- direct script entrypoint needs repository and src on sys.path
from scripts import matched_arms as pt
from scripts.aggregate_5seed import (
    BASE_ACC_THRESHOLD, _mean_pairwise_rho_vec, compute_mv_accuracy,
    compute_pairwise_rho, k_eff,
)
from scripts.correlation_estimators import c_eq3
from scripts.analyze_mv_vs_weighted import slot_cv, weighted_mv_accuracy
from scripts.cross_benchmark_common import (
    BENCHMARK_ORDER_TABLE3, TASK_ANSWER_SPACE, TASK_DISPLAY_NAMES,
    build_block_covariance_matrix, get_scorers,
)
from scripts.cross_benchmark_outputs import write_outputs
from diversity_combining.config import experiment_dir, record_path
from scripts.scoring_v2 import check_task_correct, get_scorer

OUTPUT_DIR = experiment_dir("cross_benchmark")
TASKS = tuple(t for _, tasks in BENCHMARK_ORDER_TABLE3 for t in tasks)
DOMAINS = {t: d for d, tasks in BENCHMARK_ORDER_TABLE3 for t in tasks}
N_BOOT = 10_000
RNG_SEED = 2026
# Slot-accuracy CV and MV-accuracy partitions of the accuracy-weighted voting analysis.
CV_THRESHOLD = 0.225
MV_THRESHOLD = 0.3


def fingerprint(path: Path) -> dict:
    return {"path": record_path(path), "md5": hashlib.md5(path.read_bytes()).hexdigest()}


def source_grid(path: Path, task: str, arm: str) -> list[tuple[int, int]]:
    """Supplement the GSM loader's legacy checks without changing that loader's outputs."""
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keys = [(r["seed"], r["instance_id"]) for r in records]
    seeds = pt.PT4_SEEDS if task in pt.PT4_TASKS + pt.ARCWG_TASKS else pt.EXPECTED_SEEDS
    expected = {s: pt.INSTANCES_PER_SEED for s in seeds}
    if dict(Counter(s for s, _ in keys)) != expected or len(set(keys)) != len(keys):
        raise pt.CellMismatchError(f"{path}: duplicate keys or wrong seed grid; expected {expected}")
    for r in records:
        field = "all_traces" if task == pt.TASK and arm == "PT" else "all_answers"
        size = pt.K if arm == "PT" else 32
        if not isinstance(r.get(field), list) or len(r[field]) != size:
            raise pt.CellMismatchError(f"{path}: expected {size} {field} at {r['seed'], r['instance_id']}")
        if r.get("K", size) != size:
            raise pt.CellMismatchError(f"{path}: declared K disagrees with {field} length")
        if r.get("task") != task or r.get("gold_answer") is None or r["gold_answer"] == "":
            raise pt.CellMismatchError(f"{path}: wrong task or null gold")
    return keys


def load_cell(model: str, task: str) -> tuple[list, list, dict]:
    if task == pt.TASK:
        loaded = pt.load_arms(model)
    elif task in pt.T12_TASKS:
        loaded = pt.load_t12_arms(model, task)
    elif task in pt.PT4_TASKS:
        loaded = pt.load_pt4_arms(model, task)
    elif task in pt.ARCWG_TASKS:
        loaded = pt.load_arcwg_arms(model, task)
    else:
        raise ValueError(f"unknown task: {task}")
    pt_cells, sc_cells, pt_path, sc_path = loaded
    pt_keys, sc_keys = source_grid(pt_path, task, "PT"), source_grid(sc_path, task, "SC")
    if set(pt_keys) != set(sc_keys) or len(pt_cells) != len(pt_keys) or len(sc_cells) != len(pt_keys):
        raise pt.CellMismatchError(f"{model}/{task}: unmatched grid or loader length")
    # Both loaders return their two arms in PT source order. Sort together for reproducibility.
    order = sorted(range(len(pt_keys)), key=pt_keys.__getitem__)
    if any(pt_cells[i][1] != sc_cells[i][1] for i in order):
        raise pt.CellMismatchError(f"{model}/{task}: matched gold mismatch")
    return [pt_cells[i] for i in order], [sc_cells[i] for i in order], {
        "PT": fingerprint(pt_path), "SC": fingerprint(sc_path),
        "seeds": sorted({s for s, _ in pt_keys}),
        "instances_per_seed": pt.INSTANCES_PER_SEED,
        "matched_keys_md5": hashlib.md5(json.dumps(sorted(pt_keys)).encode()).hexdigest(),
    }


def correctness_matrix(cells: list, task: str) -> np.ndarray:
    if not cells or any(len(a) != pt.K for a, _ in cells):
        raise pt.CellMismatchError(f"{task}: expected nonempty (N, {pt.K}) answers")
    return np.asarray([
        [int(a is not None and check_task_correct(a, g, task)) for a in answers]
        for answers, g in cells
    ], dtype=float)


def arm_statistics(mat: np.ndarray) -> dict:
    if mat.ndim != 2 or mat.shape[1] != pt.K or mat.shape[0] < 2 or not np.isin(mat, [0, 1]).all():
        raise pt.CellMismatchError(f"expected binary (N, {pt.K}) matrix, got {mat.shape}")
    rho, c = compute_pairwise_rho(mat), c_eq3(mat)
    return {
        "n_instances": len(mat), "p_bar": float(mat.mean()), "rho_hat": rho, "c_hat": c,
        # K_eff of Table 7 uses the measured Pearson rho.
        "K_eff_vote": k_eff(pt.K, rho), "K_eff_from_c": k_eff(pt.K, c),
        "MV8": compute_mv_accuracy(mat), "slot_cv": float(slot_cv(mat)),
        "weighted_MV8": float(weighted_mv_accuracy(mat)),
        "weighted_gain_pp": float((weighted_mv_accuracy(mat) - compute_mv_accuracy(mat)) * 100),
        "slot_accuracies": mat.mean(axis=0).tolist(),
        "constant_slots": np.flatnonzero(mat.std(axis=0) < 1e-10).tolist(),
    }


def relative_drho(rho_sc: float, rho_pt: float) -> float:
    return ((rho_pt - rho_sc) / abs(rho_sc) * 100
            if np.isfinite(rho_sc) and abs(rho_sc) > 0.01 else float("nan"))


def excluded_base_accuracy(p_bar_sc: float) -> bool:
    """Exact historical exclusion: only pooled SC path accuracy, strictly below cutoff."""
    return bool(np.isnan(p_bar_sc) or p_bar_sc < BASE_ACC_THRESHOLD)


def paired_bootstrap(sc: np.ndarray, pt_mat: np.ndarray, n_boot: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Use the imported fast Pearson estimator with one shared row draw per replicate.

    Retains the old bootstrap's |rho_SC| >= .01 denominator guard. Missing draws
    remain NaN so simultaneous domain replicates retain their original indices.
    """
    if sc.shape != pt_mat.shape or sc.ndim != 2 or n_boot < 1:
        raise pt.CellMismatchError("paired bootstrap requires matching matrices and positive draws")
    draws = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, len(sc), size=len(sc))
        rs, rp = _mean_pairwise_rho_vec(sc[idx]), _mean_pairwise_rho_vec(pt_mat[idx])
        if np.isfinite(rs) and np.isfinite(rp) and abs(rs) >= 0.01:
            draws[b] = (rp - rs) / abs(rs) * 100
    return draws


def mean(rows: list[dict], key: str) -> float | None:
    return float(np.mean([r[key] for r in rows])) if rows else None


def pearson(xs: list, ys: list) -> dict:
    if len(xs) < 2 or np.ptp(xs) == 0 or np.ptp(ys) == 0:
        return {"pearson_r": None, "p_value": None}
    result = stats.pearsonr(xs, ys)
    return {"pearson_r": float(result.statistic), "p_value": float(result.pvalue)}


def weighted_summary(cells: list[dict]) -> dict:
    # Deliberately screen PT independently of the SC-based primary validity rule.
    rows = [{"model": c["model"], "task": c["task"], **c["PT"]}
            for c in cells if c["PT"]["p_bar"] >= BASE_ACC_THRESHOLD]
    result = {"n_cells": len(rows), "cells": rows, "cv_threshold": CV_THRESHOLD,
              "mv_threshold": MV_THRESHOLD, "median_cv": float(np.median([r["slot_cv"] for r in rows]))}
    result.update(pearson([r["slot_cv"] for r in rows], [r["weighted_gain_pp"] for r in rows]))
    for name, subset in {
        "high_cv": [r for r in rows if r["slot_cv"] > CV_THRESHOLD],
        "low_cv": [r for r in rows if r["slot_cv"] <= CV_THRESHOLD],
        "high_mv": [r for r in rows if r["MV8"] >= MV_THRESHOLD],
        "low_mv": [r for r in rows if r["MV8"] < MV_THRESHOLD],
    }.items():
        result[name] = {"n": len(subset), "mean_gain_pp": mean(subset, "weighted_gain_pp"),
                        "mean_cv": mean(subset, "slot_cv"),
                        "with_high_cv_pct": (100 * sum(r["slot_cv"] > CV_THRESHOLD for r in subset)
                                             / len(subset)) if subset else None}
    return result


def block_statistic(mat: np.ndarray) -> dict:
    """Reuse historical matrix eligibility, then its inline sample-variance formula."""
    records = [{"all_correct": row.tolist()} for row in mat]
    checked = build_block_covariance_matrix(records, "gsm8k", True, *get_scorers())
    if checked is None:
        return {"reason": "PT p_bar below base-accuracy threshold or N < 20"}
    rho = compute_pairwise_rho(checked)  # same upper-triangle nanmean, skips constant slots
    variance = float(np.var(checked.mean(axis=1), ddof=1))
    if not np.isfinite(rho) or rho <= -1 / (pt.K - 1) or variance <= 0:
        return {"reason": "undefined Pearson correlation or degenerate vote variance"}
    keff_block = float(checked.mean() * (1 - checked.mean()) / variance)
    keff_vote = k_eff(pt.K, rho)
    return {"K_eff_block": keff_block, "K_eff_vote": keff_vote,
            "ratio": keff_block / keff_vote}


def derive(cells: list[dict], draws: dict) -> dict:
    valid = [c for c in cells if not c["excluded_base_acc"]]
    excluded = [c for c in cells if c["excluded_base_acc"]]
    result = {"valid_cells_count": len(valid), "excluded_cells_count": len(excluded),
              "excluded_cells": [{"model": c["model"], "task": c["task"], "p_bar_sc": c["SC"]["p_bar"]}
                                 for c in excluded],
              "decorrelates_count": sum(c["delta_rho_pct"] < 0 for c in valid),
              "mean_drho_pct": mean(valid, "delta_rho_pct"), "mean_dkeff": mean(valid, "delta_K_eff_vote"),
              "exceptions": [{"model": c["model"], "task": c["task"], "drho_pct": c["delta_rho_pct"],
                              "p_bar_sc": c["SC"]["p_bar"], "p_bar_pt": c["PT"]["p_bar"]}
                             for c in valid if c["delta_rho_pct"] >= 0]}
    table, domains, acc = {}, {}, {}
    for task in TASKS:
        selected = [c for c in valid if c["task"] == task]
        if not any(c["task"] == task for c in cells):
            continue
        table[task] = {"domain": DOMAINS[task], "display_name": TASK_DISPLAY_NAMES[task],
                       "answer_space": TASK_ANSWER_SPACE[task], "n_models": len(selected),
                       "mean_drho_pct": mean(selected, "delta_rho_pct"),
                       "mean_dkeff": mean(selected, "delta_K_eff_vote"),
                       "mean_rho_sc": float(np.mean([c["SC"]["rho_hat"] for c in selected])) if selected else None,
                       "mean_rho_pt": float(np.mean([c["PT"]["rho_hat"] for c in selected])) if selected else None,
                       "mean_sc_diversity": mean([c["SC"] for c in selected], "answer_diversity")}
    for dom, _ in BENCHMARK_ORDER_TABLE3:
        selected = [c for c in valid if c["domain"] == dom]
        ds = [c["delta_rho_pct"] for c in selected]
        domains[dom] = {"n_cells": len(selected), "mean_drho_pct": float(np.mean(ds)) if ds else None,
                        "min_drho_pct": min(ds) if ds else None, "max_drho_pct": max(ds) if ds else None}
        deltas = [(c["PT"]["MV8"] - c["SC"]["MV8"]) * 100 for c in selected]
        acc[dom] = {"n_cells": len(selected),
                    "mv_sc_mean": float(np.mean([c["SC"]["MV8"] * 100 for c in selected])) if selected else None,
                    "mv_pt_mean": float(np.mean([c["PT"]["MV8"] * 100 for c in selected])) if selected else None,
                    "delta_acc_mean": float(np.mean(deltas)) if deltas else None,
                    "delta_acc_std": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else None}
    math_cells, qa_cells = ([c for c in valid if c["domain"] == d] for d in ("Math", "QA"))
    mw = {"n_math": len(math_cells), "n_qa": len(qa_cells), "alternative": "two-sided"}
    if math_cells and qa_cells:
        observed = stats.mannwhitneyu([c["delta_rho_pct"] for c in math_cells],
                                     [c["delta_rho_pct"] for c in qa_cells], alternative="two-sided")
        arrays = [np.column_stack([draws[c["id"]] for c in group]) for group in (math_cells, qa_cells)]
        finite = np.isfinite(arrays[0]).all(axis=1) & np.isfinite(arrays[1]).all(axis=1)
        # SciPy auto selects exact for <=8 untied observations, asymptotic otherwise.
        ps = np.asarray([stats.mannwhitneyu(x, y, alternative="two-sided").pvalue
                         for x, y in zip(arrays[0][finite], arrays[1][finite])])
        mw.update(u_statistic=float(observed.statistic), p_value=float(observed.pvalue),
                  bootstrap_replicates_used=int(finite.sum()),
                  bootstrap_replicates_dropped=int((~finite).sum()),
                  bootstrap_fraction_p_lt_0001=float(np.mean(ps < 0.001)))
    widths = [c["delta_rho_ci95"][1] - c["delta_rho_ci95"][0] for c in valid]
    bs = {"ci_width_median": float(np.median(widths)),
          "ci_width_iqr_25": float(np.percentile(widths, 25)), "ci_width_iqr_75": float(np.percentile(widths, 75)),
          "cells_excluding_zero_count": sum(c["delta_rho_ci95"][0] > 0 or c["delta_rho_ci95"][1] < 0 for c in valid),
          "drho_min": min(c["delta_rho_pct"] for c in valid), "drho_max": max(c["delta_rho_pct"] for c in valid),
          "named_cells": {c["id"]: {"drho_pct": c["delta_rho_pct"], "ci95": c["delta_rho_ci95"],
                                    "p_bar_sc": c["SC"]["p_bar"]}
                          for c in valid if c["task"] in ("triviaqa", "hotpotqa", "math")}}
    bs["benchmark_precision"] = {}
    for task in ("triviaqa", "hotpotqa"):
        selected = [c for c in valid if c["task"] == task]
        if selected:
            bs["benchmark_precision"][task] = {
                "n_cells": len(selected),
                "cells_excluding_zero": sum(c["delta_rho_ci95"][0] > 0 or c["delta_rho_ci95"][1] < 0 for c in selected),
                "ci_low_min": min(c["delta_rho_ci95"][0] for c in selected),
                "ci_high_max": max(c["delta_rho_ci95"][1] for c in selected),
            }
    points = [r for r in table.values() if r["n_models"]]
    entropy = {**pearson([r["mean_sc_diversity"] for r in points], [r["mean_drho_pct"] for r in points]),
               "n_points": len(points)}
    sc_weighting = [{"model": c["model"], "task": c["task"], **c["SC"]} for c in valid]
    blocks = [{"model": c["model"], "task": c["task"], **c["block"]} for c in valid if "ratio" in c["block"]]
    block_excluded = [{"model": c["model"], "task": c["task"], **c["block"]} for c in valid if "ratio" not in c["block"]]
    result.update(table3_rows=table, domain_means=domains, table8_rows=acc, mann_whitney=mw,
                  bootstrap_stats=bs, entropy_predictor=entropy, weighted_mv_stats=weighted_summary(cells),
                  sc_weighting={"n_cells": len(sc_weighting),
                                "max_slot_cv": max(r["slot_cv"] for r in sc_weighting) if sc_weighting else None,
                                "min_cell": min(sc_weighting, key=lambda r: r["weighted_gain_pp"]) if sc_weighting else None,
                                "max_cell": max(sc_weighting, key=lambda r: r["weighted_gain_pp"]) if sc_weighting else None},
                  block_covariance_stats={"n_cells": len(blocks), "excluded_count": len(block_excluded),
                                          "excluded": block_excluded, "population": "SC-valid cells, then PT eligibility",
                                          "mean_ratio": mean(blocks, "ratio"),
                                          "triviaqa_qwen7b_ratio": next((b["ratio"] for b in blocks
                                                                          if b["task"] == "triviaqa" and b["model"] == "qwen7b"), None),
                                          "std_ratio": float(np.std([b["ratio"] for b in blocks], ddof=0)) if blocks else None,
                                          "max_cell": max(blocks, key=lambda b: b["ratio"]) if blocks else None,
                                          "cells": blocks})
    return result


def build(tasks: Sequence[str], n_boot: int = N_BOOT) -> dict:
    if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - set(TASKS):
        raise ValueError("tasks must be a nonempty unique selection of paper benchmarks")
    # Canonical order makes an invocation insensitive to the ordering of --tasks.
    tasks = [t for t in TASKS if t in tasks]
    cells, draws = [], {}
    for task in tasks:
        for model in pt.T12_MODELS:
            print(f"Loading {model}/{task}", flush=True)
            pt_cells, sc_cells, source = load_cell(model, task)
            mats = {"PT": correctness_matrix(pt_cells, task), "SC": correctness_matrix(sc_cells, task)}
            arms = {name: arm_statistics(mat) for name, mat in mats.items()}
            arms["SC"]["answer_diversity"] = float(np.mean([
                len({str(a).strip().lower() for a in ans}) / pt.K for ans, _ in sc_cells]))
            excluded = excluded_base_accuracy(arms["SC"]["p_bar"])
            cell_id = f"{model}_{task}"
            drho = relative_drho(arms["SC"]["rho_hat"], arms["PT"]["rho_hat"])
            ci, n_finite = None, 0
            if not excluded:
                if not np.isfinite(drho):
                    raise ValueError(f"{cell_id}: SC-valid cell has undefined relative rho")
                # Stable independent stream per cell; adding ARC/WG cannot change existing CIs.
                rng = np.random.default_rng(np.random.SeedSequence([
                    RNG_SEED, TASKS.index(task), pt.T12_MODELS.index(model)]))
                draws[cell_id] = paired_bootstrap(mats["SC"], mats["PT"], n_boot, rng)
                finite = draws[cell_id][np.isfinite(draws[cell_id])]
                n_finite = len(finite)
                if not n_finite:
                    raise ValueError(f"{cell_id}: no finite bootstrap draws")
                ci = np.percentile(finite, [2.5, 97.5]).tolist()
            cells.append({"id": cell_id, "model": model, "task": task, "domain": DOMAINS[task],
                          **arms, "source": source, "scorer": get_scorer(task).name,
                          "excluded_base_acc": excluded, "delta_rho_pct": drho,
                          "delta_K_eff_vote": arms["PT"]["K_eff_vote"] - arms["SC"]["K_eff_vote"],
                          "delta_rho_ci95": ci, "bootstrap_finite_replicates": n_finite,
                          "bootstrap_dropped_replicates": n_boot - n_finite if not excluded else None,
                          "block": block_statistic(mats["PT"])})
    return {"metadata": {"tasks": tasks, "models": list(pt.T12_MODELS), "n_cells": len(cells),
                         "K": pt.K, "bootstrap_replicates": n_boot, "rng_seed": RNG_SEED,
                         "base_accuracy_threshold": BASE_ACC_THRESHOLD,
                         "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                         "scipy": scipy.__version__, "platform": platform.platform()},
                         "code_sources": [fingerprint(ROOT / "scripts" / name) for name in (
                             "reproduce_cross_benchmark.py", "cross_benchmark_outputs.py",
                             "matched_arms.py", "aggregate_5seed.py", "analyze_mv_vs_weighted.py",
                             "correlation_estimators.py", "cross_benchmark_common.py",
                             "scoring_v2.py", "run_sc_kvar_v2.py", "tasks12.py")]},
            "cells": cells, "derived": derive(cells, draws)}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    args = parser.parse_args(argv)
    numbers = build(args.tasks)
    write_outputs(numbers, OUTPUT_DIR)
    print(json.dumps({k: numbers["derived"][k] for k in (
        "valid_cells_count", "excluded_cells_count", "decorrelates_count", "mean_drho_pct", "mean_dkeff")}))
    print(f"Wrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
