#!/usr/bin/env python3
"""Canonical analysis script for SC K-var evaluation (v2 protocol).

Consumes append-only JSONL produced by run_sc_kvar_v2.py with 32 ordered paths per record.
Performs protocol gating, prefix-sliced Exact-K diversity metrics, Adaptive-K evaluation
with held-out validation and baseline comparisons, clustered bootstrap uncertainty estimation,
and comprehensive provenance tracking.

Usage:
    python scripts/aggregate_kvar_v2.py
    python scripts/aggregate_kvar_v2.py --in-dir cache --out-dir results
    python scripts/aggregate_kvar_v2.py --models qwen7b llama8b --tasks gsm8k boolq
"""
from __future__ import annotations

import argparse

import sys
from pathlib import Path
_SRC = Path(__file__).resolve().parent.parent
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from diversity_combining.config import RESULTS_DIR, experiment_dir, record_path
import csv
from collections import Counter, defaultdict
from itertools import combinations
import math
import os
from pathlib import Path
import re
import sys

import numpy as np
from scipy.special import beta as beta_func, comb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.scoring_v2 import (
    canonicalize_answer,
    check_numeric_equivalence,
    check_task_correct,
    get_scorer,
    get_task_scorer,
    SCORERS,
)


# ── Answer scoring and majority vote helpers ────────────────────────────────

def compute_majority_vote(
    answers: list[str | None], gold: str, task: str
) -> tuple[str | None, int]:
    """Compute majority vote winner with ties broken by first occurrence.

    Returns:
        (winning_answer, is_correct) where is_correct is 1 if winner matches gold, else 0.
    """
    buckets: list[dict] = []
    for a in answers:
        if a is None:
            continue
        canon = canonicalize_answer(a, task)
        if canon is None:
            continue
        matched_bucket = None
        for b in buckets:
            if task in ("gsm8k", "math"):
                if canon == b["canon"] or check_numeric_equivalence(canon, b["canon"]):
                    matched_bucket = b
                    break
            else:
                if canon == b["canon"]:
                    matched_bucket = b
                    break
        if matched_bucket is not None:
            matched_bucket["count"] += 1
        else:
            buckets.append({"canon": canon, "raw": a, "count": 1})

    if not buckets:
        return None, 0

    max_count = max(b["count"] for b in buckets)
    winner_bucket = next(b for b in buckets if b["count"] == max_count)
    winner = winner_bucket["raw"]
    correct = 1 if check_task_correct(winner, gold, task) else 0
    return winner, correct


def binary_collapse_mv(answers: list, gold, task: str) -> float:
    """Binary-collapse majority: the outcome bb_mv_acc/binom_mv_acc actually predict.

    Returns 1.0 when strictly more than half the paths are individually correct,
    0.5 on an exact tie with an even number of paths, else 0.0. Mirrors the tie
    handling in bb_mv_acc.
    """
    k = len(answers)
    if k == 0:
        return 0.0
    n_correct = sum(
        1 for a in answers if a is not None and check_task_correct(a, gold, task)
    )
    if n_correct > k / 2:
        return 1.0
    elif n_correct == k / 2 and k % 2 == 0:
        return 0.5
    else:
        return 0.0


# ── Diversity and theoretical estimator functions ───────────────────────────

def pairwise_rho(mat: np.ndarray) -> float:
    """Pairwise correctness correlation c estimator (matching aggregate_canonical_sc.py)."""
    n_rows, k_cols = mat.shape
    if k_cols < 2 or n_rows < 2:
        return float("nan")
    rhos = []
    for i, j in combinations(range(k_cols), 2):
        ci, cj = mat[:, i], mat[:, j]
        if ci.std() < 1e-10 or cj.std() < 1e-10:
            continue
        rhos.append(np.corrcoef(ci, cj)[0, 1])
    return float(np.mean(rhos)) if rhos else float("nan")


def mean_pairwise_agree(mat: np.ndarray) -> float:
    """Mean pairwise correctness agreement across all path pairs."""
    n_rows, k_cols = mat.shape
    if k_cols < 2 or n_rows < 1:
        return float("nan")
    out = []
    for i, j in combinations(range(k_cols), 2):
        out.append(np.mean(mat[:, i] == mat[:, j]))
    return float(np.mean(out)) if out else float("nan")


def keff(k: int, c: float) -> float:
    """Effective diversity K_eff = K / (1 + (K - 1) * c)."""
    if k == 1:
        return 1.0
    if np.isnan(c) or c <= 0:
        return float("nan")
    return k / (1.0 + (k - 1) * c)


def bb_mv_acc(k: int, p: float, c: float) -> float:
    """Beta-binomial majority vote accuracy prediction."""
    if c <= 0 or c >= 1 or p <= 0 or p >= 1:
        return float("nan")
    alpha = p * (1.0 - c) / c
    beta_p = (1.0 - p) * (1.0 - c) / c
    acc = 0.0
    for j in range(k + 1):
        prob = comb(k, j, exact=True) * beta_func(j + alpha, k - j + beta_p) / beta_func(alpha, beta_p)
        if j > k / 2:
            acc += prob
        elif j == k / 2 and k % 2 == 0:
            acc += 0.5 * prob
    return float(acc)


def binom_mv_acc(k: int, p: float) -> float:
    """Independence baseline (c=0) majority vote accuracy prediction."""
    if p < 0 or p > 1:
        return float("nan")
    acc = 0.0
    for j in range(k + 1):
        prob = comb(k, j, exact=True) * (p ** j) * ((1.0 - p) ** (k - j))
        if j > k / 2:
            acc += prob
        elif j == k / 2 and k % 2 == 0:
            acc += 0.5 * prob
    return float(acc)


# ── File loading and provenance validation ──────────────────────────────────

MIN_SEEDS = 3
MIN_INSTANCES = 100

PROVENANCE_CHECK_FIELDS = [
    "model_id",
    "max_new_tokens",
    "temperature",
    "top_p",
    "script_version",
]


def load_and_validate_file(filepath: Path) -> list[dict]:
    """Read JSONL file and strictly validate provenance uniformity across records."""
    records = []
    seen_values: dict[str, set] = {f: set() for f in PROVENANCE_CHECK_FIELDS}

    with open(filepath, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                import json
                r = json.loads(line)
            except Exception as e:
                sys.exit(f"FATAL: Unparseable JSON at {filepath}:{lineno}: {e}")

            records.append(r)
            prov = r.get("provenance") if isinstance(r.get("provenance"), dict) else {}
            for field in PROVENANCE_CHECK_FIELDS:
                val = prov.get(field)
                if val is None:
                    val = r.get(field)
                seen_values[field].add(val)

    for field, vals in seen_values.items():
        if len(vals) > 1:
            sys.exit(
                f"FATAL: Provenance mismatch in file {filepath.name} for field '{field}'. "
                f"Conflicting values found: {sorted(list(vals))}. Aborting."
            )

    return records


def find_candidate_cells(
    in_dir: Path,
    filter_models: list[str] | None = None,
    filter_tasks: list[str] | None = None,
) -> list[tuple[str, str, Path]]:
    """Discover candidate (model, task, path) tuples matching {model}_sc_kvar_v2_{task}.jsonl."""
    pattern = re.compile(r"^([a-zA-Z0-9_]+)_sc_kvar_v2_([a-zA-Z0-9_]+)\.jsonl$")
    candidates = []

    if not in_dir.exists() or not in_dir.is_dir():
        return []

    for item in sorted(in_dir.iterdir()):
        if not item.is_file():
            continue
        m = pattern.match(item.name)
        if not m:
            continue
        model_slug, task_slug = m.group(1), m.group(2)
        if filter_models and model_slug not in filter_models:
            continue
        if filter_tasks and task_slug not in filter_tasks:
            continue
        candidates.append((model_slug, task_slug, item))

    # Also check if explicit models and tasks were requested but file wasn't matched
    if filter_models and filter_tasks:
        for mod in filter_models:
            for tsk in filter_tasks:
                target = in_dir / f"{mod}_sc_kvar_v2_{tsk}.jsonl"
                entry = (mod, tsk, target)
                if entry not in candidates:
                    candidates.append(entry)

    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates


# ── Protocol Gate ────────────────────────────────────────────────────────────

def evaluate_protocol_gate(
    model: str,
    task: str,
    filepath: Path,
    records: list[dict],
    min_seeds: int = MIN_SEEDS,
    min_instances: int = MIN_INSTANCES,
) -> dict:
    """Evaluate admissibility requirements according to the v2 protocol gate."""
    reasons: list[str] = []
    n_records = len(records)

    if not filepath.exists():
        return {
            "model": model,
            "task": task,
            "filepath": record_path(filepath),
            "n_records": 0,
            "expected_records": 0,
            "records_complete": False,
            "all_32_answers": False,
            "malformed_records": 0,
            "extraction_rate": 0.0,
            "harness_loss_rate": 0.0,
            "noncompliance_rate": 0.0,
            "cap_hit_rate": 0.0,
            "cap_threshold": 0.05 if "qwen3_5" in model else 0.01,
            "admissible": False,
            "reasons": ["File does not exist"],
        }

    if n_records == 0:
        return {
            "model": model,
            "task": task,
            "filepath": record_path(filepath),
            "n_records": 0,
            "expected_records": 0,
            "records_complete": False,
            "all_32_answers": False,
            "malformed_records": 0,
            "extraction_rate": 0.0,
            "harness_loss_rate": 0.0,
            "noncompliance_rate": 0.0,
            "cap_hit_rate": 0.0,
            "cap_threshold": 0.05 if "qwen3_5" in model else 0.01,
            "admissible": False,
            "reasons": ["File is empty (0 records)"],
        }

    unique_seeds = sorted(list({r.get("seed") for r in records if r.get("seed") is not None}))
    unique_insts = sorted(list({r.get("instance_id") for r in records if r.get("instance_id") is not None}))

    # F4: Reject any cell with fewer than two unique instance IDs
    has_min_instances = len(unique_insts) >= 2
    if not has_min_instances:
        reasons.append(
            f"Fewer than 2 unique instance IDs found ({len(unique_insts)} found); minimum 2 required for split"
        )

    if len(unique_seeds) < min_seeds:
        reasons.append(
            f"Only {len(unique_seeds)} unique seed(s); minimum {min_seeds} required"
        )
    if len(unique_insts) < min_instances:
        reasons.append(
            f"Only {len(unique_insts)} unique instance(s); minimum {min_instances} required"
        )

    n_seeds = len(unique_seeds)
    max_inst_id = max(unique_insts) if unique_insts else -1
    expected_inst_count = max_inst_id + 1
    expected_records = n_seeds * expected_inst_count

    # Check for missing instances or duplicate keys
    pair_counts = Counter((r.get("seed"), r.get("instance_id")) for r in records)
    duplicates = [k for k, count in pair_counts.items() if count > 1]
    missing_pairs = []
    for s in unique_seeds:
        for i in range(expected_inst_count):
            if (s, i) not in pair_counts:
                missing_pairs.append((s, i))

    records_complete = True
    if len(unique_insts) != expected_inst_count:
        records_complete = False
        reasons.append(f"Non-contiguous instance IDs: {len(unique_insts)} unique, max ID is {max_inst_id}")
    if duplicates:
        records_complete = False
        reasons.append(f"Duplicate records found for {len(duplicates)} (seed, instance) pairs")
    if missing_pairs:
        records_complete = False
        reasons.append(f"Incomplete records: missing {len(missing_pairs)} of {expected_records} expected pairs")
    if n_records != expected_records:
        records_complete = False
        reasons.append(f"Record count mismatch: got {n_records}, expected {expected_records}")

    # F3: Require every record to carry exactly 32 boolean flags and 32 answers
    malformed_records = sum(
        1 for r in records
        if not (
            isinstance(r.get("all_answers"), list)
            and len(r["all_answers"]) == 32
            and isinstance(r.get("truncated"), list)
            and len(r["truncated"]) == 32
            and all(isinstance(t, bool) for t in r["truncated"])
        )
    )
    all_32_answers = all(
        isinstance(r.get("all_answers"), list) and len(r["all_answers"]) == 32
        for r in records
    )
    if malformed_records > 0:
        reasons.append(
            f"{malformed_records}/{n_records} records are malformed (require exactly 32 answers and 32 boolean truncation flags)"
        )

    # Answer extraction rate
    total_answers = n_records * 32
    non_null_answers = sum(
        1 for r in records for a in r.get("all_answers", []) if a is not None
    )
    extraction_rate = non_null_answers / total_answers if total_answers > 0 else 0.0

    # Extraction loss split by attribution, matching scripts/gate_watch.py.
    # A null answer on a path the token cap cut off is HARNESS LOSS and gates the cell.
    # A null answer on a path that stopped on its own is MODEL NON-COMPLIANCE: the model
    # declined to state an answer in the required format. That is a property of the system
    # under test, so it is reported for every cell and never gates.
    harness_loss = sum(
        1
        for r in records
        for a, t in zip(r.get("all_answers", []), r.get("truncated", []))
        if a is None and bool(t)
    )
    noncompliance = sum(
        1
        for r in records
        for a, t in zip(r.get("all_answers", []), r.get("truncated", []))
        if a is None and not bool(t)
    )
    harness_loss_rate = harness_loss / total_answers if total_answers > 0 else 0.0
    noncompliance_rate = noncompliance / total_answers if total_answers > 0 else 0.0
    if harness_loss_rate > 0.05:
        reasons.append(
            f"Harness loss {harness_loss_rate * 100:.2f}% > 5.0% "
            f"(paths cut off by the token cap with no answer)"
        )

    # Cap-hit rate
    truncated_flags = sum(
        1 for r in records for t in r.get("truncated", []) if bool(t)
    )
    cap_hit_rate = truncated_flags / total_answers if total_answers > 0 else 0.0
    cap_threshold = 0.05 if "qwen3_5" in model else 0.01
    # Cap-hit is reported, not gated. It guarded a failure mode that no longer exists: a
    # truncated path used to contribute a number lifted from an unfinished calculation.
    # The extractor now refuses every heuristic on a truncated path, so such a path either
    # carries an explicit answer marker and votes, or abstains and is counted as harness
    # loss, which is the condition that gates.

    admissible = (
        has_min_instances
        and (len(unique_seeds) >= min_seeds)
        and (len(unique_insts) >= min_instances)
        and records_complete
        and (malformed_records == 0)
        and (harness_loss_rate <= 0.05)
    )

    return {
        "model": model,
        "task": task,
        "harness_loss_rate": harness_loss_rate,
        "noncompliance_rate": noncompliance_rate,
        "filepath": record_path(filepath),
        "n_records": n_records,
        "expected_records": expected_records,
        "records_complete": records_complete,
        "all_32_answers": all_32_answers,
        "malformed_records": malformed_records,
        "extraction_rate": extraction_rate,
        "cap_hit_rate": cap_hit_rate,
        "cap_threshold": cap_threshold,
        "admissible": admissible,
        "reasons": reasons,
    }


def apply_reporting_policy(gate_info: dict) -> dict:
    """Apply the reporting-only harness-loss policy in place."""
    reasons = gate_info["reasons"]
    harness_flag = gate_info["harness_loss_rate"] > 0.05
    gate_info["harness_flag"] = harness_flag
    if (
        not gate_info["admissible"]
        and harness_flag
        and reasons
        and all(reason.startswith("Harness loss") for reason in reasons)
        and gate_info["malformed_records"] == 0
        and gate_info["n_records"] == gate_info["expected_records"]
    ):
        gate_info["admissible"] = True
    return gate_info


# ── Exact-K Metrics ──────────────────────────────────────────────────────────

def compute_exact_k(
    records: list[dict],
    task: str,
    ks: list[int] = (1, 2, 4, 8, 16, 32),
) -> list[dict]:
    """Compute Exact-K metrics via prefix slicing for each requested K."""
    results = []
    n_records = len(records)
    unique_seeds = len({r.get("seed") for r in records})

    # Pre-build full 32 correctness matrix and majority votes
    # Row i, Col j: 1 if path j is correct, else 0
    full_corr_mat = np.zeros((n_records, 32), dtype=float)
    for i, r in enumerate(records):
        gold = r.get("gold_answer", "")
        for j in range(32):
            ans = r["all_answers"][j] if j < len(r.get("all_answers", [])) else None
            full_corr_mat[i, j] = 1.0 if check_task_correct(ans, gold, task) else 0.0

    for k in ks:
        sub_mat = full_corr_mat[:, :k]
        mean_path_acc = float(sub_mat.mean())

        # Majority vote with ties broken by first occurrence
        mv_corrects = []
        for r in records:
            ans_slice = r["all_answers"][:k]
            _, is_corr = compute_majority_vote(ans_slice, r.get("gold_answer", ""), task)
            mv_corrects.append(is_corr)
        mv_accuracy = float(np.mean(mv_corrects))

        agr = mean_pairwise_agree(sub_mat)
        c = pairwise_rho(sub_mat)
        kef = keff(k, c)
        ceil = 1.0 / c if (c > 0 and not np.isnan(c)) else float("nan")
        pct_ceil = (kef / ceil) if (ceil > 0 and not np.isnan(ceil) and not np.isnan(kef)) else float("nan")

        results.append({
            "K": k,
            "N": n_records,
            "n_seeds": unique_seeds,
            "mean_path_acc": mean_path_acc,
            "mv_acc": mv_accuracy,
            "mean_pairwise_agree": agr,
            "c": c,
            "keff": kef,
            "ceiling": ceil,
            "pct_ceiling": pct_ceil,
        })

    return results


# ── Adaptive-K and Baselines ─────────────────────────────────────────────────

def evaluate_adaptive_k_and_baselines(
    records: list[dict],
    task: str,
    eps: float = 0.025,
    pilot_frac: float = 0.5,
    split_seed: int = 2026,
) -> list[dict]:
    """Evaluate Adaptive-K rule alongside all required baselines on disjoint instances."""
    unique_inst_ids = sorted(list({r["instance_id"] for r in records}))
    m_insts = len(unique_inst_ids)
    n_seeds = len({r["seed"] for r in records})

    rng = np.random.default_rng(split_seed)
    shuffled_inst_ids = list(unique_inst_ids)
    rng.shuffle(shuffled_inst_ids)

    split_idx = int(m_insts * pilot_frac)
    if m_insts >= 2:
        split_idx = max(1, min(m_insts - 1, split_idx))

    half_a = set(shuffled_inst_ids[:split_idx])
    half_b = set(shuffled_inst_ids[split_idx:])

    folds = [("A", half_a, half_b), ("B", half_b, half_a)]
    fold_results: list[dict] = []

    for fold_label, pilot_inst_ids, eval_inst_ids in folds:
        pilot_records = [r for r in records if r["instance_id"] in pilot_inst_ids]
        eval_records = [r for r in records if r["instance_id"] in eval_inst_ids]

        assert len(pilot_records) > 0, (
            f"Pilot split is empty (n_instances={m_insts}, pilot_inst_ids={pilot_inst_ids})"
        )
        assert len(eval_records) > 0, (
            f"Evaluation split is empty (n_instances={m_insts}, eval_inst_ids={eval_inst_ids})"
        )

        # Pilot K=4 correctness matrix
        pilot_corr_k4 = np.array(
            [
                [
                    1.0 if check_task_correct(a, r.get("gold_answer", ""), task) else 0.0
                    for a in r["all_answers"][:4]
                ]
                for r in pilot_records
            ],
            dtype=float,
        )

        pilot_acc = float(pilot_corr_k4.mean())
        pilot_var = float(pilot_corr_k4.var())
        is_degenerate = (pilot_acc <= 0.0 or pilot_acc >= 1.0 or pilot_var < 1e-12)

        if not is_degenerate:
            c_pilot = pairwise_rho(pilot_corr_k4)
            if np.isnan(c_pilot):
                is_degenerate = True
                c_pilot = float("nan")
                k_star = 1
            else:
                c_clipped = float(np.clip(c_pilot, 0.05, 0.99))
                k_star_raw = math.ceil((math.sqrt((1.0 - c_clipped) / eps) - 1.0) / c_clipped + 1.0)
                k_star = max(1, min(32, int(k_star_raw)))
        else:
            c_pilot = float("nan")
            k_star = 1

        # Same-instance evaluation (on pilot records)
        same_mv_kstar = float(
            np.mean([
                compute_majority_vote(r["all_answers"][:k_star], r.get("gold_answer", ""), task)[1]
                for r in pilot_records
            ])
        )
        same_mv_32 = float(
            np.mean([
                compute_majority_vote(r["all_answers"][:32], r.get("gold_answer", ""), task)[1]
                for r in pilot_records
            ])
        )
        same_retention_pct = (
            (100.0 * same_mv_kstar / same_mv_32) if same_mv_32 > 0 else float("nan")
        )
        same_cost = k_star

        # Held-out evaluation (on eval records)
        heldout_mv_kstar = float(
            np.mean([
                compute_majority_vote(r["all_answers"][:k_star], r.get("gold_answer", ""), task)[1]
                for r in eval_records
            ])
        )
        heldout_mv_32 = float(
            np.mean([
                compute_majority_vote(r["all_answers"][:32], r.get("gold_answer", ""), task)[1]
                for r in eval_records
            ])
        )
        heldout_retention_pct = (
            (100.0 * heldout_mv_kstar / heldout_mv_32) if heldout_mv_32 > 0 else float("nan")
        )
        heldout_cost = k_star

        # Fixed K baselines evaluated on held-out split
        fixed_k_results: dict[int, float] = {}
        for fixed_k in (1, 2, 4, 8, 16, 32):
            acc = float(
                np.mean([
                    compute_majority_vote(r["all_answers"][:fixed_k], r.get("gold_answer", ""), task)[1]
                    for r in eval_records
                ])
            )
            fixed_k_results[fixed_k] = acc

        # Held-out split metrics for baselines
        heldout_mv_4 = float(
            np.mean([
                compute_majority_vote(r["all_answers"][:4], r.get("gold_answer", ""), task)[1]
                for r in eval_records
            ])
        )
        heldout_binary_mv_32 = float(
            np.mean([
                binary_collapse_mv(r["all_answers"][:32], r.get("gold_answer", ""), task)
                for r in eval_records
            ])
        )

        # Held-out K=4 correctness matrix, path accuracy, and pairwise rho
        eval_corr_k4 = np.array(
            [
                [
                    1.0 if check_task_correct(a, r.get("gold_answer", ""), task) else 0.0
                    for a in r["all_answers"][:4]
                ]
                for r in eval_records
            ],
            dtype=float,
        )
        heldout_acc_k4 = float(eval_corr_k4.mean())
        eval_var = float(eval_corr_k4.var())
        is_eval_degenerate = (heldout_acc_k4 <= 0.0 or heldout_acc_k4 >= 1.0 or eval_var < 1e-12)
        if not is_eval_degenerate:
            heldout_c_k4 = pairwise_rho(eval_corr_k4)
            if np.isnan(heldout_c_k4):
                heldout_c_k4 = float("nan")
        else:
            heldout_c_k4 = float("nan")

        # Pilot split MV@4
        pilot_mv_4 = float(
            np.mean([
                compute_majority_vote(r["all_answers"][:4], r.get("gold_answer", ""), task)[1]
                for r in pilot_records
            ])
        )

        # Six baseline outputs
        reuse_k4_same_pred = heldout_mv_4
        reuse_k4_same_err_pp = (
            abs(reuse_k4_same_pred - heldout_mv_32) * 100.0
            if not np.isnan(reuse_k4_same_pred)
            else float("nan")
        )

        reuse_k4_deploy_pred = pilot_mv_4
        reuse_k4_deploy_err_pp = (
            abs(reuse_k4_deploy_pred - heldout_mv_32) * 100.0
            if not np.isnan(reuse_k4_deploy_pred)
            else float("nan")
        )

        bb_same_pred = bb_mv_acc(32, heldout_acc_k4, heldout_c_k4)
        bb_same_err_pp = (
            abs(bb_same_pred - heldout_binary_mv_32) * 100.0
            if not np.isnan(bb_same_pred)
            else float("nan")
        )

        bb_deploy_pred = bb_mv_acc(32, pilot_acc, c_pilot)
        bb_deploy_err_pp = (
            abs(bb_deploy_pred - heldout_binary_mv_32) * 100.0
            if not np.isnan(bb_deploy_pred)
            else float("nan")
        )

        indep_same_pred = binom_mv_acc(32, heldout_acc_k4)
        indep_same_err_pp = (
            abs(indep_same_pred - heldout_binary_mv_32) * 100.0
            if not np.isnan(indep_same_pred)
            else float("nan")
        )

        indep_deploy_pred = binom_mv_acc(32, pilot_acc)
        indep_deploy_err_pp = (
            abs(indep_deploy_pred - heldout_binary_mv_32) * 100.0
            if not np.isnan(indep_deploy_pred)
            else float("nan")
        )

        fold_results.append({
            "fold": fold_label,
            "n_instances": m_insts,
            "n_seeds": n_seeds,
            "n_pilot_inst": len(pilot_inst_ids),
            "n_eval_inst": len(eval_inst_ids),
            "eps": eps,
            "pilot_acc": pilot_acc,
            "c_pilot": c_pilot,
            "degenerate": is_degenerate,
            "k_star": k_star,
            "same_mv_kstar": same_mv_kstar,
            "same_mv_32": same_mv_32,
            "same_retention_pct": same_retention_pct,
            "same_cost": same_cost,
            "heldout_mv_kstar": heldout_mv_kstar,
            "heldout_mv_32": heldout_mv_32,
            "heldout_retention_pct": heldout_retention_pct,
            "heldout_cost": heldout_cost,
            "fixed_k_accs": fixed_k_results,
            "heldout_mv_4": heldout_mv_4,
            "heldout_binary_mv_32": heldout_binary_mv_32,
            "reuse_k4_same_pred": reuse_k4_same_pred,
            "reuse_k4_same_err_pp": reuse_k4_same_err_pp,
            "reuse_k4_deploy_pred": reuse_k4_deploy_pred,
            "reuse_k4_deploy_err_pp": reuse_k4_deploy_err_pp,
            "bb_same_pred": bb_same_pred,
            "bb_same_err_pp": bb_same_err_pp,
            "bb_deploy_pred": bb_deploy_pred,
            "bb_deploy_err_pp": bb_deploy_err_pp,
            "indep_same_pred": indep_same_pred,
            "indep_same_err_pp": indep_same_err_pp,
            "indep_deploy_pred": indep_deploy_pred,
            "indep_deploy_err_pp": indep_deploy_err_pp,
        })

    return fold_results


# ── Clustered Bootstrap Uncertainty ──────────────────────────────────────────

def compute_clustered_bootstrap(
    records: list[dict],
    task: str,
    n_boot: int = 1000,
    seed: int = 2026,
) -> dict:
    """Compute 95% confidence intervals via clustered bootstrap over unique instance IDs."""
    by_inst: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_inst[r["instance_id"]].append(r)

    inst_ids = sorted(by_inst.keys())
    m_clusters = len(inst_ids)

    # Precompute cluster-level matrices for fast resampling
    inst_mat4 = [
        np.array(
            [
                [
                    1.0 if check_task_correct(a, r.get("gold_answer", ""), task) else 0.0
                    for a in r["all_answers"][:4]
                ]
                for r in by_inst[i]
            ],
            dtype=float,
        )
        for i in inst_ids
    ]

    inst_mat32 = [
        np.array(
            [
                [
                    1.0 if check_task_correct(a, r.get("gold_answer", ""), task) else 0.0
                    for a in r["all_answers"][:32]
                ]
                for r in by_inst[i]
            ],
            dtype=float,
        )
        for i in inst_ids
    ]

    inst_mv4 = [
        np.array(
            [
                compute_majority_vote(r["all_answers"][:4], r.get("gold_answer", ""), task)[1]
                for r in by_inst[i]
            ],
            dtype=float,
        )
        for i in inst_ids
    ]

    inst_mv32 = [
        np.array(
            [
                compute_majority_vote(r["all_answers"][:32], r.get("gold_answer", ""), task)[1]
                for r in by_inst[i]
            ],
            dtype=float,
        )
        for i in inst_ids
    ]

    # Full-data point estimates
    full_mat4 = np.concatenate(inst_mat4)
    full_mat32 = np.concatenate(inst_mat32)
    full_mv4_vec = np.concatenate(inst_mv4)
    full_mv32_vec = np.concatenate(inst_mv32)

    point_c_k4 = pairwise_rho(full_mat4)
    point_c_k32 = pairwise_rho(full_mat32)
    point_mv32 = float(np.mean(full_mv32_vec))
    point_mv4 = float(np.mean(full_mv4_vec))
    point_diff = point_mv32 - point_mv4

    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, m_clusters, size=(n_boot, m_clusters))

    boots_c_k4 = []
    boots_c_k32 = []
    boots_mv32 = []
    boots_diff = []

    for b in range(n_boot):
        idxs = boot_indices[b]
        b_mat4 = np.concatenate([inst_mat4[i] for i in idxs])
        b_mat32 = np.concatenate([inst_mat32[i] for i in idxs])
        b_mv4 = float(np.concatenate([inst_mv4[i] for i in idxs]).mean())
        b_mv32 = float(np.concatenate([inst_mv32[i] for i in idxs]).mean())

        boots_c_k4.append(pairwise_rho(b_mat4))
        boots_c_k32.append(pairwise_rho(b_mat32))
        boots_mv32.append(b_mv32)
        boots_diff.append(b_mv32 - b_mv4)

    def safe_nanpercentile(vals: list[float], q: float) -> float:
        arr = np.array(vals, dtype=float)
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return float("nan")
        return float(np.percentile(valid, q))

    return {
        "n_boot": n_boot,
        "seed": seed,
        "c_k4": {
            "point": point_c_k4,
            "ci_lower": safe_nanpercentile(boots_c_k4, 2.5),
            "ci_upper": safe_nanpercentile(boots_c_k4, 97.5),
        },
        "c_k32": {
            "point": point_c_k32,
            "ci_lower": safe_nanpercentile(boots_c_k32, 2.5),
            "ci_upper": safe_nanpercentile(boots_c_k32, 97.5),
        },
        "mv32": {
            "point": point_mv32,
            "ci_lower": safe_nanpercentile(boots_mv32, 2.5),
            "ci_upper": safe_nanpercentile(boots_mv32, 97.5),
        },
        "diff_mv32_mv4": {
            "point": point_diff,
            "ci_lower": safe_nanpercentile(boots_diff, 2.5),
            "ci_upper": safe_nanpercentile(boots_diff, 97.5),
        },
    }


# ── Output Writers ───────────────────────────────────────────────────────────

def write_csv_file(out_path: Path, headers: list[str], rows: list[dict]):
    """Write list of row dictionaries to CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in headers})


def write_provenance_markdown(
    out_path: Path,
    gate_results: list[dict],
    admissible_data: dict[tuple[str, str], dict],
):
    """Write comprehensive results/provenance_v2.md documenting inputs, seeds, scorers, and numbers."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Provenance Report (v2 Protocol Analysis)")
    lines.append("")
    lines.append("This document records the exact sources, records, seeds, scorers, and parameters")
    lines.append("for every figure and table produced by `scripts/aggregate_kvar_v2.py`.")
    lines.append("")

    lines.append("## 1. Protocol Gate and Cell Admissibility")
    lines.append("")
    lines.append("| Model | Task | Source File | Records | Expected | 32 Ans | Malformed | Extraction | Cap-Hit | Cap Thresh | Status | Reasons |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for g in gate_results:
        status_str = "ADMISSIBLE" if g["admissible"] else "EXCLUDED"
        reasons_str = "; ".join(g["reasons"]) if g["reasons"] else "None"
        lines.append(
            f"| {g['model']} | {g['task']} | `{g['filepath']}` | {g['n_records']} | {g['expected_records']} | "
            f"{'Yes' if g['all_32_answers'] else 'No'} | {g.get('malformed_records', 0)} | {g['extraction_rate']*100:.2f}% | "
            f"{g['cap_hit_rate']*100:.2f}% | {g['cap_threshold']*100:.1f}% | **{status_str}** | {reasons_str} |"
        )
    lines.append("")

    for (model, task), cell_data in admissible_data.items():
        gate_info = cell_data["gate"]
        prov = cell_data["provenance"]
        lines.append(f"## 2. Cell: {model} / {task}")
        lines.append("")
        lines.append(f"- **Source File**: `{gate_info['filepath']}`")
        lines.append(f"- **Total Records**: {gate_info['n_records']} (Expected: {gate_info['expected_records']})")
        lines.append(f"- **Seeds Evaluated**: `{sorted(list(cell_data['seeds']))}`")
        lines.append(f"- **Instances Evaluated**: {cell_data['n_instances']} (range 0 to {cell_data['n_instances']-1})")
        lines.append(f"- **Scorer Name**: `{cell_data['scorer_name']}`")
        lines.append(f"- **Model ID**: `{prov.get('model_id', 'unknown')}`")
        lines.append(f"- **Max New Tokens**: `{prov.get('max_new_tokens', 'unknown')}`")
        lines.append(f"- **Sampling Params**: temperature=`{prov.get('temperature', 'unknown')}`, top_p=`{prov.get('top_p', 'unknown')}`")
        lines.append(f"- **Script Version**: `{prov.get('script_version', 'unknown')}`")
        lines.append("")

        lines.append("### Exact-K Prefix Metrics")
        lines.append("| K | N (records) | Seeds | Path Acc | MV Acc | Pairwise Agree | Correctness c | K_eff | Ceiling (1/c) | % Ceil Reached |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for ek in cell_data["exact_k"]:
            c_str = f"{ek['c']:.4f}" if not np.isnan(ek["c"]) else "NaN"
            agr_str = f"{ek['mean_pairwise_agree']:.4f}" if not np.isnan(ek["mean_pairwise_agree"]) else "NaN"
            kef_str = f"{ek['keff']:.2f}" if not np.isnan(ek["keff"]) else "NaN"
            ceil_str = f"{ek['ceiling']:.2f}" if not np.isnan(ek["ceiling"]) else "NaN"
            pct_str = f"{ek['pct_ceiling']*100:.1f}%" if not np.isnan(ek["pct_ceiling"]) else "NaN"
            lines.append(
                f"| {ek['K']} | {ek['N']} | {ek['n_seeds']} | {ek['mean_path_acc']*100:.2f}% | "
                f"{ek['mv_acc']*100:.2f}% | {agr_str} | {c_str} | {kef_str} | {ceil_str} | {pct_str} |"
            )
        lines.append("")

        lines.append("### Adaptive-K & Baselines (Disjoint Pilot vs Evaluation)")
        for ak in cell_data["adaptive_k"]:
            lines.append(f"#### Fold {ak['fold']}")
            c_pilot_str = f"{ak['c_pilot']:.4f}" if not np.isnan(ak["c_pilot"]) else "NaN"
            lines.append(f"- **Split Instances**: Pilot={ak['n_pilot_inst']} instances, Held-Out={ak['n_eval_inst']} instances")
            lines.append(f"- **Pilot ĉ(K=4)**: {c_pilot_str} (Degenerate: {ak['degenerate']})")
            lines.append(f"- **Selected K***: {ak['k_star']} (at epsilon = {ak['eps']})")
            lines.append(f"- **Same-Instance (Pilot Split)**: MV@K* = {ak['same_mv_kstar']*100:.2f}%, MV@32 = {ak['same_mv_32']*100:.2f}%, Retained = {ak['same_retention_pct']:.1f}%, Cost = {ak['same_cost']}")
            lines.append(f"- **Held-Out (Evaluation Split)**: MV@K* = {ak['heldout_mv_kstar']*100:.2f}%, MV@32 = {ak['heldout_mv_32']*100:.2f}%, Retained = {ak['heldout_retention_pct']:.1f}%, Cost = {ak['heldout_cost']}")
            lines.append("")
            lines.append("##### Baselines on Held-Out Split:")
            for fk in (1, 2, 4, 8, 16, 32):
                lines.append(f"- Fixed K={fk}: Acc = {ak['fixed_k_accs'][fk]*100:.2f}%, Cost = {fk}")
            r4_same_err = f"{ak['reuse_k4_same_err_pp']:.2f} pp" if not np.isnan(ak['reuse_k4_same_err_pp']) else "NaN"
            lines.append(f"- Reuse K=4 Same: Pred = {ak['reuse_k4_same_pred']*100:.2f}%, Error = {r4_same_err}")
            r4_dep_err = f"{ak['reuse_k4_deploy_err_pp']:.2f} pp" if not np.isnan(ak['reuse_k4_deploy_err_pp']) else "NaN"
            lines.append(f"- Reuse K=4 Deploy: Pred = {ak['reuse_k4_deploy_pred']*100:.2f}%, Error = {r4_dep_err}")
            bb_same_str = f"{ak['bb_same_pred']*100:.2f}%" if not np.isnan(ak["bb_same_pred"]) else "NaN"
            bb_same_err = f"{ak['bb_same_err_pp']:.2f} pp" if not np.isnan(ak["bb_same_err_pp"]) else "NaN"
            lines.append(f"- Beta-Binomial Same: Pred = {bb_same_str}, Error = {bb_same_err}")
            bb_dep_str = f"{ak['bb_deploy_pred']*100:.2f}%" if not np.isnan(ak["bb_deploy_pred"]) else "NaN"
            bb_dep_err = f"{ak['bb_deploy_err_pp']:.2f} pp" if not np.isnan(ak["bb_deploy_err_pp"]) else "NaN"
            lines.append(f"- Beta-Binomial Deploy: Pred = {bb_dep_str}, Error = {bb_dep_err}")
            indep_same_str = f"{ak['indep_same_pred']*100:.2f}%" if not np.isnan(ak["indep_same_pred"]) else "NaN"
            indep_same_err = f"{ak['indep_same_err_pp']:.2f} pp" if not np.isnan(ak["indep_same_err_pp"]) else "NaN"
            lines.append(f"- Independence Same: Pred = {indep_same_str}, Error = {indep_same_err}")
            indep_dep_str = f"{ak['indep_deploy_pred']*100:.2f}%" if not np.isnan(ak["indep_deploy_pred"]) else "NaN"
            indep_dep_err = f"{ak['indep_deploy_err_pp']:.2f} pp" if not np.isnan(ak["indep_deploy_err_pp"]) else "NaN"
            lines.append(f"- Independence Deploy: Pred = {indep_dep_str}, Error = {indep_dep_err}")
            lines.append("")

        lines.append("### Clustered Bootstrap Uncertainty (1000 Resamples, Clustered on Instance ID)")
        bs = cell_data["bootstrap"]
        lines.append(
            f"- **c (K=4)**: {bs['c_k4']['point']:.4f}  (95% CI: [{bs['c_k4']['ci_lower']:.4f}, {bs['c_k4']['ci_upper']:.4f}])"
        )
        lines.append(
            f"- **c (K=32)**: {bs['c_k32']['point']:.4f}  (95% CI: [{bs['c_k32']['ci_lower']:.4f}, {bs['c_k32']['ci_upper']:.4f}])"
        )
        lines.append(
            f"- **MV@32**: {bs['mv32']['point']*100:.2f}%  (95% CI: [{bs['mv32']['ci_lower']*100:.2f}%, {bs['mv32']['ci_upper']*100:.2f}%])"
        )
        lines.append(
            f"- **(MV@32 - MV@4)**: {bs['diff_mv32_mv4']['point']*100:+.2f} pp  (95% CI: [{bs['diff_mv32_mv4']['ci_lower']*100:+.2f} pp, {bs['diff_mv32_mv4']['ci_upper']*100:+.2f} pp])"
        )
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ── Stdout Formatter ─────────────────────────────────────────────────────────

def print_stdout_summary(
    gate_results: list[dict],
    admissible_data: dict[tuple[str, str], dict],
):
    """Print readable summary tables directly to stdout."""
    print("=" * 95)
    print("PROTOCOL GATE SUMMARY")
    print("=" * 95)
    hdr_gate = f"{'Model':<12} {'Task':<10} {'Records':<12} {'32-Ans':<8} {'Malform':<8} {'Extract%':<10} {'Harness%':<10} {'NonComp%':<10} {'CapHit%':<10} {'Thresh':<8} {'Status':<12}"
    print(hdr_gate)
    print("-" * len(hdr_gate))
    for g in gate_results:
        rec_str = f"{g['n_records']}/{g['expected_records']}"
        ans_str = "Yes" if g["all_32_answers"] else "No"
        mal_str = str(g.get("malformed_records", 0))
        ext_str = f"{g['extraction_rate']*100:.1f}%"
        cap_str = f"{g['cap_hit_rate']*100:.2f}%"
        har_str = f"{g.get('harness_loss_rate', 0.0)*100:.2f}%"
        ncp_str = f"{g.get('noncompliance_rate', 0.0)*100:.2f}%"
        thr_str = f"{g['cap_threshold']*100:.1f}%"
        status_str = "ADMISSIBLE" if g["admissible"] else "EXCLUDED"
        print(f"{g['model']:<12} {g['task']:<10} {rec_str:<12} {ans_str:<8} {mal_str:<8} {ext_str:<10} {har_str:<10} {ncp_str:<10} {cap_str:<10} {thr_str:<8} {status_str:<12}")

    excluded = [g for g in gate_results if not g["admissible"]]
    if excluded:
        print("\nExcluded Non-Admissible Cells:")
        for ex in excluded:
            reasons_str = "; ".join(ex["reasons"]) if ex["reasons"] else "Unknown"
            print(f"  - ({ex['model']}, {ex['task']}): {reasons_str}")

    if not admissible_data:
        print("\nNo cells were admissible. All downstream tables excluded.")
        print("=" * 95)
        return

    print("\n" + "=" * 95)
    print("EXACT-K METRICS BY PREFIX SLICING (ADMISSIBLE CELLS)")
    print("=" * 95)
    hdr_ek = f"{'Model':<12} {'Task':<10} {'K':>3} {'N':>5} {'PathAcc':>9} {'MV@K':>9} {'Agree':>8} {'c':>8} {'K_eff':>7} {'Ceil':>7} {'%Ceil':>7}"
    print(hdr_ek)
    print("-" * len(hdr_ek))
    for (model, task), cell_data in admissible_data.items():
        for ek in cell_data["exact_k"]:
            c_str = f"{ek['c']:.3f}" if not np.isnan(ek["c"]) else "NaN"
            agr_str = f"{ek['mean_pairwise_agree']:.3f}" if not np.isnan(ek["mean_pairwise_agree"]) else "NaN"
            kef_str = f"{ek['keff']:.2f}" if not np.isnan(ek["keff"]) else "NaN"
            ceil_str = f"{ek['ceiling']:.2f}" if not np.isnan(ek["ceiling"]) else "NaN"
            pct_str = f"{ek['pct_ceiling']*100:.0f}%" if not np.isnan(ek["pct_ceiling"]) else "NaN"
            print(
                f"{model:<12} {task:<10} {ek['K']:>3} {ek['N']:>5} "
                f"{ek['mean_path_acc']*100:>8.1f}% {ek['mv_acc']*100:>8.1f}% "
                f"{agr_str:>8} {c_str:>8} {kef_str:>7} {ceil_str:>7} {pct_str:>7}"
            )

    print("\n" + "=" * 95)
    print("ADAPTIVE-K & BASELINES (DISJOINT PILOT VS HELD-OUT EVALUATION)")
    print("=" * 95)
    hdr_ak = f"{'Model':<12} {'Task':<10} {'Fold':<5} {'K*':>3} {'SameMV@K*':>10} {'HeldMV@K*':>10} {'HeldMV@32':>10} {'Retain%':>8} {'Fixed4':>8} {'R4DepErr':>9} {'BBDepErr':>9} {'IndDepErr':>10}"
    print(hdr_ak)
    print("-" * len(hdr_ak))
    for (model, task), cell_data in admissible_data.items():
        for ak in cell_data["adaptive_k"]:
            same_str = f"{ak['same_mv_kstar']*100:.1f}%"
            held_k_str = f"{ak['heldout_mv_kstar']*100:.1f}%"
            held_32_str = f"{ak['heldout_mv_32']*100:.1f}%"
            ret_str = f"{ak['heldout_retention_pct']:.0f}%" if not np.isnan(ak["heldout_retention_pct"]) else "NaN"
            f4_str = f"{ak['fixed_k_accs'][4]*100:.1f}%"
            r4_dep_err_str = f"{ak['reuse_k4_deploy_err_pp']:.1f}pp" if not np.isnan(ak["reuse_k4_deploy_err_pp"]) else "NaN"
            bb_dep_err_str = f"{ak['bb_deploy_err_pp']:.1f}pp" if not np.isnan(ak["bb_deploy_err_pp"]) else "NaN"
            indep_dep_err_str = f"{ak['indep_deploy_err_pp']:.1f}pp" if not np.isnan(ak["indep_deploy_err_pp"]) else "NaN"
            print(
                f"{model:<12} {task:<10} {ak['fold']:<5} {ak['k_star']:>3} {same_str:>10} {held_k_str:>10} "
                f"{held_32_str:>10} {ret_str:>8} {f4_str:>8} {r4_dep_err_str:>9} {bb_dep_err_str:>9} {indep_dep_err_str:>10}"
            )

    print("\n" + "=" * 95)
    print("CLUSTERED BOOTSTRAP UNCERTAINTY (95% CI OVER INSTANCE CLUSTERS)")
    print("=" * 95)
    hdr_bs = f"{'Model':<12} {'Task':<10} {'c(K=4) [95% CI]':<22} {'c(K=32) [95% CI]':<22} {'MV@32 [95% CI]':<22} {'Δ(32-4) [95% CI]':<20}"
    print(hdr_bs)
    print("-" * len(hdr_bs))
    for (model, task), cell_data in admissible_data.items():
        bs = cell_data["bootstrap"]
        c4_ci = f"{bs['c_k4']['point']:.3f} [{bs['c_k4']['ci_lower']:.3f},{bs['c_k4']['ci_upper']:.3f}]"
        c32_ci = f"{bs['c_k32']['point']:.3f} [{bs['c_k32']['ci_lower']:.3f},{bs['c_k32']['ci_upper']:.3f}]"
        mv32_ci = f"{bs['mv32']['point']*100:.1f}% [{bs['mv32']['ci_lower']*100:.1f},{bs['mv32']['ci_upper']*100:.1f}]"
        diff_ci = f"{bs['diff_mv32_mv4']['point']*100:+.1f}pp [{bs['diff_mv32_mv4']['ci_lower']*100:+.1f},{bs['diff_mv32_mv4']['ci_upper']*100:+.1f}]"
        print(f"{model:<12} {task:<10} {c4_ci:<22} {c32_ci:<22} {mv32_ci:<22} {diff_ci:<20}")
    print("=" * 95 + "\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate and analyze v2 SC K-var evaluation logs."
    )
    parser.add_argument(
        "--in-dir",
        "--cache-dir",
        type=Path,
        default=None,
        help="Input cache directory containing JSONL files (default: $DC_RECORDS_DIR or config.CACHE_DIR)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Output directory for CSV tables and provenance markdown (default: config.RESULTS_DIR)",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional filter for model slugs (e.g. qwen7b llama8b)",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional filter for task slugs (e.g. gsm8k boolq)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.025,
        help="Epsilon parameter for Adaptive-K formula (default: 0.025)",
    )
    parser.add_argument(
        "--pilot-frac",
        type=float,
        default=0.5,
        help="Fraction of instances to use as pilot split (default: 0.5)",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=1000,
        help="Number of clustered bootstrap resamples (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for bootstrap reproducibility (default: 2026)",
    )
    parser.add_argument(
        "--min-seeds",
        type=int,
        default=MIN_SEEDS,
        help=f"Minimum number of unique seeds required by protocol gate (default: {MIN_SEEDS})",
    )
    parser.add_argument(
        "--min-instances",
        type=int,
        default=MIN_INSTANCES,
        help=f"Minimum number of unique instances required by protocol gate (default: {MIN_INSTANCES})",
    )
    args = parser.parse_args()

    if args.in_dir is not None:
        in_dir = args.in_dir.resolve()
    elif os.environ.get("DC_RECORDS_DIR"):
        in_dir = Path(os.environ["DC_RECORDS_DIR"]).resolve()
    else:
        in_dir = (experiment_dir("cache")).resolve()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = find_candidate_cells(in_dir, filter_models=args.models, filter_tasks=args.tasks)
    if not candidates:
        print(f"No matching files found in {in_dir}.")
        print("Expected naming format: {model}_sc_kvar_v2_{task}.jsonl")
        sys.exit(0)

    # 1. Evaluate protocol gate on all candidate cells
    gate_results: list[dict] = []
    admissible_data: dict[tuple[str, str], dict] = {}

    for model, task, filepath in candidates:
        if not filepath.exists():
            records = []
        else:
            records = load_and_validate_file(filepath)

        gate_info = evaluate_protocol_gate(
            model, task, filepath, records, min_seeds=args.min_seeds, min_instances=args.min_instances
        )
        gate_info = apply_reporting_policy(gate_info)
        gate_results.append(gate_info)

        if not gate_info["admissible"]:
            continue

        prov = {}
        for r in records:
            if isinstance(r.get("provenance"), dict):
                prov = r["provenance"]
                break

        scorer_name = prov.get("scorer") or get_task_scorer(task)
        unique_seeds = set(r["seed"] for r in records)
        unique_insts = set(r["instance_id"] for r in records)

        # 2. Exact-K metrics
        exact_k = compute_exact_k(records, task, ks=[1, 2, 4, 8, 16, 32])

        # 3 & 4. Adaptive-K and baselines
        ak = evaluate_adaptive_k_and_baselines(
            records, task, eps=args.eps, pilot_frac=args.pilot_frac
        )

        # 5. Clustered bootstrap
        bs = compute_clustered_bootstrap(
            records, task, n_boot=args.n_boot, seed=args.seed
        )

        admissible_data[(model, task)] = {
            "gate": gate_info,
            "provenance": prov,
            "scorer_name": scorer_name,
            "seeds": unique_seeds,
            "n_instances": len(unique_insts),
            "exact_k": exact_k,
            "adaptive_k": ak,
            "bootstrap": bs,
        }

    # Print human-readable summary to stdout
    print_stdout_summary(gate_results, admissible_data)

    # 6. Write CSV tables
    # Gate CSV
    gate_headers = [
        "model", "task", "filepath", "n_records", "expected_records",
        "all_32_answers", "malformed_records", "extraction_rate",
        "harness_loss_rate", "noncompliance_rate",
        "cap_hit_rate", "cap_threshold",
        "admissible", "harness_flag", "reasons"
    ]
    gate_rows = []
    for g in gate_results:
        row = dict(g)
        row["reasons"] = "; ".join(g["reasons"])
        gate_rows.append(row)
    write_csv_file(out_dir / "kvar_v2_gate.csv", gate_headers, gate_rows)

    # Exact-K CSV
    exact_k_headers = [
        "model", "task", "K", "N", "n_seeds", "mean_path_acc",
        "mv_acc", "mean_pairwise_agree", "c", "keff", "ceiling", "pct_ceiling"
    ]
    exact_k_rows = []
    for (model, task), cell_data in admissible_data.items():
        for ek in cell_data["exact_k"]:
            row = {"model": model, "task": task}
            row.update(ek)
            exact_k_rows.append(row)
    write_csv_file(out_dir / "kvar_v2_exact_k.csv", exact_k_headers, exact_k_rows)

    # Adaptive-K and Baselines CSV
    ak_headers = [
        "model", "task", "fold", "n_instances", "n_seeds", "n_pilot_inst", "n_eval_inst",
        "eps", "pilot_acc", "c_pilot", "degenerate", "k_star",
        "same_mv_kstar", "same_mv_32", "same_retention_pct", "same_cost",
        "heldout_mv_kstar", "heldout_mv_32", "heldout_retention_pct", "heldout_cost",
        "fixed_k1_acc", "fixed_k1_cost",
        "fixed_k2_acc", "fixed_k2_cost",
        "fixed_k4_acc", "fixed_k4_cost",
        "fixed_k8_acc", "fixed_k8_cost",
        "fixed_k16_acc", "fixed_k16_cost",
        "fixed_k32_acc", "fixed_k32_cost",
        "heldout_mv_4", "heldout_binary_mv_32",
        "reuse_k4_same_pred", "reuse_k4_same_err_pp",
        "reuse_k4_deploy_pred", "reuse_k4_deploy_err_pp",
        "bb_same_pred", "bb_same_err_pp",
        "bb_deploy_pred", "bb_deploy_err_pp",
        "indep_same_pred", "indep_same_err_pp",
        "indep_deploy_pred", "indep_deploy_err_pp",
    ]
    ak_rows = []
    for (model, task), cell_data in admissible_data.items():
        for ak in cell_data["adaptive_k"]:
            row = {
                "model": model,
                "task": task,
                "fold": ak["fold"],
                "n_instances": ak["n_instances"],
                "n_seeds": ak["n_seeds"],
                "n_pilot_inst": ak["n_pilot_inst"],
                "n_eval_inst": ak["n_eval_inst"],
                "eps": ak["eps"],
                "pilot_acc": ak["pilot_acc"],
                "c_pilot": ak["c_pilot"],
                "degenerate": ak["degenerate"],
                "k_star": ak["k_star"],
                "same_mv_kstar": ak["same_mv_kstar"],
                "same_mv_32": ak["same_mv_32"],
                "same_retention_pct": ak["same_retention_pct"],
                "same_cost": ak["same_cost"],
                "heldout_mv_kstar": ak["heldout_mv_kstar"],
                "heldout_mv_32": ak["heldout_mv_32"],
                "heldout_retention_pct": ak["heldout_retention_pct"],
                "heldout_cost": ak["heldout_cost"],
                "fixed_k1_acc": ak["fixed_k_accs"][1],
                "fixed_k1_cost": 1,
                "fixed_k2_acc": ak["fixed_k_accs"][2],
                "fixed_k2_cost": 2,
                "fixed_k4_acc": ak["fixed_k_accs"][4],
                "fixed_k4_cost": 4,
                "fixed_k8_acc": ak["fixed_k_accs"][8],
                "fixed_k8_cost": 8,
                "fixed_k16_acc": ak["fixed_k_accs"][16],
                "fixed_k16_cost": 16,
                "fixed_k32_acc": ak["fixed_k_accs"][32],
                "fixed_k32_cost": 32,
                "heldout_mv_4": ak["heldout_mv_4"],
                "heldout_binary_mv_32": ak["heldout_binary_mv_32"],
                "reuse_k4_same_pred": ak["reuse_k4_same_pred"],
                "reuse_k4_same_err_pp": ak["reuse_k4_same_err_pp"],
                "reuse_k4_deploy_pred": ak["reuse_k4_deploy_pred"],
                "reuse_k4_deploy_err_pp": ak["reuse_k4_deploy_err_pp"],
                "bb_same_pred": ak["bb_same_pred"],
                "bb_same_err_pp": ak["bb_same_err_pp"],
                "bb_deploy_pred": ak["bb_deploy_pred"],
                "bb_deploy_err_pp": ak["bb_deploy_err_pp"],
                "indep_same_pred": ak["indep_same_pred"],
                "indep_same_err_pp": ak["indep_same_err_pp"],
                "indep_deploy_pred": ak["indep_deploy_pred"],
                "indep_deploy_err_pp": ak["indep_deploy_err_pp"],
            }
            ak_rows.append(row)
    write_csv_file(out_dir / "kvar_v2_adaptive_k.csv", ak_headers, ak_rows)

    # Bootstrap CSV
    boot_headers = [
        "model", "task", "n_boot", "seed",
        "c_k4_point", "c_k4_ci_lower", "c_k4_ci_upper",
        "c_k32_point", "c_k32_ci_lower", "c_k32_ci_upper",
        "mv32_point", "mv32_ci_lower", "mv32_ci_upper",
        "diff_mv32_mv4_point", "diff_mv32_mv4_ci_lower", "diff_mv32_mv4_ci_upper"
    ]
    boot_rows = []
    for (model, task), cell_data in admissible_data.items():
        bs = cell_data["bootstrap"]
        row = {
            "model": model,
            "task": task,
            "n_boot": bs["n_boot"],
            "seed": bs["seed"],
            "c_k4_point": bs["c_k4"]["point"],
            "c_k4_ci_lower": bs["c_k4"]["ci_lower"],
            "c_k4_ci_upper": bs["c_k4"]["ci_upper"],
            "c_k32_point": bs["c_k32"]["point"],
            "c_k32_ci_lower": bs["c_k32"]["ci_lower"],
            "c_k32_ci_upper": bs["c_k32"]["ci_upper"],
            "mv32_point": bs["mv32"]["point"],
            "mv32_ci_lower": bs["mv32"]["ci_lower"],
            "mv32_ci_upper": bs["mv32"]["ci_upper"],
            "diff_mv32_mv4_point": bs["diff_mv32_mv4"]["point"],
            "diff_mv32_mv4_ci_lower": bs["diff_mv32_mv4"]["ci_lower"],
            "diff_mv32_mv4_ci_upper": bs["diff_mv32_mv4"]["ci_upper"],
        }
        boot_rows.append(row)
    write_csv_file(out_dir / "kvar_v2_bootstrap.csv", boot_headers, boot_rows)

    # Provenance Markdown
    write_provenance_markdown(out_dir / "provenance_v2.md", gate_results, admissible_data)
    print(f"Results written to {out_dir}/:")
    print("  - kvar_v2_gate.csv")
    print("  - kvar_v2_exact_k.csv")
    print("  - kvar_v2_adaptive_k.csv")
    print("  - kvar_v2_bootstrap.csv")
    print("  - provenance_v2.md")


if __name__ == "__main__":
    main()
