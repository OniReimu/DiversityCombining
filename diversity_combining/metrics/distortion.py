"""Distortion metric: D = fidelity loss vs. explicit multi-path search.

Distortion measures how much accuracy is lost when using continuous aggregation
compared to the discrete SC-32 reference at matched compute.
"""

from typing import Callable

import numpy as np


def compute_distortion(
    method_accuracy: float,
    reference_accuracy: float,
) -> float:
    """D = accuracy gap relative to SC-32 reference.

    D < 0 means the method outperforms the reference (negative distortion = better).
    D > 0 means the method underperforms (positive distortion = worse).

    Args:
        method_accuracy: Accuracy of the evaluated method.
        reference_accuracy: Accuracy of SC-32 at matched compute.
    """
    return reference_accuracy - method_accuracy


def compute_distortion_with_components(
    method_answers: list[str],
    reference_answers: list[str],
    ground_truth: list[str],
    method_all_candidates: list[list[str]] | None = None,
    n_reruns: int = 5,
    rerun_answers: list[list[str]] | None = None,
    match_fn: Callable[[str, str], bool] | None = None,
    score_fn: Callable[[str, str], float] | None = None,
) -> dict:
    """Compute distortion with component breakdown.

    Args:
        match_fn: Callable(prediction, gold) -> bool for correctness (used for
                  coverage failure). Defaults to exact string equality.
        score_fn: Callable(prediction, gold) -> float for accuracy scoring.
                  If provided, accuracy is the mean of score_fn outputs (e.g. raw F1).
                  If None, accuracy is the mean of match_fn outputs (0/1).

    Returns:
        Dict with keys:
        - distortion: Overall D
        - coverage_failure_rate: Fraction where correct answer was in candidates
                                  but aggregation selected wrong answer
        - instability: Disagreement rate across independent reruns
    """
    if match_fn is None:
        match_fn = lambda m, g: m == g

    n = len(ground_truth)

    if score_fn is not None:
        method_acc = sum(score_fn(m, g) for m, g in zip(method_answers, ground_truth)) / n
        ref_acc = sum(score_fn(r, g) for r, g in zip(reference_answers, ground_truth)) / n
    else:
        method_acc = sum(match_fn(m, g) for m, g in zip(method_answers, ground_truth)) / n
        ref_acc = sum(match_fn(r, g) for r, g in zip(reference_answers, ground_truth)) / n

    D = ref_acc - method_acc

    # Coverage failure: correct in candidates but aggregation picked wrong
    coverage_failure = 0.0
    if method_all_candidates is not None:
        n_failures = 0
        for i, (ans, gt, cands) in enumerate(
            zip(method_answers, ground_truth, method_all_candidates)
        ):
            if not match_fn(ans, gt) and any(match_fn(c, gt) for c in cands):
                n_failures += 1
        coverage_failure = n_failures / n

    # Instability: disagreement across reruns
    instability = 0.0
    if rerun_answers is not None and len(rerun_answers) >= 2:
        disagreements = 0
        total_pairs = 0
        for i in range(n):
            instance_answers = [run[i] for run in rerun_answers]
            for j in range(len(instance_answers)):
                for k in range(j + 1, len(instance_answers)):
                    total_pairs += 1
                    if instance_answers[j] != instance_answers[k]:
                        disagreements += 1
        instability = disagreements / max(total_pairs, 1)

    return {
        "distortion": D,
        "method_accuracy": method_acc,
        "reference_accuracy": ref_acc,
        "coverage_failure_rate": coverage_failure,
        "instability": instability,
    }
