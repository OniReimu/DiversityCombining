#!/usr/bin/env python3
"""Correlation, majority-vote and K_eff estimators on a binary (N, K) correctness matrix.

compute_pairwise_rho is the mean pairwise Pearson correlation over the K columns
(constant columns skipped) and _mean_pairwise_rho_vec its vectorized form used by the
bootstrap; compute_mv_accuracy is the strict majority vote with half credit at exact
ties; k_eff(K, rho) = K / (1 + (K - 1) rho). BASE_ACC_THRESHOLD is the base-accuracy
exclusion rule of the paper.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

BASE_ACC_THRESHOLD = 0.02  # base-accuracy exclusion rule of the paper


# ---------- primitives ----------

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
