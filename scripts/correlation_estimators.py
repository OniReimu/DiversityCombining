#!/usr/bin/env python3
"""Pooled correctness correlation c-hat (Eq. 3 of the paper) of a binary (N, K) matrix."""
from __future__ import annotations

from itertools import combinations

import numpy as np


# ------------------------------------------------------------------ estimators

def c_eq3(mat: np.ndarray) -> float:
    """Manuscript Eq. (3): pooled p_bar in every path pair."""
    n, k = mat.shape
    if n < 2 or k < 2:
        return float("nan")
    p_bar = float(mat.mean())
    denom = p_bar * (1.0 - p_bar)
    if denom <= 0:
        return float("nan")
    cross = [float((mat[:, j] * mat[:, j2]).mean()) - p_bar ** 2
             for j, j2 in combinations(range(k), 2)]
    return float(np.mean(cross)) / denom
