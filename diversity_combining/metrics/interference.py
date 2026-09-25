"""Interference metrics: I_sep, I_head, I_conf.

Measures superposition-induced cross-talk that limits the model's
ability to recover individual hypotheses from the aggregated representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression


# ── I_sep: Separability ─────────────────────────────────────────────────────

def separability(
    aggregated_embeddings: torch.Tensor,
    dominant_labels: torch.Tensor,
) -> float:
    """I_sep = linear probe accuracy predicting which hypothesis dominates.

    High separability (→1) = downstream layers can recover individual hypotheses.
    Low separability (→1/K) = destructive interference, hypotheses indistinguishable.

    Args:
        aggregated_embeddings: [N, D] aggregated multiplex embeddings.
        dominant_labels: [N] integer labels indicating which of K candidates
                         had the highest probability at each step.
    """
    if aggregated_embeddings.shape[0] < 10:
        return 0.5

    X = aggregated_embeddings.numpy()
    y = dominant_labels.numpy()

    n_classes = len(np.unique(y))
    if n_classes < 2:
        return 1.0

    # Stratified train/test split
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import balanced_accuracy_score

    clf = LogisticRegression(max_iter=500, multi_class="multinomial")
    try:
        scores = cross_val_score(clf, X, y, cv=min(5, aggregated_embeddings.shape[0] // 2),
                                 scoring='balanced_accuracy')
        return float(np.mean(scores))
    except ValueError:
        # Not enough samples for cross-validation
        clf.fit(X, y)
        y_pred = clf.predict(X)
        return float(balanced_accuracy_score(y, y_pred))


# ── I_head: Head Specialization ─────────────────────────────────────────────

def head_specialization(attention_patterns) -> float:
    """I_head = inverse of mean pairwise cosine similarity among attention heads.

    High specialization (→1) = different heads attend to different hypotheses.
    Low specialization (→0) = heads collapse to similar patterns.

    Handles variable-length attention patterns (S_t varies per step),
    which occurs in autoregressive decoding where each step attends to
    all previous positions.

    Args:
        attention_patterns: [N, H, S] attention weights across N steps,
                           H heads, S source positions.
                           Or a list of [H, S_t] tensors (variable length).
    """
    if isinstance(attention_patterns, list):
        if len(attention_patterns) == 0:
            return 0.5
        # Check if all tensors have the same shape (can stack)
        shapes = set(tuple(t.shape) for t in attention_patterns)
        if len(shapes) == 1:
            attention_patterns = torch.stack(attention_patterns)  # [N, H, S]
        else:
            # Variable-length: compute per-step similarity and average
            per_step_sims = []
            for attn in attention_patterns:
                # attn: [H, S_t]
                if attn.dim() != 2:
                    continue
                H = attn.shape[0]
                if H < 2:
                    continue
                normed = F.normalize(attn.float(), p=2, dim=-1)  # [H, S_t]
                sim = normed @ normed.T  # [H, H]
                mask = ~torch.eye(H, dtype=torch.bool)
                mean_sim = sim[mask].mean().item()
                per_step_sims.append(mean_sim)
            if not per_step_sims:
                return 0.5
            return float(1.0 - np.mean(per_step_sims))

    # Stacked tensor path (original)
    if attention_patterns.dim() != 3:
        return 0.5

    N, H, S = attention_patterns.shape

    if H < 2:
        return 1.0

    # Compute mean attention pattern per head across steps
    mean_patterns = attention_patterns.mean(dim=0)  # [H, S]

    # Normalize
    mean_patterns = F.normalize(mean_patterns.float(), p=2, dim=-1)

    # Pairwise cosine similarity
    sim_matrix = mean_patterns @ mean_patterns.T  # [H, H]

    # Mean off-diagonal similarity
    mask = ~torch.eye(H, dtype=torch.bool)
    mean_sim = sim_matrix[mask].mean().item()

    # Specialization = 1 - mean similarity
    return float(1.0 - mean_sim)


# ── I_conf: Confusability ───────────────────────────────────────────────────

def confusability(
    candidate_embeddings: torch.Tensor,
    candidate_probs: torch.Tensor,
) -> float:
    """I_conf = probability-weighted mean pairwise cosine similarity.

    High confusability (→1) = candidates geometrically overlap → destructive interference.
    Low confusability (→0) = candidates well-separated → decodable.

    Args:
        candidate_embeddings: [K, D] embeddings of K sampled tokens.
        candidate_probs: [K] probabilities of each candidate.
    """
    K = candidate_embeddings.shape[0]
    if K < 2:
        return 0.0

    # Normalize embeddings
    normed = F.normalize(candidate_embeddings.float(), p=2, dim=-1)  # [K, D]

    # Pairwise cosine similarity
    sim_matrix = normed @ normed.T  # [K, K]

    # Probability-weighted mean of off-diagonal entries
    probs = candidate_probs.float()
    probs = probs / probs.sum()

    total_sim = 0.0
    total_weight = 0.0
    for i in range(K):
        for j in range(i + 1, K):
            w = probs[i] * probs[j]
            total_sim += w * sim_matrix[i, j].item()
            total_weight += w

    if total_weight < 1e-10:
        return 0.0

    return float(total_sim / total_weight)


# ── Aggregate interference from per-step measurements ───────────────────────

def aggregate_interference(
    step_confusabilities: list[float],
    step_separabilities: list[float],
    head_spec: float,
) -> dict:
    """Aggregate per-step interference into summary statistics.

    Returns:
        Dict with mean, std, and phase transition indicators.
    """
    conf = np.array(step_confusabilities)
    sep = np.array(step_separabilities)

    return {
        "I_conf_mean": float(conf.mean()),
        "I_conf_std": float(conf.std()),
        "I_sep_mean": float(sep.mean()),
        "I_sep_std": float(sep.std()),
        "I_head": head_spec,
        # Phase transition indicators
        "semantic_collapse_fraction": float((conf > 0.8).mean()),
        "destructive_interference_fraction": float((sep < 0.4).mean()),
    }
