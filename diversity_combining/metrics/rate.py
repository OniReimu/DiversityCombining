"""Rate metrics: R_dist, R_geom, R_IT.

Reasoning Rate measures how much hypothesis information a single
continuous-thought step carries.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats


# ── R_dist: Distributional Rate ─────────────────────────────────────────────

def distributional_rate(probs: torch.Tensor) -> float:
    """R_dist = H(p) = -Σ p_k log₂ p_k.

    Entropy of the candidate probability distribution.
    Range: 0 bits (single candidate) to log₂(K) bits (uniform over K).

    Args:
        probs: [K] probability distribution over K candidates.
    """
    p = probs[probs > 1e-15].float()
    p = p / p.sum()  # Normalize to valid distribution before computing entropy
    return float(-torch.sum(p * torch.log2(p)).item())


def distributional_rate_from_logits(logits: torch.Tensor, K: int) -> float:
    """Compute R_dist from raw logits by taking top-K probabilities.

    Args:
        logits: [V] next-token logits.
        K: Number of candidates to consider.
    """
    probs = F.softmax(logits, dim=-1)
    topk_probs, _ = torch.topk(probs, K)
    topk_probs = topk_probs / topk_probs.sum()  # Renormalize
    return distributional_rate(topk_probs)


# ── R_geom: Geometric Rate ──────────────────────────────────────────────────

def geometric_rate(embeddings: torch.Tensor) -> float:
    """R_geom = participation ratio of aggregated representation covariance.

    Effective dimensionality of information-carrying subspace.
    Formula: (Σ λ_i)² / Σ λ_i²

    Args:
        embeddings: [N, D] collection of aggregated embeddings across steps.
                    N = number of reasoning steps, D = embedding dimension.
    """
    if embeddings.shape[0] < 2:
        return 1.0

    # Center the embeddings
    embeddings = embeddings.float()
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)

    # Covariance matrix
    cov = (centered.T @ centered) / (centered.shape[0] - 1)

    # Eigenvalues
    eigenvalues = torch.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) == 0:
        return 1.0

    # Participation ratio
    sum_lambda = eigenvalues.sum()
    sum_lambda_sq = (eigenvalues ** 2).sum()

    return float((sum_lambda ** 2 / sum_lambda_sq).item())


# ── R_IT: Information-Theoretic Rate ────────────────────────────────────────

class MIEstimator(nn.Module):
    """Variational MI estimator using Barber-Agakov bound.

    Estimates I(Z; Y) where Z is the latent state and Y is the correct answer.
    Uses a small 2-layer MLP as the variational posterior q(y|z).

    MI ≥ E[log q(y|z)] + H(Y)  (Barber-Agakov bound)
    """

    def __init__(self, input_dim: int, n_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict log q(y|z)."""
        return F.log_softmax(self.net(z), dim=-1)


def information_theoretic_rate(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
    compute_budget: float,
    hidden_dim: int = 256,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cuda",
) -> float:
    """R_IT = I(Z; Y) / C_budget.

    Estimates mutual information between latent state Z and correct answer Y,
    normalized by compute budget for fair comparison.

    Uses variational lower bound (Barber-Agakov):
    I(Z;Y) ≥ E_{p(z,y)}[log q(y|z)] + H(Y)

    Args:
        hidden_states: [N, D] latent representations.
        labels: [N] integer class labels.
        n_classes: Number of unique answer classes.
        compute_budget: Forward passes used (for normalization).
        hidden_dim: MLP hidden dimension.
        epochs: Training epochs for MI estimator.
        lr: Learning rate.
    """
    if hidden_states.shape[0] < 10:
        return 0.0

    estimator = MIEstimator(hidden_states.shape[1], n_classes, hidden_dim).to(device)
    optimizer = torch.optim.Adam(estimator.parameters(), lr=lr)

    hidden_states = hidden_states.to(device)
    labels = labels.to(device)

    # Train variational posterior
    for _ in range(epochs):
        log_qy_z = estimator(hidden_states)
        loss = F.nll_loss(log_qy_z, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Estimate MI lower bound
    with torch.no_grad():
        log_qy_z = estimator(hidden_states)
        # E[log q(y|z)]
        conditional_entropy_bound = -F.nll_loss(log_qy_z, labels).item()

        # H(Y) — marginal entropy of labels
        label_counts = torch.bincount(labels, minlength=n_classes).float()
        label_probs = label_counts / label_counts.sum()
        h_y = -torch.sum(label_probs[label_probs > 0] * torch.log2(label_probs[label_probs > 0]))
        h_y = h_y.item()

    # MI ≥ E[log q(y|z)] + H(Y)  (in bits, convert from nats)
    mi_lower_bound = (conditional_entropy_bound / np.log(2)) + h_y
    mi_lower_bound = max(0.0, mi_lower_bound)

    # Normalize by compute budget
    return mi_lower_bound / max(compute_budget, 1.0)
