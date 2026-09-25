"""Base class for reasoning methods."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch


@dataclass
class ReasoningOutput:
    """Output from a single reasoning invocation."""

    answer: str
    is_correct: bool | None = None
    # Per-step data for metric computation
    candidate_logits: list[torch.Tensor] = field(default_factory=list)  # [steps] x [V]
    candidate_probs: list[torch.Tensor] = field(default_factory=list)  # [steps] x [K]
    candidate_embeddings: list[torch.Tensor] = field(default_factory=list)  # [steps] x [K, D]
    aggregated_embeddings: list[torch.Tensor] = field(default_factory=list)  # [steps] x [D]
    hidden_states: list[torch.Tensor] = field(default_factory=list)  # [steps] x [L, D]
    attention_patterns: list[torch.Tensor] = field(default_factory=list)  # [steps] x [H, S, S]
    # Compute accounting
    forward_passes: int = 0
    total_tokens_generated: int = 0
    metadata: dict = field(default_factory=dict)


class ReasoningMethod(ABC):
    """Abstract base class for all reasoning methods.

    Each method must:
    1. Generate reasoning traces from a model given a prompt
    2. Track compute cost (forward passes) for fair comparison
    3. Expose intermediate representations for Diversity Combining metric computation
    """

    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        **kwargs,
    ) -> ReasoningOutput:
        """Generate a reasoning trace and return structured output."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Method identifier for logging."""
        ...

    def compute_budget(self, output: ReasoningOutput) -> float:
        """Normalized compute cost = forward_passes / base_forward_passes."""
        return output.forward_passes
