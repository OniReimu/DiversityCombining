"""Soft-K Aggregation / Multiplex Thinking method.

This is the core continuous-thought method that Diversity Combining primarily evaluates.
It wraps the Multiplex Thinking mechanism from Tang et al. (2026).

Two execution modes:
1. Native (via patched SGLang) — uses the upstream multiplex thinking inference
2. Standalone (pure PyTorch) — post-hoc implementation for models without SGLang
"""

import torch
import torch.nn.functional as F

from diversity_combining.methods.base import ReasoningMethod, ReasoningOutput


class SoftKMethod(ReasoningMethod):
    """Soft-K / Multiplex Thinking: Sample K tokens, aggregate embeddings.

    At each reasoning step:
    1. Get next-token distribution p(v | context)
    2. Sample K tokens independently: k_1, ..., k_K ~ p
    3. Look up embeddings: e_1, ..., e_K
    4. Aggregate: c = Σ p(k_j) * e(k_j) / Σ p(k_j)  (probability-weighted)
    5. Feed c as next input embedding

    This is the "Soft Aggregation" method in Diversity Combining = the same mechanism as
    Multiplex Thinking (Tang et al., 2026).
    """

    def __init__(self, model, tokenizer, K: int = 8, temperature: float = 0.7,
                 top_p: float = 0.95, aggregation: str = "prob_weighted",
                 device: str = "cuda"):
        super().__init__(model, tokenizer, device)
        self.K = K
        self.temperature = temperature
        self.top_p = top_p
        self.aggregation = aggregation  # "prob_weighted", "mean", "learned_gate"

    @property
    def name(self) -> str:
        return f"soft-{self.K}"

    def _sample_and_aggregate(
        self, logits: torch.Tensor, embed_weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample K tokens and aggregate their embeddings.

        Args:
            logits: [V] next-token logits
            embed_weight: [V, D] embedding matrix

        Returns:
            aggregated: [D] the multiplex embedding
            probs: [K] probability of each sampled token
            indices: [K] token IDs
            candidate_embeds: [K, D] embeddings of sampled tokens
        """
        # Apply temperature
        scaled_logits = logits / self.temperature
        probs_full = F.softmax(scaled_logits, dim=-1)

        # Top-p (nucleus) filtering
        sorted_probs, sorted_indices = torch.sort(probs_full, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs - sorted_probs > self.top_p
        sorted_probs[mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum()

        # Sample K tokens
        sampled_positions = torch.multinomial(sorted_probs, num_samples=self.K, replacement=True)
        indices = sorted_indices[sampled_positions]  # [K]
        sample_probs = probs_full[indices]  # [K] — use original probs for weighting

        # Look up embeddings
        candidate_embeds = embed_weight[indices]  # [K, D]

        # Aggregate
        if self.aggregation == "prob_weighted":
            weights = sample_probs / sample_probs.sum()
            aggregated = torch.sum(weights.unsqueeze(-1) * candidate_embeds, dim=0)  # [D]
        elif self.aggregation == "mean":
            aggregated = candidate_embeds.mean(dim=0)  # [D]
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return aggregated, sample_probs, indices, candidate_embeds

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        collect_attentions: bool = False,
        **kwargs,
    ) -> ReasoningOutput:
        """Generate using step-by-step soft-K aggregation.

        This is the standalone (pure PyTorch) implementation.
        For native SGLang execution, use SoftKSGLangMethod instead.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids  # [1, S]
        embed_layer = self.model.get_input_embeddings()
        embed_weight = embed_layer.weight  # [V, D]

        # Start with standard embeddings for the prompt
        past_key_values = None
        input_embeds = embed_layer(input_ids)  # [1, S, D]

        all_candidate_probs = []
        all_candidate_embeddings = []
        all_aggregated_embeddings = []
        all_hidden_states = []
        all_attention_patterns = []
        generated_token_ids = []

        for step in range(max_tokens):
            # Forward pass
            model_output = self.model(
                inputs_embeds=input_embeds,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=collect_attentions,
            )

            # Get logits for next token
            logits = model_output.logits[0, -1, :]  # [V]
            past_key_values = model_output.past_key_values

            # Collect hidden states and attention
            last_hidden = model_output.hidden_states[-1][0, -1, :].cpu()
            all_hidden_states.append(last_hidden)

            if model_output.attentions is not None and model_output.attentions[-1] is not None:
                last_attn = model_output.attentions[-1][0, :, -1, :].cpu()
                all_attention_patterns.append(last_attn)

            # Check for EOS
            top_token = logits.argmax().item()
            if top_token == self.tokenizer.eos_token_id:
                break

            # Sample K tokens and aggregate
            aggregated, probs, indices, candidate_embeds = self._sample_and_aggregate(
                logits, embed_weight
            )

            all_candidate_probs.append(probs.cpu())
            all_candidate_embeddings.append(candidate_embeds.cpu())
            all_aggregated_embeddings.append(aggregated.cpu())

            # Use the most probable sampled token as the "discrete" output
            best_idx = probs.argmax()
            generated_token_ids.append(indices[best_idx].item())

            # Feed aggregated embedding as next input
            input_embeds = aggregated.unsqueeze(0).unsqueeze(0)  # [1, 1, D]

        answer = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        n_steps = len(generated_token_ids)

        return ReasoningOutput(
            answer=answer,
            candidate_probs=all_candidate_probs,
            candidate_embeddings=all_candidate_embeddings,
            aggregated_embeddings=all_aggregated_embeddings,
            hidden_states=all_hidden_states,
            attention_patterns=all_attention_patterns,
            # Compute accounting: each step uses ONE forward pass to get logits,
            # then samples K tokens from the distribution (no additional FPs).
            # The K-sampling is just indexing into the softmax output, not K
            # separate forward passes. +1 for the initial prompt encoding pass.
            forward_passes=n_steps + 1,
            total_tokens_generated=n_steps,
            metadata={
                "method": self.name,
                "K": self.K,
                "aggregation": self.aggregation,
                "temperature": self.temperature,
            },
        )
