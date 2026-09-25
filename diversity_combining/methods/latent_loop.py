"""Latent Loop (LL-T): Feed hidden state back for T iterations without generating tokens.

Inspired by Coconut (Hao et al., 2025) and pause tokens (Goyal et al., 2024).
The model processes the same position T times, each time feeding the last hidden
state back as input, before generating the next discrete token.
"""

import copy

import torch

from diversity_combining.methods.base import ReasoningMethod, ReasoningOutput


class LatentLoopMethod(ReasoningMethod):
    """Latent Loop: Internal recurrence without token generation.

    For each reasoning position:
    1. Run forward pass to get hidden state h
    2. Project h to vocab logits, take argmax to get a discrete token
    3. Feed that token's embedding as input for another forward pass
    4. Repeat for T iterations
    5. Use final h to predict next token

    The projection uses argmax over the vocabulary (straight-through discrete),
    ensuring the loop embedding is always a real token embedding that the model
    can process. The soft distribution is kept for R-D metric computation.

    Compute cost per token: T forward passes (vs 1 for greedy).
    """

    def __init__(self, model, tokenizer, T: int = 8, device: str = "cuda"):
        super().__init__(model, tokenizer, device)
        self.T = T

    @property
    def name(self) -> str:
        return f"ll-{self.T}"

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        **kwargs,
    ) -> ReasoningOutput:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        embed_layer = self.model.get_input_embeddings()
        embed_weight = embed_layer.weight  # [V, D]

        past_key_values = None
        input_embeds = embed_layer(input_ids)

        all_hidden_states = []
        all_attention_patterns = []
        all_candidate_probs = []
        all_candidate_embeddings = []
        all_aggregated_embeddings = []
        generated_token_ids = []
        total_forward_passes = 0

        for step in range(max_tokens):
            # First forward pass: standard autoregressive with full KV cache
            model_output = self.model(
                inputs_embeds=input_embeds,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=(self.T == 1),  # Only need attention if no loop
            )
            total_forward_passes += 1

            # Save KV cache that includes prompt + all prior tokens + current token
            step_kv = model_output.past_key_values
            last_hidden = model_output.hidden_states[-1][0, -1, :]  # [D_hidden]

            # Latent loop: T-1 additional iterations at this position
            loop_output = None
            for t in range(self.T - 1):
                # Project hidden state to vocab space via logits
                similarities = torch.matmul(
                    last_hidden.unsqueeze(0),
                    embed_weight.T,
                )  # [1, V]

                # Compute soft distribution for metric tracking
                soft_probs = torch.softmax(similarities / 1.0, dim=-1)

                # Discrete projection: argmax token → real embedding
                loop_token_id = similarities.argmax(dim=-1)  # [1]
                loop_embed = embed_layer(loop_token_id).unsqueeze(0)  # [1, 1, D]

                # Capture candidate distribution from the last projection step
                if t == self.T - 2:
                    K = min(self.T, soft_probs.shape[-1])
                    topk_probs, topk_indices = torch.topk(soft_probs.squeeze(0), K)
                    topk_embeds = embed_weight[topk_indices]  # [K, D]
                    all_candidate_probs.append(topk_probs.cpu())
                    all_candidate_embeddings.append(topk_embeds.cpu())
                    all_aggregated_embeddings.append(
                        embed_weight[loop_token_id.squeeze()].cpu()
                    )

                # Feed loop_embed with a SNAPSHOT of step_kv (full context preserved).
                # DynamicCache.update() mutates in-place even with use_cache=False,
                # so we deep-clone to keep step_kv pristine for the next loop iter
                # and for the next autoregressive step.
                loop_kv = copy.deepcopy(step_kv)
                loop_output = self.model(
                    inputs_embeds=loop_embed,
                    past_key_values=loop_kv,
                    use_cache=False,
                    output_hidden_states=True,
                    output_attentions=(t == self.T - 2),  # Attention on last loop iter
                )
                total_forward_passes += 1
                last_hidden = loop_output.hidden_states[-1][0, -1, :]

            # After loop: use step_kv for next autoregressive step
            past_key_values = step_kv

            # Get logits from final output (loop_output if we looped, else model_output)
            final_output = loop_output if loop_output is not None else model_output
            logits = final_output.logits[0, -1, :]

            # Collect metrics data
            all_hidden_states.append(last_hidden.cpu())
            if final_output.attentions is not None:
                last_attn = final_output.attentions[-1][0, :, -1, :].cpu()
                all_attention_patterns.append(last_attn)

            # Greedy decode from final logits
            next_token = logits.argmax().item()
            if next_token == self.tokenizer.eos_token_id:
                break
            generated_token_ids.append(next_token)

            # Standard embedding for next step
            input_embeds = embed_layer(
                torch.tensor([[next_token]], device=self.device)
            )

        answer = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)

        return ReasoningOutput(
            answer=answer,
            candidate_probs=all_candidate_probs,
            candidate_embeddings=all_candidate_embeddings,
            aggregated_embeddings=all_aggregated_embeddings,
            hidden_states=all_hidden_states,
            attention_patterns=all_attention_patterns,
            forward_passes=total_forward_passes,
            total_tokens_generated=len(generated_token_ids),
            metadata={
                "method": self.name,
                "T": self.T,
                "iterations_per_token": self.T,
            },
        )
