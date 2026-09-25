"""Greedy single-path decoding baseline."""

import torch

from diversity_combining.methods.base import ReasoningMethod, ReasoningOutput


class GreedyMethod(ReasoningMethod):
    """Standard greedy autoregressive decoding (1 forward pass per token)."""

    @property
    def name(self) -> str:
        return "greedy"

    def _get_stop_token_ids(self) -> set[int]:
        """Collect all stop token IDs including chat template markers."""
        stop_ids = set()
        # Standard EOS
        eos = self.tokenizer.eos_token_id
        if isinstance(eos, list):
            stop_ids.update(eos)
        elif eos is not None:
            stop_ids.add(eos)
        # Chat template stop tokens (Qwen: <|im_end|>, <|endoftext|>)
        for tok_name in ["<|im_end|>", "<|im_start|>", "<|endoftext|>",
                         "<|eot_id|>", "</s>"]:
            tok_id = self.tokenizer.convert_tokens_to_ids(tok_name)
            if tok_id is not None and tok_id != self.tokenizer.unk_token_id:
                stop_ids.add(tok_id)
        return stop_ids

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        collect_internals: bool = True,
        **kwargs,
    ) -> ReasoningOutput:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids  # [1, S]

        hidden_states = []
        attention_patterns = []
        candidate_probs = []
        candidate_embeddings = []
        aggregated_embeddings = []
        generated_token_ids = []

        past_key_values = None
        current_input = input_ids
        stop_ids = self._get_stop_token_ids()

        embed_weight = self.model.get_input_embeddings().weight

        for step in range(max_tokens):
            model_output = self.model(
                input_ids=current_input,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=collect_internals,
                output_attentions=collect_internals,
            )

            logits = model_output.logits[0, -1, :]  # [V]
            past_key_values = model_output.past_key_values

            # Greedy: pick argmax
            next_token = logits.argmax().item()
            if next_token in stop_ids:
                break

            generated_token_ids.append(next_token)

            # Collect internals
            if collect_internals:
                if model_output.hidden_states is not None:
                    last_layer = model_output.hidden_states[-1]
                    hidden_states.append(last_layer[0, -1, :].cpu())

                if model_output.attentions is not None and model_output.attentions[-1] is not None:
                    last_attn = model_output.attentions[-1]
                    attention_patterns.append(last_attn[0, :, -1, :].cpu())

            # Single candidate, prob = 1.0
            candidate_probs.append(torch.tensor([1.0]))
            emb = embed_weight[next_token].detach().cpu()
            candidate_embeddings.append(emb.unsqueeze(0))
            aggregated_embeddings.append(emb)

            current_input = torch.tensor([[next_token]], device=self.device)

        answer = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        n_tokens = len(generated_token_ids)

        return ReasoningOutput(
            answer=answer,
            candidate_probs=candidate_probs,
            candidate_embeddings=candidate_embeddings,
            aggregated_embeddings=aggregated_embeddings,
            hidden_states=hidden_states,
            attention_patterns=attention_patterns,
            forward_passes=n_tokens,
            total_tokens_generated=n_tokens,
            metadata={"method": "greedy"},
        )
