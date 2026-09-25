"""Self-Consistency (SC-K): Sample K paths, majority vote."""

import re
from collections import Counter

import torch

from diversity_combining.methods.base import ReasoningMethod, ReasoningOutput


def _sc_strip_artifacts(text: str) -> str:
    """Strip chat template turn markers that leak on truncation."""
    for pat in [r"\nuser\n", r"\nUser:", r"\n<\|im_start\|>", r"\nassistant\n"]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            text = text[:m.start()]
    return text.strip()


def _sc_extract_number(text: str) -> str | None:
    """Extract final number — lightweight version for majority vote."""
    text = _sc_strip_artifacts(text)
    # \boxed{X}
    boxed = list(re.finditer(r"\\boxed\{([^}]+)\}", text))
    if boxed:
        nums = re.findall(r"-?[\d,]+\.?\d*", boxed[-1].group(1))
        if nums:
            return nums[-1].replace(",", "")
    # #### X
    m = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if m:
        return m.group(1).replace(",", "")
    # "answer is X"
    m = re.search(r"(?:answer|result|total)\s*(?:is|=|:)\s*\$?(-?[\d,]+\.?\d*)", text, re.I)
    if m:
        return m.group(1).replace(",", "")
    # Last number
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


class SelfConsistencyMethod(ReasoningMethod):
    """Self-Consistency with K independent reasoning paths.

    Wang et al. (2023) — Sample K CoT paths, extract answers, majority vote.
    Compute cost: K * base_forward_passes.
    """

    def __init__(self, model, tokenizer, K: int = 8, temperature: float = 0.7,
                 top_p: float = 0.95, device: str = "cuda",
                 temperatures: list[float] | None = None):
        super().__init__(model, tokenizer, device)
        self.K = K
        self.temperature = temperature
        self.top_p = top_p
        # Multi-temperature mode: each path k uses temperatures[k]
        # When set, creates non-exchangeable paths for GLS combining
        self.temperatures = temperatures

    @property
    def name(self) -> str:
        return f"sc-{self.K}"

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        extract_answer_fn=None,
        **kwargs,
    ) -> ReasoningOutput:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]

        all_answers = []
        all_traces = []
        total_forward_passes = 0
        total_tokens = 0

        # Collect hidden states and embeddings from all K paths
        path_embeddings = []

        # Collect stop token IDs for chat models
        stop_ids = []
        for tok_name in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>"]:
            tid = self.tokenizer.convert_tokens_to_ids(tok_name)
            if tid is not None and tid != self.tokenizer.unk_token_id:
                stop_ids.append(tid)

        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        if stop_ids:
            gen_kwargs["eos_token_id"] = stop_ids

        for k in range(self.K):
            # Multi-temperature: override temperature for path k
            if self.temperatures is not None and k < len(self.temperatures):
                gen_kwargs["temperature"] = self.temperatures[k]
            outputs = self.model.generate(**inputs, **gen_kwargs)

            generated_ids = outputs.sequences[0, input_len:]
            trace = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            all_traces.append(trace)

            n_tokens = len(generated_ids)
            total_forward_passes += n_tokens
            total_tokens += n_tokens

            # Extract last hidden state of last generated token
            # outputs.hidden_states is a tuple of (n_generated_tokens) tuples of
            # (n_layers) tensors, each of shape [batch, seq_len, D]
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                # Last generated token, last layer, batch=0, last position
                last_step_hidden = outputs.hidden_states[-1][-1][0, -1, :]  # [D]
                path_embeddings.append(last_step_hidden.cpu())

            # Extract answer from trace — use numeric extraction for
            # robust majority vote (raw text comparison fails across paths)
            trace = _sc_strip_artifacts(trace)
            if extract_answer_fn is not None:
                answer = extract_answer_fn(trace)
            else:
                extracted = _sc_extract_number(trace)
                answer = extracted if extracted is not None else trace.strip().split("\n")[-1]
            all_answers.append(answer)

        # Majority vote
        answer_counts = Counter(all_answers)
        majority_answer = answer_counts.most_common(1)[0][0]

        # Build candidate probability distribution — per-path uniform weights
        # Each of the K paths gets equal probability 1/K, matching the [K, D]
        # candidate_embeddings so that I_conf can zip them without length mismatch.
        unique_answers = list(answer_counts.keys())
        per_path_probs = torch.full((self.K,), 1.0 / self.K)

        # Build path-level candidate embeddings: [K, D] — one embedding per path
        path_embeds_tensor = torch.stack(path_embeddings) if path_embeddings else torch.empty(0)

        return ReasoningOutput(
            answer=majority_answer,
            candidate_probs=[per_path_probs],  # [K] — one per path, matching embeddings
            candidate_embeddings=[path_embeds_tensor] if len(path_embeddings) > 0 else [],
            aggregated_embeddings=[path_embeds_tensor.mean(dim=0)] if len(path_embeddings) > 0 else [],
            hidden_states=path_embeddings,  # One [D] tensor per path
            forward_passes=total_forward_passes,
            total_tokens_generated=total_tokens,
            metadata={
                "method": f"sc-{self.K}",
                "K": self.K,
                "all_answers": all_answers,
                "answer_distribution": dict(answer_counts),
                "all_traces": all_traces,
                "n_unique_answers": len(unique_answers),
                "temperatures": self.temperatures,
            },
        )
