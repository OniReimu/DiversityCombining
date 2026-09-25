"""Tree-of-Thought (ToT): BFS with pruning."""

import torch

from diversity_combining.methods.base import ReasoningMethod, ReasoningOutput


class TreeOfThoughtMethod(ReasoningMethod):
    """Tree-of-Thought with breadth-first search.

    Yao et al. (2023) — At each step, expand b candidates, evaluate, prune to b,
    repeat for d depth levels.
    Compute cost: b * d forward passes (plus evaluation calls).
    """

    def __init__(self, model, tokenizer, breadth: int = 3, depth: int = 2,
                 temperature: float = 0.7, device: str = "cuda"):
        super().__init__(model, tokenizer, device)
        self.breadth = breadth
        self.depth = depth
        self.temperature = temperature

    @property
    def name(self) -> str:
        return f"tot-b{self.breadth}-d{self.depth}"

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        step_tokens: int = 64,
        evaluate_fn=None,
        **kwargs,
    ) -> ReasoningOutput:
        """BFS tree search over reasoning steps.

        Args:
            step_tokens: Tokens per thought step (before pruning).
            evaluate_fn: Callable(partial_trace) -> float score.
                         If None, uses log-probability as heuristic.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        total_forward_passes = 0

        # Collectors for per-depth-level representations
        all_candidate_probs = []
        all_candidate_embeddings = []
        all_aggregated_embeddings = []
        all_hidden_states = []

        # Initialize beam with the prompt
        beams = [{"text": prompt, "input_ids": inputs.input_ids[0], "score": 0.0}]

        for d in range(self.depth):
            candidates = []
            depth_hidden = []  # Hidden states for candidates at this depth
            for beam in beams:
                beam_input = beam["input_ids"].unsqueeze(0).to(self.device)
                # Generate b continuations
                for _ in range(self.breadth):
                    outputs = self.model.generate(
                        input_ids=beam_input,
                        max_new_tokens=step_tokens,
                        do_sample=True,
                        temperature=self.temperature,
                        return_dict_in_generate=True,
                        output_scores=True,
                        output_hidden_states=True,
                    )
                    new_ids = outputs.sequences[0]
                    new_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
                    n_new_tokens = len(new_ids) - len(beam["input_ids"])
                    total_forward_passes += n_new_tokens

                    # Score: use evaluate_fn or log-prob
                    if evaluate_fn is not None:
                        score = evaluate_fn(new_text)
                    else:
                        # Sum of log-probs from generation scores
                        score = beam["score"]
                        if outputs.scores:
                            for step_scores in outputs.scores:
                                step_logprobs = torch.log_softmax(step_scores[0], dim=-1)
                                score += step_logprobs.max().item()

                    # Extract last hidden state of last generated token
                    candidate_hidden = None
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                        # Last generated token, last layer, batch=0, last position
                        candidate_hidden = outputs.hidden_states[-1][-1][0, -1, :].cpu()

                    candidates.append({
                        "text": new_text,
                        "input_ids": new_ids,
                        "score": score,
                        "_hidden": candidate_hidden,
                    })

            # Prune to top-b
            candidates.sort(key=lambda c: c["score"], reverse=True)
            beams = candidates[: self.breadth]

            # Collect representations from the pruned (surviving) candidates at this depth
            depth_scores = torch.tensor([c["score"] for c in beams])
            depth_probs = torch.softmax(depth_scores.float(), dim=0)
            all_candidate_probs.append(depth_probs.cpu())

            depth_hidden = [c["_hidden"] for c in beams if c["_hidden"] is not None]
            if depth_hidden:
                depth_embeds = torch.stack(depth_hidden)  # [breadth, D]
                all_candidate_embeddings.append(depth_embeds)
                all_hidden_states.extend(depth_hidden)
                # Aggregated embedding: weighted mean by beam-score probabilities
                weights = depth_probs.unsqueeze(-1)  # [breadth, 1]
                all_aggregated_embeddings.append((weights * depth_embeds).sum(dim=0))

            # Clean up the internal _hidden key from beams to avoid carrying large tensors
            for beam in beams:
                beam.pop("_hidden", None)

        # Best beam is the answer
        best = beams[0]
        # Extract just the generated portion
        answer_text = best["text"][len(prompt) :].strip()

        return ReasoningOutput(
            answer=answer_text,
            candidate_probs=all_candidate_probs,
            candidate_embeddings=all_candidate_embeddings,
            aggregated_embeddings=all_aggregated_embeddings,
            hidden_states=all_hidden_states,
            forward_passes=total_forward_passes,
            total_tokens_generated=total_forward_passes,  # Approximate
            metadata={
                "method": self.name,
                "breadth": self.breadth,
                "depth": self.depth,
                "n_candidates_explored": self.breadth * self.depth * self.breadth,
                "best_score": best["score"],
            },
        )
