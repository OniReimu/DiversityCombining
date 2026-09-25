"""Diversity Combining Evaluation Runner.

Implements Algorithm 1 from the paper:
For each (method, task, compute_budget) triple:
1. Generate reasoning traces with N random seeds
2. Compute R = (R_dist, R_geom, R_IT)
3. Compute D relative to SC-32 reference
4. Compute I = (I_sep, I_head, I_conf)
5. Construct R-D curves by varying free parameters at matched compute
"""

import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from diversity_combining.methods.base import ReasoningMethod, ReasoningOutput
from diversity_combining.metrics.rate import distributional_rate, geometric_rate, information_theoretic_rate
from diversity_combining.metrics.distortion import compute_distortion_with_components
from diversity_combining.metrics.interference import separability, head_specialization, confusability


# ── Answer extraction ─────────────────────────────────────────────────────

def _strip_think_tags(text: str) -> str:
    """Strip <think>...</think> reasoning blocks, return only the final answer portion.

    DeepSeek-R1 models wrap internal reasoning in <think> tags.
    If tags are present, extract only the text after the last </think>.
    If no closing tag found (truncated), return the text after <think> block.
    """
    # Find the last </think> tag
    idx = text.rfind("</think>")
    if idx != -1:
        return text[idx + len("</think>"):].strip()
    # If <think> exists but no closing tag (truncated response), try text before <think>
    idx = text.find("<think>")
    if idx != -1 and idx > 0:
        return text[:idx].strip()
    return text


def _strip_template_artifacts(text: str) -> str:
    """Strip chat template turn markers that leak into output on truncation.

    When models hit max_tokens, they may generate next-turn markers like
    'user\\nContinue...' or '<|im_start|>user'. Strip everything after
    the first such marker.
    """
    # Common patterns: "user\n", "User:", "<|im_start|>", "assistant\n"
    patterns = [
        r"\nuser\n",           # Qwen-style turn boundary
        r"\nUser:",            # Generic turn boundary
        r"\n<\|im_start\|>",   # Qwen special token as text
        r"\nassistant\n",      # Assistant turn boundary
        r"\n<\|eot_id\|>",     # Llama-style
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            text = text[:m.start()]
    return text.strip()


def extract_number(text: str) -> str | None:
    """Extract the final number from a model response.

    Priority order:
    1. Strip <think> tags to isolate the final answer portion
    2. \\boxed{X} (DeepSeek-R1 / MATH standard format)
    3. #### X (GSM8K gold format)
    4. "answer is X" patterns
    5. "= X" at end of line
    6. Last number in the (stripped) text
    """
    # Step 1: Strip reasoning and template artifacts
    answer_text = _strip_template_artifacts(_strip_think_tags(text))
    # If stripping left nothing useful, fall back to full text
    if len(answer_text) < 2:
        answer_text = _strip_template_artifacts(text)

    # Step 2: \boxed{X} — highest priority (DeepSeek-R1, MATH standard)
    # Search in answer_text first, then full text as fallback
    for search_text in [answer_text, text]:
        boxed_matches = list(re.finditer(r"\\boxed\{([^}]+)\}", search_text))
        if boxed_matches:
            content = boxed_matches[-1].group(1).strip()
            # Extract number from boxed content (may contain formatting like $, commas)
            nums = re.findall(r"-?[\d,]+\.?\d*", content)
            if nums:
                return nums[-1].replace(",", "")
            return content

    # Step 3: "#### X" format (GSM8K gold)
    m = re.search(r"####\s*(-?[\d,]+\.?\d*)", answer_text)
    if m:
        return m.group(1).replace(",", "")
    # Step 4: "answer is X"
    m = re.search(r"(?:answer|result|total)\s*(?:is|=|:)\s*\$?(-?[\d,]+\.?\d*)", answer_text, re.I)
    if m:
        return m.group(1).replace(",", "")
    # Step 5: "= X" at end of line
    m = re.search(r"=\s*\$?(-?[\d,]+\.?\d*)\s*$", answer_text, re.MULTILINE)
    if m:
        return m.group(1).replace(",", "")
    # Step 6: Last number in the stripped text (not full text — avoids intermediate numbers)
    numbers = re.findall(r"-?[\d,]+\.?\d*", answer_text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def extract_algorithmic_answer(text: str, task: str) -> str:
    """Extract the final answer from an algorithmic task's reasoning output.

    Models produce verbose reasoning but the gold answer is a short string
    like "even", "odd", "[1, 3, 9]", "valid", "invalid".
    """
    text_lower = text.lower().strip()

    if task == "algorithmic_parity":
        # Look for explicit "answer is even/odd" patterns first
        m = re.search(
            r"(?:answer|parity|result|conclusion)\s*(?:is|:)\s*(even|odd)",
            text_lower,
        )
        if m:
            return m.group(1)
        # Look for "the parity is even/odd"
        m = re.search(r"the\s+parity\s+is\s+(even|odd)", text_lower)
        if m:
            return m.group(1)
        # Fallback: last occurrence of "even" or "odd" in the text
        matches = list(re.finditer(r"\b(even|odd)\b", text_lower))
        if matches:
            return matches[-1].group(1)

    elif task == "algorithmic_sorting":
        # Look for a Python-style list as the final answer
        # "answer is [1, 3, 9]" or "sorted list: [1, 3, 9]"
        m = re.search(
            r"(?:answer|result|sorted\s*(?:list|array)?)\s*(?:is|:)?\s*(\[[0-9,\s]+\])",
            text_lower,
        )
        if m:
            return m.group(1).replace(" ", "")
        # Fallback: last bracketed list of numbers in the text
        matches = list(re.finditer(r"\[([0-9]+(?:\s*,\s*[0-9]+)*)\]", text))
        if matches:
            nums = matches[-1].group(0)
            return nums.replace(" ", "")

    elif task == "algorithmic_dyck":
        # Look for "valid" or "invalid" verdict
        m = re.search(
            r"(?:answer|sequence|result|conclusion)\s*(?:is|:)\s*(invalid|valid|not\s+valid|balanced|not\s+balanced|unbalanced)",
            text_lower,
        )
        if m:
            verdict = m.group(1)
            if verdict in ("valid", "balanced"):
                return "valid"
            return "invalid"
        # Fallback: last occurrence of "valid"/"invalid" (check "invalid" first since it contains "valid")
        matches_invalid = list(re.finditer(r"\b(invalid|not\s+valid|unbalanced|not\s+balanced)\b", text_lower))
        matches_valid = list(re.finditer(r"\b(valid|balanced)\b", text_lower))
        if matches_invalid or matches_valid:
            last_invalid = matches_invalid[-1].start() if matches_invalid else -1
            last_valid = matches_valid[-1].start() if matches_valid else -1
            if last_invalid > last_valid:
                return "invalid"
            elif last_valid > last_invalid:
                return "valid"

    # If nothing extracted, return the original text for exact match fallback
    return text.strip()


def compute_f1(prediction: str, gold: str) -> float:
    """Compute token-level F1 between prediction and gold answer.

    This is the standard SQuAD/HotpotQA F1 metric: tokenize both strings
    (lowercased, split on whitespace and punctuation), then compute
    token-level precision, recall, and F1.
    """
    def _tokenize(text: str) -> list[str]:
        # Lowercase and split on whitespace + punctuation
        return re.findall(r"\w+", text.lower())

    pred_tokens = _tokenize(prediction)
    gold_tokens = _tokenize(gold)

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)

    # Number of shared tokens (taking minimum counts)
    common = sum((pred_counter & gold_counter).values())

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def check_answer(model_answer: str, gold_answer: str, task: str = "") -> bool:
    """Check if model's extracted answer matches gold.

    For HotpotQA, uses token-level F1 >= 0.5 (standard SQuAD/HotpotQA metric).
    For numeric tasks (GSM8K, MATH), uses numeric extraction and comparison.
    For all other tasks, falls back to exact string match.

    Args:
        model_answer: The model's response string.
        gold_answer: The ground-truth answer string.
        task: Task name (e.g. "hotpotqa", "gsm8k", "math"). Controls
              which matching strategy is used.
    """
    if task.lower() == "hotpotqa":
        return compute_f1(model_answer, gold_answer) >= 0.5

    # Algorithmic tasks: extract final answer from reasoning, then exact match
    if task.lower().startswith("algorithmic_"):
        extracted = extract_algorithmic_answer(model_answer, task.lower())
        # Normalize spacing for list comparison (sorting task)
        return extracted.replace(" ", "") == gold_answer.strip().replace(" ", "")

    model_num = extract_number(model_answer)
    gold_num = extract_number(gold_answer)
    if model_num is None or gold_num is None:
        return model_answer.strip() == gold_answer.strip()
    try:
        return abs(float(model_num) - float(gold_num)) < 1e-3
    except ValueError:
        return model_num == gold_num


@dataclass
class TrialResult:
    """Result for a single (method, task, K, seed) trial."""

    method: str
    task: str
    K: int
    seed: int
    accuracy: float
    # Rate metrics
    R_dist: float = 0.0
    R_geom: float = 0.0
    R_IT: float = 0.0
    # Distortion
    D: float = 0.0
    coverage_failure: float = 0.0
    instability: float = 0.0
    # Interference
    I_sep: float = 0.0
    I_head: float = 0.0
    I_conf: float = 0.0
    # Compute
    forward_passes: int = 0
    total_tokens: int = 0
    wall_time_s: float = 0.0


@dataclass
class EvalConfig:
    """Configuration for a full Diversity Combining evaluation run."""

    methods: list[str] = field(default_factory=lambda: [
        "greedy", "sc", "tot", "soft_k", "latent_loop",
    ])
    tasks: list[str] = field(default_factory=lambda: [
        "gsm8k", "math", "hotpotqa",
    ])
    K_values: list[int] = field(default_factory=lambda: [4, 8, 12, 16])
    seeds: list[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])
    max_instances: int = 200
    max_tokens: int = 512
    reference_method: str = "sc"
    reference_K: int = 32
    output_dir: str = "results"
    device: str = "cuda"


class EvalRunner:
    """Main evaluation runner for Diversity Combining experiments."""

    def __init__(self, config: EvalConfig, model, tokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint file for crash-safe resumption
        self.checkpoint_path = self.output_dir / "checkpoint.jsonl"
        self.completed = self._load_checkpoint()

    def _load_checkpoint(self) -> set[str]:
        """Load completed trial IDs from checkpoint."""
        completed = set()
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path) as f:
                for line in f:
                    trial = json.loads(line)
                    key = f"{trial['method']}_{trial['task']}_{trial['K']}_{trial['seed']}"
                    completed.add(key)
        return completed

    def _save_trial(self, result: TrialResult):
        """Append trial result to checkpoint (crash-safe)."""
        with open(self.checkpoint_path, "a") as f:
            f.write(json.dumps(asdict(result)) + "\n")
        key = f"{result.method}_{result.task}_{result.K}_{result.seed}"
        self.completed.add(key)

    def _build_method(self, method_name: str, K: int) -> ReasoningMethod:
        """Instantiate a reasoning method with the given K."""
        from diversity_combining.methods import get_method

        common_kwargs = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "device": self.config.device,
        }

        if method_name == "greedy":
            return get_method("greedy", **common_kwargs)
        elif method_name == "sc":
            return get_method("sc", K=K, **common_kwargs)
        elif method_name == "tot":
            # Map K to breadth/depth: K≈b*d, prefer balanced
            b = max(2, int(np.sqrt(K)))
            d = max(1, K // b)
            return get_method("tot", breadth=b, depth=d, **common_kwargs)
        elif method_name == "soft_k":
            return get_method("soft_k", K=K, **common_kwargs)
        elif method_name == "latent_loop":
            return get_method("latent_loop", T=K, **common_kwargs)
        else:
            raise ValueError(f"Unknown method: {method_name}")

    def _compute_metrics(
        self,
        outputs: list[ReasoningOutput],
        ground_truth: list[str],
        reference_outputs: list[ReasoningOutput] | None = None,
        task: str = "",
    ) -> dict:
        """Compute all Diversity Combining metrics from a batch of outputs."""
        n = len(outputs)

        # Accuracy (task-aware: raw F1 mean for HotpotQA, exact match for others)
        if task.lower() == "hotpotqa":
            accuracy = sum(
                compute_f1(o.answer, g) for o, g in zip(outputs, ground_truth)
            ) / max(n, 1)
        else:
            correct = sum(
                1 for o, g in zip(outputs, ground_truth)
                if check_answer(o.answer, g, task=task)
            )
            accuracy = correct / max(n, 1)

        # --- Rate metrics ---

        # R_dist: Average distributional rate across instances and steps
        r_dist_values = []
        for output in outputs:
            for probs in output.candidate_probs:
                if len(probs) > 1:
                    r_dist_values.append(distributional_rate(probs))
        R_dist = float(np.mean(r_dist_values)) if r_dist_values else 0.0

        # R_geom: Geometric rate from aggregated embeddings
        all_agg = []
        for output in outputs:
            for emb in output.aggregated_embeddings:
                all_agg.append(emb)
        if len(all_agg) >= 2:
            R_geom = geometric_rate(torch.stack(all_agg))
        else:
            R_geom = 1.0

        # R_IT: MI between hidden states and answer labels (not binary correctness)
        all_hidden = []
        answer_labels = []
        unique_answers = sorted(set(ground_truth))

        # For high-cardinality tasks (e.g., math with unique numeric answers),
        # bin answers to keep n_classes manageable for MI estimation
        MAX_CLASSES = 20
        if len(unique_answers) <= MAX_CLASSES:
            answer_to_id = {ans: i for i, ans in enumerate(unique_answers)}
        else:
            # Try numeric binning first
            numeric_vals = []
            for ans in ground_truth:
                num = extract_number(ans)
                try:
                    numeric_vals.append(float(num) if num is not None else None)
                except (ValueError, TypeError):
                    numeric_vals.append(None)

            if sum(v is not None for v in numeric_vals) > len(ground_truth) * 0.8:
                # Numeric task: quantile binning
                valid_vals = [v for v in numeric_vals if v is not None]
                quantiles = np.quantile(valid_vals, np.linspace(0, 1, MAX_CLASSES + 1))
                answer_to_id = {}
                for ans in ground_truth:
                    num = extract_number(ans)
                    try:
                        val = float(num) if num is not None else 0.0
                    except (ValueError, TypeError):
                        val = 0.0
                    bin_idx = min(np.searchsorted(quantiles[1:], val), MAX_CLASSES - 1)
                    answer_to_id[ans] = int(bin_idx)
            else:
                # Text task: hash-based bucketing
                answer_to_id = {ans: hash(ans) % MAX_CLASSES for ans in unique_answers}

        n_answer_classes = len(set(answer_to_id.values()))

        for output, gt in zip(outputs, ground_truth):
            if output.hidden_states:
                last_h = output.hidden_states[-1] if output.hidden_states else None
                if last_h is not None:
                    all_hidden.append(last_h)
                    answer_labels.append(answer_to_id[gt])

        R_IT = 0.0
        if (len(all_hidden) >= max(10, 2 * n_answer_classes)
                and n_answer_classes >= 2
                and len(set(answer_labels)) >= 2):
            hidden_stack = torch.stack(all_hidden)
            label_tensor = torch.tensor(answer_labels, dtype=torch.long)
            compute_budget = sum(o.forward_passes for o in outputs)
            try:
                R_IT = information_theoretic_rate(
                    hidden_stack, label_tensor, n_classes=n_answer_classes,
                    compute_budget=compute_budget, device="cpu",
                )
            except Exception:
                R_IT = 0.0

        # --- Interference metrics ---

        # I_conf: Average confusability across steps
        conf_values = []
        for output in outputs:
            for embs, probs in zip(output.candidate_embeddings, output.candidate_probs):
                if embs is not None and len(probs) > 1:
                    conf_values.append(confusability(embs, probs))
        I_conf = float(np.mean(conf_values)) if conf_values else 0.0

        # I_head: Head specialization from attention patterns
        all_attn = []
        for output in outputs:
            all_attn.extend(output.attention_patterns)
        I_head = head_specialization(all_attn) if all_attn else 0.5

        # I_sep: Separability via linear probe
        # Predicts which candidate dominates from aggregated embedding
        all_agg_for_sep = []
        dominant_labels = []
        for output in outputs:
            for step_idx in range(len(output.aggregated_embeddings)):
                agg = output.aggregated_embeddings[step_idx]
                if step_idx < len(output.candidate_probs):
                    probs = output.candidate_probs[step_idx]
                    if len(probs) > 1:
                        all_agg_for_sep.append(agg)
                        dominant_labels.append(int(probs.argmax().item()))

        I_sep = 0.5
        if len(all_agg_for_sep) >= 10 and len(set(dominant_labels)) >= 2:
            try:
                agg_stack = torch.stack(all_agg_for_sep)
                label_tensor = torch.tensor(dominant_labels, dtype=torch.long)
                I_sep = separability(agg_stack, label_tensor)
            except Exception:
                I_sep = 0.5

        # --- Distortion ---
        D = 0.0
        if reference_outputs is not None:
            ref_answers = [o.answer.strip() for o in reference_outputs]
            method_answers = [o.answer.strip() for o in outputs]
            dist_kwargs = {
                "match_fn": lambda m, g: check_answer(m, g, task=task),
            }
            if task.lower() == "hotpotqa":
                dist_kwargs["score_fn"] = lambda m, g: compute_f1(m, g)
            dist_result = compute_distortion_with_components(
                method_answers, ref_answers, [g.strip() for g in ground_truth],
                **dist_kwargs,
            )
            D = dist_result["distortion"]

        # --- Compute stats ---
        total_fwd = sum(o.forward_passes for o in outputs)
        total_tok = sum(o.total_tokens_generated for o in outputs)

        return {
            "accuracy": accuracy,
            "R_dist": R_dist,
            "R_geom": R_geom,
            "R_IT": R_IT,
            "D": D,
            "I_sep": I_sep,
            "I_head": I_head,
            "I_conf": I_conf,
            "total_forward_passes": total_fwd,
            "total_tokens": total_tok,
        }

    def run(self, dataset_loader):
        """Execute the full Diversity Combining evaluation protocol.

        Args:
            dataset_loader: Callable(task_name) -> list[dict] with keys
                           'prompt', 'answer'.
        """
        all_results = []

        for task in self.config.tasks:
            print(f"\n{'='*60}")
            print(f"Task: {task}")
            print(f"{'='*60}")

            # Load dataset
            dataset = dataset_loader(task)
            instances = dataset[: self.config.max_instances]
            prompts = [inst["prompt"] for inst in instances]
            ground_truth = [inst["answer"] for inst in instances]

            # Generate SC-32 reference outputs per seed for distortion computation
            reference_cache = {}  # seed -> list[ReasoningOutput]
            ref_method = self._build_method(
                self.config.reference_method, self.config.reference_K
            )
            for seed in self.config.seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)
                ref_outputs = []
                for prompt in tqdm(prompts, desc=f"  ref SC-{self.config.reference_K} s={seed}"):
                    try:
                        output = ref_method.generate(prompt, max_tokens=self.config.max_tokens)
                        ref_outputs.append(output)
                    except Exception as e:
                        print(f"    Ref error: {e}")
                        ref_outputs.append(ReasoningOutput(answer="ERROR"))
                reference_cache[seed] = ref_outputs

            for method_name in self.config.methods:
                K_range = [1] if method_name == "greedy" else self.config.K_values

                for K in K_range:
                    for seed in self.config.seeds:
                        trial_key = f"{method_name}_{task}_{K}_{seed}"
                        if trial_key in self.completed:
                            print(f"  [skip] {trial_key} (already completed)")
                            continue

                        print(f"  Running: {method_name} K={K} seed={seed}")
                        torch.manual_seed(seed)
                        np.random.seed(seed)

                        method = self._build_method(method_name, K)

                        t0 = time.time()
                        outputs = []
                        for prompt in tqdm(prompts, desc=f"  {method.name}"):
                            try:
                                output = method.generate(
                                    prompt,
                                    max_tokens=self.config.max_tokens,
                                )
                                outputs.append(output)
                            except Exception as e:
                                print(f"    Error on instance: {e}")
                                outputs.append(ReasoningOutput(answer="ERROR"))
                        wall_time = time.time() - t0

                        # Compute metrics (pass reference outputs for distortion)
                        ref_outputs = reference_cache.get(seed)
                        metrics = self._compute_metrics(outputs, ground_truth, reference_outputs=ref_outputs, task=task)

                        result = TrialResult(
                            method=method_name,
                            task=task,
                            K=K,
                            seed=seed,
                            accuracy=metrics["accuracy"],
                            R_dist=metrics["R_dist"],
                            R_geom=metrics["R_geom"],
                            R_IT=metrics["R_IT"],
                            D=metrics["D"],
                            I_sep=metrics["I_sep"],
                            I_head=metrics["I_head"],
                            I_conf=metrics["I_conf"],
                            forward_passes=metrics["total_forward_passes"],
                            total_tokens=metrics["total_tokens"],
                            wall_time_s=wall_time,
                        )

                        self._save_trial(result)
                        all_results.append(result)

                        print(f"    Acc={metrics['accuracy']:.3f} "
                              f"R_dist={metrics['R_dist']:.2f} "
                              f"I_conf={metrics['I_conf']:.2f} "
                              f"({wall_time:.1f}s)")

        # Save aggregated results
        results_path = self.output_dir / "all_results.json"
        with open(results_path, "w") as f:
            json.dump([asdict(r) for r in all_results], f, indent=2)

        print(f"\nResults saved to {results_path}")
        return all_results
