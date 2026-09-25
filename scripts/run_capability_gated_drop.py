#!/usr/bin/env python3
"""Capability-gated diversity experiments — DROP version (discrete reasoning over paragraphs).

Separate from run_capability_gated_qa.py because:
- Dataset has multi-span gold answers (take max F1 across all spans)
- Answer types include numbers, spans, and dates
- Prompt templates are adapted for discrete reasoning (counting, sorting, arithmetic)

Usage:
    uv run python scripts/run_capability_gated_drop.py --model qwen7b --method both
"""
from __future__ import annotations

import argparse
import json
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_SRC = Path(__file__).resolve().parent.parent
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from diversity_combining.config import CACHE_DIR

MODEL_CONFIGS = {
    "qwen05b":   ("Qwen/Qwen2.5-0.5B-Instruct", 16),
    "qwen7b":    ("Qwen/Qwen2.5-7B-Instruct", 64),
    "qwen32b":   ("Qwen/Qwen2.5-32B-Instruct", 96),
    "llama8b":   ("meta-llama/Llama-3.1-8B-Instruct", 64),
    "mistral7b": ("mistralai/Mistral-7B-Instruct-v0.3", 64),
}

K = 8
DEFAULT_SEEDS = [42, 123]  # backward-compat default; pass --seeds to override
N_INSTANCES = 50
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.7

# Discrete-reasoning prompt templates (counting, sorting, arithmetic within text)
PROMPT_TEMPLATES_DROP = [
    # Slot 0: Standard
    "Read the passage and answer the question.\n\n{question}\n\nAnswer:",
    # Slot 1: Step by step with counting/comparing
    "Think step by step, count or compare as needed to answer the question.\n\n{question}\n\nLet me think step by step:",
    # Slot 2: Find relevant numbers/facts
    "Find the relevant numbers or facts in the passage, then answer the question.\n\n{question}\n\nRelevant facts:",
    # Slot 3: Direct
    "Give a short, direct answer to this question.\n\n{question}\n\nShort answer:",
    # Slot 4: Show arithmetic
    "Show your arithmetic or reasoning, then answer the question.\n\n{question}\n\nReasoning:",
    # Slot 5: Focus on details
    "Focus on the specific details asked about in the question.\n\n{question}\n\nAnswer:",
    # Slot 6: Extract then compute
    "Extract the relevant information from the passage, then compute the answer.\n\n{question}\n\nExtracted info:",
    # Slot 7: Trace quantities
    "Carefully trace the quantities mentioned in the passage to answer the question.\n\n{question}\n\nTracing quantities:",
]


def normalize_answer(s: str) -> str:
    """Normalize answer string for comparison (from SQuAD evaluation)."""
    s = str(s).lower().strip()
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # Remove punctuation
    s = s.translate(str.maketrans('', '', string.punctuation))
    # Collapse whitespace
    s = ' '.join(s.split())
    return s


def compute_f1(prediction: str, gold: str) -> float:
    """Token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_max_f1(prediction: str, gold_spans: list[str]) -> float:
    """Compute F1 against all gold spans and return the maximum."""
    if not gold_spans:
        return 0.0
    return max(compute_f1(prediction, span) for span in gold_spans)


def extract_answer(text: str) -> str:
    """Extract answer from model output."""
    text = str(text).strip()
    # Try to find "answer is X" pattern
    for pat in [
        r'(?:final\s+)?answer\s*(?:is|:)\s*(.+?)(?:\n|$)',
        r'(?:therefore|thus|so)\s*,?\s*(.+?)(?:\.|$)',
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            ans = m.group(1).strip().rstrip('.')
            if 0 < len(ans) < 200:
                return ans
    # Fallback: last line
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if lines:
        return lines[-1].rstrip('.')
    return text.strip()


def format_question(passage: str, question: str) -> str:
    """Format passage + question into a single prompt string."""
    return f"{passage}\n\nQuestion: {question}\n\nAnswer:"


def generate_text(model, tokenizer, prompt: str, seed: int) -> str:
    import torch
    torch.manual_seed(seed + hash(prompt) % 10000)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            top_p=0.95,
        )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def run_sc(model, tokenizer, prompt_text: str, seed: int, instance_id: int, gold_spans: list[str]) -> dict:
    """Standard SC: same prompt, K paths."""
    chat = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    all_answers = []
    all_f1s = []
    all_correct = []
    for k in range(K):
        path_seed = seed * 1000 + instance_id * 100 + k
        trace = generate_text(model, tokenizer, formatted, path_seed)
        ans = extract_answer(trace)
        f1 = compute_max_f1(ans, gold_spans)
        correct = f1 >= 0.5
        all_answers.append(ans)
        all_f1s.append(f1)
        all_correct.append(correct)

    return {"method": "sc", "all_answers": all_answers, "all_f1s": all_f1s, "all_correct": all_correct}


def run_pt(model, tokenizer, prompt_text: str, seed: int, instance_id: int, gold_spans: list[str]) -> dict:
    """Prompt-template SC: different prompt per slot."""
    all_answers = []
    all_f1s = []
    all_correct = []
    for k in range(K):
        try:
            prompt = PROMPT_TEMPLATES_DROP[k].format(question=prompt_text)
        except (KeyError, IndexError, ValueError):
            prompt = PROMPT_TEMPLATES_DROP[k].replace("{question}", prompt_text)
        chat = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        path_seed = seed * 1000 + instance_id * 100 + k
        trace = generate_text(model, tokenizer, formatted, path_seed)
        ans = extract_answer(trace)
        f1 = compute_max_f1(ans, gold_spans)
        correct = f1 >= 0.5
        all_answers.append(ans)
        all_f1s.append(f1)
        all_correct.append(correct)

    return {"method": "sc_prompttpl", "all_answers": all_answers, "all_f1s": all_f1s, "all_correct": all_correct}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--method", default="both", choices=["sc", "pt", "both"])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help=f"Random seeds to run (default: {DEFAULT_SEEDS}). "
                             f"Production example: --seeds 42 123 456 789 1024")
    args = parser.parse_args()
    seeds = args.seeds
    print(f"Running with seeds: {seeds}")

    model_id, gpu_gb = MODEL_CONFIGS[args.model]
    cache_file = CACHE_DIR / f"{args.model}_capgated_drop.jsonl"

    # Load existing entries for resume
    existing_keys: set[tuple[str, int, int]] = set()
    if cache_file.exists():
        with open(cache_file) as f:
            for line in f:
                try:
                    e = json.loads(line)
                    existing_keys.add((e["method"], e["seed"], e["instance_id"]))
                except Exception:
                    pass
    print(f"Existing: {len(existing_keys)} entries in {cache_file}")

    # Load model
    print(f"Loading {model_id}...")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto", "trust_remote_code": True}
    if gpu_gb > 80:
        try:
            import bitsandbytes  # noqa: F401
            load_kwargs["load_in_4bit"] = True
            print("Using 4-bit quantization")
        except ImportError:
            print("WARNING: bitsandbytes not installed, loading in bf16")
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()
    try:
        print(f"Model loaded on {model.device}")
    except Exception:
        print("Model loaded (multi-device)")

    # Load DROP dataset
    from datasets import load_dataset
    ds = load_dataset("drop", split="validation")
    print(f"DROP validation: {len(ds)} instances, using first {N_INSTANCES}")

    methods_to_run = []
    if args.method in ("sc", "both"):
        methods_to_run.append("sc")
    if args.method in ("pt", "both"):
        methods_to_run.append("pt")

    total = 0
    for method in methods_to_run:
        method_key = "sc" if method == "sc" else "sc_prompttpl"
        for seed in seeds:
            for i in range(min(N_INSTANCES, len(ds))):
                key = (method_key, seed, i)
                if key in existing_keys:
                    continue

                row = ds[i]
                passage = row["passage"]
                question = row["question"]
                answers_spans = row["answers_spans"]
                gold_spans = answers_spans["spans"]
                # Primary gold: first span (used for display)
                gold_primary = gold_spans[0] if gold_spans else ""

                prompt_text = format_question(passage, question)
                t0 = time.time()

                if method == "sc":
                    result = run_sc(model, tokenizer, prompt_text, seed, i, gold_spans)
                else:
                    result = run_pt(model, tokenizer, prompt_text, seed, i, gold_spans)

                entry = {
                    "model_slug": args.model,
                    "method": result["method"],
                    "task": "drop",
                    "K": K,
                    "seed": seed,
                    "instance_id": i,
                    "gold_answer": gold_primary,
                    "gold_spans": gold_spans,
                    "all_answers": result["all_answers"],
                    "all_f1s": result["all_f1s"],
                    "all_correct": result["all_correct"],
                    "timestamp": time.time(),
                    "elapsed": time.time() - t0,
                }

                with open(cache_file, "a") as f:
                    f.write(json.dumps(entry) + "\n")

                total += 1
                elapsed = time.time() - t0
                avg_f1 = np.mean(result["all_f1s"])
                n_correct = sum(result["all_correct"])
                print(f"  [{total}] {method_key} drop seed={seed} i={i}: "
                      f"{n_correct}/{K} correct (avg_f1={avg_f1:.2f}, {elapsed:.1f}s)")

    print(f"\nDone. {total} new entries written to {cache_file}")


if __name__ == "__main__":
    main()
