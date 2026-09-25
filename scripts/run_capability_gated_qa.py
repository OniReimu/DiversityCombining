#!/usr/bin/env python3
"""Capability-gated diversity experiments — QA version (HotpotQA).

Separate from run_capability_gated.py (math version) because:
- Answer extraction is text-based (F1 matching), not numeric
- Prompt templates are adapted for QA tasks

Usage:
    uv run python scripts/run_capability_gated_qa.py --model qwen7b --method both
"""
from __future__ import annotations

import argparse
import json
import re
import string
import sys
import time
from pathlib import Path
from collections import Counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diversity_combining.evaluation.data_loader import load_task
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
MAX_NEW_TOKENS = 2048  # Match math version for fair comparison
TEMPERATURE = 0.7

# QA-adapted prompt templates
PROMPT_TEMPLATES_QA = [
    # Slot 0: Standard
    "Answer the following question based on the provided context.\n\n{question}\n\nAnswer:",
    # Slot 1: Chain of thought
    "Read the context carefully and reason step by step to answer the question.\n\n{question}\n\nLet me think step by step:",
    # Slot 2: Extract then answer
    "First identify the key facts in the context, then answer the question.\n\n{question}\n\nKey facts:",
    # Slot 3: Direct
    "Give a short, direct answer to this question.\n\n{question}\n\nShort answer:",
    # Slot 4: Verify
    "Answer the question, then verify your answer against the context.\n\n{question}\n\nAnswer and verification:",
    # Slot 5: Decompose
    "Break the question into sub-questions, answer each, then combine.\n\n{question}\n\nSub-questions:",
    # Slot 6: Concise
    "Answer in as few words as possible.\n\n{question}\n\nAnswer:",
    # Slot 7: Explain
    "Explain your reasoning thoroughly, then give the final answer.\n\n{question}\n\nExplanation:",
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


def extract_answer_qa(text: str) -> str:
    """Extract answer from QA model output."""
    text = str(text).strip()
    # Try to find "Answer:" or "answer is" patterns
    for pat in [
        r'(?:final\s+)?answer\s*(?:is|:)\s*(.+?)(?:\n|$)',
        r'(?:therefore|thus|so)\s*,?\s*(.+?)(?:\.|$)',
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            ans = m.group(1).strip().rstrip('.')
            if len(ans) > 0 and len(ans) < 200:
                return ans
    # Fallback: last line
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if lines:
        return lines[-1].rstrip('.')
    return text.strip()


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


def run_sc(model, tokenizer, task_data, seed: int, instance_id: int, gold: str) -> dict:
    """Standard SC: same prompt, K paths."""
    prompt = task_data["prompt"]
    chat = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    all_answers = []
    all_f1s = []
    for k in range(K):
        path_seed = seed * 1000 + instance_id * 100 + k
        trace = generate_text(model, tokenizer, formatted, path_seed)
        ans = extract_answer_qa(trace)
        f1 = compute_f1(ans, gold)
        all_answers.append(ans)
        all_f1s.append(f1)

    return {"method": "sc", "all_answers": all_answers, "all_f1s": all_f1s}


def run_pt(model, tokenizer, task_data, seed: int, instance_id: int, gold: str) -> dict:
    """Prompt-template SC: different prompt per slot."""
    question_text = task_data["prompt"]
    all_answers = []
    all_f1s = []
    for k in range(K):
        try:
            prompt = PROMPT_TEMPLATES_QA[k].format(question=question_text)
        except (KeyError, IndexError, ValueError):
            prompt = PROMPT_TEMPLATES_QA[k].replace("{question}", question_text)
        chat = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        path_seed = seed * 1000 + instance_id * 100 + k
        trace = generate_text(model, tokenizer, formatted, path_seed)
        ans = extract_answer_qa(trace)
        f1 = compute_f1(ans, gold)
        all_answers.append(ans)
        all_f1s.append(f1)

    return {"method": "sc_prompttpl", "all_answers": all_answers, "all_f1s": all_f1s}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--method", default="both", choices=["sc", "pt", "both"])
    parser.add_argument("--task", default="hotpotqa", choices=["hotpotqa", "triviaqa"])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help=f"Random seeds to run (default: {DEFAULT_SEEDS}). "
                             f"Production example: --seeds 42 123 456 789 1024")
    args = parser.parse_args()
    seeds = args.seeds
    print(f"Running with seeds: {seeds}")

    model_id, gpu_gb = MODEL_CONFIGS[args.model]
    task_suffix = "qa" if args.task == "hotpotqa" else args.task
    cache_file = CACHE_DIR / f"{args.model}_capgated_{task_suffix}.jsonl"

    # Load existing
    existing_keys = set()
    if cache_file.exists():
        with open(cache_file) as f:
            for line in f:
                try:
                    e = json.loads(line)
                    existing_keys.add((e["method"], e["seed"], e["instance_id"]))
                except:
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

    # Load QA task
    task_data = load_task(args.task)

    methods_to_run = []
    if args.method in ("sc", "both"):
        methods_to_run.append("sc")
    if args.method in ("pt", "both"):
        methods_to_run.append("pt")

    total = 0
    for method in methods_to_run:
        method_key = "sc" if method == "sc" else "sc_prompttpl"
        for seed in seeds:
            for i in range(min(N_INSTANCES, len(task_data))):
                key = (method_key, seed, i)
                if key in existing_keys:
                    continue

                gold = str(task_data[i]["answer"])
                t0 = time.time()

                if method == "sc":
                    result = run_sc(model, tokenizer, task_data[i], seed, i, gold)
                else:
                    result = run_pt(model, tokenizer, task_data[i], seed, i, gold)

                entry = {
                    "model_slug": args.model,
                    "method": result["method"],
                    "task": args.task,
                    "K": K,
                    "seed": seed,
                    "instance_id": i,
                    "gold_answer": gold,
                    "all_answers": result["all_answers"],
                    "all_f1s": result["all_f1s"],
                    "timestamp": time.time(),
                    "elapsed": time.time() - t0,
                }

                with open(cache_file, "a") as f:
                    f.write(json.dumps(entry) + "\n")

                total += 1
                elapsed = time.time() - t0
                avg_f1 = np.mean(result["all_f1s"])
                n_correct = sum(1 for f in result["all_f1s"] if f >= 0.5)
                print(f"  [{total}] {method_key} {args.task} seed={seed} i={i}: {n_correct}/{K} correct (avg_f1={avg_f1:.2f}, {elapsed:.1f}s)")

    print(f"\nDone. {total} new entries written to {cache_file}")


if __name__ == "__main__":
    main()
