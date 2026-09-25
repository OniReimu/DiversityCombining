#!/usr/bin/env python3
"""Capability-gated diversity experiments.

Runs standard SC (K=8) and/or prompt-template SC (K=8) on a given model.
Designed for the paired comparison: same model, same K, two diversity methods.

Usage:
    uv run python scripts/run_capability_gated.py --model qwen05b --method both
    uv run python scripts/run_capability_gated.py --model qwen32b --method sc
    uv run python scripts/run_capability_gated.py --model llama8b --method pt
"""
from __future__ import annotations

import argparse
import json
import re
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
TASKS = ["gsm8k"]  # Override with --task math for MATH benchmark
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.7

PROMPT_TEMPLATES = [
    "Solve the following problem step by step.\n\nProblem: {question}\n\nSolution:",
    "Use algebraic equations to solve this problem. Define variables, write equations, and solve.\n\nProblem: {question}\n\nSolution:",
    "First give a rough estimate of the answer, then solve precisely to get the exact answer.\n\nProblem: {question}\n\nSolution:",
    "Break this problem into smaller sub-problems. Solve each sub-problem, then combine.\n\nProblem: {question}\n\nSolution:",
    "Work backwards from a hypothetical answer to verify. Then solve forward.\n\nProblem: {question}\n\nSolution:",
    "Solve this problem, then verify your answer by substituting back.\n\nProblem: {question}\n\nSolution:",
    "Solve this problem concisely. Show key steps only.\n\nProblem: {question}\n\nSolution:",
    "Explain your solution as if teaching a student. Be thorough.\n\nProblem: {question}\n\nSolution:",
]


def extract_number(text: str) -> str | None:
    text = str(text).strip()
    for pat in [r'\\boxed\{([^}]+)\}', r'\\\\boxed\{([^}]+)\}']:
        boxed = list(re.finditer(pat, text))
        if boxed:
            nums = re.findall(r'-?[\d,]+\.?\d*', boxed[-1].group(1))
            if nums:
                return nums[-1].replace(',', '')
    m = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if m:
        return m.group(1).replace(',', '')
    numbers = re.findall(r'-?[\d,]+\.?\d*', text)
    if numbers:
        return numbers[-1].replace(',', '')
    return None


def generate_text(model, tokenizer, prompt: str, seed: int) -> str:
    import torch
    torch.manual_seed(seed + hash(prompt) % 10000)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
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
    """Standard SC: same prompt, K paths with temperature sampling."""
    question = task_data["prompt"]
    prompt = f"Solve the following problem step by step.\n\nProblem: {question}\n\nSolution:"

    chat = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    all_answers = []
    for k in range(K):
        path_seed = seed * 1000 + instance_id * 100 + k
        trace = generate_text(model, tokenizer, formatted, path_seed)
        ans = extract_number(trace)
        all_answers.append(ans)

    return {"method": "sc", "all_answers": all_answers, "gold_answer": gold}


def run_pt(model, tokenizer, task_data, seed: int, instance_id: int, gold: str) -> dict:
    """Prompt-template SC: different prompt per slot."""
    question = task_data["prompt"]
    all_answers = []
    for k in range(K):
        try:
            prompt = PROMPT_TEMPLATES[k].format(question=question)
        except (KeyError, IndexError, ValueError):
            prompt = PROMPT_TEMPLATES[k].replace("{question}", question)
        chat = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        path_seed = seed * 1000 + instance_id * 100 + k
        trace = generate_text(model, tokenizer, formatted, path_seed)
        ans = extract_number(trace)
        all_answers.append(ans)

    return {"method": "sc_prompttpl", "all_answers": all_answers, "gold_answer": gold}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--method", default="both", choices=["sc", "pt", "both"])
    parser.add_argument("--task", default="gsm8k", choices=["gsm8k", "math"])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help=f"Random seeds to run (default: {DEFAULT_SEEDS}). "
                             f"Production example: --seeds 42 123 456 789 1024")
    args = parser.parse_args()
    seeds = args.seeds
    print(f"Running with seeds: {seeds}")

    model_id, gpu_gb = MODEL_CONFIGS[args.model]
    cache_file = CACHE_DIR / f"{args.model}_capgated.jsonl"

    # Load existing results to skip completed
    existing_keys = set()
    if cache_file.exists():
        with open(cache_file) as f:
            for line in f:
                try:
                    e = json.loads(line)
                    existing_keys.add((e["method"], e["task"], e["seed"], e["instance_id"]))
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
            print("WARNING: bitsandbytes not installed, loading in bf16 (may OOM)")
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()
    try:
        print(f"Model loaded on {model.device}")
    except Exception:
        print("Model loaded (multi-device)")

    # Load task data
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
                key = (method_key, args.task, seed, i)
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
                    "timestamp": time.time(),
                    "elapsed": time.time() - t0,
                }

                with open(cache_file, "a") as f:
                    f.write(json.dumps(entry) + "\n")

                total += 1
                elapsed = time.time() - t0
                gold_num = extract_number(gold)
                correct = 0
                if gold_num is not None:
                    for a in result["all_answers"]:
                        if a is not None:
                            try:
                                if abs(float(a) - float(gold_num)) < 1e-3:
                                    correct += 1
                            except (ValueError, TypeError):
                                pass
                print(f"  [{total}] {method_key} {args.task} seed={seed} i={i}: {correct}/{K} correct ({elapsed:.1f}s)")

    print(f"\nDone. {total} new entries written to {cache_file}")


if __name__ == "__main__":
    main()
