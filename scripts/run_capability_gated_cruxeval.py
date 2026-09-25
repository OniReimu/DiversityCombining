#!/usr/bin/env python3
"""Capability-gated diversity — Code understanding (CRUXEval).

Task: given Python code + input, predict the output. Exact string match.
No code execution needed — pure reasoning about code behavior.

Usage:
    uv run python scripts/run_capability_gated_cruxeval.py --model qwen7b --method both
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
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

PROMPT_TEMPLATES = [
    "What is the output of the following Python code?\n\n```python\n{code}\n```\n\nInput: `{input}`\n\nOutput:",
    "Trace through this code step by step, then give the output.\n\n```python\n{code}\n```\n\nInput: `{input}`\n\nStep-by-step trace:",
    "Execute this Python function mentally and predict the result.\n\n```python\n{code}\n```\n\nInput: `{input}`\n\nResult:",
    "Give only the output value, nothing else.\n\n```python\n{code}\n```\n\nInput: `{input}`\n\nOutput value:",
    "What does `f({input})` return?\n\n```python\n{code}\n```\n\nReturn value:",
    "Analyze the code logic, then predict the output.\n\n```python\n{code}\n```\n\nInput: `{input}`\n\nCode analysis:",
    "Think about edge cases in this code, then give the output.\n\n```python\n{code}\n```\n\nInput: `{input}`\n\nEdge case analysis and output:",
    "Simulate a Python interpreter running this code.\n\n```python\n{code}\n```\n\nInput: `{input}`\n\nInterpreter output:",
]


def extract_output(text: str, gold: str) -> str:
    """Extract predicted output from model response."""
    text = text.strip()
    # Try to find content in backticks
    m = re.search(r'`([^`]+)`', text)
    if m:
        return m.group(1).strip()
    # Try "output is X" pattern
    m = re.search(r'(?:output|result|return|returns)\s*(?:is|=|:)\s*(.+?)(?:\n|$)', text, re.I)
    if m:
        return m.group(1).strip().rstrip('.')
    # Last line
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        return lines[-1].rstrip('.')
    return text


def check_match(predicted: str, gold: str) -> bool:
    """Check if predicted output matches gold (flexible matching)."""
    pred = predicted.strip().strip('`').strip("'").strip('"')
    gold = gold.strip().strip('`').strip("'").strip('"')
    # Exact match
    if pred == gold:
        return True
    # Try eval-based comparison (handles formatting differences)
    try:
        if repr(eval(pred)) == repr(eval(gold)):
            return True
    except:
        pass
    # String containment (gold in pred)
    if gold in pred:
        return True
    return False


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


def run_method(model, tokenizer, item, seed, instance_id, method, gold):
    all_answers = []
    all_correct = []
    for k in range(K):
        if method == "sc":
            prompt_text = PROMPT_TEMPLATES[0].format(code=item['code'], input=item['input'])
        else:
            try:
                prompt_text = PROMPT_TEMPLATES[k].format(code=item['code'], input=item['input'])
            except (KeyError, IndexError):
                prompt_text = PROMPT_TEMPLATES[k].replace("{code}", item['code']).replace("{input}", str(item['input']))

        chat = [{"role": "user", "content": prompt_text}]
        formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        path_seed = seed * 1000 + instance_id * 100 + k
        trace = generate_text(model, tokenizer, formatted, path_seed)
        pred = extract_output(trace, gold)
        correct = check_match(pred, gold)
        all_answers.append(pred[:200])
        all_correct.append(correct)

    method_key = "sc" if method == "sc" else "sc_prompttpl"
    return {"method": method_key, "all_answers": all_answers, "all_correct": all_correct}


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
    cache_file = CACHE_DIR / f"{args.model}_capgated_cruxeval.jsonl"

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

    print(f"Loading {model_id}...")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto", "trust_remote_code": True}
    if gpu_gb > 80:
        try:
            import bitsandbytes
            load_kwargs["load_in_4bit"] = True
        except ImportError:
            pass
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()
    try:
        print(f"Model loaded on {model.device}")
    except Exception:
        print("Model loaded (multi-device)")

    from datasets import load_dataset
    ds = load_dataset("cruxeval-org/cruxeval", split="test")
    task_data = list(ds)

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

                item = task_data[i]
                gold = str(item["output"])
                t0 = time.time()

                result = run_method(model, tokenizer, item, seed, i, method, gold)

                entry = {
                    "model_slug": args.model,
                    "method": result["method"],
                    "task": "cruxeval",
                    "K": K,
                    "seed": seed,
                    "instance_id": i,
                    "gold_answer": gold,
                    "all_answers": result["all_answers"],
                    "all_correct": result["all_correct"],
                    "timestamp": time.time(),
                    "elapsed": time.time() - t0,
                }

                with open(cache_file, "a") as f:
                    f.write(json.dumps(entry) + "\n")

                total += 1
                elapsed = time.time() - t0
                n_correct = sum(result["all_correct"])
                print(f"  [{total}] {method_key} cruxeval seed={seed} i={i}: {n_correct}/{K} correct ({elapsed:.1f}s)")

    print(f"\nDone. {total} new entries written to {cache_file}")


if __name__ == "__main__":
    main()
