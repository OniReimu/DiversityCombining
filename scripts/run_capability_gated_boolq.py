#!/usr/bin/env python3
"""Capability-gated diversity — BoolQ (yes/no reading comprehension).

Answer matching: extract Yes/No from model output, compare with gold boolean.

Usage:
    uv run python scripts/run_capability_gated_boolq.py --model qwen7b --method both
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

PROMPT_TEMPLATES_YESNO = [
    # Slot 0: Standard
    "Based on the passage, answer Yes or No.\n\n{question}\n\nAnswer:",
    # Slot 1: Chain of thought
    "Read carefully and reason step by step, then answer Yes or No.\n\n{question}\n\nLet me reason step by step:",
    # Slot 2: Evidence first
    "Find the relevant evidence in the passage, then answer.\n\n{question}\n\nRelevant evidence:",
    # Slot 3: Direct
    "Answer only Yes or No, nothing else.\n\n{question}\n\nAnswer:",
    # Slot 4: Quote then answer
    "First quote the relevant part of the passage, then answer.\n\n{question}\n\nQuote:",
    # Slot 5: Support/contradict
    "Consider whether the passage supports or contradicts the question.\n\n{question}\n\nAnalysis:",
    # Slot 6: Stated vs implied
    "Think about what the passage actually says vs what it implies.\n\n{question}\n\nThinking:",
    # Slot 7: Explicit or inferred
    "Determine if the answer is explicitly stated or can be inferred.\n\n{question}\n\nDetermination:",
]


def extract_yesno(text: str) -> str | None:
    """Extract Yes/No from model output."""
    text = text.strip()
    # Direct match at start: "Yes" or "No"
    m = re.match(r'^(yes|no)\b', text, re.IGNORECASE)
    if m:
        return m.group(1).capitalize()
    # "answer is yes/no" pattern
    m = re.search(r'answer\s*(?:is|:)\s*(yes|no)\b', text, re.IGNORECASE)
    if m:
        return m.group(1).capitalize()
    # Standalone "yes" or "no" anywhere in text
    matches = re.findall(r'\b(yes|no)\b', text, re.IGNORECASE)
    if matches:
        return matches[-1].capitalize()
    return None


def format_question(passage: str, question: str) -> str:
    """Format BoolQ passage and question into a single prompt."""
    return f"Passage: {passage}\n\nQuestion: {question}\n\nAnswer (Yes or No):"


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


def run_sc(model, tokenizer, item, seed, instance_id, gold_yesno):
    question_text = format_question(item["passage"], item["question"])
    prompt_text = PROMPT_TEMPLATES_YESNO[0].format(question=question_text)
    chat = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    all_answers = []
    for k in range(K):
        path_seed = seed * 1000 + instance_id * 100 + k
        trace = generate_text(model, tokenizer, formatted, path_seed)
        ans = extract_yesno(trace)
        all_answers.append(ans)

    return {"method": "sc", "all_answers": all_answers}


def run_pt(model, tokenizer, item, seed, instance_id, gold_yesno):
    question_text = format_question(item["passage"], item["question"])
    all_answers = []
    for k in range(K):
        prompt_text = PROMPT_TEMPLATES_YESNO[k].format(question=question_text)
        chat = [{"role": "user", "content": prompt_text}]
        formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        path_seed = seed * 1000 + instance_id * 100 + k
        trace = generate_text(model, tokenizer, formatted, path_seed)
        ans = extract_yesno(trace)
        all_answers.append(ans)

    return {"method": "sc_prompttpl", "all_answers": all_answers}


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
    cache_file = CACHE_DIR / f"{args.model}_capgated_boolq.jsonl"

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

    # Load BoolQ dataset
    from datasets import load_dataset
    ds = load_dataset("google/boolq", split="validation")
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
                gold_yesno = "Yes" if item["answer"] else "No"
                t0 = time.time()

                if method == "sc":
                    result = run_sc(model, tokenizer, item, seed, i, gold_yesno)
                else:
                    result = run_pt(model, tokenizer, item, seed, i, gold_yesno)

                all_correct = [1 if a == gold_yesno else 0 for a in result["all_answers"]]

                entry = {
                    "model_slug": args.model,
                    "method": result["method"],
                    "task": "boolq",
                    "K": K,
                    "seed": seed,
                    "instance_id": i,
                    "gold_answer": gold_yesno,
                    "all_answers": result["all_answers"],
                    "all_correct": all_correct,
                    "timestamp": time.time(),
                    "elapsed": time.time() - t0,
                }

                with open(cache_file, "a") as f:
                    f.write(json.dumps(entry) + "\n")

                total += 1
                elapsed = time.time() - t0
                n_correct = sum(all_correct)
                print(f"  [{total}] {method_key} boolq seed={seed} i={i}: {n_correct}/{K} correct ({elapsed:.1f}s)")

    print(f"\nDone. {total} new entries written to {cache_file}")


if __name__ == "__main__":
    main()
