#!/usr/bin/env python3
"""Capability-gated diversity — WinoGrande (commonsense pronoun resolution).

Binary choice: extract A/B (or 1/2), exact match against gold answer.
Dataset: allenai/winogrande, config="winogrande_xl", split="validation".

Usage:
    uv run python scripts/run_capability_gated_winogrande.py --model qwen7b --method both
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

# Pronoun resolution prompt templates (8 slots for K=8)
PROMPT_TEMPLATES_WINO = [
    # Slot 0: Standard
    "Which option best fills the blank?\n\n{sentence}\n\n{choices}\n\nAnswer:",
    # Slot 1: Context clues
    "Think about the context clues step by step to determine the correct option.\n\n{sentence}\n\n{choices}\n\nLet me reason step by step:",
    # Slot 2: Elimination
    "Consider both options and eliminate the wrong one.\n\n{sentence}\n\n{choices}\n\nElimination:",
    # Slot 3: Direct
    "Give just A or B.\n\n{sentence}\n\n{choices}\n\nThe answer is:",
    # Slot 4: Real-world knowledge
    "Use real-world knowledge to decide which option fills the blank.\n\n{sentence}\n\n{choices}\n\nBased on real-world knowledge:",
    # Slot 5: Careful reading
    "Read the sentence carefully, focus on meaning, and select the right option.\n\n{sentence}\n\n{choices}\n\nAfter careful reading:",
    # Slot 6: Logical consistency
    "Which option makes the sentence logically coherent?\n\n{sentence}\n\n{choices}\n\nThe logically coherent option is:",
    # Slot 7: Grammatical/semantic
    "Think about what makes grammatical and semantic sense.\n\n{sentence}\n\n{choices}\n\nConsidering grammar and meaning:",
]


def extract_binary_answer(text: str) -> str | None:
    """Extract A/B (or 1/2) from model output, return '1' or '2'."""
    text = text.strip()

    # Direct match: starts with A or B
    m = re.match(r'^([AB])\b', text)
    if m:
        return "1" if m.group(1) == "A" else "2"

    # Direct match: starts with 1 or 2
    m = re.match(r'^([12])\b', text)
    if m:
        return m.group(1)

    # "answer is A/B" or "correct is A/B"
    m = re.search(r'(?:answer|correct)\s*(?:is|:)\s*\(?([AB])\)?', text, re.I)
    if m:
        return "1" if m.group(1).upper() == "A" else "2"

    # "answer is 1/2" or "correct is 1/2"
    m = re.search(r'(?:answer|correct)\s*(?:is|:)\s*\(?([12])\)?', text, re.I)
    if m:
        return m.group(1)

    # Last standalone A/B
    letters = re.findall(r'\b([AB])\b', text)
    if letters:
        return "1" if letters[-1] == "A" else "2"

    # Last standalone 1/2
    digits = re.findall(r'\b([12])\b', text)
    if digits:
        return digits[-1]

    return None


def format_winogrande_question(sentence: str, option1: str, option2: str) -> tuple[str, str]:
    """Format WinoGrande item into sentence display and choices string."""
    choices = f"A) {option1}\nB) {option2}"
    return sentence, choices


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


def run_sc(model, tokenizer, item, seed, instance_id, gold):
    """Standard SC: same prompt, K paths."""
    sentence, choices = format_winogrande_question(
        item["sentence"], item["option1"], item["option2"]
    )
    prompt_text = PROMPT_TEMPLATES_WINO[0].format(sentence=sentence, choices=choices)
    chat = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    all_answers = []
    for k in range(K):
        path_seed = seed * 1000 + instance_id * 100 + k
        trace = generate_text(model, tokenizer, formatted, path_seed)
        ans = extract_binary_answer(trace)
        all_answers.append(ans)

    return {"method": "sc", "all_answers": all_answers}


def run_pt(model, tokenizer, item, seed, instance_id, gold):
    """Prompt-template SC: different prompt per slot."""
    sentence, choices = format_winogrande_question(
        item["sentence"], item["option1"], item["option2"]
    )
    all_answers = []
    for k in range(K):
        prompt_text = PROMPT_TEMPLATES_WINO[k].format(sentence=sentence, choices=choices)
        chat = [{"role": "user", "content": prompt_text}]
        formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        path_seed = seed * 1000 + instance_id * 100 + k
        trace = generate_text(model, tokenizer, formatted, path_seed)
        ans = extract_binary_answer(trace)
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
    cache_file = CACHE_DIR / f"{args.model}_capgated_winogrande.jsonl"

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

    # Load WinoGrande
    from datasets import load_dataset
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
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
                gold = item["answer"]  # "1" or "2"
                t0 = time.time()

                if method == "sc":
                    result = run_sc(model, tokenizer, item, seed, i, gold)
                else:
                    result = run_pt(model, tokenizer, item, seed, i, gold)

                all_correct = [1 if a == gold else 0 for a in result["all_answers"]]

                entry = {
                    "model_slug": args.model,
                    "method": result["method"],
                    "task": "winogrande",
                    "K": K,
                    "seed": seed,
                    "instance_id": i,
                    "gold_answer": gold,
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
                print(f"  [{total}] {method_key} winogrande seed={seed} i={i}: {n_correct}/{K} correct ({elapsed:.1f}s)")

    print(f"\nDone. {total} new entries written to {cache_file}")


if __name__ == "__main__":
    main()
