#!/usr/bin/env python3
"""Capability-gated diversity experiments — Code generation (MBPP).

Correctness = pass all test cases (executed in subprocess with timeout).

Usage:
    uv run python scripts/run_capability_gated_mbpp.py --model qwen7b --method both
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
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
N_INSTANCES = 50  # HumanEval has 164 total
MAX_NEW_TOKENS = 2048  # Match other scripts for fair comparison
TEMPERATURE = 0.7
EXEC_TIMEOUT = 10  # seconds per test execution

PROMPT_TEMPLATES_CODE = [
    # Slot 0: Standard
    "Complete the following Python function.\n\n{prompt}",
    # Slot 1: Think first
    "Think about the approach step by step, then complete the function.\n\n{prompt}",
    # Slot 2: Test-driven
    "Consider what test cases this function should handle, then implement it.\n\n{prompt}",
    # Slot 3: Concise
    "Write the most concise implementation possible.\n\n{prompt}",
    # Slot 4: Defensive
    "Write a robust implementation with edge case handling.\n\n{prompt}",
    # Slot 5: Algorithmic
    "Choose the most efficient algorithm and implement it.\n\n{prompt}",
    # Slot 6: Readable
    "Write clean, readable code with meaningful variable names.\n\n{prompt}",
    # Slot 7: Alternative
    "Think of an alternative approach to the obvious solution, then implement.\n\n{prompt}",
]


def extract_code(text: str, entry_point: str) -> str:
    """Extract the function body from model output."""
    # Try to find a complete function definition
    lines = text.split('\n')
    code_lines = []
    in_func = False
    for line in lines:
        if f'def {entry_point}' in line:
            in_func = True
            code_lines = [line]
            continue
        if in_func:
            if line.strip() == '' or line[0] == ' ' or line[0] == '\t':
                code_lines.append(line)
            elif line.startswith('def ') or line.startswith('class '):
                break
            else:
                code_lines.append(line)
                break

    if code_lines:
        return '\n'.join(code_lines)

    # Fallback: return everything (the prompt already has the function signature)
    return text


def check_correctness_mbpp(completion: str, test_code: str, imports: str) -> bool:
    """Execute MBPP: completion is full code, test_code is assert statements."""
    # Extract code blocks if model wrapped in ```python
    code = completion
    m = re.search(r'```python\s*\n(.*?)```', code, re.DOTALL)
    if m:
        code = m.group(1)
    # Also try extracting just def blocks
    lines = code.split('\n')
    code_lines = []
    for line in lines:
        if line.strip().startswith('#') or line.strip() == '':
            code_lines.append(line)
        elif line[0:1] in (' ', '\t', '') or line.startswith('def ') or line.startswith('import ') or line.startswith('from '):
            code_lines.append(line)
        elif code_lines:
            break
    if code_lines:
        code = '\n'.join(code_lines)

    full_code = imports + "\n" + code + "\n" + test_code
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_code)
        f.flush()
        try:
            result = subprocess.run(
                ['python3', f.name],
                capture_output=True,
                timeout=EXEC_TIMEOUT,
                text=True,
            )
            if result.returncode == 0:
                os.unlink(f.name)
                return True
        except (subprocess.TimeoutExpired, Exception):
            pass
        finally:
            try:
                os.unlink(f.name)
            except OSError:
                pass
    return False


def check_correctness(prompt: str, completion: str, test: str, entry_point: str) -> bool:
    """Execute generated code against test cases."""
    # The model output may contain the full function (including signature from prompt).
    # Strategy: try completion alone first (model repeated the signature),
    # then try prompt + completion (model only gave the body).
    candidates = []

    # Option 1: completion contains full function definition
    if f'def {entry_point}' in completion:
        # Extract everything from the function def onward
        idx = completion.find(f'def {entry_point}')
        candidates.append(completion[idx:])

    # Option 2: completion is just the function body (indented code)
    candidates.append(prompt + completion)

    # Option 3: raw completion
    candidates.append(completion)

    for code in candidates:
        full_code = code + "\n" + test + f"\ncheck({entry_point})\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            f.flush()
            try:
                result = subprocess.run(
                    ['python3', f.name],
                    capture_output=True,
                    timeout=EXEC_TIMEOUT,
                    text=True,
                )
                if result.returncode == 0:
                    os.unlink(f.name)
                    return True
            except (subprocess.TimeoutExpired, Exception):
                pass
            finally:
                try:
                    os.unlink(f.name)
                except OSError:
                    pass

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


def run_sc(model, tokenizer, item, seed, instance_id):
    """Standard SC: same prompt, K paths."""
    prompt_text = f"Write a Python function for the following task.\n\n{item['prompt']}\n\nWrite only the function code:"
    chat = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    test_code = "\n".join(item.get("test_list", []))
    imports = "\n".join(item.get("test_imports", []))

    all_passed = []
    all_completions = []
    for k in range(K):
        path_seed = seed * 1000 + instance_id * 100 + k
        completion = generate_text(model, tokenizer, formatted, path_seed)
        passed = check_correctness_mbpp(completion, test_code, imports)
        all_passed.append(passed)
        all_completions.append(completion[:500])

    return {"method": "sc", "all_correct": all_passed, "all_completions": all_completions}


def run_pt(model, tokenizer, item, seed, instance_id):
    """Prompt-template SC: different prompt per slot."""
    test_code = "\n".join(item.get("test_list", []))
    imports = "\n".join(item.get("test_imports", []))

    all_passed = []
    all_completions = []
    for k in range(K):
        prompt_text = PROMPT_TEMPLATES_CODE[k].format(prompt=item['prompt'])
        chat = [{"role": "user", "content": prompt_text}]
        formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        path_seed = seed * 1000 + instance_id * 100 + k
        completion = generate_text(model, tokenizer, formatted, path_seed)
        passed = check_correctness_mbpp(completion, test_code, imports)
        all_passed.append(passed)
        all_completions.append(completion[:500])

    return {"method": "sc_prompttpl", "all_correct": all_passed, "all_completions": all_completions}


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
    cache_file = CACHE_DIR / f"{args.model}_capgated_mbpp.jsonl"

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
            print("Using 4-bit quantization")
        except ImportError:
            print("WARNING: bitsandbytes not installed, loading in bf16")
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()
    try:
        print(f"Model loaded on {model.device}")
    except Exception:
        print("Model loaded (multi-device)")

    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
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
                t0 = time.time()

                if method == "sc":
                    result = run_sc(model, tokenizer, item, seed, i)
                else:
                    result = run_pt(model, tokenizer, item, seed, i)

                entry = {
                    "model_slug": args.model,
                    "method": result["method"],
                    "task": "mbpp",
                    "K": K,
                    "seed": seed,
                    "instance_id": i,
                    "task_id": item.get("task_id", i),
                    "all_correct": result["all_correct"],
                    "timestamp": time.time(),
                    "elapsed": time.time() - t0,
                }

                with open(cache_file, "a") as f:
                    f.write(json.dumps(entry) + "\n")

                total += 1
                elapsed = time.time() - t0
                n_pass = sum(result["all_correct"])
                print(f"  [{total}] {method_key} humaneval seed={seed} i={i}: {n_pass}/{K} pass ({elapsed:.1f}s)")

    print(f"\nDone. {total} new entries written to {cache_file}")


if __name__ == "__main__":
    main()
