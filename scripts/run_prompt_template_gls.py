#!/usr/bin/env python3
"""Generate the prompt-template arm: K=8 paths per question, path k under template k.

Every slot uses the k-th template of the benchmark's template family (TEMPLATE_FAMILIES)
at one shared sampling temperature. One JSONL record per (seed, instance) is appended to
<out_dir>/<model>_prompt_template_<task>.jsonl, where <out_dir> is DC_RECORDS_DIR when set
and experiment_dir("cache") otherwise; an interrupted run resumes from the stored keys.
After generation the script prints per-slot accuracies and a weighted-vote summary; the
paper's numbers come from scripts/reproduce_cross_benchmark.py.

Usage:
    python scripts/run_prompt_template_gls.py --model qwen7b --tasks math --seeds 42 123 456 789 1024 --n-instances 100
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO
for _path in (str(_SRC), str(_REPO)):
    while _path in sys.path:
        sys.path.remove(_path)
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_SRC))

import diversity_combining


def _assert_local_diversity_combining_import() -> None:
    """Fail before generation if an editable install shadows this checkout."""
    imported = Path(diversity_combining.__file__).resolve()
    expected = _SRC.resolve()
    try:
        imported.relative_to(expected)
    except ValueError as exc:
        raise SystemExit(
            "FATAL: imported diversity_combining from the wrong checkout: "
            f"imported={imported}; expected_under={expected}"
        ) from exc


_assert_local_diversity_combining_import()
from diversity_combining.config import CACHE_DIR
from scripts.run_sc_kvar_v2 import ANSWER_FORMAT_SUFFIX, extract_answer, load_task_instances
from scripts.scoring_v2 import canonicalize_answer, check_task_correct
from scripts.tasks12 import (
    canonicalize_for_instance,
    evaluate_mbpp_completion,
    require_nonempty_slice,
)


MODEL_CONFIGS = {
    "qwen7b": ("Qwen/Qwen2.5-7B-Instruct", 64),
    "qwen05b": ("Qwen/Qwen2.5-0.5B-Instruct", 16),
    "qwen32b": ("Qwen/Qwen2.5-32B-Instruct", 96),
    "llama8b": ("meta-llama/Llama-3.1-8B-Instruct", 64),
    "mistral7b": ("mistralai/Mistral-7B-Instruct-v0.3", 64),
    "qwen3_5_9b": ("Qwen/Qwen3.5-9B", 96),
}

TASKS = [
    "gsm8k", "math", "hotpotqa", "triviaqa", "boolq",
    "drop", "mmlu", "hellaswag", "cruxeval", "mbpp",
    "arc_challenge", "winogrande",
]
TASK_DEFAULT_MAX_TOKENS = {
    "gsm8k": 2048, "math": 2048, "hotpotqa": 512, "triviaqa": 512,
    "boolq": 256, "drop": 512, "mmlu": 4096, "hellaswag": 512,
    "cruxeval": 1024, "mbpp": 1024, "arc_challenge": 1024, "winogrande": 512,
}
K_VALUES = [8]  # 8 slots = 8 prompt templates
SEEDS = [42, 123]  # default seeds; the paper's runs pass --seeds
N_INSTANCES = 50  # default instances per seed; the paper's runs pass --n-instances 100


# ── Prompt Templates (K=8 fixed slots) ──────────────────────────────────
# Each template encodes a distinct reasoning strategy.
# The {question} placeholder is replaced with the actual problem.

PROMPT_TEMPLATES = [
    # Slot 0: Standard CoT
    "Solve the following problem step by step.\n\nProblem: {question}\n\nSolution:",

    # Slot 1: Algebra-first
    "Use algebraic equations to solve this problem. Define variables, "
    "write equations, and solve.\n\nProblem: {question}\n\nSolution:",

    # Slot 2: Estimation then precise
    "First give a rough estimate of the answer, then solve precisely "
    "to get the exact answer.\n\nProblem: {question}\n\nSolution:",

    # Slot 3: Decomposition
    "Break this problem into smaller subproblems. Solve each subproblem "
    "separately, then combine the results.\n\nProblem: {question}\n\nSolution:",

    # Slot 4: Work backwards
    "Think about what the answer should look like, then work backwards "
    "to verify. Show your reasoning.\n\nProblem: {question}\n\nSolution:",

    # Slot 5: Verify and check
    "Solve this problem carefully. After finding the answer, verify it "
    "by substituting back or using an alternative method.\n\nProblem: {question}\n\nSolution:",

    # Slot 6: Concise direct
    "Solve this problem. Be concise and direct. Give the final numerical "
    "answer.\n\nProblem: {question}\n\nSolution:",

    # Slot 7: Explain to a student
    "Explain how to solve this problem as if teaching a student. "
    "Show every step clearly.\n\nProblem: {question}\n\nSolution:",
]

assert len(PROMPT_TEMPLATES) == 8, f"Need exactly 8 templates, got {len(PROMPT_TEMPLATES)}"

PROMPT_TEMPLATES_QA = [
    "Answer the following question based on the provided context.\n\n{question}\n\nAnswer:",
    "Read the context carefully and reason step by step to answer the question.\n\n{question}\n\nLet me think step by step:",
    "First identify the key facts in the context, then answer the question.\n\n{question}\n\nKey facts:",
    "Give a short, direct answer to this question.\n\n{question}\n\nShort answer:",
    "Answer the question, then verify your answer against the context.\n\n{question}\n\nAnswer and verification:",
    "Break the question into sub-questions, answer each, then combine.\n\n{question}\n\nSub-questions:",
    "Answer in as few words as possible.\n\n{question}\n\nAnswer:",
    "Explain your reasoning thoroughly, then give the final answer.\n\n{question}\n\nExplanation:",
]
PROMPT_TEMPLATES_BOOLQ = [
    "Based on the passage, answer Yes or No.\n\n{question}\n\nAnswer:",
    "Read carefully and reason step by step, then answer Yes or No.\n\n{question}\n\nLet me reason step by step:",
    "Find the relevant evidence in the passage, then answer.\n\n{question}\n\nRelevant evidence:",
    "Answer only Yes or No, nothing else.\n\n{question}\n\nAnswer:",
    "First quote the relevant part of the passage, then answer.\n\n{question}\n\nQuote:",
    "Consider whether the passage supports or contradicts the question.\n\n{question}\n\nAnalysis:",
    "Think about what the passage actually says vs what it implies.\n\n{question}\n\nThinking:",
    "Determine if the answer is explicitly stated or can be inferred.\n\n{question}\n\nDetermination:",
]
PROMPT_TEMPLATES_DROP = [
    "Read the passage and answer the question.\n\n{question}\n\nAnswer:",
    "Think step by step, count or compare as needed to answer the question.\n\n{question}\n\nLet me think step by step:",
    "Find the relevant numbers or facts in the passage, then answer the question.\n\n{question}\n\nRelevant facts:",
    "Give a short, direct answer to this question.\n\n{question}\n\nShort answer:",
    "Show your arithmetic or reasoning, then answer the question.\n\n{question}\n\nReasoning:",
    "Focus on the specific details asked about in the question.\n\n{question}\n\nAnswer:",
    "Extract the relevant information from the passage, then compute the answer.\n\n{question}\n\nExtracted info:",
    "Carefully trace the quantities mentioned in the passage to answer the question.\n\n{question}\n\nTracing quantities:",
]
PROMPT_TEMPLATES_MC = [
    "Answer the following multiple choice question.\n\n{question}\n\nAnswer:",
    "Think step by step, then select the best answer.\n\n{question}\n\nLet me reason through this:",
    "Eliminate wrong answers first, then choose.\n\n{question}\n\nElimination:",
    "Give the answer directly with just the letter.\n\n{question}\n\nThe answer is:",
    "Explain why each option is right or wrong, then select.\n\n{question}\n\nAnalysis:",
    "Use your scientific knowledge to answer.\n\n{question}\n\nBased on scientific principles:",
    "Consider real-world examples to determine the answer.\n\n{question}\n\nReal-world reasoning:",
    "What would a teacher say is the correct answer?\n\n{question}\n\nA teacher would say:",
]
PROMPT_TEMPLATES_HELLASWAG = [
    "Choose the most likely continuation of the following scenario.\n\n{question}\n\nAnswer:",
    "Think step by step about what happens next, then select the best continuation.\n\n{question}\n\nLet me reason through this:",
    "Eliminate unlikely continuations first, then choose the best one.\n\n{question}\n\nElimination:",
    "Give just the letter of the most likely continuation.\n\n{question}\n\nThe answer is:",
    "Consider real-world common sense to determine the best continuation.\n\n{question}\n\nBased on common sense:",
    "Visualize the scenario, then choose what would happen next.\n\n{question}\n\nVisualizing the scene:",
    "Think about cause and effect to select the correct continuation.\n\n{question}\n\nCause and effect reasoning:",
    "What would naturally follow in this situation? Select the best continuation.\n\n{question}\n\nNaturally, what follows is:",
]
PROMPT_TEMPLATES_WINOGRANDE = [
    "Which option best fills the blank?\n\n{question}\n\nAnswer:",
    "Think about the context clues step by step to determine the correct option.\n\n{question}\n\nLet me reason step by step:",
    "Consider both options and eliminate the wrong one.\n\n{question}\n\nElimination:",
    "Give just A or B.\n\n{question}\n\nThe answer is:",
    "Use real-world knowledge to decide which option fills the blank.\n\n{question}\n\nBased on real-world knowledge:",
    "Read the sentence carefully, focus on meaning, and select the right option.\n\n{question}\n\nAfter careful reading:",
    "Which option makes the sentence logically coherent?\n\n{question}\n\nThe logically coherent option is:",
    "Think about what makes grammatical and semantic sense.\n\n{question}\n\nConsidering grammar and meaning:",
]
PROMPT_TEMPLATES_CRUXEVAL = [
    "What is the output of the following Python code?\n\n{question}\n\nOutput:",
    "Trace through this code step by step, then give the output.\n\n{question}\n\nStep-by-step trace:",
    "Execute this function mentally and predict the result.\n\n{question}\n\nResult:",
    "Give only the output value, nothing else.\n\n{question}\n\nOutput value:",
    "What does the function return?\n\n{question}\n\nReturn value:",
    "Analyze the code logic, then predict the output.\n\n{question}\n\nCode analysis:",
    "Think about edge cases, then give the output.\n\n{question}\n\nEdge case analysis and output:",
    "Simulate a Python interpreter running this code.\n\n{question}\n\nInterpreter output:",
]
PROMPT_TEMPLATES_MBPP = [
    "Complete the following Python function.\n\n{question}",
    "Think about the approach step by step, then complete the function.\n\n{question}",
    "Consider what test cases this function should handle, then implement it.\n\n{question}",
    "Write the most concise implementation possible.\n\n{question}",
    "Write a robust implementation with edge case handling.\n\n{question}",
    "Choose the most efficient algorithm and implement it.\n\n{question}",
    "Write clean, readable code with meaningful variable names.\n\n{question}",
    "Think of an alternative approach to the obvious solution, then implement.\n\n{question}",
]

TEMPLATE_FAMILIES = {
    "gsm8k": ("math", PROMPT_TEMPLATES), "math": ("math", PROMPT_TEMPLATES),
    "hotpotqa": ("qa", PROMPT_TEMPLATES_QA), "triviaqa": ("qa", PROMPT_TEMPLATES_QA),
    "boolq": ("reading_comprehension", PROMPT_TEMPLATES_BOOLQ),
    "drop": ("discrete_reasoning", PROMPT_TEMPLATES_DROP),
    "mmlu": ("science_mc", PROMPT_TEMPLATES_MC),
    "hellaswag": ("commonsense", PROMPT_TEMPLATES_HELLASWAG),
    "arc_challenge": ("science_mc", PROMPT_TEMPLATES_MC),
    "winogrande": ("commonsense", PROMPT_TEMPLATES_WINOGRANDE),
    "cruxeval": ("code_understanding", PROMPT_TEMPLATES_CRUXEVAL),
    "mbpp": ("code_generation", PROMPT_TEMPLATES_MBPP),
}


# ── Answer Extraction ────────────────────────────────────────────────────

def strip_artifacts(text: str) -> str:
    """Strip chat template turn markers that leak on truncation."""
    for pat in [r"\nuser\n", r"\nUser:", r"\n<\|im_start\|>", r"\nassistant\n"]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            text = text[:m.start()]
    return text.strip()


def extract_number(text: str) -> str | None:
    """Extract final number from reasoning trace."""
    text = strip_artifacts(text)
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


def answers_match(pred: str | None, gold: str | None) -> bool:
    """Check if predicted answer matches gold, handling numeric comparison."""
    if pred is None or gold is None:
        return False
    try:
        p = float(pred)
        g = float(gold)
        return abs(p - g) < 1e-3
    except (ValueError, TypeError):
        return str(pred).strip() == str(gold).strip()


# ── Generation ───────────────────────────────────────────────────────────

def _trim_terminal_ids(generated_ids: list[int], terminal_ids: set[int]) -> tuple[list[int], bool]:
    """Trim terminal padding/EOS ids and report whether generation ended naturally."""
    end = len(generated_ids)
    while end > 0 and generated_ids[end - 1] in terminal_ids:
        end -= 1
    return generated_ids[:end], end < len(generated_ids)


def build_template_prompt(template: str, question: str, task: str) -> str:
    """Apply a slot template and the SC driver's task-specific answer contract."""
    return template.format(question=question) + ANSWER_FORMAT_SUFFIX.get(task, "")


def run_prompt_template_sc(
    model, tokenizer, question: str, K: int,
    templates: list[str], max_tokens: int, device: str,
    has_chat: bool, temperature: float = 0.7,
    task: str = "gsm8k", item: dict | None = None,
) -> tuple[list[str | None], list[str], list[str], list[int], list[bool]]:
    """Generate K paths, each with a different prompt template.

    Returns (answers, traces, template_names, generation lengths, cap-hit flags).
    """
    import torch

    # Collect stop token IDs
    stop_ids = []
    for tok_name in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>"]:
        tid = tokenizer.convert_tokens_to_ids(tok_name)
        if tid is not None and tid != tokenizer.unk_token_id:
            stop_ids.append(tid)

    all_answers = []
    all_traces = []
    gen_lens = []
    truncated = []

    for k in range(K):
        # Build prompt with slot-specific template
        template = templates[k % len(templates)]
        prompt = build_template_prompt(template, question, task)

        # Apply chat template
        if has_chat:
            messages = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs.input_ids.shape[1]

        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
        )
        if stop_ids:
            gen_kwargs["eos_token_id"] = stop_ids

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        generated_ids = outputs[0, input_len:]
        if task == "gsm8k":
            # Preserve every stored GSM8K field byte-for-byte.
            gen_len = int(generated_ids.shape[0])
            is_truncated = gen_len >= max_tokens
            decode_ids = generated_ids
        else:
            generated_list = generated_ids.tolist()
            terminal_ids = set(stop_ids)
            if tokenizer.eos_token_id is not None:
                eos_ids = tokenizer.eos_token_id
                terminal_ids.update(eos_ids if isinstance(eos_ids, list) else [eos_ids])
            if tokenizer.pad_token_id is not None:
                terminal_ids.add(tokenizer.pad_token_id)
            decode_ids, ended_naturally = _trim_terminal_ids(generated_list, terminal_ids)
            gen_len = len(decode_ids)
            is_truncated = not ended_naturally and gen_len >= max_tokens
        trace = tokenizer.decode(decode_ids, skip_special_tokens=True)
        all_traces.append(trace)
        gen_lens.append(gen_len)
        truncated.append(is_truncated)

        # Extract answer
        trace_clean = strip_artifacts(trace)
        if task == "gsm8k":
            extracted = extract_number(trace_clean)
            answer = extracted if extracted is not None else trace_clean.strip().split("\n")[-1]
        elif task == "mbpp":
            if item is None:
                raise ValueError("MBPP generation requires the dataset item")
            answer = None if is_truncated else evaluate_mbpp_completion(
                trace_clean,
                list(item["assertions"]),
                list(item.get("test_imports", [])),
            )
        else:
            extracted = extract_answer(trace_clean, task=task, is_truncated=is_truncated)
            if task in ("math", "triviaqa", "drop", "cruxeval"):
                if item is None:
                    raise ValueError(f"{task} generation requires the dataset item")
                answer = canonicalize_for_instance(extracted, item, task)
            else:
                answer = canonicalize_answer(extracted, task)
        all_answers.append(answer)

    return all_answers, all_traces, [f"slot{k}" for k in range(K)], gen_lens, truncated


# ── Combiners ────────────────────────────────────────────────────────────

def majority_vote(answers: list[str | None]) -> str | None:
    """Standard majority vote in which abstentions cast no vote."""
    counts = Counter(answer for answer in answers if answer is not None)
    return counts.most_common(1)[0][0] if counts else None


def gls_weighted_vote(answers: list[str | None], weights: np.ndarray) -> str | None:
    """GLS-weighted majority vote in which abstentions cast no vote."""
    weighted_counts: dict[str, float] = defaultdict(float)
    for ans, w in zip(answers, weights):
        if ans is not None:
            weighted_counts[ans] += w
    return max(weighted_counts, key=weighted_counts.get) if weighted_counts else None


# ── Main ─────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen7b", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument(
        "--attn-implementation",
        choices=["eager", "sdpa", "default"],
        default="default",
    )
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--n-instances", type=int, default=N_INSTANCES)
    args = parser.parse_args(argv)
    if args.n_instances <= 0:
        parser.error("--n-instances must be positive")
    if not args.seeds:
        parser.error("--seeds requires at least one value")
    if args.max_tokens is not None and args.max_tokens <= 0:
        parser.error("--max_tokens must be positive")
    return args


def build_model_kwargs(attn_implementation: str, torch_dtype) -> dict:
    """Build model-loading kwargs, omitting attention selection for default."""
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if attn_implementation != "default":
        model_kwargs["attn_implementation"] = attn_implementation
    return model_kwargs


def resolve_out_file(model: str, task: str | None = None) -> Path:
    """Resolve the output file, honoring the DC_RECORDS_DIR override."""
    out_dir = Path(os.environ["DC_RECORDS_DIR"]) if os.environ.get("DC_RECORDS_DIR") else CACHE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{task}" if task else ""
    return out_dir / f"{model}_prompt_template{suffix}.jsonl"


def check_resume_provenance(
    out_file: Path, max_tokens: int, temperature: float,
    attn_implementation: str, task: str | None = None,
) -> int:
    """Validate existing records before appending and return their count."""
    if not out_file.exists():
        return 0

    record_count = 0
    with open(out_file) as f:
        for lineno, line in enumerate(f, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"FATAL: {out_file}:{lineno} is not valid JSON; refusing to resume"
                ) from exc
            record_count += 1
            provenance = record.get("provenance", {}) if isinstance(record, dict) else {}
            record_max_tokens = record.get("max_tokens", provenance.get("max_tokens"))
            if not isinstance(record, dict) or record_max_tokens is None:
                raise SystemExit(
                    f"FATAL: {out_file}:{lineno} lacks max_tokens; refusing to resume"
                )
            if record_max_tokens != max_tokens:
                raise SystemExit(
                    f"FATAL: {out_file}:{lineno} max_tokens mismatch "
                    f"(cached={record_max_tokens!r}, current={max_tokens!r}); "
                    "refusing to resume"
                )
            if "attn_implementation" not in record:
                raise SystemExit(
                    f"FATAL: {out_file}:{lineno} lacks attn_implementation; "
                    "refusing to resume"
                )
            if record["attn_implementation"] != attn_implementation:
                raise SystemExit(
                    f"FATAL: {out_file}:{lineno} attn_implementation mismatch "
                    f"(cached={record['attn_implementation']!r}, "
                    f"current={attn_implementation!r}); refusing to resume"
                )
            if "temperature" in record and record["temperature"] != temperature:
                raise SystemExit(
                    f"FATAL: {out_file}:{lineno} temperature mismatch "
                    f"(cached={record['temperature']!r}, current={temperature!r}); "
                    "refusing to resume"
                )
            if task is not None and record.get("task") != task:
                raise SystemExit(
                    f"FATAL: {out_file}:{lineno} task mismatch "
                    f"(cached={record.get('task')!r}, current={task!r}); refusing to resume"
                )
    return record_count


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id, ram_gb = MODEL_CONFIGS[args.model]
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model_kwargs = build_model_kwargs(args.attn_implementation, torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()
    resolved_attn = getattr(model.config, "_attn_implementation", "<unknown>")
    print(f"Resolved attention implementation: {resolved_attn}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"

    has_chat = hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None
    print(f"Chat template: {has_chat}")

    print(f"\n{'='*60}")
    print(f"PROMPT-TEMPLATE-PER-SLOT SC + GLS EXPERIMENT")
    print(f"Model: {model_id}")
    print(f"Templates: {K_VALUES[0]} slots per task family")
    print(f"Temperature: {args.temperature} (uniform across slots)")
    print(f"{'='*60}")

    output_files: list[Path] = []
    for task in args.tasks:
        max_tokens = args.max_tokens or TASK_DEFAULT_MAX_TOKENS[task]
        dataset = require_nonempty_slice(load_task_instances(task), args.n_instances, task)
        out_file = resolve_out_file(args.model, task)
        output_files.append(out_file)
        existing_count = check_resume_provenance(
            out_file, max_tokens, args.temperature, args.attn_implementation, task,
        )
        out_file.touch(exist_ok=True)
        existing_keys: set[str] = set()
        with open(out_file) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    existing_keys.add(json.loads(line).get("key", ""))
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"FATAL: invalid JSON in {out_file}: {exc}") from exc

        family_name, family_templates = TEMPLATE_FAMILIES[task]
        print(f"\nTask: {task} ({len(dataset)} instances)")
        print(
            f"  family={family_name} max_tokens={max_tokens} "
            f"resume={existing_count} out_file={out_file}"
        )

        for K in K_VALUES:
            templates = family_templates[:K]
            print(f"\n  K={K}, templates: {[f'slot{i}' for i in range(K)]}")

            for seed in args.seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)

                for i, item in enumerate(dataset):
                    key = f"{args.model}_sc_prompttpl_{task}_K{K}_s{seed}_i{i}"
                    if key in existing_keys:
                        continue

                    # Extract raw question from the prompt
                    # (strip the data_loader prefix to inject our own template)
                    raw_question = item["prompt"]
                    # Remove common prefixes from data_loader
                    for prefix in [
                        "Solve the following math problem step by step.\n\nProblem: ",
                        "Solve the following math problem. Show your work and give the final answer.\n\nProblem: ",
                    ]:
                        if raw_question.startswith(prefix):
                            raw_question = raw_question[len(prefix):]
                            break
                    # Remove trailing "\n\nSolution:"
                    if raw_question.endswith("\n\nSolution:"):
                        raw_question = raw_question[:-len("\n\nSolution:")]

                    answers, traces, slot_names, gen_lens, truncated = run_prompt_template_sc(
                        model, tokenizer, raw_question, K,
                        templates, max_tokens, device,
                        has_chat, args.temperature, task=task, item=item,
                    )

                    if task in ("math", "triviaqa", "drop", "cruxeval"):
                        gold = canonicalize_for_instance(item["answer"], item, task)
                        if gold is None:
                            raise ValueError(f"{task} instance {i} has empty canonical gold")
                    else:
                        gold = str(item["answer"])

                    provenance = {
                        "max_tokens": max_tokens,
                        "temperature": args.temperature,
                        "attn_implementation": args.attn_implementation,
                        "template_family": family_name,
                    }
                    entry = {
                        "key": key,
                        "model_slug": args.model,
                        "method": "sc_prompttpl",
                        "task": task,
                        "K": K,
                        "seed": seed,
                        "instance_id": i,
                        "gold_answer": gold,
                        "all_answers": answers,
                        "all_traces": traces,
                        "gen_lens": gen_lens,
                        "truncated": truncated,
                        "slot_names": slot_names,
                        "template_family": family_name,
                        "temperature": args.temperature,
                        "max_tokens": max_tokens,
                        "attn_implementation": args.attn_implementation,
                        "provenance": provenance,
                        "timestamp": time.time(),
                    }
                    with open(out_file, "a") as f:
                        f.write(json.dumps(entry, default=str) + "\n")
                    existing_keys.add(key)

                    if (i + 1) % 20 == 0:
                        print(f"    seed={seed}: {i+1}/{len(dataset)}")

                print(f"    seed={seed}: done")

    # ── Analysis ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"ANALYSIS: Prompt-Template GLS vs Uniform MV")
    print(f"{'='*60}")

    all_data = []
    for out_file in output_files:
        with open(out_file) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    all_data.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"FATAL: invalid JSON in {out_file}: {exc}") from exc

    for task in args.tasks:
        print(f"\n--- {task.upper()} ---")
        for K in K_VALUES:
            # Cross-seed analysis: use leave-one-seed-out
            for test_seed in args.seeds:
                train_seeds = [s for s in args.seeds if s != test_seed]
                train_entries = [
                    e for e in all_data
                    if e["task"] == task and e["K"] == K and e["seed"] in train_seeds
                ]
                test_entries = [
                    e for e in all_data
                    if e["task"] == task and e["K"] == K and e["seed"] == test_seed
                ]

                if not train_entries or not test_entries:
                    continue

                n_train = len(train_entries)
                n_test = len(test_entries)

                # Train: estimate g and Sigma
                train_correct = np.zeros((n_train, K))
                for idx, e in enumerate(train_entries):
                    gold = str(e["gold_answer"])
                    for k_idx, a in enumerate(e["all_answers"]):
                        if check_task_correct(a, gold, task):
                            train_correct[idx, k_idx] = 1.0

                g_train = train_correct.mean(axis=0)
                Y_c = train_correct - g_train[np.newaxis, :]
                Sigma_train = (Y_c.T @ Y_c) / max(n_train - 1, 1)

                # Test: MV accuracy
                test_mv_correct = 0
                for e in test_entries:
                    gold = str(e["gold_answer"])
                    mv_ans = majority_vote(e["all_answers"])
                    if check_task_correct(mv_ans, gold, task):
                        test_mv_correct += 1
                test_mv_acc = test_mv_correct / n_test * 100

                # Per-slot accuracy on train set
                print(f"\n  K={K} test_seed={test_seed} (train={n_train}, test={n_test}):")
                print(f"    Train per-slot accuracy: {np.round(g_train * 100, 1)}")
                print(f"    Test MV: {test_mv_acc:.1f}%")

                # Rank stability check: Spearman correlation of slot quality
                if len(train_seeds) >= 2:
                    from scipy.stats import spearmanr
                    seed_accs = []
                    for s in train_seeds:
                        s_entries = [e for e in train_entries if e["seed"] == s]
                        s_correct = np.zeros((len(s_entries), K))
                        for idx, e in enumerate(s_entries):
                            gold = str(e["gold_answer"])
                            for k_idx, a in enumerate(e["all_answers"]):
                                if check_task_correct(a, gold, task):
                                    s_correct[idx, k_idx] = 1.0
                        seed_accs.append(s_correct.mean(axis=0))
                    rho, pval = spearmanr(seed_accs[0], seed_accs[1])
                    print(f"    Slot rank stability (Spearman ρ): {rho:.3f} (p={pval:.3f})")

                # GLS with lambda sweep
                for lam in [0.1, 0.5, 1.0, 5.0]:
                    Sigma_reg = Sigma_train + lam * np.eye(K)
                    try:
                        Sigma_inv_g = np.linalg.solve(Sigma_reg, g_train)
                        denom = g_train @ Sigma_inv_g
                        w_gls = Sigma_inv_g / denom if abs(denom) > 1e-10 else np.ones(K) / K
                    except np.linalg.LinAlgError:
                        w_gls = np.ones(K) / K

                    gls_correct = 0
                    for e in test_entries:
                        gold = str(e["gold_answer"])
                        gls_ans = gls_weighted_vote(e["all_answers"], w_gls)
                        if check_task_correct(gls_ans, gold, task):
                            gls_correct += 1
                    gls_acc = gls_correct / n_test * 100
                    delta = gls_acc - test_mv_acc
                    print(f"    GLS(λ={lam:4.2f}): {gls_acc:.1f}%  Δ={delta:+.1f}%")

                # Oracle: any-correct headroom
                oracle_correct = 0
                for e in test_entries:
                    gold = str(e["gold_answer"])
                    if any(check_task_correct(a, gold, task) for a in e["all_answers"]):
                        oracle_correct += 1
                oracle_acc = oracle_correct / n_test * 100
                print(f"    Oracle (any-correct): {oracle_acc:.1f}%  (headroom: {oracle_acc - test_mv_acc:+.1f}%)")


if __name__ == "__main__":
    main()
