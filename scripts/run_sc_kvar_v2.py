#!/usr/bin/env python3
"""K=32 SC generation script (v2 protocol) for reasoning and non-reasoning models.

Generates K=32 sampled reasoning paths once per (model, task, seed, instance)
using batched sampling with num_return_sequences in chunks of --batch-paths
under one torch seed per (seed, instance). Never regenerates per K; prefix-
subsampling derives K in [4, 8, 16, 32] downstream.

Applies each tokenizer's own chat template directly to the task loader prompt
without double wrapping. Reasoning behavior and final-answer segmentation are
declared in MODEL_CONFIGS; answer extraction never sees the thinking segment.

Records per record: model_slug, method ("sc"), task, K (32), seed, instance_id,
gold_answer, all_answers (length 32), all_traces (length 32), gen_lens (length 32),
truncated (length 32), timestamp, elapsed, max_new_tokens, and a provenance block.

Answer extraction & truncation rule:
- Explicit final-answer markers (balanced \boxed{...}, ####, or "final answer is X" /
  "the answer is X" phrases) are honoured even on truncated paths (a path can hit the
  token cap after declaring an answer).
- Heuristic fallbacks that read an unfinished calculation (last number for gsm8k/math,
  "= X at end of line", last yes/no for boolq, last line for hotpotqa, last standalone letter for MC)
  are refused on truncated paths and return None.
- Untruncated paths keep every rule and fallback.
This defines the denominator of every reported accuracy.

Output: append-only JSONL at {out_dir}/{model}_sc_kvar_v2_{task}.jsonl, keyed on
(seed, instance_id), skipping completed keys on startup. Startup refuses to append
if existing records disagree on model_id, max_new_tokens, script_version,
temperature, top_p, batch_paths, scorer, or prompt_sha.

Usage:
    python scripts/run_sc_kvar_v2.py --model qwen7b --task gsm8k
    python scripts/run_sc_kvar_v2.py --model llama8b --task gsm8k --seeds 42 123 456 789 1024
    python scripts/run_sc_kvar_v2.py --model qwen3_5_9b --task gsm8k
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
import re
import string
import sys
import time
from pathlib import Path

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
from diversity_combining.evaluation.data_loader import load_task
from scripts.scoring_v2 import check_task_correct, get_scorer
from scripts.tasks12 import (
    NEW_TASKS,
    canonicalize_for_instance,
    evaluate_mbpp_completion,
    normalize_latex_thousands_separators,
    require_nonempty_slice,
)

SCRIPT_VERSION = "v2.0"
EXTRACTOR_VERSION = "v2.2"
REASONING_MAX_NEW_TOKENS = 16384

@dataclass(frozen=True)
class ModelConfig:
    """Generation and reasoning-output contract for one model."""

    model_id: str
    ram_gb: int
    is_reasoning: bool = False
    thinking_start: str | None = None
    thinking_end: str | None = None
    final_channel: str | None = None
    chat_template_flag: str | None = None
    system_prompt: str | None = None
    template_kwargs: tuple[tuple[str, object], ...] = ()
    reasoning_max_new_tokens: int | None = None
    thinking_channel: str | None = None

    def __iter__(self):
        """Preserve legacy ``model_id, ram_gb = MODEL_CONFIGS[slug]`` callers."""
        return iter((self.model_id, self.ram_gb))

    def __getitem__(self, index):
        """Preserve legacy tuple indexing without exposing new fields positionally."""
        return (self.model_id, self.ram_gb)[index]

    def __len__(self) -> int:
        return 2


MODEL_CONFIGS = {
    "qwen05b": ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", 16),
    "qwen7b": ModelConfig("Qwen/Qwen2.5-7B-Instruct", 64),
    "qwen32b": ModelConfig("Qwen/Qwen2.5-32B-Instruct", 96),
    "llama8b": ModelConfig("meta-llama/Llama-3.1-8B-Instruct", 64),
    "mistral7b": ModelConfig("mistralai/Mistral-7B-Instruct-v0.3", 64),
    "qwen3_5_9b": ModelConfig(
        "Qwen/Qwen3.5-9B", 64, is_reasoning=True,
        thinking_start="<think>", thinking_end="</think>",
        chat_template_flag="enable_thinking=True",
        template_kwargs=(("enable_thinking", True),),
        reasoning_max_new_tokens=REASONING_MAX_NEW_TOKENS,
    ),
    "gptoss20b": ModelConfig(
        "openai/gpt-oss-20b", 80, is_reasoning=True, final_channel="final",
        chat_template_flag="reasoning_effort=medium",
        template_kwargs=(("reasoning_effort", "medium"),),
        reasoning_max_new_tokens=REASONING_MAX_NEW_TOKENS,
        thinking_channel="analysis",
    ),
    "r1qwen15b": ModelConfig(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 32, is_reasoning=True,
        thinking_start="<think>", thinking_end="</think>",
        chat_template_flag="add_generation_prompt=True",
        reasoning_max_new_tokens=REASONING_MAX_NEW_TOKENS,
    ),
}

TASK_CHOICES = [
    "gsm8k", "math", "hotpotqa", "triviaqa", "boolq",
    "drop", "mmlu", "hellaswag", "cruxeval", "mbpp",
]
DEFAULT_SEEDS = [42, 123, 456, 789, 1024]
DEFAULT_N = 100

TASK_DEFAULT_MAX_NEW_TOKENS = {
    "gsm8k": 1024,
    "math": 2048,
    "hotpotqa": 512,
    "triviaqa": 512,
    "boolq": 256,
    "drop": 512,
    "mmlu": 4096,
    "hellaswag": 512,
    "cruxeval": 1024,
    "mbpp": 1024,
}
ANSWER_FORMAT_SUFFIX = {
    "mmlu": "\n\nReason step by step, then end your response with the letter of the correct option "
            "on its own final line, in the form: #### <letter>",
    "hellaswag": "\n\nReason step by step, then end your response with the letter of the correct option "
                 "on its own final line, in the form: #### <letter>",
    "boolq": "\n\nEnd your response with your answer on its own final line, in the form: #### yes "
             "or #### no",
    "hotpotqa": "\n\nEnd your response with the short answer span on its own final line, "
                "in the form: #### <answer>",
    "triviaqa": "\n\nEnd your response with the short answer span on its own final line, "
                "in the form: #### <answer> If you are unsure, give your single best guess. "
                "Never leave the answer blank.",
    "drop": "\n\nEnd your response with the short answer on its own final line, "
            "in the form: #### <answer>",
    "math": "\n\nEnd your response with the final answer in the form: \\boxed{<answer>}",
    "cruxeval": "\n\nEnd your response with the predicted Python literal on its own final line, "
                "in the form: #### <literal>",
    "mbpp": "\n\nReturn only the complete Python function code.",
}


REASONING_GSM8K_SUFFIX = (
    "\n\nEnd your response with the final numeric answer on its own final line, "
    "in the form: #### <number>"
)


def build_prompt(question: str, task: str, model_slug: str | None = None) -> str:
    """Build the task prompt, adding the GSM8K contract only for reasoning models."""
    suffix = ANSWER_FORMAT_SUFFIX.get(task, "")
    if (
        task == "gsm8k"
        and model_slug is not None
        and MODEL_CONFIGS[model_slug].is_reasoning
    ):
        suffix += REASONING_GSM8K_SUFFIX
    return question + suffix

DEFAULT_BATCH_PATHS = 8
REASONING_BATCH_PATHS = 4
K = 32
TEMPERATURE = 0.7
TOP_P = 0.95
THINK_END_TAG = "</think>"
HARMONY_FINAL_MARKER = "<|channel|>final<|message|>"
REASONING_EXTRACTION_SLUGS = frozenset({"gptoss20b", "r1qwen15b", "qwen3_5_9b"})


# ── Answer extraction & truncation rule ─────────────────────────────────────
# Truncation rule (defines the denominator of every reported accuracy):
# A truncated path may still return an answer when the trace contains an EXPLICIT
# final-answer marker: a balanced \boxed{...}, a #### marker, or an explicit
# "final answer is X" / "the answer is X" phrase (using balanced braces for \boxed{...}).
# A truncated path must NOT fall back to heuristics reading an unfinished calculation:
# the last-number-in-text rule for gsm8k/math, the "= X at end of line" rule,
# the last yes/no occurrence for boolq, or the last-line rule for hotpotqa. Those return None.
# Untruncated paths keep every rule and fallback.

def _strip_template_artifacts(text: str) -> str:
    """Strip chat template turn markers that leak into output on truncation."""
    patterns = [
        r"\nuser\n",
        r"\nUser:",
        r"\n<\|im_start\|>",
        r"\nassistant\n",
        r"\n<\|eot_id\|>",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            text = text[:m.start()]
    return text.strip()


def _strip_think_tags(text: str) -> str:
    """Strip <think>...</think> reasoning blocks if present."""
    idx = text.rfind("</think>")
    if idx != -1:
        return text[idx + len("</think>"):].strip()
    idx = text.find("<think>")
    if idx != -1 and idx > 0:
        return text[:idx].strip()
    return text


def build_chat_prompt(tokenizer, question: str, model_config: ModelConfig) -> str:
    """Render a prompt using the model registry's tokenizer-template contract."""
    chat = [{"role": "user", "content": question}]
    if model_config.system_prompt is not None:
        chat.insert(0, {"role": "system", "content": model_config.system_prompt})
    if getattr(tokenizer, "chat_template", None) is None:
        return question
    return tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
        **dict(model_config.template_kwargs),
    )


def extract_final_segment(text: str, model_slug: str) -> str | None:
    """Return only the declared final-answer segment for a reasoning model."""
    config = MODEL_CONFIGS[model_slug]
    if not config.is_reasoning:
        return text
    if config.final_channel == "final":
        idx = text.rfind(HARMONY_FINAL_MARKER)
        if idx == -1:
            return None
        final = text[idx + len(HARMONY_FINAL_MARKER):]
        end_positions = [
            pos for marker in ("<|return|>", "<|end|>")
            if (pos := final.find(marker)) != -1
        ]
        return final[:min(end_positions) if end_positions else None].strip(" \t\r")
    if config.thinking_end is not None:
        idx = text.rfind(config.thinking_end)
        if idx == -1:
            return None
        return text[idx + len(config.thinking_end):].strip(" \t\r")
    raise ValueError(f"Reasoning model {model_slug!r} has no final-segment convention")


def _clean_num(val: str) -> str:
    s = val.replace(",", "").strip()
    if s.endswith("."):
        s = s[:-1]
    return s


def extract_boxed_answer(solution: str) -> str | None:
    r"""Extract the content of \boxed{...} requiring a balanced closing brace.

    Returns None if no \boxed{ is found or if braces are unbalanced.
    """
    idx = solution.rfind(r"\boxed{")
    if idx == -1:
        return None

    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(solution) and depth > 0:
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            depth -= 1
        i += 1

    if depth == 0:
        return solution[start : i - 1].strip()
    return None


def _terminated_label_candidate(text: str, start: int) -> str | None:
    """Return a labelled answer span only when its line or sentence is complete."""
    newline = text.find("\n", start)
    sentence = re.search(r"[.!?](?=\s|$)", text[start:])
    sentence_end = start + sentence.start() if sentence is not None else -1
    ends = [end for end in (newline, sentence_end) if end != -1]
    if not ends:
        return None
    return text[start:min(ends)].strip()


def _extract_complete_label_number(candidate: str) -> str | None:
    """Extract a complete numeric result from a terminated labelled span."""
    bold_spans = list(re.finditer(r"\*\*(.+?)\*\*", candidate))
    for bold in bold_spans:
        content = bold.group(1).strip()
        nums = re.findall(r"-?\d[\d,]*(?:\.\d*)?", content)
        if len(nums) == 1 and not re.search(r"[+\-*/^=([{]\s*$", content):
            return _clean_num(nums[0])

    stripped = candidate.strip()
    if re.search(r"[+\-*/^=([{]\s*$", stripped):
        return None
    if any(stripped.count(opening) > stripped.count(closing) for opening, closing in (
        ("(", ")"), ("[", "]"), ("{", "}"),
    )):
        return None
    nums = re.findall(r"-?\d[\d,]*(?:\.\d*)?", stripped)
    if len(nums) == 1:
        return _clean_num(nums[0])
    if len(nums) > 1 and re.search(r"=\s*\$?" + re.escape(nums[-1]) + r"\s*$", stripped):
        return _clean_num(nums[-1])
    return None


def _answer_label_candidate(text: str, start: int) -> tuple[str, int] | None:
    """Return the label-line remainder or the next line after at most two blanks."""
    line_end = text.find("\n", start)
    candidate_end = line_end if line_end != -1 else len(text)
    candidate = text[start:candidate_end]
    if candidate.strip():
        return candidate, start
    if line_end == -1:
        return None

    blank_lines = 0
    line_start = line_end + 1
    while line_start <= len(text):
        line_end = text.find("\n", line_start)
        candidate_end = line_end if line_end != -1 else len(text)
        candidate = text[line_start:candidate_end]
        if candidate.strip():
            return candidate, line_start
        blank_lines += 1
        if blank_lines > 2 or line_end == -1:
            return None
        line_start = line_end + 1
    return None


def _is_concluding_answer_label(text: str, label: re.Match[str]) -> bool:
    """Return whether a fix8 label introduces the final non-empty line."""
    nonempty_lines = list(re.finditer(r"(?m)^[^\r\n]*\S[^\r\n]*$", text))
    if not nonempty_lines:
        return False
    last_line = nonempty_lines[-1]
    label_text = re.search(r"\S", label.group(0))
    if label_text is None:
        return False
    label_start = label.start() + label_text.start()
    label_line_start = text.rfind("\n", 0, label_start) + 1
    if label_line_start == last_line.start():
        return True

    label_line_end = text.find("\n", label.end())
    if label_line_end == -1 or text[label.end():label_line_end].strip():
        return False
    continuation = _answer_label_candidate(text, label.end())
    if continuation is None:
        return False
    _, continuation_start = continuation
    if continuation_start == last_line.start():
        return True

    continuation_line_end = text.find("\n", continuation_start)
    tail_start = continuation_line_end + 1 if continuation_line_end != -1 else len(text)
    tail_lines = [line.strip() for line in text[tail_start:].splitlines() if line.strip()]
    postscript = " ".join(tail_lines)
    return postscript.startswith("(") and postscript.endswith(")")


def _extract_single_numeric_bold_span(line: str) -> str | None:
    """Extract one number only when exactly one bold span is numeric."""
    numeric_spans: list[str] = []
    for bold in re.finditer(r"\*\*(.+?)\*\*", line):
        nums = re.findall(r"-?\d[\d,]*(?:\.\d*)?", bold.group(1))
        if nums:
            if len(nums) != 1:
                return None
            numeric_spans.append(nums[0])
    if len(numeric_spans) != 1:
        return None
    return _clean_num(numeric_spans[0])


def extract_gsm8k_answer(
    text: str,
    is_truncated: bool = False,
    model_slug: str | None = None,
) -> str | None:
    """Extract numeric answer for GSM8K and MATH matching runner conventions."""
    normalized_text = normalize_latex_thousands_separators(text)
    preclean_text = _strip_think_tags(normalized_text)
    had_terminal_newline = preclean_text.endswith(("\n", "\r"))
    clean_text = _strip_template_artifacts(preclean_text)
    if len(clean_text) < 2:
        clean_text = _strip_template_artifacts(normalized_text)

    reasoning_rules = model_slug in REASONING_EXTRACTION_SLUGS

    # 1. #### X (take last standalone answer line). Under the v2.2 reasoning
    # contract this marker has precedence over every other rule.
    if reasoning_rules:
        matches = list(re.finditer(
            r"^[ \t]*####[ \t]+(-?[\d,]+\.?\d*)[ \t]*[.!]?[ \t]*$",
            clean_text,
            re.MULTILINE,
        ))
        if matches:
            return _clean_num(matches[-1].group(1))

    # 2. \boxed{...} (requires balanced closing brace, scans from last occurrence)
    boxed = extract_boxed_answer(clean_text)
    if boxed is None and clean_text != normalized_text:
        boxed = extract_boxed_answer(normalized_text)
    if boxed is not None:
        if reasoning_rules:
            numeric_boxed = normalize_latex_thousands_separators(boxed)
            numeric_boxed = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", numeric_boxed)
            numeric_boxed = (
                numeric_boxed.replace(r"\$", "")
                .replace("$", "")
                .replace(r"\,", "")
                .replace("%", "")
                .replace(",", "")
            )
            nums = re.findall(r"-?\d+(?:\.\d*)?", numeric_boxed)
        else:
            nums = re.findall(r"-?[\d,]+\.?\d*", boxed)
        if nums:
            return _clean_num(nums[-1])
        if boxed and not reasoning_rules:
            return _clean_num(boxed)

    # Preserve the broad legacy marker match outside the declared reasoning contract.
    if not reasoning_rules:
        matches = list(re.finditer(r"####\s*(-?[\d,]+\.?\d*)", clean_text))
        if matches:
            return _clean_num(matches[-1].group(1))

    # 3. An explicit answer declaration (take last match, word boundaries on bare words)
    matches = list(re.finditer(
        r"\b(?:final\s+answer|answer)\b\s*(?:is\b|=|:)\s*\$?(-?[\d,]+\.?\d*)",
        clean_text,
        re.IGNORECASE,
    ))
    if matches and (not is_truncated or not reasoning_rules):
        return _clean_num(matches[-1].group(1))

    # 4. A labelled answer line/sentence. The label makes this an explicit declaration,
    # including on truncated paths. Prefer the first numeric bold span, then the first
    # number after the label. Taking the last labelled candidate avoids earlier summaries.
    answer_labels: list[re.Match[str]] = []
    if model_slug in REASONING_EXTRACTION_SLUGS:
        legacy_labels = list(re.finditer(
            r"(?:^|(?<=[.!?])\s+)[ \t]*(?:\*\*)?[ \t]*"
            r"(?:final[ \t]+answer|answer)"
            r"\b[ \t]*(?::[ \t]*)?(?:\*\*)?"
            r"[ \t]*(?::[ \t]*)?",
            clean_text,
            re.IGNORECASE | re.MULTILINE,
        ))
        fix8_labels = [
            label for label in re.finditer(
                r"(?:^|(?<=[.!?])\s+)[ \t]*(?:\*\*)?[ \t]*"
                r"(?:final[ \t]+position|conclusion|result)"
                r"\b[ \t]*(?::[ \t]*)?(?:\*\*)?"
                r"[ \t]*(?::[ \t]*)?",
                clean_text,
                re.IGNORECASE | re.MULTILINE,
            )
            if _is_concluding_answer_label(clean_text, label)
        ]
        answer_labels = sorted((*legacy_labels, *fix8_labels), key=lambda label: label.start())
    if answer_labels:
        label = answer_labels[-1]
        labelled_candidate = _answer_label_candidate(clean_text, label.end())
        if labelled_candidate is None:
            return None
        candidate, candidate_start = labelled_candidate
        if is_truncated:
            termination_text = clean_text + ("\n" if had_terminal_newline else "")
            terminated = _terminated_label_candidate(termination_text, candidate_start)
            if terminated is not None:
                answer = _extract_complete_label_number(terminated)
                if answer is not None:
                    return answer
        else:
            for bold in re.finditer(r"\*\*(.+?)\*\*", candidate):
                nums = re.findall(r"-?\d[\d,]*(?:\.\d*)?", bold.group(1))
                if nums:
                    return _clean_num(nums[0])
            nums = re.findall(r"-?\d[\d,]*(?:\.\d*)?", candidate)
            if nums:
                return _clean_num(nums[0])

    # 5. A single numeric bold span on the last non-empty line of a reasoning
    # model's final segment. Reuse the fix5 completion guard when truncated.
    if reasoning_rules:
        nonempty_lines = list(re.finditer(r"(?m)^[^\r\n]*\S[^\r\n]*$", clean_text))
        if nonempty_lines:
            last_line = nonempty_lines[-1]
            candidate = last_line.group(0)
            if is_truncated:
                termination_text = clean_text + ("\n" if had_terminal_newline else "")
                terminated = _terminated_label_candidate(termination_text, last_line.start())
                if terminated is not None:
                    bold_answer = _extract_single_numeric_bold_span(terminated)
                    if bold_answer is not None:
                        return bold_answer
            else:
                bold_answer = _extract_single_numeric_bold_span(candidate)
                if bold_answer is not None:
                    return bold_answer

    # Heuristic fallbacks below read an unfinished calculation; refuse them when truncated.
    if is_truncated:
        return None

    # Result or total rule (take last match, word boundaries on bare words)
    matches = list(re.finditer(
        r"\b(?:result|total)\b\s*(?:is\b|=|:)\s*\$?(-?[\d,]+\.?\d*)",
        clean_text,
        re.IGNORECASE,
    ))
    if matches:
        return _clean_num(matches[-1].group(1))

    # 4. "= X" at end of line (take last match)
    matches = list(re.finditer(r"=\s*\$?(-?[\d,]+\.?\d*)\s*$", clean_text, re.MULTILINE))
    if matches:
        return _clean_num(matches[-1].group(1))

    # 5. Last number in text
    numbers = re.findall(r"-?[\d,]+\.?\d*", clean_text)
    if numbers:
        return _clean_num(numbers[-1])

    return None


def extract_math_answer(text: str, is_truncated: bool = False) -> str | None:
    """Extract the required last balanced boxed MATH answer."""
    clean_text = _strip_template_artifacts(_strip_think_tags(text))
    return extract_boxed_answer(clean_text)


def _strip_cruxeval_code_delimiters(literal: str) -> str | None:
    """Remove balanced Markdown code delimiters around a CruxEval literal."""
    candidate = literal.strip()
    while candidate:
        fenced = re.fullmatch(
            r"```(?:python|json)?[ \t]*\r?\n(.*?)(?:\r?\n)?```",
            candidate,
            flags=re.DOTALL,
        )
        if fenced is not None:
            candidate = fenced.group(1).strip()
            continue

        if set(candidate) == {"`"}:
            return None if len(candidate) % 2 == 0 else candidate

        leading = len(candidate) - len(candidate.lstrip("`"))
        trailing = len(candidate) - len(candidate.rstrip("`"))
        if leading == 0 or leading != trailing or 2 * leading > len(candidate):
            break
        candidate = candidate[leading:-trailing].strip()
    return candidate or None


def extract_literal_answer(text: str, is_truncated: bool = False) -> str | None:
    """Extract a Python literal only from the required #### marker."""
    clean_text = _strip_template_artifacts(_strip_think_tags(text)).strip()
    markers = list(re.finditer(r"####\s*", clean_text))
    if not markers:
        return None

    remainder = clean_text[markers[-1].end():]
    fenced = re.match(
        r"```(?:python|json)?[ \t]*\r?\n.*?(?:\r?\n)?```",
        remainder,
        flags=re.DOTALL,
    )
    literal = fenced.group(0) if fenced is not None else remainder.split("\n", 1)[0]
    return _strip_cruxeval_code_delimiters(literal)


def extract_boolq_answer(text: str, is_truncated: bool = False) -> str | None:
    """Extract Yes/No answer for BoolQ."""
    clean_text = _strip_template_artifacts(_strip_think_tags(text))
    boxed = extract_boxed_answer(clean_text)
    if boxed is None and clean_text != text:
        boxed = extract_boxed_answer(text)
    if boxed is not None:
        matches = re.findall(r"\b(yes|no)\b", boxed, re.IGNORECASE)
        if matches:
            return matches[-1].capitalize()
    matches = list(re.finditer(r"####\s*\b(yes|no)\b", clean_text, re.IGNORECASE))
    if matches:
        return matches[-1].group(1).capitalize()
    matches = list(re.finditer(
        r"\b(?:final\s+)?answer\b\s*(?:is\b|=|:)\s*\b(yes|no)\b",
        clean_text,
        re.IGNORECASE,
    ))
    if matches:
        return matches[-1].group(1).capitalize()
    # A leading yes/no is the model answering directly, so it counts as a declaration and
    # survives truncation. This matches the start-anchored letter rule in extract_mc_answer.
    m = re.match(r"^(yes|no)\b", clean_text.strip(), re.IGNORECASE)
    if m:
        return m.group(1).capitalize()
    if is_truncated:
        return None
    matches = re.findall(r"\b(yes|no)\b", clean_text, re.IGNORECASE)
    if matches:
        return matches[-1].capitalize()
    return None


def _strip_answer_lead_in(ans: str) -> str:
    """Strip a leading answer declaration from a candidate span."""
    return re.sub(
        r"^(?:the\s+final\s+answer\s+is|the\s+answer\s+is|it\s+is)\b\s*[:,-]?\s*",
        "",
        (ans or "").strip(),
        flags=re.IGNORECASE,
    ).strip()


def _is_answer_placeholder(candidate: str) -> bool:
    """Return whether a short-answer candidate is only a prompt placeholder."""
    stripped = candidate.strip()
    without_marker = re.sub(r"^####\s*", "", stripped).strip()
    return (
        without_marker.lower() == "<answer>"
        or re.fullmatch(r"(?:####\s*)*", without_marker) is not None
    )


def _clean_hotpotqa_candidate(
    ans: str,
    reject_placeholders: bool = False,
) -> str | None:
    """Clean a HotpotQA candidate span and reject if empty or >15 words."""
    if not ans:
        return None
    cand = ans.strip()
    cand = re.sub(
        r"^(?:the\s+final\s+answer\s+is|the\s+answer\s+is|it\s+is)\b\s*[:,-]?\s*",
        "",
        cand,
        flags=re.IGNORECASE,
    ).strip()
    cand = cand.rstrip(".").strip()
    if not cand or (reject_placeholders and _is_answer_placeholder(cand)):
        return None
    if len(cand.split()) > 15:
        return None
    return cand


def extract_hotpotqa_answer(
    text: str,
    is_truncated: bool = False,
    reject_placeholders: bool = False,
) -> str | None:
    """Extract answer for HotpotQA."""
    clean_text = _strip_template_artifacts(_strip_think_tags(text))
    boxed = extract_boxed_answer(clean_text)
    if boxed is None and clean_text != text:
        boxed = extract_boxed_answer(text)
    if boxed is not None:
        cand = _clean_hotpotqa_candidate(boxed, reject_placeholders=reject_placeholders)
        if cand:
            return cand
    matches = list(re.finditer(r"####\s*(.+?)(?:\n|$)", clean_text))
    for m in reversed(matches):
        cand = _clean_hotpotqa_candidate(
            m.group(1), reject_placeholders=reject_placeholders
        )
        if cand:
            return cand
    matches = list(re.finditer(
        r"\b(?:final\s+)?answer\b\s*(?:is\b|=|:)\s*(.+?)(?:\n|$)",
        clean_text,
        re.IGNORECASE,
    ))
    for m in reversed(matches):
        cand = _clean_hotpotqa_candidate(
            m.group(1), reject_placeholders=reject_placeholders
        )
        if cand:
            return cand
    if is_truncated:
        return None
    matches = list(re.finditer(
        r"\b(?:therefore|thus|so)\b\s*,?\s*(.+?)(?:\.|$)",
        clean_text,
        re.IGNORECASE,
    ))
    for m in reversed(matches):
        cand = _clean_hotpotqa_candidate(
            m.group(1), reject_placeholders=reject_placeholders
        )
        if cand:
            return cand
    # Last-line fallback. This one is NOT subject to the short-span cap: when the model
    # declared no answer, its last line IS the answer it gave, and a long line should score
    # a low F1 rather than be recorded as an abstention. Capping here would move the
    # denominator of every reported accuracy instead of the numerator.
    lines = [line.strip() for line in clean_text.split("\n") if line.strip()]
    if lines:
        cand = _strip_answer_lead_in(lines[-1]).rstrip(".").strip()
        if cand and (not reject_placeholders or not _is_answer_placeholder(cand)):
            return cand
    return None


def extract_mc_answer(text: str, is_truncated: bool = False) -> str | None:
    """Extract option letter (A/B/C/D) for multiple choice tasks (MMLU, HellaSwag)."""
    clean_text = _strip_template_artifacts(_strip_think_tags(text)).strip()
    boxed = extract_boxed_answer(clean_text)
    if boxed is None and clean_text != text:
        boxed = extract_boxed_answer(text)
    if boxed is not None:
        letters = re.findall(r"\b([A-D])\b", boxed, re.IGNORECASE)
        if letters:
            return letters[-1].upper()
    matches = list(re.finditer(r"####\s*\(?\b([A-D])\b\)?", clean_text, re.IGNORECASE))
    if matches:
        return matches[-1].group(1).upper()
    # Declared answer: take last match, word boundaries on bare words
    matches = list(re.finditer(
        r"\b(?:final\s+answer|answer|correct(?:\s+answer)?)\b\s*(?:is\b|=|:)\s*\(?\b([A-D])\b\)?",
        clean_text,
        re.IGNORECASE,
    ))
    if matches:
        return matches[-1].group(1).upper()
    # Direct match: "A", "B", "C", "D" at start
    m = re.match(r"^([A-D])\b", clean_text)
    if m:
        return m.group(1).upper()
    if is_truncated:
        return None
    # Last standalone letter
    letters = re.findall(r"\b([A-D])\b", clean_text)
    if letters:
        return letters[-1].upper()
    return None


def extract_answer(
    text: str,
    task: str = "gsm8k",
    is_truncated: bool = False,
    model_slug: str | None = None,
) -> str | None:
    """Route answer extraction based on task.

    TRUNCATION RULE, which defines the denominator of every reported accuracy.
    A path that hit the token cap may still have declared a final answer before the cap,
    so an EXPLICIT marker is honoured on a truncated path: a balanced \boxed{...}, a ####
    marker, or an explicit "the answer is X" phrase. The heuristic fallbacks that read an
    unfinished calculation are refused on a truncated path and return None: the
    last-number rule, the "= X at end of line" rule, the last yes/no occurrence, the last
    standalone option letter, and the last-line rule. An untruncated path keeps every rule.
    """
    if task == "gsm8k":
        answer = extract_gsm8k_answer(
            text,
            is_truncated=is_truncated,
            model_slug=model_slug,
        )
    elif task == "math":
        answer = extract_math_answer(text, is_truncated=is_truncated)
    elif task == "boolq":
        answer = extract_boolq_answer(text, is_truncated=is_truncated)
    elif task == "hotpotqa":
        answer = extract_hotpotqa_answer(text, is_truncated=is_truncated)
    elif task == "triviaqa":
        answer = extract_hotpotqa_answer(
            text,
            is_truncated=is_truncated,
            reject_placeholders=True,
        )
    elif task == "drop":
        answer = extract_hotpotqa_answer(text, is_truncated=is_truncated)
    elif task in ("mmlu", "hellaswag"):
        answer = extract_mc_answer(text, is_truncated=is_truncated)
    elif task == "cruxeval":
        answer = extract_literal_answer(text, is_truncated=is_truncated)
    elif task == "mbpp":
        answer = None if is_truncated else text.strip() or None
    else:
        raise ValueError(f"Unknown task for extraction: {task}")
    return answer if answer is None or answer.strip() else None


# ── Dataset loader ────────────────────────────────────────────────────────

def load_task_instances(task: str) -> list[dict]:
    """Load task instances with uniform {prompt, answer, metadata} schema."""
    if task in ("gsm8k", "math", "hotpotqa", "triviaqa", "boolq", "drop", "cruxeval", "mbpp"):
        return load_task(task)
    elif task == "mmlu":
        from datasets import load_dataset
        ds = load_dataset("cais/mmlu", "all", split="test")
        instances = []
        labels = ["A", "B", "C", "D"]
        for item in ds:
            choices_str = "\n".join(f"{label}) {text}" for label, text in zip(labels, item["choices"]))
            prompt = (
                f"Answer the following multiple choice question.\n\n"
                f"{item['question']}\n\n"
                f"{choices_str}\n\n"
                f"Answer:"
            )
            gold = chr(ord("A") + int(item["answer"]))
            instances.append({
                "prompt": prompt,
                "answer": gold,
                "metadata": {"subject": item.get("subject", "")},
            })
        return instances
    elif task == "hellaswag":
        from datasets import load_dataset
        ds = load_dataset("Rowan/hellaswag", split="validation")
        instances = []
        labels = ["A", "B", "C", "D"]
        label_to_letter = {"0": "A", "1": "B", "2": "C", "3": "D"}
        for item in ds:
            choices_str = "\n".join(f"{label}) {text}" for label, text in zip(labels, item["endings"]))
            prompt = (
                f"Choose the most likely continuation of the following scenario.\n\n"
                f"{item['ctx']}\n\n"
                f"{choices_str}\n\n"
                f"Answer:"
            )
            raw_label = str(item["label"])
            gold = label_to_letter.get(raw_label, raw_label)
            instances.append({
                "prompt": prompt,
                "answer": gold,
                "metadata": {"source_id": item.get("source_id", "")},
            })
        return instances
    else:
        raise ValueError(f"Unknown task: {task}. Available: {TASK_CHOICES}")


# ── Summary reporting ─────────────────────────────────────────────────────

def print_summary(cache_file: Path, seeds: list[int], task: str, model_slug: str):
    """Print per-seed summary: instances, mean tokens/path, cap-hit rate, accuracy."""
    print("\n" + "=" * 80)
    print(f"SUMMARY: model={model_slug}, task={task}, cache={cache_file}")
    print("=" * 80)

    records_by_seed: dict[int, list[dict]] = {s: [] for s in seeds}
    if cache_file.exists():
        with open(cache_file) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                    s = e.get("seed")
                    if s in records_by_seed:
                        records_by_seed[s].append(e)
                except (json.JSONDecodeError, TypeError) as exc:
                    raise ValueError(f"Malformed summary record in {cache_file}: {exc}") from exc

    if task in ("gsm8k", "math"):
        hdr = f"{'Seed':<8}{'Instances':<12}{'Mean Tok/Path':<16}{'Cap-Hit Rate':<14}{'Numeric MV@32':<16}{'Path Acc':<12}"
    else:
        hdr = f"{'Seed':<8}{'Instances':<12}{'Mean Tok/Path':<16}{'Cap-Hit Rate':<14}{'MV@32 Acc':<16}{'Path Acc':<12}"
    print(hdr)
    print("-" * len(hdr))

    overall_inst = 0
    overall_toks = 0
    overall_paths = 0
    overall_trunc = 0
    overall_mv_correct = 0
    overall_path_correct = 0

    for s in seeds:
        recs = sorted(records_by_seed[s], key=lambda x: x["instance_id"])
        n_done = len(recs)
        if n_done == 0:
            print(f"{s:<8}{0:<12}{'N/A':<16}{'N/A':<14}{'N/A':<16}{'N/A':<12}")
            continue

        all_lens = [l for r in recs for l in r.get("gen_lens", [])]
        all_trunc = [t for r in recs for t in r.get("truncated", [])]
        mean_tok = sum(all_lens) / len(all_lens) if all_lens else 0.0
        cap_hit = sum(1 for t in all_trunc if t) / len(all_trunc) if all_trunc else 0.0

        mv_correct = 0
        path_correct = 0
        n_paths = 0
        for r in recs:
            gold = r.get("gold_answer", "")
            answers = [a for a in r.get("all_answers", []) if a is not None]
            if answers:
                mv_ans = Counter(answers).most_common(1)[0][0]
                if check_task_correct(mv_ans, gold, task):
                    mv_correct += 1
            for a in r.get("all_answers", []):
                n_paths += 1
                if check_task_correct(a, gold, task):
                    path_correct += 1

        mv_acc = mv_correct / n_done if n_done else 0.0
        p_acc = path_correct / n_paths if n_paths else 0.0

        overall_inst += n_done
        overall_toks += sum(all_lens)
        overall_paths += len(all_lens)
        overall_trunc += sum(1 for t in all_trunc if t)
        overall_mv_correct += mv_correct
        overall_path_correct += path_correct

        s_tok = f"{mean_tok:.1f}"
        s_cap = f"{cap_hit * 100:.1f}%"
        s_mv = f"{mv_acc * 100:.1f}%"
        s_p = f"{p_acc * 100:.1f}%"
        print(f"{s:<8}{n_done:<12}{s_tok:<16}{s_cap:<14}{s_mv:<16}{s_p:<12}")

    print("-" * len(hdr))
    if overall_inst > 0:
        tot_mean_tok = overall_toks / overall_paths if overall_paths else 0.0
        tot_cap_hit = overall_trunc / overall_paths if overall_paths else 0.0
        tot_mv_acc = overall_mv_correct / overall_inst if overall_inst else 0.0
        tot_p_acc = overall_path_correct / overall_paths if overall_paths else 0.0
        s_tot_tok = f"{tot_mean_tok:.1f}"
        s_tot_cap = f"{tot_cap_hit * 100:.1f}%"
        s_tot_mv = f"{tot_mv_acc * 100:.1f}%"
        s_tot_p = f"{tot_p_acc * 100:.1f}%"
        print(f"{'Total':<8}{overall_inst:<12}{s_tot_tok:<16}{s_tot_cap:<14}{s_tot_mv:<16}{s_tot_p:<12}")
    print("=" * 80 + "\n", flush=True)


# ── Main execution ────────────────────────────────────────────────────────

def _temperature_arg(value: str) -> float:
    temperature = float(value)
    if not 0.0 < temperature <= 2.0:
        raise argparse.ArgumentTypeError(
            f"temperature must be in (0.0, 2.0], got {value}"
        )
    return temperature


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SC K=32 generation (v2 protocol)")
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()),
                        help=f"Model to run ({', '.join(MODEL_CONFIGS.keys())})")
    parser.add_argument("--task", default="gsm8k", choices=TASK_CHOICES,
                        help=f"Task to evaluate ({', '.join(TASK_CHOICES)}; default: gsm8k)")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help=f"Random seeds (default: {DEFAULT_SEEDS})")
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help=f"Max instances to evaluate (default: {DEFAULT_N})")
    parser.add_argument("--k", type=int, choices=[4, K], default=K,
                        help=f"Paths per instance (default: {K}; 4 is for smoke testing only)")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Max new tokens per path (defaults to per-task / per-model budget)")
    parser.add_argument("--batch-paths", type=int, default=None,
                        help="Batch size for paths per generate() call (default: 8, 4 for reasoning models)")
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "default"], default="default",
                        help="Attention implementation. Default: 'default' for every model. "
                             "Recorded in every record and checked by the resume guard.")
    parser.add_argument("--temperature", type=_temperature_arg, default=TEMPERATURE,
                        help=f"Sampling temperature in (0.0, 2.0] (default: {TEMPERATURE})")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (overrides DC_RECORDS_DIR and default cache dir)")
    return parser.parse_args(argv)


def load_existing_keys(
    cache_file: Path,
    *,
    model_id: str,
    max_new_tokens: int,
    temperature: float,
    batch_paths: int,
    scorer_name: str,
    expected_prompt_shas: dict[int, str],
    attn_implementation: str | None = None,
    expected_k: int = K,
) -> set[tuple]:
    """Read resumable records, refusing any provenance mismatch."""
    existing_keys = set()
    if not cache_file.exists():
        return existing_keys

    with open(cache_file, "rb") as f:
        raw_lines = f.readlines()

    non_empty = []
    curr_offset = 0
    for lineno, line in enumerate(raw_lines, 1):
        next_offset = curr_offset + len(line)
        if line.strip():
            non_empty.append((lineno, line, curr_offset, next_offset))
        curr_offset = next_offset

    last_complete_offset = 0
    dropped_final_line = False

    for idx, (lineno, line, _start_offset, end_offset) in enumerate(non_empty):
        is_final = (idx == len(non_empty) - 1)
        try:
            if not line.endswith(b"\n"):
                raise ValueError("Incomplete line without terminating newline")
            e = json.loads(line.decode("utf-8"))
        except Exception:
            if is_final:
                print(f"WARN: Dropping unparseable final line at {cache_file}:{lineno}")
                dropped_final_line = True
                continue
            sys.exit(
                f"FATAL: {cache_file}:{lineno} is not a valid JSON record. "
                f"Repair or archive the file before rerunning."
            )

        prov = e.get("provenance") if isinstance(e.get("provenance"), dict) else {}
        rec_model_id = prov.get("model_id", e.get("model_id"))
        rec_max_tokens = prov.get("max_new_tokens", e.get("max_new_tokens"))
        rec_max_tokens_alias = prov.get("max_tokens", e.get("max_tokens"))
        rec_version = prov.get("script_version", e.get("script_version"))
        rec_temp = prov.get("temperature")
        rec_top_p = prov.get("top_p")
        rec_batch_paths = prov.get("batch_paths")
        rec_scorer = prov.get("scorer")
        rec_prompt_sha = prov.get("prompt_sha")
        rec_attn = prov.get("attn_implementation", e.get("attn_implementation"))
        rec_inst_id = e.get("instance_id")
        rec_k = e.get("K")

        mismatches = []
        if rec_model_id != model_id:
            mismatches.append(f"model_id (cached={rec_model_id!r}, current={model_id!r})")
        if rec_max_tokens != max_new_tokens:
            mismatches.append(f"max_new_tokens (cached={rec_max_tokens!r}, current={max_new_tokens!r})")
        if rec_version != SCRIPT_VERSION:
            mismatches.append(f"script_version (cached={rec_version!r}, current={SCRIPT_VERSION!r})")
        if rec_temp != temperature:
            mismatches.append(f"temperature (cached={rec_temp!r}, current={temperature!r})")
        if rec_top_p != TOP_P:
            mismatches.append(f"top_p (cached={rec_top_p!r}, current={TOP_P!r})")
        if rec_batch_paths != batch_paths:
            mismatches.append(f"batch_paths (cached={rec_batch_paths!r}, current={batch_paths!r})")
        if rec_scorer != scorer_name:
            mismatches.append(f"scorer (cached={rec_scorer!r}, current={scorer_name!r})")
        if rec_k != expected_k:
            mismatches.append(f"K (cached={rec_k!r}, current={expected_k!r})")
        if attn_implementation is not None and rec_attn != attn_implementation:
            mismatches.append(
                f"attn_implementation (cached={rec_attn!r}, current={attn_implementation!r})"
            )
        if attn_implementation is not None and rec_max_tokens_alias != max_new_tokens:
            mismatches.append(
                f"max_tokens (cached={rec_max_tokens_alias!r}, current={max_new_tokens!r})"
            )
        if rec_inst_id is not None and rec_inst_id in expected_prompt_shas:
            expected_sha = expected_prompt_shas[rec_inst_id]
            if rec_prompt_sha != expected_sha:
                mismatches.append(
                    f"prompt_sha for instance_id={rec_inst_id} "
                    f"(cached={rec_prompt_sha!r}, current={expected_sha!r})"
                )

        if mismatches:
            sys.exit(
                f"FATAL: {cache_file}:{lineno} provenance mismatch with current run:\n"
                + "\n".join(f"  - {m}" for m in mismatches) + "\n"
                f"Startup refused to append to file with disagreeing provenance block. "
                f"Archive or remove {cache_file} before rerunning."
            )

        last_complete_offset = end_offset
        key = (e.get("seed"), e.get("instance_id"))
        if e.get("K") == expected_k and len(e.get("all_answers", [])) == expected_k:
            existing_keys.add(key)

    if dropped_final_line:
        os.truncate(cache_file, last_complete_offset)
        print(f"Dropped unparseable final line; truncated {cache_file} to {last_complete_offset} bytes")

    return existing_keys


def main():
    args = parse_args()

    model_config = MODEL_CONFIGS[args.model]
    is_reasoning = model_config.is_reasoning

    if args.max_new_tokens is not None:
        max_new_tokens = args.max_new_tokens
    elif is_reasoning:
        max_new_tokens = model_config.reasoning_max_new_tokens
        assert max_new_tokens is not None
    else:
        max_new_tokens = TASK_DEFAULT_MAX_NEW_TOKENS.get(args.task, 1024)

    if args.batch_paths is not None:
        batch_paths = args.batch_paths
    elif is_reasoning:
        batch_paths = REASONING_BATCH_PATHS
    else:
        batch_paths = DEFAULT_BATCH_PATHS

    if batch_paths < 1:
        sys.exit(f"FATAL: --batch-paths must be at least 1, got {batch_paths}")

    scorer = get_scorer(args.task)
    scorer_name = scorer.name
    seeds = list(dict.fromkeys(args.seeds))
    model_id = model_config.model_id

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    elif os.environ.get("DC_RECORDS_DIR"):
        out_dir = Path(os.environ["DC_RECORDS_DIR"])
    else:
        out_dir = CACHE_DIR
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_file = out_dir / f"{args.model}_sc_kvar_v2_{args.task}.jsonl"

    print(f"Model: {args.model} ({model_id})")
    print(f"Task: {args.task}, Scorer: {scorer_name}, Seeds: {seeds}, n={args.n}, "
          f"max_new_tokens={max_new_tokens}, batch_paths={batch_paths}, "
          f"temperature={args.temperature}")
    if is_reasoning:
        print(
            "Reasoning convention: "
            f"chat_template={model_config.chat_template_flag}; "
            f"thinking_tags=({model_config.thinking_start!r}, {model_config.thinking_end!r}); "
            f"thinking_channel={model_config.thinking_channel!r}; "
            f"final_channel={model_config.final_channel!r}"
        )
    print(f"Output directory: {out_dir}")
    print(f"Cache target: {cache_file}")

    # Load task data
    loaded_task_data = load_task_instances(args.task)
    task_data = require_nonempty_slice(loaded_task_data, args.n, args.task)
    n = len(task_data)
    print(f"Loaded {len(loaded_task_data)} instances for {args.task}, running {n}")

    # Normalize gold answers. New tasks store their canonical correct vote bucket.
    golds = []
    for item in task_data[:n]:
        if args.task in ("math", "triviaqa", "drop", "cruxeval"):
            g = canonicalize_for_instance(item["answer"], item, args.task)
            if g is None:
                sys.exit(f"FATAL: {args.task} instance has an empty canonical gold answer")
        else:
            g = str(item["answer"]).replace(",", "").strip()
        golds.append(g)

    # Load tokenizer for prompt formatting and resume fingerprint validation
    print(f"Loading tokenizer for {model_id}...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    expected_prompt_shas = {}
    for idx_i in range(n):
        q = build_prompt(task_data[idx_i]["prompt"], args.task, args.model)
        fmt = build_chat_prompt(tokenizer, q, model_config)
        expected_prompt_shas[idx_i] = hashlib.sha256(fmt.encode("utf-8")).hexdigest()[:16]

    # Resume logic: read completed (seed, instance_id) pairs with provenance validation
    requested_attn = args.attn_implementation
    existing_keys = load_existing_keys(
        cache_file,
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        temperature=args.temperature,
        batch_paths=batch_paths,
        scorer_name=scorer_name,
        expected_prompt_shas=expected_prompt_shas,
        attn_implementation=requested_attn,
        expected_k=args.k,
    )

    print(f"Existing: {len(existing_keys)} valid entries in {cache_file}")

    # Check if all instances for all requested seeds are already completed
    all_needed = {(s, i) for s in seeds for i in range(n)}
    if all_needed.issubset(existing_keys):
        print("All requested (seed, instance_id) pairs are already present in cache.")
        print_summary(cache_file, seeds, args.task, args.model)
        return

    # Load model
    print(f"Loading {model_id}...")
    import torch
    from transformers import AutoModelForCausalLM

    # Stop and pad token setup
    stop_ids = set()
    if isinstance(tokenizer.eos_token_id, list):
        stop_ids.update(tokenizer.eos_token_id)
    elif tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)

    for tok_name in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>"]:
        tid = tokenizer.convert_tokens_to_ids(tok_name)
        if tid is not None and tid != tokenizer.unk_token_id:
            stop_ids.add(tid)

    stop_ids = sorted(list(stop_ids))

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif stop_ids:
            tokenizer.pad_token = tokenizer.convert_ids_to_tokens(stop_ids[0])
    pad_id = tokenizer.pad_token_id

    terminal_ids = set(stop_ids)
    if pad_id is not None:
        terminal_ids.add(pad_id)

    special_strs = [t for t in tokenizer.all_special_tokens if t and t != THINK_END_TAG]

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if requested_attn != "default":
        model_kwargs["attn_implementation"] = requested_attn

    model_obj = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model_obj.eval()
    print("Model loaded.")

    # Determine chunk sizes for batched sampling up to K=32
    chunk_sizes = []
    rem = args.k
    while rem > 0:
        c = min(batch_paths, rem)
        chunk_sizes.append(c)
        rem -= c

    total_new = 0
    for seed in seeds:
        for i in range(n):
            key = (seed, i)
            if key in existing_keys:
                continue

            item = task_data[i]
            question = build_prompt(item["prompt"], args.task, args.model)
            gold = golds[i]

            # Direct prompt usage without double wrapping
            formatted = build_chat_prompt(tokenizer, question, model_config)

            prompt_sha = hashlib.sha256(formatted.encode("utf-8")).hexdigest()[:16]

            inputs = tokenizer(formatted, return_tensors="pt").to(model_obj.device)
            input_len = inputs.input_ids.shape[1]

            # Single torch seed per (seed, instance)
            torch.manual_seed(seed * 100003 + i)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed * 100003 + i)

            all_answers = []
            all_traces = []
            gen_lens = []
            truncated = []
            t0 = time.time()

            for chunk_size in chunk_sizes:
                gen_kwargs = dict(
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=TOP_P,
                    num_return_sequences=chunk_size,
                    pad_token_id=pad_id,
                )
                if stop_ids:
                    gen_kwargs["eos_token_id"] = stop_ids

                with torch.no_grad():
                    outputs = model_obj.generate(**inputs, **gen_kwargs)

                for j in range(outputs.shape[0]):
                    gen_ids = outputs[j, input_len:].tolist()
                    gen_len = len(gen_ids)
                    while gen_len > 0 and gen_ids[gen_len - 1] in terminal_ids:
                        gen_len -= 1
                    is_truncated = (gen_len == len(gen_ids) and gen_len >= max_new_tokens)

                    if is_reasoning:
                        trace = tokenizer.decode(gen_ids[:gen_len], skip_special_tokens=False)
                        answer_text = extract_final_segment(trace, args.model)
                        if answer_text is not None:
                            for sp in special_strs:
                                answer_text = answer_text.replace(sp, " ")
                            ans = extract_answer(
                                answer_text,
                                task=args.task,
                                is_truncated=is_truncated,
                                model_slug=args.model,
                            )
                        else:
                            ans = None
                    else:
                        trace = tokenizer.decode(gen_ids[:gen_len], skip_special_tokens=True)
                        ans = extract_answer(
                            trace,
                            task=args.task,
                            is_truncated=is_truncated,
                            model_slug=args.model,
                        )

                    if args.task == "mbpp" and ans is not None:
                        ans = evaluate_mbpp_completion(
                            ans,
                            list(item["assertions"]),
                            list(item.get("test_imports", [])),
                        )
                    elif args.task in ("math", "triviaqa", "drop", "cruxeval"):
                        ans = canonicalize_for_instance(ans, item, args.task)

                    all_answers.append(ans)
                    all_traces.append(trace)
                    gen_lens.append(gen_len)
                    truncated.append(is_truncated)

            elapsed = time.time() - t0
            provenance = {
                "script_version": SCRIPT_VERSION,
                "extractor_version": EXTRACTOR_VERSION,
                "model_id": model_id,
                "max_new_tokens": max_new_tokens,
                "max_tokens": max_new_tokens,
                "temperature": args.temperature,
                "top_p": TOP_P,
                "batch_paths": batch_paths,
                "prompt_sha": prompt_sha,
                "scorer": scorer_name,
                "attn_implementation": requested_attn,
                "reasoning": is_reasoning,
                "chat_template_flag": model_config.chat_template_flag,
                "thinking_start": model_config.thinking_start,
                "thinking_end": model_config.thinking_end,
                "thinking_channel": model_config.thinking_channel,
                "final_channel": model_config.final_channel,
            }
            entry = {
                "model_slug": args.model,
                "method": "sc",
                "task": args.task,
                "K": args.k,
                "seed": seed,
                "instance_id": i,
                "gold_answer": gold,
                "all_answers": all_answers,
                "all_traces": all_traces,
                "gen_lens": gen_lens,
                "truncated": truncated,
                "timestamp": time.time(),
                "elapsed": elapsed,
                "max_new_tokens": max_new_tokens,
                "max_tokens": max_new_tokens,
                "temperature": args.temperature,
                "attn_implementation": requested_attn,
                "provenance": provenance,
            }

            with open(cache_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
                f.flush()
            existing_keys.add(key)

            total_new += 1
            n_correct = sum(1 for a in all_answers if check_task_correct(a, gold, args.task))
            n_trunc = sum(1 for t in truncated if t)
            n_answered = sum(1 for a in all_answers if a is not None)
            print(f"seed={seed} i={i}: {n_correct}/{args.k} correct, gold={gold}, "
                  f"answered={n_answered}/{args.k}, truncated={n_trunc}/{args.k}, "
                  f"elapsed={elapsed:.1f}s [{total_new}]", flush=True)

    print(f"\nDone. {total_new} new entries written to {cache_file}")
    print_summary(cache_file, seeds, args.task, args.model)


if __name__ == "__main__":
    main()
