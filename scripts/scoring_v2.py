#!/usr/bin/env python3
"""Centralised scoring logic for evaluation tasks (v2 protocol).

Provides one scorer per task with a canonical `name` attribute recorded
in result provenance.
"""
from __future__ import annotations

import re
import string

from diversity_combining.evaluation.runner import compute_f1
from scripts.tasks12 import (
    canonicalize_math,
    canonicalize_numeric_or_text,
    canonicalize_python_literal,
    normalize_latex_thousands_separators,
    normalize_squad,
)

try:
    from math_verify import parse, verify
    HAS_MATH_VERIFY = True
except ImportError:
    HAS_MATH_VERIFY = False


def _clean_numeric(s: str) -> str:
    s = normalize_latex_thousands_separators(s).replace(",", "").strip()
    if s.endswith("."):
        s = s[:-1]
    return s.strip()


def check_numeric_equivalence(pred: str | None, gold: str, tolerance: float = 1e-3) -> bool:
    """Check numeric equivalence with tolerance after stripping commas and trailing periods."""
    if pred is None:
        return False
    gold_clean = _clean_numeric(gold)
    pred_clean = _clean_numeric(pred)
    try:
        return abs(float(pred_clean) - float(gold_clean)) < tolerance
    except (ValueError, TypeError):
        return pred_clean == gold_clean


class GSM8KScorer:
    name: str = "gsm8k_numeric"

    def __call__(self, pred: str | None, gold: str) -> bool:
        return check_numeric_equivalence(pred, gold)


class MathScorer:
    name: str = "math500_canonical_exact"

    def __call__(self, pred: str | None, gold: str) -> bool:
        return pred is not None and canonicalize_math(pred) == canonicalize_math(gold)


def normalize_boolq(s: str | None) -> str:
    """Normalize BoolQ string to yes/no case-insensitively with trailing punctuation removed."""
    if s is None:
        return ""
    return str(s).strip().rstrip(string.punctuation).strip().lower()


class BoolQScorer:
    name: str = "boolq_yesno"

    def __call__(self, pred: str | None, gold: str) -> bool:
        if pred is None:
            return False
        return normalize_boolq(pred) == normalize_boolq(gold)


def normalize_hotpotqa(s: str) -> str:
    """HotpotQA-style normalization: lowercase, strip articles/punct/extra whitespace."""
    s = str(s).lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


class HotpotQAScorer:
    name: str = "hotpotqa_token_f1"

    def __call__(self, pred: str | None, gold: str) -> bool:
        if pred is None:
            return False
        gold_norm = normalize_hotpotqa(gold)
        pred_norm = normalize_hotpotqa(pred)
        if gold_norm and pred_norm == gold_norm:
            return True
        return compute_f1(str(pred), str(gold)) >= 0.5


class MCScorer:
    name: str = "mc_exact"

    def __call__(self, pred: str | None, gold: str) -> bool:
        if pred is None:
            return False
        return str(pred).strip().upper() == str(gold).strip().upper()


class TriviaQAScorer:
    name: str = "triviaqa_alias_canonical_exact"

    def __call__(self, pred: str | None, gold: str) -> bool:
        return pred is not None and normalize_squad(pred) == normalize_squad(gold)


class DropScorer:
    name: str = "drop_alias_numeric_or_squad_exact"

    def __call__(self, pred: str | None, gold: str) -> bool:
        return (
            pred is not None
            and canonicalize_numeric_or_text(pred) == canonicalize_numeric_or_text(gold)
        )


class CruxEvalScorer:
    name: str = "cruxeval_literal_exact"

    def __call__(self, pred: str | None, gold: str) -> bool:
        return (
            pred is not None
            and canonicalize_python_literal(pred) == canonicalize_python_literal(gold)
        )


class MBPPScorer:
    name: str = "mbpp_per_assert_sandbox"

    def __call__(self, pred: str | None, gold: str) -> bool:
        return pred is not None and bool(gold) and str(pred) == str(gold) and set(gold) == {"P"}


SCORERS = {
    "gsm8k": GSM8KScorer,
    "math": MathScorer,
    "boolq": BoolQScorer,
    "hotpotqa": HotpotQAScorer,
    "mmlu": MCScorer,
    "hellaswag": MCScorer,
    "triviaqa": TriviaQAScorer,
    "drop": DropScorer,
    "cruxeval": CruxEvalScorer,
    "mbpp": MBPPScorer,
}


def canonicalize_answer(answer: str | None, task: str) -> str | None:
    """Canonicalize an answer string according to task-specific scoring rules."""
    if answer is None:
        return None
    if task == "gsm8k":
        clean = _clean_numeric(answer)
        if not clean:
            return None
        try:
            val = round(float(clean), 3)
            if val == 0:
                val = 0.0
            return str(int(val)) if val.is_integer() else str(val)
        except (ValueError, TypeError):
            return clean
    elif task == "math":
        return canonicalize_math(answer)
    elif task == "boolq":
        norm = normalize_boolq(answer)
        return norm if norm in ("yes", "no") else (norm if norm else None)
    elif task in ("mmlu", "hellaswag"):
        clean = str(answer).strip().upper()
        return clean if clean else None
    elif task == "hotpotqa":
        norm = normalize_hotpotqa(answer)
        return norm if norm else None
    elif task == "triviaqa":
        norm = normalize_squad(answer)
        return norm or None
    elif task == "drop":
        return canonicalize_numeric_or_text(answer)
    elif task == "cruxeval":
        return canonicalize_python_literal(answer)
    elif task == "mbpp":
        clean = str(answer).strip()
        return clean if clean else None
    raise ValueError(f"Unknown task for canonicalization: {task}")


def get_scorer(task: str):
    """Return an instantiated scorer for task, validating availability at startup."""
    if task not in SCORERS:
        raise ValueError(f"Unknown task for scoring: {task}. Available: {list(SCORERS.keys())}")
    return SCORERS[task]()


def check_task_correct(pred: str | None, gold: str, task: str) -> bool:
    """Check correctness according to repository task scorer conventions."""
    return get_scorer(task)(pred, gold)


def get_task_scorer(task: str) -> str:
    """Return canonical scorer name for task provenance."""
    return get_scorer(task).name
