#!/usr/bin/env python3
"""Benchmark metadata, the JSON helper, the canonical scorers and the block-covariance
matrix builder shared by scripts/reproduce_cross_benchmark.py and
scripts/cross_benchmark_outputs.py.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.aggregate_5seed import BASE_ACC_THRESHOLD
from scripts.scoring_v2 import GSM8KScorer, MathScorer

# Task answer space labels as printed in Table 7
TASK_ANSWER_SPACE = {
    "triviaqa": "open text",
    "hotpotqa": "open text",
    "arc_challenge": "bounded (4)",
    "mmlu": "bounded (4)",
    "hellaswag": "bounded (4)",
    "winogrande": "binary (2)",
    "drop": "open text",
    "boolq": "binary (y/n)",
    "mbpp": "code pass/fail",
    "cruxeval": "code pass/fail",
    "gsm8k": "numeric",
    "math": "numeric",
}

TASK_DISPLAY_NAMES = {
    "triviaqa": "TriviaQA",
    "hotpotqa": "HotpotQA",
    "arc_challenge": "ARC-C",
    "mmlu": "MMLU",
    "hellaswag": "HellaSwag",
    "winogrande": "WinoGrande",
    "drop": "DROP$^\\dagger$",
    "boolq": "BoolQ",
    "mbpp": "MBPP",
    "cruxeval": "CruxEval",
    "gsm8k": "GSM8K",
    "math": "MATH",
}

BENCHMARK_ORDER_TABLE3 = [
    ("QA", ["triviaqa", "hotpotqa"]),
    ("Science/MC", ["arc_challenge", "mmlu"]),
    ("Commonsense", ["hellaswag", "winogrande"]),
    ("NLU", ["drop", "boolq"]),
    ("Code", ["mbpp", "cruxeval"]),
    ("Math", ["gsm8k", "math"]),
]

def to_json_safe(obj: Any) -> Any:
    """Recursively convert NaN/Inf to None and numpy scalars to Python scalars."""
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    return obj


def get_scorers() -> Tuple[GSM8KScorer, MathScorer]:
    """Instantiate and validate the canonical scoring_v2 scorers."""
    gsm = GSM8KScorer()
    math_sc = MathScorer()
    assert isinstance(gsm, GSM8KScorer), "gsm must be instance of GSM8KScorer"
    assert isinstance(math_sc, MathScorer), "math_sc must be instance of MathScorer"
    assert gsm.name == "gsm8k_numeric", f"Unexpected GSM8K scorer name: {gsm.name}"
    assert math_sc.name == "math500_canonical_exact", f"Unexpected MATH scorer name: {math_sc.name}"
    return gsm, math_sc


def build_block_covariance_matrix(
    records: List[Dict[str, Any]],
    task: str,
    rescore_math_gsm: bool,
    gsm_scorer: GSM8KScorer,
    math_scorer: MathScorer,
) -> Optional[np.ndarray]:
    """Build a PT correctness matrix; GSM8K and MATH use scoring_v2 when rescore_math_gsm is True."""
    rows = []
    for r in records:
        if "all_correct" in r and r["all_correct"]:
            rows.append(r["all_correct"])
        elif "all_answers" in r and "gold_answer" in r:
            if rescore_math_gsm and task == "gsm8k":
                rows.append([int(bool(gsm_scorer(a, r["gold_answer"]))) for a in r["all_answers"]])
            elif rescore_math_gsm and task == "math":
                rows.append([int(bool(math_scorer(a, r["gold_answer"]))) for a in r["all_answers"]])
            else:
                g = str(r["gold_answer"]).strip().lower()
                rows.append([1 if str(a).strip().lower() == g else 0 for a in r["all_answers"]])
    if not rows:
        return None
    mat = np.array(rows, dtype=float)
    return mat if mat.shape[0] >= 20 and mat.mean() >= BASE_ACC_THRESHOLD else None
