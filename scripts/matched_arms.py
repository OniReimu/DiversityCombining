#!/usr/bin/env python3
"""Load the prompt-template (PT) and self-consistency (SC) arms of one model-benchmark cell.

Each loader returns the two arms on the same (seed, instance) pairs, with the first K=8
answers of every SC record, and raises CellMismatchError on any difference in seeds,
instances or gold answers. GSM8K PT answers are re-extracted from the stored path text
with the shipped extractor (scripts/run_sc_kvar_v2.extract_answer); every other arm uses
the stored all_answers. Records are read from experiment_dir("pt_records") and
experiment_dir("sc_records").
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_SRC = REPO_ROOT
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from diversity_combining.config import experiment_dir  # noqa: E402

from scripts.run_sc_kvar_v2 import extract_answer  # noqa: E402
from scripts.scoring_v2 import canonicalize_answer  # noqa: E402

MODELS = ("qwen05b", "qwen7b", "llama8b", "mistral7b", "qwen32b")
T12_MODELS = ("qwen05b", "qwen7b", "qwen32b", "llama8b", "mistral7b")
MODEL_NAMES = {
    "qwen05b": "Qwen2.5-0.5B", "qwen7b": "Qwen2.5-7B", "qwen32b": "Qwen2.5-32B",
    "llama8b": "Llama-3.1-8B", "mistral7b": "Mistral-7B",
}
TASK = "gsm8k"
T12_TASKS = ("math", "triviaqa", "drop", "cruxeval", "mbpp")
PT4_TASKS = ("boolq", "hotpotqa", "mmlu", "hellaswag")
ARCWG_TASKS = ("arc_challenge", "winogrande")
TASK_NAMES = {
    "math": "MATH", "triviaqa": "TriviaQA", "drop": "DROP", "cruxeval": "CruxEval", "mbpp": "MBPP",
    "boolq": "BoolQ", "hotpotqa": "HotpotQA", "mmlu": "MMLU", "hellaswag": "HellaSwag",
    "arc_challenge": "ARC-C", "winogrande": "WinoGrande",
}
K = 8
EXPECTED_SEEDS = frozenset({42, 123, 456, 789, 1024})
PT4_SEEDS = frozenset({42, 123, 456})
INSTANCES_PER_SEED = 100
class CellMismatchError(RuntimeError):
    """The two arms do not cover the same questions, so the comparison would be unmatched."""


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _match_arms(
    model: str,
    task: str,
    pt_path: Path,
    sc_path: Path,
    expected_seeds: frozenset[int] = EXPECTED_SEEDS,
    canonical_gold: bool = False,
    sc_paths: int | None = None,
) -> tuple[list, list, Path, Path]:
    """Load pre-extracted answers and require an exact PT-to-SC record match.

    With ``canonical_gold`` the two arms' gold strings need only agree after the
    task's canonicalisation, and both arms are scored against the SC gold.
    """
    if not pt_path.is_file():
        raise FileNotFoundError(f"{model}/{task}: prompt-template file is missing: {pt_path}")
    if not sc_path.is_file():
        raise FileNotFoundError(f"{model}/{task}: self-consistency partner file is missing: {sc_path}")
    pt = _read_jsonl(pt_path)
    sc_records = _read_jsonl(sc_path)
    sc: dict[tuple[object, object], dict] = {}
    for record in sc_records:
        key = (record["seed"], record["instance_id"])
        if key in sc:
            raise CellMismatchError(f"{model}/{task}: duplicate self-consistency key {key}")
        sc[key] = record

    seeds = {record["seed"] for record in pt}
    if seeds != expected_seeds:
        raise CellMismatchError(
            f"{model}/{task}: prompt-template seeds {sorted(seeds)} != {sorted(expected_seeds)}"
        )
    seen: set[tuple[object, object]] = set()
    pt_cells, sc_cells = [], []
    for record in pt:
        key = (record["seed"], record["instance_id"])
        if key in seen:
            raise CellMismatchError(f"{model}/{task}: duplicate prompt-template key {key}")
        seen.add(key)
        if key not in sc:
            raise CellMismatchError(
                f"{model}/{task}: {key} present in prompt-template arm but not in self-consistency arm"
            )
        partner = sc[key]
        if canonical_gold:
            sc_gold = canonicalize_answer(partner["gold_answer"], task)
            pt_gold_canon = canonicalize_answer(record["gold_answer"], task)
            if sc_gold is None or pt_gold_canon is None:
                raise CellMismatchError(f"{model}/{task}: gold answer is null in an arm at {key}")
            if sc_gold != pt_gold_canon:
                raise CellMismatchError(f"{model}/{task}: canonical gold answer differs between arms at {key}")
            pt_gold = partner["gold_answer"]
        else:
            if str(partner["gold_answer"]) != str(record["gold_answer"]):
                raise CellMismatchError(f"{model}/{task}: gold answer differs between arms at {key}")
            pt_gold = record["gold_answer"]
        if sc_paths is not None and (len(record["all_answers"]) != K or len(partner["all_answers"]) != sc_paths):
            raise CellMismatchError(
                f"{model}/{task}: {key} has {len(record['all_answers'])} PT / {len(partner['all_answers'])} SC answers, "
                f"expected {K} / {sc_paths}"
            )
        pt_cells.append((record["all_answers"][:K], pt_gold))
        sc_cells.append((partner["all_answers"][:K], partner["gold_answer"]))
    unmatched = set(sc) - seen
    if unmatched:
        raise CellMismatchError(
            f"{model}/{task}: {len(unmatched)} self-consistency records have no prompt-template partner, e.g. {sorted(unmatched)[:3]}"
        )
    per_seed = {seed: sum(1 for key in seen if key[0] == seed) for seed in expected_seeds}
    if any(n != INSTANCES_PER_SEED for n in per_seed.values()):
        raise CellMismatchError(
            f"{model}/{task}: matched questions per seed {per_seed} != {INSTANCES_PER_SEED}"
        )
    return pt_cells, sc_cells, pt_path, sc_path


def load_arms(model: str) -> tuple[list, list, Path, Path]:
    """Load the two arms restricted to the questions both cover."""
    pt_path = experiment_dir("pt_records") / TASK / f"{model}_prompt_template.jsonl"
    sc_path = experiment_dir("sc_records") / f"{model}_sc_kvar_v2_{TASK}.jsonl"
    pt = _read_jsonl(pt_path)
    sc = {(r["seed"], r["instance_id"]): r for r in _read_jsonl(sc_path)}

    seeds = {r["seed"] for r in pt}
    if seeds != EXPECTED_SEEDS:
        raise CellMismatchError(f"{model}: prompt-template seeds {sorted(seeds)} != {sorted(EXPECTED_SEEDS)}")

    pt_cells, sc_cells = [], []
    for record in pt:
        key = (record["seed"], record["instance_id"])
        if key not in sc:
            raise CellMismatchError(f"{model}: {key} present in prompt-template arm but not in self-consistency arm")
        partner = sc[key]
        if str(partner["gold_answer"]) != str(record["gold_answer"]):
            raise CellMismatchError(f"{model}: gold answer differs between arms at {key}")
        # A path with no extracted answer is None in both arms, so cell_statistics skips it
        # in the r estimate exactly as it does for the self-consistency arm.
        pt_cells.append(([extract_answer(t, TASK) for t in record["all_traces"]], record["gold_answer"]))
        sc_cells.append((partner["all_answers"][:K], partner["gold_answer"]))
    return pt_cells, sc_cells, pt_path, sc_path


def load_t12_arms(
    model: str,
    task: str,
    pt_root: Path | None = None,
    sc_root: Path | None = None,
    expected_seeds: frozenset[int] = EXPECTED_SEEDS,
) -> tuple[list, list, Path, Path]:
    """Load one further-task cell from the locally re-extracted caches."""
    pt_root = pt_root or experiment_dir("pt_records")
    sc_root = sc_root or experiment_dir("sc_records")
    pt_path = pt_root / task / f"{model}_prompt_template_{task}.jsonl"
    sc_path = sc_root / f"{model}_sc_kvar_v2_{task}.jsonl"
    return _match_arms(model, task, pt_path, sc_path, expected_seeds)


def pt4_sc_path(model: str, task: str) -> Path:
    """Return the self-consistency partner path for one PT4 cell."""
    if task not in PT4_TASKS:
        raise ValueError(f"unknown PT4 task: {task}")
    return experiment_dir("sc_records") / f"{model}_sc_kvar_v2_{task}.jsonl"


def load_pt4_arms(
    model: str,
    task: str,
    pt_root: Path | None = None,
) -> tuple[list, list, Path, Path]:
    """Load one PT4 cell and require the three-seed matched grid."""
    pt_root = pt_root or experiment_dir("pt_records")
    pt_path = pt_root / task / f"{model}_prompt_template_{task}.jsonl"
    return _match_arms(model, task, pt_path, pt4_sc_path(model, task), PT4_SEEDS, canonical_gold=True)


def load_arcwg_arms(
    model: str,
    task: str,
    cache_root: Path | None = None,
) -> tuple[list, list, Path, Path]:
    """Load ARC/WinoGrande with the PT4 three-seed and canonical-gold contract."""
    if task not in ARCWG_TASKS:
        raise ValueError(f"unknown ARCWG task: {task}")
    pt_root = cache_root / "pt" if cache_root else experiment_dir("pt_records")
    sc_root = cache_root / "sc" if cache_root else experiment_dir("sc_records")
    pt_path = pt_root / task / f"{model}_prompt_template_{task}.jsonl"
    sc_path = sc_root / f"{model}_sc_kvar_v2_{task}.jsonl"
    return _match_arms(model, task, pt_path, sc_path, PT4_SEEDS, canonical_gold=True, sc_paths=32)
