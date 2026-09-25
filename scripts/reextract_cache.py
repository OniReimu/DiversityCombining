#!/usr/bin/env python3
"""Re-extract cached path answers with the current SC extractor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.run_sc_kvar_v2 import (
    EXTRACTOR_VERSION,
    MODEL_CONFIGS,
    extract_answer,
    extract_final_segment,
    load_task_instances,
)
from scripts.scoring_v2 import check_task_correct
from scripts.tasks12 import canonicalize_for_instance, evaluate_mbpp_completion


TASK12_CANONICAL_TASKS = {"math", "triviaqa", "drop", "cruxeval"}


def extract_cached_path(
    trace: str,
    task: str,
    model_slug: str,
    is_truncated: bool,
    item: dict[str, Any] | None = None,
) -> str | None:
    """Apply the same final-segment, extraction, and vote-bucket path as the driver."""
    if model_slug not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_slug {model_slug!r}")
    config = MODEL_CONFIGS[model_slug]
    answer_text = (
        extract_final_segment(trace, model_slug)
        if config.is_reasoning
        else trace
    )
    if answer_text is None:
        return None
    answer = extract_answer(
        answer_text,
        task=task,
        is_truncated=is_truncated,
        model_slug=model_slug,
    )
    if task == "mbpp" and answer is not None:
        if item is None:
            raise ValueError("MBPP re-extraction requires the dataset item")
        answer = evaluate_mbpp_completion(
            answer,
            list(item["assertions"]),
            list(item.get("test_imports", [])),
        )
    elif task in TASK12_CANONICAL_TASKS:
        if item is None:
            raise ValueError(f"{task} re-extraction requires the dataset item")
        answer = canonicalize_for_instance(answer, item, task)
    return answer


def reextract_record(
    record: dict[str, Any],
    task: str,
    task_items: list[dict[str, Any]] | None = None,
) -> tuple[list[str | None], list[str | None]]:
    """Return old and newly extracted answers for one cache record."""
    traces = record.get("all_traces")
    old_answers = record.get("all_answers")
    truncated = record.get("truncated")
    if not isinstance(traces, list) or not isinstance(old_answers, list):
        raise ValueError("record must contain list-valued all_traces and all_answers")
    if not isinstance(truncated, list):
        raise ValueError("record must contain list-valued truncated flags")
    if not (len(traces) == len(old_answers) == len(truncated)):
        raise ValueError(
            "all_traces, all_answers, and truncated must have equal lengths "
            f"({len(traces)}, {len(old_answers)}, {len(truncated)})"
        )
    model_slug = record.get("model_slug")
    record_task = record.get("task")
    if record_task != task:
        raise ValueError(f"record task {record_task!r} does not match --task {task!r}")
    item = None
    if task_items is not None:
        instance_id = record.get("instance_id")
        if not isinstance(instance_id, int) or not 0 <= instance_id < len(task_items):
            raise ValueError(f"invalid instance_id {instance_id!r} for {task}")
        item = task_items[instance_id]
    new_answers = [
        extract_cached_path(
            str(trace),
            task,
            str(model_slug),
            bool(is_truncated),
            item=item,
        )
        for trace, is_truncated in zip(traces, truncated)
    ]
    return old_answers, new_answers


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def reextract_file(
    in_file: Path,
    out_file: Path,
    task: str,
    task_items: list[dict[str, Any]] | None,
) -> dict[str, int | float | str]:
    """Rewrite one JSONL file and return its re-extraction statistics."""
    paths = changed = old_null = new_null = old_correct = new_correct = 0
    old_empty = new_empty = 0
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with in_file.open(encoding="utf-8") as source, out_file.open("x", encoding="utf-8") as dest:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{in_file}:{line_number}: malformed JSON: {exc}") from exc
            old_answers, new_answers = reextract_record(record, task, task_items)
            gold = record.get("gold_answer")
            for old, new in zip(old_answers, new_answers):
                paths += 1
                changed += old != new
                old_null += old is None
                new_null += new is None
                old_empty += isinstance(old, str) and not old.strip()
                new_empty += isinstance(new, str) and not new.strip()
                old_correct += check_task_correct(old, gold, task)
                new_correct += check_task_correct(new, gold, task)
            record["all_answers_v21"] = old_answers
            record["all_answers"] = new_answers
            provenance = record.get("provenance")
            if not isinstance(provenance, dict):
                provenance = {}
                record["provenance"] = provenance
            provenance["extractor_version"] = EXTRACTOR_VERSION
            dest.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {
        "file": in_file.name,
        "paths": paths,
        "changed": changed,
        "null_before": _rate(old_null, paths),
        "null_after": _rate(new_null, paths),
        "accuracy_before": _rate(old_correct, paths),
        "accuracy_after": _rate(new_correct, paths),
        "empty_before": old_empty,
        "empty_after": new_empty,
    }


def _print_table(rows: list[dict[str, int | float | str]]) -> None:
    header = (
        f"{'file':<49} {'paths':>7} {'changed':>8} "
        f"{'null before':>11} {'null after':>10} {'acc before':>10} {'acc after':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['file']:<49} {row['paths']:>7} {row['changed']:>8} "
            f"{row['null_before']:>10.2%} {row['null_after']:>9.2%} "
            f"{row['accuracy_before']:>9.2%} {row['accuracy_after']:>8.2%}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_dir", type=Path, required=True)
    parser.add_argument("--out", dest="out_dir", type=Path, required=True)
    parser.add_argument("--task", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    in_dir = args.in_dir.resolve()
    out_dir = args.out_dir.resolve()
    if in_dir == out_dir:
        raise SystemExit("FATAL: --out must differ from --in; in-place rewriting is forbidden")
    if not in_dir.is_dir():
        raise SystemExit(f"FATAL: input directory does not exist: {in_dir}")
    files = sorted(in_dir.glob(f"*_sc_kvar_v2_{args.task}.jsonl"))
    if not files:
        raise SystemExit(f"FATAL: no {args.task} cache files found in {in_dir}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"FATAL: output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    task_items = None
    if args.task in TASK12_CANONICAL_TASKS | {"mbpp"}:
        task_items = load_task_instances(args.task)
    rows = [
        reextract_file(path, out_dir / path.name, args.task, task_items)
        for path in files
    ]
    _print_table(rows)
    print(f"empty-string answers: {sum(int(row['empty_after']) for row in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
