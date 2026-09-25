"""Dataset loaders for Diversity Combining evaluation tasks."""

import random
from pathlib import Path

from datasets import load_dataset


def extract_boxed_answer(solution: str) -> str:
    r"""Extract the content of \boxed{...} from a MATH solution string.

    Handles nested braces (e.g. \boxed{\frac{1}{2}}).
    Falls back to the full solution if no \boxed is found.
    """
    idx = solution.rfind(r"\boxed{")
    if idx == -1:
        return solution

    # Start after '\boxed{'
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
    # Mismatched braces — fall back
    return solution


def load_task(task_name: str, split: str = "test", max_instances: int | None = None) -> list[dict]:
    """Load evaluation dataset for a given task.

    Returns list of dicts with keys: 'prompt', 'answer', 'metadata'.

    Supported tasks: gsm8k, math, hotpotqa, triviaqa, aime2024, aime2025,
                     amc2023, math500, algorithmic_parity, algorithmic_sorting,
                     algorithmic_dyck.
    """
    loaders = {
        "gsm8k": _load_gsm8k,
        "math": _load_math,
        "hotpotqa": _load_hotpotqa,
        "triviaqa": _load_triviaqa,
        "aime2024": lambda split: _load_aime(split, year=2024),
        "aime2025": lambda split: _load_aime(split, year=2025),
        "amc2023": _load_amc,
        "math500": _load_math500,
        "algorithmic_parity": _load_algorithmic_parity,
        "algorithmic_sorting": _load_algorithmic_sorting,
        "algorithmic_dyck": _load_algorithmic_dyck,
    }

    loaders["mbpp"] = _load_mbpp
    loaders["boolq"] = _load_boolq
    loaders["cruxeval"] = _load_cruxeval
    loaders["drop"] = _load_drop

    if task_name not in loaders:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(loaders)}")

    instances = loaders[task_name](split)
    if max_instances is not None:
        instances = instances[:max_instances]
    return instances


def _load_gsm8k(split: str) -> list[dict]:
    """GSM8K: Grade-school math word problems."""
    ds = load_dataset("openai/gsm8k", "main", split=split)
    instances = []
    for item in ds:
        # Extract final numerical answer after ####
        answer_text = item["answer"]
        final_answer = answer_text.split("####")[-1].strip()
        instances.append({
            "prompt": f"Solve the following math problem step by step.\n\n"
                      f"Problem: {item['question']}\n\nSolution:",
            "answer": final_answer,
            "metadata": {"full_solution": answer_text},
        })
    return instances


def _load_math(split: str) -> list[dict]:
    """MATH-500: the fixed Task-12 mathematics benchmark."""
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    instances = []
    for item in ds:
        instances.append({
            "prompt": f"Solve the following math problem. Show your work and give "
                      f"the final answer.\n\nProblem: {item['problem']}\n\nSolution:",
            "answer": item["answer"].strip(),
            "metadata": {
                "level": item.get("level", ""),
                "type": item.get("subject", ""),
                "source": "HuggingFaceH4/MATH-500",
            },
        })
    return instances


def _load_hotpotqa(split: str) -> list[dict]:
    """HotpotQA: Multi-hop question answering."""
    split_map = {"test": "validation"}  # HotpotQA test is hidden
    actual_split = split_map.get(split, split)
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=actual_split)
    instances = []
    for item in ds:
        context = "\n".join(
            f"- {title}: {' '.join(sentences)}"
            for title, sentences in zip(item["context"]["title"], item["context"]["sentences"])
        )
        instances.append({
            "prompt": f"Answer the following question using the provided context.\n\n"
                      f"Context:\n{context}\n\nQuestion: {item['question']}\n\nAnswer:",
            "answer": item["answer"],
            "metadata": {"type": item.get("type", ""), "level": item.get("level", "")},
        })
    return instances


def _load_cruxeval(split: str) -> list[dict]:
    """CruxEval: Code output prediction."""
    ds = load_dataset("cruxeval-org/cruxeval", split=split)
    instances = []
    for item in ds:
        instances.append({
            "prompt": f"What is the output of the following Python code?\n\n```python\n{item['code']}\n```\n\n"
                      f"Input: {item['input']}\n\nOutput:",
            "answer": item["output"],
            "metadata": {"id": item.get("id", "")},
        })
    return instances


def _load_boolq(split: str) -> list[dict]:
    """BoolQ: Boolean Yes/No question answering."""
    split_map = {"test": "validation"}
    actual_split = split_map.get(split, split)
    ds = load_dataset("google/boolq", split=actual_split)
    instances = []
    for item in ds:
        instances.append({
            "prompt": f"Answer the following yes/no question based on the passage.\n\n"
                      f"Passage: {item['passage']}\n\nQuestion: {item['question']}\n\nAnswer (Yes or No):",
            "answer": "Yes" if item["answer"] else "No",
            "metadata": {},
        })
    return instances


def _load_mbpp(split: str) -> list[dict]:
    """MBPP: Mostly Basic Python Programming."""
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split=split)
    instances = []
    for item in ds:
        # Extract entry_point (function name) from the code solution
        import re as _re
        entry_point = ""
        m = _re.search(r'def\s+(\w+)', item.get("code", ""))
        if m:
            entry_point = m.group(1)
        assertions = list(item.get("test_list", []))
        if not assertions:
            raise ValueError(f"MBPP instance {item.get('task_id', '')!r} has no assertions")
        prompt = (
            f"{item['prompt']}\n\nYour code should pass these tests:\n"
            + "\n".join(assertions)
        )
        instances.append({
            "prompt": prompt,
            "answer": "P" * len(assertions),
            "assertions": assertions,
            "test_imports": list(item.get("test_imports", [])),
            "entry_point": entry_point,
            "code_solution": item.get("code", ""),
            "metadata": {"task_id": item.get("task_id", "")},
        })
    return instances


def _load_drop(split: str) -> list[dict]:
    """DROP validation, retaining all annotator references as accepted aliases."""
    actual_split = "validation" if split == "test" else split
    ds = load_dataset("ucinlp/drop", split=actual_split)
    instances = []
    for item in ds:
        spans = list(item["answers_spans"]["spans"])
        if not spans:
            raise ValueError("DROP instance has no answer span references")
        instances.append({
            "prompt": (
                "Answer the following question using the provided passage.\n\n"
                f"Passage: {item['passage']}\n\nQuestion: {item['question']}\n\nAnswer:"
            ),
            "answer": spans[0],
            "metadata": {
                "section_id": item.get("section_id", ""),
                "aliases": spans[1:],
            },
        })
    print(f"DROP {actual_split}: kept {len(instances)} instances")
    return instances


def _load_triviaqa(split: str) -> list[dict]:
    """TriviaQA: Single-hop factoid question answering (reading comprehension)."""
    split_map = {"test": "validation"}  # TriviaQA test is hidden
    actual_split = split_map.get(split, split)
    ds = load_dataset("trivia_qa", "rc", split=actual_split)
    instances = []
    for item in ds:
        # Use search context (shorter than full Wikipedia article)
        contexts = item.get("search_results", {}).get("search_context", [])
        if not contexts:
            contexts = item.get("entity_pages", {}).get("wiki_context", [])
        context_text = "\n".join(c[:500] for c in contexts[:3]) if contexts else ""
        # TriviaQA has multiple acceptable answers in answer.aliases
        answer = item["answer"]["value"]
        instances.append({
            "prompt": f"Answer the following question using the provided context.\n\n"
                      f"Context:\n{context_text}\n\nQuestion: {item['question']}\n\nAnswer:",
            "answer": answer,
            "metadata": {
                "aliases": item["answer"].get("aliases", []),
                "normalized_aliases": item["answer"].get("normalized_aliases", []),
            },
        })
    return instances


def _load_aime(split: str, year: int = 2024) -> list[dict]:
    """AIME (year-specific): Load from upstream Multiplex Thinking parquet files.

    The upstream deepscaler parquets use the RL schema (prompt = chat-message
    list, answer under reward_model.ground_truth), which load_from_parquet does
    not handle; they are parsed directly here. aime2025 has no HF fallback:
    silently substituting another year's data would corrupt any experiment, so
    a missing parquet raises instead.
    """
    hdfs_dir = (
        Path(__file__).resolve().parent.parent.parent
        / "multiplex_thinking_upstream" / "deepscaler" / "hdfs_data"
    )
    parquet_names = {
        2024: ["aime24.parquet", "aime.parquet"],
        2025: ["aime25.parquet", "aime2025.parquet"],
    }[year]

    for name in parquet_names:
        parquet_path = hdfs_dir / name
        if not parquet_path.exists():
            continue
        import pandas as pd
        df = pd.read_parquet(parquet_path)
        instances = []
        if "reward_model" in df.columns:
            # deepscaler RL schema
            for _, row in df.iterrows():
                instances.append({
                    "prompt": row["prompt"][0]["content"],
                    "answer": str(dict(row["reward_model"])["ground_truth"]).strip(),
                    "metadata": {"source": f"aime{year}", "file": name},
                })
        else:
            instances = load_from_parquet(str(parquet_path))
        if len(instances) != 30:
            raise RuntimeError(
                f"AIME {year} parquet {name} yielded {len(instances)} instances, expected 30"
            )
        return instances

    if year == 2024:
        # Fallback: try HuggingFace dataset (2024 only; loud, never silent-empty)
        print(f"WARNING: no AIME {year} parquet under {hdfs_dir}; falling back to HF di-zhang-fdu/AIME24")
        ds = load_dataset("di-zhang-fdu/AIME24", split="test")
        instances = []
        for item in ds:
            prompt_col = "problem" if "problem" in item else "question"
            instances.append({
                "prompt": f"Solve the following math competition problem.\n\n"
                          f"Problem: {item[prompt_col]}\n\nSolution:",
                "answer": str(item.get("answer", item.get("solution", ""))).strip(),
                "metadata": {"source": "aime2024_hf"},
            })
        if len(instances) != 30:
            raise RuntimeError(f"HF AIME24 fallback yielded {len(instances)} instances, expected 30")
        return instances

    raise RuntimeError(
        f"AIME {year} parquet not found under {hdfs_dir} (tried {parquet_names}); "
        f"sync multiplex_thinking_upstream/deepscaler/hdfs_data/ before running aime{year}"
    )


def _load_amc(split: str) -> list[dict]:
    """AMC: Load from upstream parquet or HuggingFace."""
    parquet_path = (
        Path(__file__).resolve().parent.parent.parent
        / "multiplex_thinking_upstream" / "deepscaler" / "hdfs_data" / "amc23.parquet"
    )
    if parquet_path.exists():
        return load_from_parquet(str(parquet_path))
    return []


def _load_math500(split: str) -> list[dict]:
    """MATH-500: 500-problem subset of MATH benchmark."""
    try:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        instances = []
        for item in ds:
            instances.append({
                "prompt": f"Solve the following math problem. Show your work and give "
                          f"the final answer.\n\nProblem: {item['problem']}\n\nSolution:",
                "answer": item["answer"].strip(),
                "metadata": {
                    "level": item.get("level", ""),
                    "type": item.get("type", ""),
                    "source": "math500",
                },
            })
        return instances
    except Exception:
        return []


def _load_algorithmic_parity(split: str) -> list[dict]:
    """Algorithmic task: parity of 1s in a binary string."""
    rng = random.Random(42)
    instances = []
    for i in range(1000):
        length = rng.randint(8, 16)
        bitstring = "".join(rng.choice("01") for _ in range(length))
        count_ones = bitstring.count("1")
        answer = "even" if count_ones % 2 == 0 else "odd"
        instances.append({
            "prompt": f"What is the parity (even or odd) of the number of 1s "
                      f"in: {bitstring}?",
            "answer": answer,
            "metadata": {"task": "algorithmic_parity", "instance_id": i},
        })
    return instances


def _load_algorithmic_sorting(split: str) -> list[dict]:
    """Algorithmic task: sort a list of integers."""
    rng = random.Random(42)
    instances = []
    for i in range(1000):
        length = rng.randint(5, 10)
        nums = [rng.randint(0, 99) for _ in range(length)]
        answer = str(sorted(nums))
        instances.append({
            "prompt": f"Sort the following list: {nums}",
            "answer": answer,
            "metadata": {"task": "algorithmic_sorting", "instance_id": i},
        })
    return instances


def _load_algorithmic_dyck(split: str) -> list[dict]:
    """Algorithmic task: Dyck language membership (balanced brackets)."""
    rng = random.Random(42)

    def _generate_balanced(rng: random.Random, n_pairs: int) -> str:
        """Generate a balanced bracket sequence of length 2*n_pairs."""
        seq: list[str] = []
        open_count = 0
        close_count = 0
        for _ in range(2 * n_pairs):
            if open_count == n_pairs:
                seq.append(")")
                close_count += 1
            elif close_count == open_count:
                seq.append("(")
                open_count += 1
            else:
                if rng.random() < 0.5:
                    seq.append("(")
                    open_count += 1
                else:
                    seq.append(")")
                    close_count += 1
        return "".join(seq)

    def _generate_unbalanced(rng: random.Random, length: int) -> str:
        """Generate an unbalanced bracket sequence of given length."""
        while True:
            seq = "".join(rng.choice("()") for _ in range(length))
            # Verify it is actually unbalanced
            depth = 0
            valid = True
            for ch in seq:
                depth += 1 if ch == "(" else -1
                if depth < 0:
                    valid = False
                    break
            if depth != 0:
                valid = False
            if not valid:
                return seq

    instances = []
    # 500 balanced + 500 unbalanced
    for i in range(500):
        n_pairs = rng.randint(3, 7)  # length 6-14
        seq = _generate_balanced(rng, n_pairs)
        instances.append({
            "prompt": f"Is the following bracket sequence valid? {seq}",
            "answer": "valid",
            "metadata": {"task": "algorithmic_dyck", "instance_id": i},
        })
    for i in range(500):
        length = rng.randint(6, 14)
        # Ensure even length for fairness
        if length % 2 != 0:
            length += 1
        seq = _generate_unbalanced(rng, length)
        instances.append({
            "prompt": f"Is the following bracket sequence valid? {seq}",
            "answer": "invalid",
            "metadata": {"task": "algorithmic_dyck", "instance_id": 500 + i},
        })
    # Shuffle so valid/invalid are interleaved
    rng.shuffle(instances)
    return instances


def load_from_parquet(parquet_path: str) -> list[dict]:
    """Load evaluation data from Multiplex Thinking's parquet format.

    The upstream repo stores datasets in deepscaler/hdfs_data/*.parquet
    with columns: 'prompt' (or 'question') and 'answer'.
    """
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    instances = []
    for _, row in df.iterrows():
        prompt_col = "prompt" if "prompt" in df.columns else "question"
        instances.append({
            "prompt": row[prompt_col],
            "answer": str(row["answer"]),
            "metadata": {k: row[k] for k in df.columns if k not in [prompt_col, "answer"]},
        })
    return instances
