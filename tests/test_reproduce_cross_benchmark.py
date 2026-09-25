#!/usr/bin/env python3
"""Tests for scripts/reproduce_cross_benchmark.py.

Checks the scorer contract (GSM8K numeric, MATH-500 canonical exact), the per-arm
production shape and its negative controls, the scored GSM8K value for Qwen2.5-7B,
and the deliverables written by scripts/reproduce_cross_benchmark.py (run it first).
"""
import json
import sys
from pathlib import Path

import pytest

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
_SRC = PROJECT_ROOT
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from scripts.aggregate_5seed import MODELS, BENCHMARK_FILES
from scripts.scoring_v2 import GSM8KScorer, MathScorer
from scripts.reproduce_cross_benchmark import (
    EXPECTED_SEEDS,
    INSTANCES_PER_ARM,
    OUTPUT_DIR,
    PARTIAL_ARM_SEED_COUNTS,
    assert_arm_shape,
    compute_derived_numbers,
    expected_arm_seed_counts,
    extract_correctness_vector,
    get_scorers,
    run_table3_pipeline,
)



@pytest.fixture(scope="module")
def pipeline_results():
    """Run the pipeline without bootstrap draws (n_boot=0) for fast value checks."""
    cells, pt_mats = run_table3_pipeline(n_boot=0)
    derived = compute_derived_numbers(cells, pt_mats)
    return cells, pt_mats, derived


def test_scorer_identity_and_contract():
    """Verify scorers are the exact scoring_v2 objects."""
    gsm, math_sc = get_scorers()

    # Must be exact classes from scripts.scoring_v2
    assert type(gsm) is GSM8KScorer, "Must be exact GSM8KScorer class"
    assert type(math_sc) is MathScorer, "Must be exact MathScorer class"
    assert gsm.__class__.__module__ == "scripts.scoring_v2"
    assert math_sc.__class__.__module__ == "scripts.scoring_v2"

    # Contract names
    assert gsm.name == "gsm8k_numeric"
    assert math_sc.name == "math500_canonical_exact"

    # Behavior: GSM8K accepts trailing periods and thousands commas, rejects wrong values
    assert gsm("3.", "3") is True
    assert gsm("1,200", "1200") is True
    assert gsm("3", "4") is False
    assert gsm(None, "3") is False

    # Behavior: MATH accepts LaTeX fraction shorthand and equations
    assert math_sc(r"\frac12", r"\frac{1}{2}") is True
    assert math_sc("x = 5", "5") is True
    assert math_sc("5", "6") is False
    assert math_sc(None, "5") is False


def test_production_shape_table():
    """The asserted per-arm shape pins the known partial arms of the matrix."""
    pooled_n = {
        (m, t, a): sum(c.values()) for (m, t, a), c in PARTIAL_ARM_SEED_COUNTS.items()
    }
    assert pooled_n == {
        ("qwen7b", "math", "sc"): 226, ("qwen7b", "math", "pt"): 100,
        ("qwen32b", "math", "sc"): 235, ("qwen32b", "math", "pt"): 100,
        ("qwen7b", "mmlu", "pt"): 124, ("qwen32b", "mmlu", "pt"): 158,
        ("qwen7b", "mbpp", "pt"): 223, ("llama8b", "math", "pt"): 201,
    }
    for (model, task, arm), counts in PARTIAL_ARM_SEED_COUNTS.items():
        assert expected_arm_seed_counts(model, task, arm) == counts
        assert set(counts) <= set(EXPECTED_SEEDS)


def test_arm_shape_assertion_responds():
    """Negative controls: the shape guard fires on each kind of deviation."""
    recs = [{"seed": s, "instance_id": i} for s in EXPECTED_SEEDS for i in range(INSTANCES_PER_ARM)]
    assert_arm_shape("qwen05b", "gsm8k", "sc", recs)  # full arm passes
    with pytest.raises(AssertionError, match="seed counts"):
        assert_arm_shape("qwen05b", "gsm8k", "sc", recs[:-1])
    with pytest.raises(AssertionError, match="seed counts"):
        assert_arm_shape("qwen05b", "gsm8k", "sc", recs + [{"seed": 7, "instance_id": 0}])
    with pytest.raises(AssertionError, match="seed counts"):
        assert_arm_shape("qwen7b", "math", "pt", recs)  # full arm where production is partial
    dup = recs[:-1] + [{"seed": recs[-1]["seed"], "instance_id": 0}]  # same seed, repeated id
    with pytest.raises(AssertionError, match="duplicate"):
        assert_arm_shape("qwen05b", "gsm8k", "sc", dup)
    shifted = [{"seed": r["seed"], "instance_id": r["instance_id"] + (r["seed"] == 42)} for r in recs]
    with pytest.raises(AssertionError, match="distinct instances"):
        assert_arm_shape("qwen05b", "gsm8k", "sc", shifted)


def test_gsm8k_math_scored_values(pipeline_results):
    """GSM8K and MATH cells are scored with the numeric and MATH-500 scorers."""
    cells, _, derived = pipeline_results
    assert len(cells) == 60
    assert len([k for k in cells if k[1] in ("gsm8k", "math")]) == 10
    assert cells[("qwen7b", "gsm8k")]["p_bar_sc_pooled"] == pytest.approx(0.805, abs=1e-3)
    assert derived["valid_cells_count"] == 57
    assert derived["decorrelates_count"] == 55


def _load_numbers_json():
    num_f = OUTPUT_DIR / "numbers.json"
    assert num_f.exists(), f"run scripts/reproduce_cross_benchmark.py first: missing {num_f}"

    def _reject(tok):
        raise AssertionError(f"numbers.json contains non-standard JSON constant {tok}")

    return json.loads(num_f.read_text(), parse_constant=_reject)


def test_numbers_json_is_strict_json():
    """undefined values are null, never bare NaN/Infinity."""
    data = _load_numbers_json()
    assert len(data["cells"]) == 60
    # excluded DROP cells carry null CIs
    assert data["cells"]["qwen05b_drop"]["drho_ci_low"] is None


def test_deliverables_on_disk():
    """Deliverables exist and are well-formed after running reproduce_cross_benchmark.py."""
    data = _load_numbers_json()
    assert "table3_rows" in data["derived_numbers"]
    assert data["metadata"]["valid_cells_count"] == 57

    tex = (OUTPUT_DIR / "rows_table3.tex").read_text()
    for name in ("TriviaQA", "GSM8K", "MATH"):
        assert name in tex
