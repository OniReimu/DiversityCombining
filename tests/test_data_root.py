"""The generation records must be present: these tests fail, never skip, without them.

Download the records (see README.md) and set DC_CACHE_ROOT to their directory.
"""
import os
from pathlib import Path

from diversity_combining.config import experiment_dir

LOCATION_FOR = {"sc_records": "sc_records", "reasoning_records": "reasoning_records",
                "capability_matrix": "cache"}

EXPECTED = {
    "sc_records": (
        "qwen7b_sc_kvar_v2_gsm8k.jsonl",
        "llama8b_sc_kvar_v2_gsm8k.jsonl",
        "mistral7b_sc_kvar_v2_gsm8k.jsonl",
        "llama8b_sc_kvar_v2_hotpotqa.jsonl",
        "llama8b_sc_kvar_v2_boolq.jsonl",
    ),
    "reasoning_records": (
        "qwen3_5_9b_sc_kvar_v2_gsm8k.jsonl",
    ),
    "capability_matrix": (
        "qwen05b_capgated.jsonl",
        "qwen05b_capgated_qa.jsonl",
        "qwen05b_capgated_triviaqa.jsonl",
        "qwen05b_capgated_arc.jsonl",
        "qwen05b_capgated_mmlu.jsonl",
        "qwen05b_capgated_mbpp.jsonl",
        "qwen05b_capgated_cruxeval.jsonl",
        "qwen05b_capgated_hellaswag.jsonl",
        "qwen05b_capgated_winogrande.jsonl",
        "qwen05b_capgated_boolq.jsonl",
        "qwen05b_capgated_drop.jsonl",
        "qwen7b_capgated.jsonl",
        "qwen7b_capgated_qa.jsonl",
        "qwen7b_capgated_triviaqa.jsonl",
        "qwen7b_capgated_arc.jsonl",
        "qwen7b_capgated_mmlu.jsonl",
        "qwen7b_capgated_mbpp.jsonl",
        "qwen7b_capgated_cruxeval.jsonl",
        "qwen7b_capgated_hellaswag.jsonl",
        "qwen7b_capgated_winogrande.jsonl",
        "qwen7b_capgated_boolq.jsonl",
        "qwen7b_capgated_drop.jsonl",
        "qwen32b_capgated.jsonl",
        "qwen32b_capgated_qa.jsonl",
        "qwen32b_capgated_triviaqa.jsonl",
        "qwen32b_capgated_arc.jsonl",
        "qwen32b_capgated_mmlu.jsonl",
        "qwen32b_capgated_mbpp.jsonl",
        "qwen32b_capgated_cruxeval.jsonl",
        "qwen32b_capgated_hellaswag.jsonl",
        "qwen32b_capgated_winogrande.jsonl",
        "qwen32b_capgated_boolq.jsonl",
        "qwen32b_capgated_drop.jsonl",
        "llama8b_capgated.jsonl",
        "llama8b_capgated_qa.jsonl",
        "llama8b_capgated_triviaqa.jsonl",
        "llama8b_capgated_arc.jsonl",
        "llama8b_capgated_mmlu.jsonl",
        "llama8b_capgated_mbpp.jsonl",
        "llama8b_capgated_cruxeval.jsonl",
        "llama8b_capgated_hellaswag.jsonl",
        "llama8b_capgated_winogrande.jsonl",
        "llama8b_capgated_boolq.jsonl",
        "llama8b_capgated_drop.jsonl",
        "mistral7b_capgated.jsonl",
        "mistral7b_capgated_qa.jsonl",
        "mistral7b_capgated_triviaqa.jsonl",
        "mistral7b_capgated_arc.jsonl",
        "mistral7b_capgated_mmlu.jsonl",
        "mistral7b_capgated_mbpp.jsonl",
        "mistral7b_capgated_cruxeval.jsonl",
        "mistral7b_capgated_hellaswag.jsonl",
        "mistral7b_capgated_winogrande.jsonl",
        "mistral7b_capgated_boolq.jsonl",
        "mistral7b_capgated_drop.jsonl",
    ),
}


def test_cache_root_is_set():
    value = os.environ.get("DC_CACHE_ROOT")
    assert value, "DC_CACHE_ROOT is not set; download the records (README.md) and export DC_CACHE_ROOT"
    assert Path(value).expanduser().is_dir(), f"DC_CACHE_ROOT={value} is not a directory"


def test_every_record_file_is_present():
    missing = []
    for folder, names in EXPECTED.items():
        directory = experiment_dir(LOCATION_FOR[folder])
        for name in names:
            path = directory / name
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(f"{folder}/{name}")
    assert not missing, f"{len(missing)} record file(s) missing under DC_CACHE_ROOT: {missing[:5]}"
