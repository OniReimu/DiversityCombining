"""The generation records must be present: these tests fail, never skip, without them.

Download the records (see README.md) and set DC_CACHE_ROOT to their directory.
"""
import os
from pathlib import Path

from diversity_combining.config import experiment_dir

LOCATION_FOR = {"sc_records": "sc_records", "reasoning_records": "reasoning_records",
                "pt_records": "pt_records"}

EXPECTED = {
    "sc_records": (
        "llama8b_sc_kvar_v2_arc_challenge.jsonl",
        "llama8b_sc_kvar_v2_boolq.jsonl",
        "llama8b_sc_kvar_v2_cruxeval.jsonl",
        "llama8b_sc_kvar_v2_drop.jsonl",
        "llama8b_sc_kvar_v2_gsm8k.jsonl",
        "llama8b_sc_kvar_v2_hellaswag.jsonl",
        "llama8b_sc_kvar_v2_hotpotqa.jsonl",
        "llama8b_sc_kvar_v2_math.jsonl",
        "llama8b_sc_kvar_v2_mbpp.jsonl",
        "llama8b_sc_kvar_v2_mmlu.jsonl",
        "llama8b_sc_kvar_v2_triviaqa.jsonl",
        "llama8b_sc_kvar_v2_winogrande.jsonl",
        "mistral7b_sc_kvar_v2_arc_challenge.jsonl",
        "mistral7b_sc_kvar_v2_boolq.jsonl",
        "mistral7b_sc_kvar_v2_cruxeval.jsonl",
        "mistral7b_sc_kvar_v2_drop.jsonl",
        "mistral7b_sc_kvar_v2_gsm8k.jsonl",
        "mistral7b_sc_kvar_v2_hellaswag.jsonl",
        "mistral7b_sc_kvar_v2_hotpotqa.jsonl",
        "mistral7b_sc_kvar_v2_math.jsonl",
        "mistral7b_sc_kvar_v2_mbpp.jsonl",
        "mistral7b_sc_kvar_v2_mmlu.jsonl",
        "mistral7b_sc_kvar_v2_triviaqa.jsonl",
        "mistral7b_sc_kvar_v2_winogrande.jsonl",
        "qwen05b_sc_kvar_v2_arc_challenge.jsonl",
        "qwen05b_sc_kvar_v2_boolq.jsonl",
        "qwen05b_sc_kvar_v2_cruxeval.jsonl",
        "qwen05b_sc_kvar_v2_drop.jsonl",
        "qwen05b_sc_kvar_v2_gsm8k.jsonl",
        "qwen05b_sc_kvar_v2_hellaswag.jsonl",
        "qwen05b_sc_kvar_v2_hotpotqa.jsonl",
        "qwen05b_sc_kvar_v2_math.jsonl",
        "qwen05b_sc_kvar_v2_mbpp.jsonl",
        "qwen05b_sc_kvar_v2_mmlu.jsonl",
        "qwen05b_sc_kvar_v2_triviaqa.jsonl",
        "qwen05b_sc_kvar_v2_winogrande.jsonl",
        "qwen32b_sc_kvar_v2_arc_challenge.jsonl",
        "qwen32b_sc_kvar_v2_boolq.jsonl",
        "qwen32b_sc_kvar_v2_cruxeval.jsonl",
        "qwen32b_sc_kvar_v2_drop.jsonl",
        "qwen32b_sc_kvar_v2_gsm8k.jsonl",
        "qwen32b_sc_kvar_v2_hellaswag.jsonl",
        "qwen32b_sc_kvar_v2_hotpotqa.jsonl",
        "qwen32b_sc_kvar_v2_math.jsonl",
        "qwen32b_sc_kvar_v2_mbpp.jsonl",
        "qwen32b_sc_kvar_v2_mmlu.jsonl",
        "qwen32b_sc_kvar_v2_triviaqa.jsonl",
        "qwen32b_sc_kvar_v2_winogrande.jsonl",
        "qwen7b_sc_kvar_v2_arc_challenge.jsonl",
        "qwen7b_sc_kvar_v2_boolq.jsonl",
        "qwen7b_sc_kvar_v2_cruxeval.jsonl",
        "qwen7b_sc_kvar_v2_drop.jsonl",
        "qwen7b_sc_kvar_v2_gsm8k.jsonl",
        "qwen7b_sc_kvar_v2_hellaswag.jsonl",
        "qwen7b_sc_kvar_v2_hotpotqa.jsonl",
        "qwen7b_sc_kvar_v2_math.jsonl",
        "qwen7b_sc_kvar_v2_mbpp.jsonl",
        "qwen7b_sc_kvar_v2_mmlu.jsonl",
        "qwen7b_sc_kvar_v2_triviaqa.jsonl",
        "qwen7b_sc_kvar_v2_winogrande.jsonl",
    ),
    "reasoning_records": (
        "qwen3_5_9b_sc_kvar_v2_gsm8k.jsonl",
    ),
    "pt_records": (
        "arc_challenge/llama8b_prompt_template_arc_challenge.jsonl",
        "arc_challenge/mistral7b_prompt_template_arc_challenge.jsonl",
        "arc_challenge/qwen05b_prompt_template_arc_challenge.jsonl",
        "arc_challenge/qwen32b_prompt_template_arc_challenge.jsonl",
        "arc_challenge/qwen7b_prompt_template_arc_challenge.jsonl",
        "boolq/llama8b_prompt_template_boolq.jsonl",
        "boolq/mistral7b_prompt_template_boolq.jsonl",
        "boolq/qwen05b_prompt_template_boolq.jsonl",
        "boolq/qwen32b_prompt_template_boolq.jsonl",
        "boolq/qwen7b_prompt_template_boolq.jsonl",
        "cruxeval/llama8b_prompt_template_cruxeval.jsonl",
        "cruxeval/mistral7b_prompt_template_cruxeval.jsonl",
        "cruxeval/qwen05b_prompt_template_cruxeval.jsonl",
        "cruxeval/qwen32b_prompt_template_cruxeval.jsonl",
        "cruxeval/qwen7b_prompt_template_cruxeval.jsonl",
        "drop/llama8b_prompt_template_drop.jsonl",
        "drop/mistral7b_prompt_template_drop.jsonl",
        "drop/qwen05b_prompt_template_drop.jsonl",
        "drop/qwen32b_prompt_template_drop.jsonl",
        "drop/qwen7b_prompt_template_drop.jsonl",
        "gsm8k/llama8b_prompt_template.jsonl",
        "gsm8k/mistral7b_prompt_template.jsonl",
        "gsm8k/qwen05b_prompt_template.jsonl",
        "gsm8k/qwen32b_prompt_template.jsonl",
        "gsm8k/qwen7b_prompt_template.jsonl",
        "hellaswag/llama8b_prompt_template_hellaswag.jsonl",
        "hellaswag/mistral7b_prompt_template_hellaswag.jsonl",
        "hellaswag/qwen05b_prompt_template_hellaswag.jsonl",
        "hellaswag/qwen32b_prompt_template_hellaswag.jsonl",
        "hellaswag/qwen7b_prompt_template_hellaswag.jsonl",
        "hotpotqa/llama8b_prompt_template_hotpotqa.jsonl",
        "hotpotqa/mistral7b_prompt_template_hotpotqa.jsonl",
        "hotpotqa/qwen05b_prompt_template_hotpotqa.jsonl",
        "hotpotqa/qwen32b_prompt_template_hotpotqa.jsonl",
        "hotpotqa/qwen7b_prompt_template_hotpotqa.jsonl",
        "math/llama8b_prompt_template_math.jsonl",
        "math/mistral7b_prompt_template_math.jsonl",
        "math/qwen05b_prompt_template_math.jsonl",
        "math/qwen32b_prompt_template_math.jsonl",
        "math/qwen7b_prompt_template_math.jsonl",
        "mbpp/llama8b_prompt_template_mbpp.jsonl",
        "mbpp/mistral7b_prompt_template_mbpp.jsonl",
        "mbpp/qwen05b_prompt_template_mbpp.jsonl",
        "mbpp/qwen32b_prompt_template_mbpp.jsonl",
        "mbpp/qwen7b_prompt_template_mbpp.jsonl",
        "mmlu/llama8b_prompt_template_mmlu.jsonl",
        "mmlu/mistral7b_prompt_template_mmlu.jsonl",
        "mmlu/qwen05b_prompt_template_mmlu.jsonl",
        "mmlu/qwen32b_prompt_template_mmlu.jsonl",
        "mmlu/qwen7b_prompt_template_mmlu.jsonl",
        "triviaqa/llama8b_prompt_template_triviaqa.jsonl",
        "triviaqa/mistral7b_prompt_template_triviaqa.jsonl",
        "triviaqa/qwen05b_prompt_template_triviaqa.jsonl",
        "triviaqa/qwen32b_prompt_template_triviaqa.jsonl",
        "triviaqa/qwen7b_prompt_template_triviaqa.jsonl",
        "winogrande/llama8b_prompt_template_winogrande.jsonl",
        "winogrande/mistral7b_prompt_template_winogrande.jsonl",
        "winogrande/qwen05b_prompt_template_winogrande.jsonl",
        "winogrande/qwen32b_prompt_template_winogrande.jsonl",
        "winogrande/qwen7b_prompt_template_winogrande.jsonl",
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
