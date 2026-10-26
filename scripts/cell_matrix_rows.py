"""LaTeX rows for the per-cell appendix table: all 60 model-benchmark cells.

Reads the numbers.json written by scripts/reproduce_cross_benchmark.py.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
from diversity_combining.config import experiment_dir
DIR = experiment_dir("cross_benchmark")

# Benchmark order of Table 7
TASKS = [("triviaqa", "TriviaQA"), ("hotpotqa", "HotpotQA"), ("arc_challenge", "ARC-C"),
         ("mmlu", "MMLU"), ("hellaswag", "HellaSwag"), ("winogrande", "WinoGrande"),
         ("drop", "DROP"), ("boolq", "BoolQ"), ("mbpp", "MBPP"), ("cruxeval", "CruxEval"),
         ("gsm8k", "GSM8K"), ("math", "MATH")]
MODELS = [("qwen05b", "Qwen-0.5B"), ("qwen7b", "Qwen-7B"), ("qwen32b", "Qwen-32B"),
          ("llama8b", "Llama-8B"), ("mistral7b", "Mistral-7B")]


def signed(x):
    return f"${'+' if x >= 0 else '-'}{abs(x):.1f}$"


def main():
    cells = {(c["model"], c["task"]): c for c in json.loads((DIR / "numbers.json").read_text())["cells"]}
    assert len(cells) == 60
    lines = []
    for t, (task, tname) in enumerate(TASKS):
        for m, (model, mname) in enumerate(MODELS):
            c = cells[(model, task)]
            sc, pt = c["SC"], c["PT"]
            lo, hi = c["delta_rho_ci95"]
            lines.append(
                f"{tname if m == 0 else ''} & {mname} & {sc['n_instances']} & "
                f"{100 * sc['p_bar']:.1f} & {100 * pt['p_bar']:.1f} & "
                f"{sc['rho_hat']:.2f} & {pt['rho_hat']:.2f} & {signed(c['delta_rho_pct'])} & "
                f"[{signed(lo)}, {signed(hi)}] & {100 * sc['MV8']:.1f} & {100 * pt['MV8']:.1f} \\\\")
        if t < len(TASKS) - 1:
            lines.append("\\midrule")
    out = DIR / "table_cell_matrix_rows.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
