"""Per-cell prompt-template decorrelation: one marker per (model, benchmark) cell.

Reads the numbers.json written by scripts/reproduce_cross_benchmark.py.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
from diversity_combining.config import experiment_dir
NUMBERS = experiment_dir("cross_benchmark") / "numbers.json"
OUT = experiment_dir("figures") / "drho_cells.pdf"

# Benchmark order and labels of Table 7
TASKS = ["triviaqa", "hotpotqa", "arc_challenge", "mmlu", "hellaswag", "winogrande",
         "drop", "boolq", "mbpp", "cruxeval", "gsm8k", "math"]
LABELS = ["TriviaQA", "HotpotQA", "ARC-C", "MMLU", "HellaSwag", "WinoGr.",
          "DROP", "BoolQ", "MBPP", "CruxEval", "GSM8K", "MATH"]
# Fig. 2 colours for the three shared models, Okabe-Ito for the other two
MODELS = [("qwen05b", "Qwen-0.5B", "#E69F00", "v"), ("qwen7b", "Qwen-7B", "#D55E00", "o"),
          ("qwen32b", "Qwen-32B", "#CC79A7", "^"), ("llama8b", "Llama-8B", "#0072B2", "s"),
          ("mistral7b", "Mistral-7B", "#009E73", "D")]


def main():
    cells = {(c["model"], c["task"]): c["delta_rho_pct"]
             for c in json.loads(NUMBERS.read_text())["cells"]}
    assert len(cells) == 60 and all((m, t) in cells for m, *_ in MODELS for t in TASKS)
    # Sized so that, at the same printed height as Fig. 2, the fonts match Fig. 2.
    plt.rcParams.update({"font.size": 16, "axes.labelsize": 16, "xtick.labelsize": 13,
                         "ytick.labelsize": 16, "font.family": "sans-serif",
                         "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(5.0, 2.15))
    for g in range(1, 6):  # domain separators, two benchmarks per domain
        ax.axvline(2 * g - 0.5, color="#BBBBBB", linewidth=0.8)
    ax.axhline(0, color="#555555", linestyle="--", linewidth=1.2)
    offsets = [-0.24, -0.12, 0.0, 0.12, 0.24]
    for (key, _, color, marker), dx in zip(MODELS, offsets):
        ax.scatter([i + dx for i in range(len(TASKS))], [cells[(key, t)] for t in TASKS],
                   s=34, color=color, marker=marker, edgecolors="black", linewidths=0.6, zorder=3)
    ax.set_xticks(range(len(TASKS)))
    ax.set_xticklabels(LABELS, rotation=40, ha="right")
    ax.set_xlim(-0.6, len(TASKS) - 0.4)
    ax.set_ylabel(r"$\Delta\rho$ (%)")
    ax.spines[["top", "right"]].set_visible(False)
    handles = [Line2D([], [], linestyle="", marker=m, markersize=6, markerfacecolor=c,
                      markeredgecolor="black", label=lab) for _, lab, c, m in MODELS]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False,
              fontsize=10, handletextpad=0.1, columnspacing=0.8, borderaxespad=0.1)
    ax.set_ylim(-60, 100)
    ax.set_yticks([-50, 0, 50, 100])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
