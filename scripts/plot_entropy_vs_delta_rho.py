#!/usr/bin/env python3
"""Figure 3: answer diversity (mean n_unique/K under self-consistency) vs. Δρ for 12 benchmarks.

The per-benchmark mean Δρ (%) is read from Table 3's numbers.json
(derived_numbers table3_rows[*].mean_drho_pct); the answer diversity is computed
from the capability-matrix records. Prints the Pearson r and p of the plotted
points and writes entropy_vs_delta_rho.pdf to the output directory.
"""
import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import sys
_SRC = Path(__file__).resolve().parent.parent
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from diversity_combining.config import experiment_dir

CACHE_DIR = experiment_dir("cache")
OUTDIR = experiment_dir("figures")
NUMBERS_PATH = experiment_dir("cross_benchmark") / "numbers.json"

MODELS = ["qwen05b", "qwen7b", "qwen32b", "llama8b", "mistral7b"]
BENCHMARK_FILES = {
    "": {"gsm8k": "Math", "math": "Math"},
    "_qa": {"hotpotqa": "QA"},
    "_triviaqa": {"triviaqa": "QA"},
    "_arc": {"arc_challenge": "Science/MC"},
    "_mmlu": {"mmlu": "Science/MC"},
    "_mbpp": {"mbpp": "Code"},
    "_cruxeval": {"cruxeval": "Code"},
    "_hellaswag": {"hellaswag": "Commonsense"},
    "_winogrande": {"winogrande": "Commonsense"},
    "_boolq": {"boolq": "NLU"},
    "_drop": {"drop": "NLU"},
}

def load_delta_rho(numbers_path):
    """Per-benchmark mean delta-rho (%) from Table 3's numbers.json."""
    with open(numbers_path) as f:
        numbers = json.load(f)
    rows = numbers["derived_numbers"]["table3_rows"]
    return {task: row["mean_drho_pct"] for task, row in rows.items()}


DOMAIN_COLORS = {
    "Math": "#E69F00", "QA": "#D55E00", "Science/MC": "#0072B2",
    "Code": "#009E73", "Commonsense": "#CC79A7", "NLU": "#56B4E9",
}

TASK_LABELS = {
    "hotpotqa": "HotpotQA", "triviaqa": "TriviaQA",
    "arc_challenge": "ARC-C", "mmlu": "MMLU",
    "hellaswag": "HellaSwag", "winogrande": "WinoGrande",
    "boolq": "BoolQ", "drop": "DROP",
    "mbpp": "MBPP", "cruxeval": "CruxEval",
    "gsm8k": "GSM8K", "math": "MATH",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--numbers", type=Path, default=NUMBERS_PATH,
                        help="Table 3 numbers.json (default: %(default)s)")
    parser.add_argument("--out-dir", type=Path, default=OUTDIR,
                        help="directory for entropy_vs_delta_rho.pdf (default: %(default)s)")
    args = parser.parse_args()
    DELTA_RHO = load_delta_rho(args.numbers)

    plt.rcParams.update({
        "font.size": 18, "axes.labelsize": 18, "xtick.labelsize": 14,
        "ytick.labelsize": 14, "font.family": "sans-serif",
    })

    # Compute per-benchmark answer diversity under SC
    bench_diversity = {}
    bench_domain = {}

    for model in MODELS:
        for suffix, task_map in BENCHMARK_FILES.items():
            path = CACHE_DIR / f"{model}_capgated{suffix}.jsonl"
            if not path.exists():
                continue
            records = [json.loads(l) for l in open(path) if l.strip()]
            for task_name, domain in task_map.items():
                bench_domain[task_name] = domain
                sc_recs = [r for r in records if r.get("task") == task_name and r["method"] == "sc"]
                for r in sc_recs:
                    answers = r.get("all_answers", [])
                    if not answers:
                        correct = r.get("all_correct", [])
                        if correct:
                            answers = [str(c) for c in correct]
                    if not answers:
                        continue
                    K = len(answers)
                    n_unique = len(set(str(a).strip().lower() for a in answers))
                    bench_diversity.setdefault(task_name, []).append(n_unique / K)

    if set(bench_diversity) != set(DELTA_RHO):
        raise ValueError(f"benchmarks differ: records {sorted(bench_diversity)}, "
                         f"numbers.json {sorted(DELTA_RHO)}")

    fig, ax = plt.subplots(figsize=(5, 4))

    xs, ys = [], []
    for task in sorted(bench_diversity.keys()):
        if task not in DELTA_RHO:
            continue
        x = np.mean(bench_diversity[task])
        y = DELTA_RHO[task]
        domain = bench_domain[task]
        ax.scatter(x, y, color=DOMAIN_COLORS[domain], s=100,
                   edgecolors="black", linewidths=0.8, zorder=5)
        # Label each point
        offset = (5, 5)
        if task in ("math", "gsm8k"):
            offset = (5, -12)
        elif task == "cruxeval":
            offset = (-45, 8)
        elif task == "mbpp":
            offset = (5, -12)
        elif task == "boolq":
            offset = (-35, -12)
        ax.annotate(TASK_LABELS[task], (x, y), fontsize=9,
                    xytext=offset, textcoords="offset points")
        xs.append(x)
        ys.append(y)

    # Regression line
    r_val, p_val = stats.pearsonr(xs, ys)
    print(f"pearson_r={float(r_val)!r} p_value={float(p_val)!r} n_points={len(xs)}")
    z = np.polyfit(xs, ys, 1)
    x_line = np.linspace(min(xs) - 0.02, max(xs) + 0.02, 100)
    ax.plot(x_line, np.polyval(z, x_line), color="#888888", linestyle="--",
            linewidth=1.5, alpha=0.7)

    ax.text(0.95, 0.95, f"$r = {r_val:.2f}$, $p = {p_val:.3f}$",
            transform=ax.transAxes, fontsize=14, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"))

    ax.set_xlabel("Answer diversity (mean $n_{\\mathrm{unique}}/K$)")
    ax.set_ylabel("$\\Delta\\rho$ (\\%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Domain legend
    from matplotlib.lines import Line2D
    domain_handles = [Line2D([], [], marker="o", color="w", markerfacecolor=c,
                      markersize=8, markeredgecolor="black", markeredgewidth=0.5,
                      label=d) for d, c in DOMAIN_COLORS.items()]
    ax.legend(handles=domain_handles, loc="lower left", fontsize=9,
              framealpha=0.9, ncol=2)

    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "entropy_vs_delta_rho.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
