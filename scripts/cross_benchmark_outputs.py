"""Write numbers.json and the LaTeX rows of Table 7 (per-benchmark means) and Table 9
(per-domain accuracy) for the cross-benchmark analysis.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.cross_benchmark_common import BENCHMARK_ORDER_TABLE3, to_json_safe


def full_matrix_rows(rows: dict) -> str:
    lines = []
    for domain, tasks in BENCHMARK_ORDER_TABLE3:
        tasks = [t for t in tasks if t in rows]
        if not tasks:
            continue
        if lines:
            lines.append(r"\midrule")
        lines.append(f"\\multirow{{{len(tasks)}}}{{*}}{{{domain}}}")
        for task in tasks:
            r = rows[task]
            if not r["n_models"]:
                values = [str(r["n_models"])] + ["--"] * 4
            else:
                values = [str(r["n_models"]), f"${r['mean_drho_pct']:+.1f}$",
                          f"${r['mean_dkeff']:+.1f}$", f"{r['mean_rho_sc']:.2f}", f"{r['mean_rho_pt']:.2f}"]
            lines.append("  & " + " & ".join([r["display_name"], r["answer_space"], *values]) + r" \\")
    return "\n".join(lines) + "\n"


def accuracy_rows(rows: dict) -> str:
    lines = []
    for domain, r in rows.items():
        label = "Sci/MC" if domain == "Science/MC" else domain
        vals = ([f"{r['mv_sc_mean']:.1f}\\%", f"{r['mv_pt_mean']:.1f}\\%",
                 f"${r['delta_acc_mean']:+.1f} \\pm {r['delta_acc_std']:.1f}$"]
                if r["n_cells"] > 1 else ["--", "--", "--"])
        lines.append(" & ".join([label, str(r["n_cells"]), *vals]) + r" \\")
    return "\n".join(lines) + "\n"


def compact(value: object) -> str:
    return json.dumps(to_json_safe(value), ensure_ascii=False, separators=(", ", ": ")).replace("|", r"\|")


def write_outputs(numbers: dict, output_dir: Path) -> None:
    # Prepare all content before writing. Reject unexpected non-finite valid results upstream.
    json_text = json.dumps(to_json_safe(numbers), indent=2, allow_nan=False) + "\n"
    d = numbers["derived"]
    texts = {"numbers.json": json_text, "table_full_matrix_rows.tex": full_matrix_rows(d["table3_rows"]),
             "table_pt_acc_delta_rows.tex": accuracy_rows(d["table8_rows"])}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in texts.items():
        (output_dir / name).write_text(content)
