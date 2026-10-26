# Diversity Combining for Multi-Path LLM Reasoning

Code for the NeurIPS 2026 paper *Diversity Combining for Multi-Path LLM Reasoning*.
It measures the pairwise correctness correlation `c` among sampled reasoning
paths, the effective number of independent votes `K_eff = K / (1 + (K - 1) c)`,
beta-binomial predictions of majority-vote accuracy, the Adaptive-K operating
point, and the decorrelation produced by prompt-template perturbation across
5 models and 12 benchmarks.

## Contents

```text
diversity_combining/     shared package: paths (config.py), data loading, scoring helpers,
                         method and metric implementations
scripts/
  reproduce_tables.py            Tables 1, 2, 3, 10, 11, 12 and Figure 2
  reproduce_cross_benchmark.py   Tables 7 and 9 and the cross-benchmark statistics
  plot_drho_cells.py             Figure 3
  cell_matrix_rows.py            Table 8 (all 60 model-benchmark cells)
  matched_arms.py                loads the matched prompt-template and self-consistency arms
  cross_benchmark_outputs.py     writes numbers.json and the LaTeX rows of Tables 7 and 9
  cross_benchmark_common.py      benchmark metadata, scorers, block-covariance matrix builder
  correlation_estimators.py      pooled c-hat (Eq. 3)
  aggregate_canonical_sc.py      estimators: majority vote, beta-binomial and binomial predictions
  aggregate_kvar_v2.py           record loader and validator, c-hat, K_eff, exact-K table
  aggregate_5seed.py             pairwise correlation, majority vote and K_eff estimators
  analyze_mv_vs_weighted.py      uniform and accuracy-weighted majority vote
  scoring_v2.py, tasks12.py      task scorers and answer normalization
  run_sc_kvar_v2.py              generation driver for the K=32 self-consistency records
  run_prompt_template_gls.py     generation driver for the K=8 prompt-template records
  reextract_cache.py             re-applies the answer extractor to existing records
tests/                   tests for the two reproduction scripts and the data location
```

## Installation

Python 3.11 or newer. The dependency versions are pinned to the environment
that produced the paper's numbers.

```text
uv venv
uv pip install -e ".[test]"
```

or `python -m pip install -e ".[test]"` in a virtual environment.

## Data layout

The generation records are published as a Hugging Face dataset. Download them
into `data/` and point `DC_CACHE_ROOT` at that directory:

```text
hf download UTSCybeR/DiversityCombining --repo-type dataset --local-dir data
export DC_CACHE_ROOT=data
```

Records omit wall-clock fields. All locations are set by environment variables
(defaults in parentheses):

| Variable | Default | Contents |
|---|---|---|
| `DC_CACHE_ROOT` | `./data` | generation records: `sc_records/`, `reasoning_records/`, `pt_records/` |
| `DC_RESULTS_ROOT` | `./results` | outputs: `tables/`, `cross_benchmark/`, `figures/` |
| `DC_REFERENCE_ROOT` | `./reference` | `kvar_v2_exact_k.csv`, read by one test (included in the download as `data/reference/`) |
| `DC_RECORDS_DIR` | unset | optional input or output directory override for `run_sc_kvar_v2.py`, `run_prompt_template_gls.py` and `aggregate_kvar_v2.py` |

- `sc_records/` holds the K=32 self-consistency records
  `{model}_sc_kvar_v2_{task}.jsonl` of the five instruction models
  (`qwen05b`, `qwen7b`, `qwen32b`, `llama8b`, `mistral7b`) on the twelve
  benchmarks. Five of them (qwen7b, llama8b and mistral7b on GSM8K, llama8b on
  HotpotQA and BoolQ) are also the primary cells of Tables 1 to 3.
- `pt_records/{task}/` holds the K=8 prompt-template records
  `{model}_prompt_template_{task}.jsonl` of the same 60 cells
  (`{model}_prompt_template.jsonl` for GSM8K).
- `reasoning_records/` holds `qwen3_5_9b_sc_kvar_v2_gsm8k.jsonl`.
- Records keep the full text of every path (`all_traces`) except on HellaSwag:
  the HellaSwag records in `sc_records/` and `pt_records/hellaswag/` carry the
  extracted answers only, since the model text can restate HellaSwag context
  text. The analysis reads `all_answers` for HellaSwag, so every number is
  reproduced from them.

## Reproducing the paper

```text
export DC_CACHE_ROOT="$PWD/data" DC_RESULTS_ROOT="$PWD/results" DC_REFERENCE_ROOT="$PWD/data/reference"
python scripts/reproduce_tables.py
python scripts/reproduce_cross_benchmark.py
python scripts/plot_drho_cells.py
python scripts/cell_matrix_rows.py
pytest -q
python scripts/aggregate_kvar_v2.py --in-dir "$DC_CACHE_ROOT/sc_records" --out-dir "$DC_RESULTS_ROOT/reference"
```

`reproduce_tables.py` writes `results/tables/numbers.json`, the LaTeX rows
`rows_diversity.tex`, `rows_cross_arch.tex`, `rows_adaptive_k.tex`,
`rows_pilot_sensitivity.tex`, `rows_eps_sensitivity.tex` and `bb_calibration_cr.pdf`.
`reproduce_cross_benchmark.py` writes `results/cross_benchmark/numbers.json`,
`table_full_matrix_rows.tex` (Table 7) and `table_pt_acc_delta_rows.tex` (Table 9).
`plot_drho_cells.py` and `cell_matrix_rows.py` read that `numbers.json` and write
`results/figures/drho_cells.pdf` (Figure 3) and
`results/cross_benchmark/table_cell_matrix_rows.tex` (Table 8). The exact-K table
`kvar_v2_exact_k.csv` that the tests compare against ships with the records; the
last command regenerates it from `sc_records/` together with the gate, adaptive-K
and bootstrap tables, and its rows for the five primary cells are byte-identical to
`data/reference/kvar_v2_exact_k.csv`. The tests read the outputs of the first two commands, so run
them first. On a laptop CPU the two reproduction scripts take several minutes
each; none of the commands needs a GPU.

| Paper item | Script | Output (`numbers.json` key or file) |
|---|---|---|
| Table 1, effective diversity (Qwen2.5-7B, GSM8K) | `reproduce_tables.py` | `tab_diversity`; `rows_diversity.tex` |
| Table 2, cross-architecture calibration | `reproduce_tables.py` | `tab_cross_arch`; `rows_cross_arch.tex` |
| Figure 2, held-out beta-binomial calibration | `reproduce_tables.py` | `heldout_bb`; `bb_calibration_cr.pdf` |
| Table 3, Adaptive-K | `reproduce_tables.py` | `tab_adaptive_k`; `rows_adaptive_k.tex` |
| Section 5.2.4 counts, domain medians, Mann-Whitney test, answer diversity | `reproduce_cross_benchmark.py` | `derived.valid_cells_count`, `decorrelates_count`, `mean_drho_pct`, `mean_dkeff`, `exceptions`, `domain_means`, `mann_whitney`, `entropy_predictor` |
| Figure 3, per-cell decorrelation | `plot_drho_cells.py` | `drho_cells.pdf` (points: `cells[*].delta_rho_pct`) |
| Appendix I, accuracy-weighted voting and the SC comparison | `reproduce_cross_benchmark.py` | `derived.weighted_mv_stats`, `derived.sc_weighting` |
| Appendix J, pilot cost amortization | `reproduce_tables.py` | `amortized_accounting` |
| Appendix J, block-covariance K_eff | `reproduce_cross_benchmark.py` | `derived.block_covariance_stats`; per-cell `block` |
| Appendix K, bootstrap precision of the decorrelation | `reproduce_cross_benchmark.py` | `derived.bootstrap_stats`; per-cell `delta_rho_ci95` |
| Table 7 (Appendix L), per-benchmark means | `reproduce_cross_benchmark.py` | `derived.table3_rows`; `table_full_matrix_rows.tex` |
| Table 8 (Appendix L), all 60 cells | `cell_matrix_rows.py` | `cells`; `table_cell_matrix_rows.tex` |
| Table 9 (Appendix M), accuracy deltas | `reproduce_cross_benchmark.py` | `derived.table8_rows`; `table_pt_acc_delta_rows.tex` |
| Table 10 (Appendix N), reasoning model | `reproduce_tables.py` | `reasoning_model_e1` |
| Table 11 (Appendix Q), pilot-size sensitivity | `reproduce_tables.py` | `tab_pilot_sensitivity`; `rows_pilot_sensitivity.tex` |
| Table 12 (Appendix T), threshold sensitivity | `reproduce_tables.py` | `tab_eps_sensitivity`; `rows_eps_sensitivity.tex` |
| Figure 1, Tables 4-6, other appendices | none | analytical or descriptive content |

The keys `derived.table3_rows` and `derived.table8_rows` keep the names of the
analysis code; they contain the rows of Tables 7 and 9.

## Generation protocol

The primary and reasoning-model cells record their sampling settings in every
record's `provenance` field. The table lists the distinct values found in the
records used by the paper; "not recorded" means the field is absent.

| Cell | Model | Seeds | Instances per seed | K | Temperature | top_p | max_new_tokens | Attention implementation | Scorer | Extractor version |
|---|---|---|---|---|---|---|---|---|---|---|
| `sc_records/qwen7b_sc_kvar_v2_gsm8k.jsonl` | Qwen/Qwen2.5-7B-Instruct | 42, 123, 456, 789, 1024 | 100 | 32 | 0.7 | 0.95 | 1024 | not recorded | gsm8k_numeric | v2.1 |
| `sc_records/llama8b_sc_kvar_v2_gsm8k.jsonl` | meta-llama/Llama-3.1-8B-Instruct | 42, 123, 456, 789, 1024 | 100 | 32 | 0.7 | 0.95 | 1024 | not recorded | gsm8k_numeric | v2.1 |
| `sc_records/mistral7b_sc_kvar_v2_gsm8k.jsonl` | mistralai/Mistral-7B-Instruct-v0.3 | 42, 123, 456, 789, 1024 | 100 | 32 | 0.7 | 0.95 | 1024 | not recorded | gsm8k_numeric | v2.1 |
| `sc_records/llama8b_sc_kvar_v2_hotpotqa.jsonl` | meta-llama/Llama-3.1-8B-Instruct | 42, 123, 456 | 100 | 32 | 0.7 | 0.95 | 512 | not recorded | hotpotqa_token_f1 | v2.1 |
| `sc_records/llama8b_sc_kvar_v2_boolq.jsonl` | meta-llama/Llama-3.1-8B-Instruct | 42, 123, 456 | 100 | 32 | 0.7 | 0.95 | 256 | not recorded | boolq_yesno | v2.1 |
| `reasoning_records/qwen3_5_9b_sc_kvar_v2_gsm8k.jsonl` | Qwen/Qwen3.5-9B | 42, 123, 456 | 100 | 32 | 0.7 | 0.95 | 16384 | default | gsm8k_numeric | v2.2 |

Cross-benchmark cells (five models per row: Qwen/Qwen2.5-0.5B-Instruct (`qwen05b`), Qwen/Qwen2.5-7B-Instruct (`qwen7b`), Qwen/Qwen2.5-32B-Instruct (`qwen32b`), meta-llama/Llama-3.1-8B-Instruct (`llama8b`), mistralai/Mistral-7B-Instruct-v0.3 (`mistral7b`)). The self-consistency (SC) arm holds K=32 paths per record, of which the analysis uses the first 8; the prompt-template (PT) arm holds K=8 paths, path k under template k. Values are the distinct ones found in the records of the five models (provenance first, then the record's own field). `all_answers_v21` (present in all, none or the given share of the records) holds the answers before a re-extraction with `reextract_cache.py`, which writes the new answers to `all_answers` and the extractor version to the provenance.

| Benchmark | Arm | Seeds | Instances per seed | K | Temperature | top_p | Max new tokens | Attention implementation | Extractor version | `all_answers_v21` |
|---|---|---|---|---|---|---|---|---|---|---|
| triviaqa | SC | 42, 123, 456, 789, 1024 | 100 | 32 | 0.7 | 0.95 | 512 | default | v2.2 | yes |
| triviaqa | PT | 42, 123, 456, 789, 1024 | 100 | 8 | 0.7 | not recorded | 512 | default | v2.2 | yes |
| hotpotqa | SC | 42, 123, 456 | 100 | 32 | 0.7 | 0.95 | 512 | not recorded | v2.1 | no |
| hotpotqa | PT | 42, 123, 456 | 100 | 8 | 0.7 | not recorded | 512 | default | v2.2 | yes |
| arc_challenge | SC | 42, 123, 456 | 100 | 32 | 0.7 | 0.95 | 1024 | default | v2.2 | no |
| arc_challenge | PT | 42, 123, 456 | 100 | 8 | 0.7 | not recorded | 1024 | default | not recorded | no |
| mmlu | SC | 42, 123, 456 | 100 | 32 | 0.7 | 0.95 | 4096 | not recorded | v2.1 | no |
| mmlu | PT | 42, 123, 456 | 100 | 8 | 0.7 | not recorded | 4096 | default | v2.2 | yes |
| hellaswag | SC | 42, 123, 456 | 100 | 32 | 0.7 | 0.95 | 512 | not recorded | v2.1 | no |
| hellaswag | PT | 42, 123, 456 | 100 | 8 | 0.7 | not recorded | 512 | default | v2.2 | yes |
| winogrande | SC | 42, 123, 456 | 100 | 32 | 0.7 | 0.95 | 512 | default | v2.2 | no |
| winogrande | PT | 42, 123, 456 | 100 | 8 | 0.7 | not recorded | 512 | default | not recorded | no |
| drop | SC | 42, 123, 456, 789, 1024 | 100 | 32 | 0.7 | 0.95 | 512 | default | v2.2 | yes |
| drop | PT | 42, 123, 456, 789, 1024 | 100 | 8 | 0.7 | not recorded | 512 | default | v2.2 | yes |
| boolq | SC | 42, 123, 456 | 100 | 32 | 0.7 | 0.95 | 256 | not recorded | v2.1 | no |
| boolq | PT | 42, 123, 456 | 100 | 8 | 0.7 | not recorded | 256 | default | v2.2 | yes |
| mbpp | SC | 42, 123, 456, 789, 1024 | 100 | 32 | 0.7 | 0.95 | 1024 | default | v2.2 | yes |
| mbpp | PT | 42, 123, 456, 789, 1024 | 100 | 8 | 0.7 | not recorded | 1024 | default | v2.2 | yes |
| cruxeval | SC | 42, 123, 456, 789, 1024 | 100 | 32 | 0.7 | 0.95 | 1024 | default | v2.2 | yes |
| cruxeval | PT | 42, 123, 456, 789, 1024 | 100 | 8 | 0.7 | not recorded | 1024 | default | v2.2 | yes |
| gsm8k | SC | 42, 123, 456, 789, 1024 | 100 | 32 | 0.7 | 0.95 | 1024 | not recorded | v2.1 | no |
| gsm8k | PT | 42, 123, 456, 789, 1024 | 100 | 8 | 0.7 | not recorded | 1024 | default | not recorded | no |
| math | SC | 42, 123, 456, 789, 1024 | 100 | 32 | 0.7 | 0.95 | 2048 | default | v2.2 | yes |
| math | PT | 42, 123, 456, 789, 1024 | 100 | 8 | 0.7 | not recorded | 2048 | default | v2.2 | yes |

The PT records do not store `top_p`; the prompt-template driver samples with top_p = 0.95 (read from the driver source). Both drivers load the models in bfloat16. The GSM8K PT records store no provenance; `scripts/matched_arms.py` re-extracts their answers from `all_traces` with the shipped extractor (`run_sc_kvar_v2.extract_answer`).

### Generation commands

The K=32 self-consistency records were written by `run_sc_kvar_v2.py`. The
primary records were then re-extracted with `reextract_cache.py`, which stamps
the extractor version into their provenance (v2.1); the shipped extractor is
v2.2, so re-extracting raw records with it is not guaranteed to reproduce the
v2.1 answers exactly. The reasoning-model records were written directly by the
driver at extractor v2.2. For example:

```text
python scripts/run_sc_kvar_v2.py --model qwen7b --task gsm8k --seeds 42 123 456 789 1024 --n 100 --k 32 --temperature 0.7 --out-dir "$DC_CACHE_ROOT/sc_raw"
python scripts/reextract_cache.py --in "$DC_CACHE_ROOT/sc_raw" --out "$DC_CACHE_ROOT/sc_records" --task gsm8k
python scripts/run_sc_kvar_v2.py --model qwen3_5_9b --task gsm8k --seeds 42 123 456 --n 100 --k 32 --temperature 0.7 --out-dir "$DC_CACHE_ROOT/reasoning_records"
```

Reasoning models default to a 16384-token budget. HotpotQA and BoolQ use seeds
`42 123 456`. The prompt-template records were written by
`run_prompt_template_gls.py` (one template per slot, K=8), for example:

```text
python scripts/run_prompt_template_gls.py --model qwen7b --tasks arc_challenge --seeds 42 123 456 --n-instances 100 --max_tokens 1024 --attn-implementation default
python scripts/run_sc_kvar_v2.py --model qwen7b --task arc_challenge --seeds 42 123 456 --n 100
```

The second command writes the matching self-consistency arm. The settings of
every benchmark are listed in the table above; records that carry
`all_answers_v21` were passed through `reextract_cache.py` after generation.
`reextract_cache.py` re-extracts answers from `all_traces`, so it applies to
every published record except the HellaSwag records, which carry extracted
answers only.

## Tests

`pytest -q` runs the tests in `tests/` against the records under
`DC_CACHE_ROOT`, the reference table under `DC_REFERENCE_ROOT` and the outputs
under `DC_RESULTS_ROOT`. `tests/test_data_root.py` fails when `DC_CACHE_ROOT` is
unset or lacks a record file.

## License

MIT (see `LICENSE`). Third-party code is listed in `THIRD_PARTY_NOTICES.md`.
