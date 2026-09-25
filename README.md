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
  reproduce_tables.py            Tables 1, 2, 4, 9, 10, 11 and Figure 2
  reproduce_cross_benchmark.py   Table 3, Table 8 and the cross-benchmark statistics
  aggregate_canonical_sc.py      estimators: majority vote, beta-binomial and binomial predictions
  aggregate_kvar_v2.py           record loader and validator, c-hat, K_eff, exact-K table
  aggregate_5seed.py             capability-matrix loader and pooled estimators
  analyze_mv_vs_weighted.py      accuracy-weighted voting per prompt-template cell
  analyze_block_covariance.py    block-covariance K_eff per prompt-template cell
  plot_entropy_vs_delta_rho.py   Figure 3
  scoring_v2.py, tasks12.py      task scorers and answer normalization
  run_sc_kvar_v2.py              generation driver for the K=32 self-consistency records
  reextract_cache.py             re-applies the answer extractor to existing records
  run_capability_gated*.py       generation drivers for the 5 x 12 capability matrix
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
hf download OniReimu/DiversityCombining --repo-type dataset --local-dir data
export DC_CACHE_ROOT=data
```

Records omit wall-clock fields. All locations are set by environment variables
(defaults in parentheses):

| Variable | Default | Contents |
|---|---|---|
| `DC_CACHE_ROOT` | `./data` | generation records: `sc_records/`, `reasoning_records/`, `capability_matrix/` |
| `DC_RESULTS_ROOT` | `./results` | outputs: `tables/`, `cross_benchmark/`, `figures/` |
| `DC_REFERENCE_ROOT` | `./reference` | `kvar_v2_exact_k.csv`, read by one test (included in the download as `data/reference/`) |
| `DC_RECORDS_DIR` | unset | optional input or output directory override for `run_sc_kvar_v2.py` and `aggregate_kvar_v2.py` |

- `sc_records/` holds `{model}_sc_kvar_v2_{task}.jsonl` for qwen7b, llama8b and
  mistral7b on GSM8K and llama8b on HotpotQA and BoolQ.
- `reasoning_records/` holds `qwen3_5_9b_sc_kvar_v2_gsm8k.jsonl`.
- `capability_matrix/` holds `{model}_capgated{suffix}.jsonl` for the five
  instruction models and the eleven suffixes `""`, `_qa`, `_triviaqa`, `_arc`,
  `_mmlu`, `_mbpp`, `_cruxeval`, `_hellaswag`, `_winogrande`, `_boolq`, `_drop`.

## Reproducing the paper

```text
export DC_CACHE_ROOT="$PWD/data" DC_RESULTS_ROOT="$PWD/results" DC_REFERENCE_ROOT="$PWD/data/reference"
python scripts/reproduce_tables.py
python scripts/reproduce_cross_benchmark.py
python scripts/plot_entropy_vs_delta_rho.py
pytest -q
python scripts/aggregate_kvar_v2.py --in-dir "$DC_CACHE_ROOT/sc_records" --out-dir "$DC_RESULTS_ROOT/reference"
```

`reproduce_tables.py` writes `results/tables/numbers.json`, the LaTeX rows
`rows_diversity.tex`, `rows_cross_arch.tex`, `rows_adaptive_k.tex`,
`rows_pilot_sensitivity.tex`, `rows_eps_sensitivity.tex` and `bb_calibration_cr.pdf`.
`reproduce_cross_benchmark.py` writes `results/cross_benchmark/numbers.json` and
`rows_table3.tex`. `plot_entropy_vs_delta_rho.py` reads that `numbers.json` and the
capability-matrix records, prints the Pearson r and p of the plotted points and
writes `results/figures/entropy_vs_delta_rho.pdf`. The exact-K table
`kvar_v2_exact_k.csv` that the tests compare against ships with the records; the
last command regenerates it from `sc_records/` (byte-identical to
`data/reference/kvar_v2_exact_k.csv`), together with the gate, adaptive-K and
bootstrap tables. The tests read the outputs of the first two commands, so run
them first. On a laptop CPU the two reproduction
scripts take several minutes each; none of the commands needs a GPU.

| Paper item | Script | Output (`numbers.json` key or file) |
|---|---|---|
| Table 1, effective diversity (Qwen2.5-7B, GSM8K) | `reproduce_tables.py` | `tab_diversity`; `rows_diversity.tex` |
| Table 2, cross-architecture calibration | `reproduce_tables.py` | `tab_cross_arch`; `rows_cross_arch.tex` |
| Figure 2, held-out beta-binomial calibration | `reproduce_tables.py` | `heldout_bb`; `bb_calibration_cr.pdf` |
| Table 3, 5 x 12 prompt-template matrix | `reproduce_cross_benchmark.py` | `derived_numbers.table3_rows`, `cells`; `rows_table3.tex` |
| Section 5 headline counts and Mann-Whitney test | `reproduce_cross_benchmark.py` | `derived_numbers.valid_cells_count`, `decorrelates_count`, `mean_drho_pct`, `mean_dkeff`, `domain_means`, `mann_whitney` |
| Figure 3, answer diversity vs. decorrelation | `plot_entropy_vs_delta_rho.py` | `entropy_vs_delta_rho.pdf` (points: `derived_numbers.table3_rows[*].mean_drho_pct` against mean SC answer diversity; the printed r and p equal `derived_numbers.entropy_predictor` from `reproduce_cross_benchmark.py`) |
| Table 4, Adaptive-K | `reproduce_tables.py` | `tab_adaptive_k`; `rows_adaptive_k.tex` |
| Appendix I, accuracy-weighted voting | `reproduce_cross_benchmark.py` | `derived_numbers.weighted_mv_stats` (per-cell listing: `analyze_mv_vs_weighted.py`) |
| Appendix J, pilot cost amortization | `reproduce_tables.py` | `amortized_accounting` |
| Appendix J, block-covariance K_eff | `reproduce_cross_benchmark.py` | `derived_numbers.block_covariance_stats` (per-cell listing: `analyze_block_covariance.py`) |
| Appendix K, bootstrap precision of the decorrelation | `reproduce_cross_benchmark.py` | `derived_numbers.bootstrap_stats`; per-cell `drho_ci_low`, `drho_ci_high` |
| Table 8 (Appendix L), accuracy deltas | `reproduce_cross_benchmark.py` | `derived_numbers.table8_rows` |
| Table 9 (Appendix M), reasoning model | `reproduce_tables.py` | `reasoning_model_e1` |
| Table 10 (Appendix P), pilot-size sensitivity | `reproduce_tables.py` | `tab_pilot_sensitivity`; `rows_pilot_sensitivity.tex` |
| Table 11 (Appendix S), threshold sensitivity | `reproduce_tables.py` | `tab_eps_sensitivity`; `rows_eps_sensitivity.tex` |
| Figure 1, Tables 5-7, other appendices | none | analytical or descriptive content |

`analyze_mv_vs_weighted.py` and `analyze_block_covariance.py` print per-cell
listings and score every task from the stored answers; the published values
come from `reproduce_cross_benchmark.py`, which scores GSM8K with the numeric
scorer and MATH with the MATH-500 canonical scorer.

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

The capability-matrix records (Table 3) store answers, seeds and K but no
sampling fields:

| Model | Record files | Arms | Seeds present | K | Records |
|---|---|---|---|---|---|
| Qwen/Qwen2.5-0.5B-Instruct (`qwen05b`) | 11 | sc, sc_prompttpl | 42, 123, 456, 789, 1024 | 8 | 6000 |
| Qwen/Qwen2.5-7B-Instruct (`qwen7b`) | 11 | sc, sc_prompttpl | 42, 123, 456, 789, 1024 | 8 | 5673 |
| Qwen/Qwen2.5-32B-Instruct (`qwen32b`) | 11 | sc, sc_prompttpl | 42, 123, 456, 789, 1024 | 8 | 5743 |
| meta-llama/Llama-3.1-8B-Instruct (`llama8b`) | 11 | sc, sc_prompttpl | 42, 123, 456, 789, 1024 | 8 | 5951 |
| mistralai/Mistral-7B-Instruct-v0.3 (`mistral7b`) | 11 | sc, sc_prompttpl | 42, 123, 456, 789, 1024 | 8 | 6000 |

Their sampling settings are the constants shared by all ten capability drivers (read from the driver sources): K = 8, 50 instances per seed, temperature 0.7, top_p 0.95, max_new_tokens 2048. The drivers load models in bfloat16 with the default attention implementation. Their optional 4-bit path requires `bitsandbytes`, which is not part of the pinned environment, so every model, Qwen2.5-32B included, runs in bfloat16. The partial arms (fewer than five seeds or 50 instances) are listed in `PARTIAL_ARM_SEED_COUNTS` in `scripts/reproduce_cross_benchmark.py` and pooled as they are.

### Generation commands

The K=32 self-consistency records were written by `run_sc_kvar_v2.py`. The
primary records (`sc_records`) were then re-extracted with `reextract_cache.py`,
which stamps the extractor version into their provenance (v2.1); the shipped
extractor is v2.2, so re-extracting raw records with it is not guaranteed to
reproduce the v2.1 answers exactly. The reasoning-model records were written
directly by the driver at extractor v2.2. For example:

```text
python scripts/run_sc_kvar_v2.py --model qwen7b --task gsm8k --seeds 42 123 456 789 1024 --n 100 --k 32 --temperature 0.7 --out-dir "$DC_CACHE_ROOT/sc_raw"
python scripts/reextract_cache.py --in "$DC_CACHE_ROOT/sc_raw" --out "$DC_CACHE_ROOT/sc_records" --task gsm8k
python scripts/run_sc_kvar_v2.py --model qwen3_5_9b --task gsm8k --seeds 42 123 456 --n 100 --k 32 --temperature 0.7 --out-dir "$DC_CACHE_ROOT/reasoning_records"
```

Reasoning models default to a 16384-token budget. HotpotQA and BoolQ use seeds
`42 123 456`. The capability-matrix records were written by the ten
`run_capability_gated*.py` drivers into `$DC_CACHE_ROOT/capability_matrix`, for example:

```text
python scripts/run_capability_gated.py --model qwen7b --task gsm8k --method both --seeds 42 123 456 789 1024
python scripts/run_capability_gated_qa.py --model qwen7b --task triviaqa --method both --seeds 42 123 456 789 1024
python scripts/run_capability_gated_mc.py --model qwen7b --task arc --method both --seeds 42 123 456 789 1024
```

The remaining drivers (`_mmlu`, `_mbpp`, `_cruxeval`, `_hellaswag`,
`_winogrande`, `_boolq`, `_drop`) each serve one benchmark and take only
`--model`, `--method` and `--seeds`, for example
`python scripts/run_capability_gated_mmlu.py --model qwen7b --method both --seeds 42 123 456 789 1024`.
The MBPP driver
executes generated code in a sandboxed temporary directory.

## Tests

`pytest -q` runs the tests in `tests/` against the records under
`DC_CACHE_ROOT`, the reference table under `DC_REFERENCE_ROOT` and the outputs
under `DC_RESULTS_ROOT`. `tests/test_data_root.py` fails when `DC_CACHE_ROOT` is
unset or lacks a record file.

## License

MIT (see `LICENSE`). Third-party code is listed in `THIRD_PARTY_NOTICES.md`.
