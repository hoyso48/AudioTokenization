### Purpose
`eval/run_results_ft_suite.sh` runs finetuned result evaluation for one or more `run_dir` paths.

For each run directory, it does the following:

1. Runs `utils/update_legacy_config.py` on `hydra/config.yaml`.
2. Detects whether DTP is enabled and whether the selected `dtp_cls` supports `fixed_tau`.
3. If supported and `--tau_finetune` is enabled (default), runs `eval/dtp_stats_search.py`.
4. Runs `eval/eval.py` (using searched `fixed_tau` when applicable).

If DTP is disabled or the class does not support `fixed_tau`, it automatically falls back to eval-only mode.

### Prerequisites

```bash
cd /home/hoyso/projects/AudioTokenization
conda activate speech_eval
```

### Basic usage

Positional run directories:

```bash
bash eval/run_results_ft_suite.sh \
  results/default_PLE_50hz_vq16384 \
  results/default_fixedpattern_50hz_vq65536
```

List file input (`#` comments allowed):

```bash
bash eval/run_results_ft_suite.sh --run_list eval/run_dirs.txt
```

### Common options

- `--tau_finetune` / `--no_tau_finetune`
  - default: `--tau_finetune`
  - tau search runs only when DTP is enabled and class supports `fixed_tau`
- `--bootstrap_iters <int>`
  - default: `1`
  - number of bootstrap passes for `update_test_time` warm-start
- `--bootstrap_only` / `--no_bootstrap_only`
  - default: `--no_bootstrap_only`
  - optional: use bootstrap final `tau_end` directly (skip binary search)
- `--cfg_override <dotlist>` (repeatable)
  - applies to both tau-search and eval
- `--metrics <list|all>`
  - default: `all`
  - example: `--metrics stoi,pesq_wb,wer,utmos_v2`
  - if `utmos_v2` is requested and missing, scripts auto-install UTMOSv2 by default
- `--throughput_warmup_items <int>`
  - default: `5`
  - excludes initial torch-compile warmup iterations from throughput aggregation
- `--eval_subdir`, `--stats_subdir`, `--name_suffix`
  - controls output subdirectory names under each run directory
- `--force`
  - re-runs eval even if `<eval_out>/metrics.json` already has requested metrics

### Output files

For each run directory:

- Tau search output (when used)
  - `<run_dir>/<stats_subdir><name_suffix>/summary.json`
  - `<run_dir>/<stats_subdir><name_suffix>/trials.jsonl`
- Eval output
  - `<run_dir>/<eval_subdir><name_suffix>/metrics.json`
  - plus `manifest.jsonl`, `audio_metrics.json`, and related outputs
  - `manifest.jsonl` now includes prediction throughput fields (`prediction_elapsed_sec`, `prediction_samples`, `prediction_samples_per_sec`, `prediction_items_per_sec`)
  - `metrics.json` includes `utmos_v2` when selected and no longer includes `avg_sim`
  - when `use_dtp=True`, `metrics.json` also includes final realized `dtp_avg_r_mean/std` and `dtp_tau_used_mean/std`
  - if rerun with a new metric subset, only missing metric keys are added/updated (existing keys are preserved)

### Usage examples

Run with explicit `--run_dir` and cfg overrides:

```bash
bash eval/run_results_ft_suite.sh \
  --run_dir results/default_random_50hz_vq16384 \
  --cfg_override model.resampler.dtp_cls=BatchTopK \
  --cfg_override model.resampler.dtp_params.max_s=4 \
  --name_suffix _topk_ms4
```

Eval-only mode (skip tau search):

```bash
bash eval/run_results_ft_suite.sh \
  --no_tau_finetune \
  --run_dir results/default_random_50hz_vq16384
```

Custom tau search range:

```bash
bash eval/run_results_ft_suite.sh \
  --run_dir results/default_random_50hz_vq16384 \
  --tau_min 0.001 \
  --tau_max 2.0 \
  --tau_step 0.002 \
  --target_avg_r 0.5
```

### Notes

- Tau search assumes `avg_r` is monotonic with respect to `fixed_tau`.
- Bootstrap warm-start (`update_test_time`) is enabled by default when supported, with `--bootstrap_iters 1`.
- Bootstrap always computes `avg_r` on the full dataset (not capped by `--max_samples`).
- Bootstrap starts from previous final tau when available (prior `summary.json` tau_end, then current fixed_tau, then cached trials fallback).
- If `<stats_subdir>/summary.json` already exists and `--force` is not set, scripts reuse that `fixed_tau` instead of rerunning bootstrap/search.
- For `--metrics all`, scripts rerun only when at least one expected metric key is missing in `metrics.json`.
- By default, search continues after bootstrap warm-start; use `--bootstrap_only` to use final bootstrap `tau_end` directly.
- Search defaults to no-resume behavior (`--no_resume_search`).
- First metrics run may download HF weights (`facebook/hubert-large-ls960-ft`).
