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
- `--cfg_override <dotlist>` (repeatable)
  - applies to both tau-search and eval
- `--eval_subdir`, `--stats_subdir`, `--name_suffix`
  - controls output subdirectory names under each run directory
- `--force`
  - re-runs eval even if `<eval_out>/metrics.json` exists

### Output files

For each run directory:

- Tau search output (when used)
  - `<run_dir>/<stats_subdir><name_suffix>/summary.json`
  - `<run_dir>/<stats_subdir><name_suffix>/trials.jsonl`
- Eval output
  - `<run_dir>/<eval_subdir><name_suffix>/metrics.json`
  - plus `manifest.jsonl`, `audio_metrics.json`, and related outputs

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
- Bootstrap warm-start (`update_test_time`) is enabled by default when supported.
- Search defaults to no-resume behavior (`--no_resume_search`).
- First metrics run may download HF weights (`facebook/hubert-large-ls960-ft`).
