### Purpose
This folder contains a repeatable runner for your **results0117** experiments:

- **Step A**: run `eval/dtp_stats_search.py` to find a `fixed_tau` that hits **`target_avg_r=0.5`**
- **(Bootstrap)**: the suite first runs a single pass with **`update_test_time=True`** so the DTP controller can
  adapt `tau` towards the target ratio, then uses that as a good starting point for the discrete (0.001-step) search.
- **Step B**: run `eval/eval.py` using that `fixed_tau` via `--cfg_override model.resampler.dtp_params.fixed_tau=...`

This automates exactly the workflow you described for run dirs (1)–(5), including the 6-way grid for (5).

### Prerequisites
- Run from: `/home/hoyso/projects/AudioTokenization`
- Activate env first:

```bash
conda activate speech_eval
```

The runner checks that `torch`, `torchaudio`, and `omegaconf` import successfully.

### Run

```bash
cd /home/hoyso/projects/AudioTokenization
bash eval/run_results0117_ft_suite.sh
```

### Legacy config auto-conversion (important)
Before running any tau-search/eval, the runner automatically executes:
- `python utils/update_legacy_config.py --path <run_dir>/hydra/config.yaml`

This converts legacy quantizer fields under `model.codec_decoder` into the new `model.quantizer` section
and creates a backup `config_legacy.yaml` next to the original `config.yaml`.

### What gets created (outputs)
For each run directory:
- **Tau search outputs** (from `dtp_stats_search.py`):
  - `<output_dir>/summary.json` (contains `best.fixed_tau` and `best.avg_r_mean`)
  - `<output_dir>/trials.jsonl` (all tried taus)
- **Eval outputs** (from `eval.py`):
  - `<output_dir>/metrics.json` (final metrics)
  - plus `manifest.jsonl`, `audio_metrics.json`, reconstructed wavs, etc. depending on `eval.py` defaults.

### Folder naming convention (as requested)
The suite follows:
- `dtp_stats_ft` / `eval_ft` for run dirs **1–3**
- `eval_ft` for run dir **4** (no overrides)
- For run dir **5**:
  - `dtp_stats_{algorithm}_ft` and `eval_{algorithm}_ft`
  - optional `_ms4` suffix when `--cfg_override model.resampler.dtp_params.max_s=4` is used

Algorithms used:
- **PLE**: default (no `dtp_cls` override) → `*_ple_*`
- **TopK**: `--cfg_override model.resampler.dtp_cls=BatchTopK` → `*_topk_*`
- **Greedy**: `--cfg_override model.resampler.dtp_cls=BatchGreedy` → `*_greedy_*`

### How to extend
- **Change target avg_r**: edit `TARGET_AVG_R` in `run_results0117_ft_suite.sh`
- **Change tau search range/step**: edit `TAU_MIN`, `TAU_MAX`, `TAU_STEP`
- **Add more overrides**:
  - Add more strings to the `run_search_and_eval ... "<override>" "<override>" ...` call.
  - Each string must be a valid OmegaConf dotlist assignment, e.g.:
    - `model.resampler.dtp_params.sample_prob=1`
    - `dataset.multiple_of=320`

### Notes / gotchas
- The search assumes `avg_r` is **monotonic** in `fixed_tau` for the selected DTP method.
  - If that assumption breaks for some configuration, `dtp_stats_search.py` may fail to bracket the target.
  - In that case, widen `TAU_MAX` or adjust the configuration (or switch to a scan-based search).
- The runner enables bootstrap explicitly:
  - `dtp_stats_search.py --bootstrap_update_test_time --bootstrap_override_update_test_time`
  - This uses the internal Robbins–Monro controller to estimate a good initial tau (see `tau_progress.end` in `summary.json`).
- The runner also uses `--no_resume` so that if you re-run the suite after code/config changes,
  it won’t reuse stale `trials.jsonl` results.

### Why `pytorch_model.bin` gets downloaded
`eval/eval.py` loads HuggingFace Transformers models (HuBERT ASR: `facebook/hubert-large-ls960-ft`) during the **metrics stage**,
so the first time you run it, it downloads weights like `pytorch_model.bin` and caches them.
Subsequent runs should reuse the cache.


