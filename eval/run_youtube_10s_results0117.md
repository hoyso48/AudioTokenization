### Goal
Download specific YouTube audio segments as **10s MP3 clips** (filenames include the **video title**), resample them to **16 kHz mono WAV**, then run `eval/eval.py` on your **results0117** run directories and produce:

- **Reconstructions** (`pred_16k/*.wav`)
- **Metrics** (`metrics.json`, `audio_metrics.json`)

All evaluations are done with the **pre-computed** DTP tau from each run’s `dtp_stats*/summary.json` (**`best.fixed_tau`**), applied via:

- `--cfg_override model.resampler.dtp_params.fixed_tau=<best.fixed_tau>`

### Script
Run:

```bash
cd /home/hoyso/projects/AudioTokenization
bash eval/run_youtube_10s_results0117.sh
```

### Inputs (YouTube → clips)
The script hard-codes these pairs (URL, start time), crops **10 seconds** from each:

- `https://youtu.be/9TxtTF_dUHQ?si=4kPSZ5HNpA5iYxHZ` @ `2:05`
- `https://youtu.be/jNQXAC9IVRw?si=k7sNX3mM8lYMW3cU` @ `0`
- `https://youtu.be/sP5ElraFHHE?si=1DUetO19MwoFLKW3` @ `44`
- `https://youtu.be/gfHEOL-sDy4?si=vZAtCdpFv1TrHWjx` @ `26`
- `https://youtu.be/g4xoe5Ccuzc?si=v9JdeTYoZk-Y5P9P` @ `0`
- `https://youtu.be/YWq_CGT0QPY?si=G2UmlikXbSM-kzTs` @ `1:30`

### Clip outputs
Created under:

- `eval/youtube_10s/raw_mp3/`: full audio MP3 (downloaded by `yt-dlp`)
- `eval/youtube_10s/mp3_10s/`: 10s MP3 clips (cropped with `ffmpeg`)
- `eval/youtube_10s/wav_16k/`: 10s WAV clips (**mono**, **16 kHz**) used as `eval/eval.py --input`
- `eval/youtube_10s/manifest.jsonl`: mapping from URL/start → produced files

### Eval outputs (per run directory)
The script writes new eval folders (so it won’t overwrite your existing `eval_ft/`):

- `results/results0117/default_PLE_25hz_vq16384/eval_youtube_10s/`
  - tau from: `.../dtp_stats_ft/summary.json` (best.fixed_tau)
- `results/results0117/default_PLE_50hz_vq16384/eval_youtube_10s/`
  - tau from: `.../dtp_stats_ft/summary.json` (best.fixed_tau)
- `results/results0117/default_fixedpattern_50hz_vq65536/eval_youtube_10s/`
  - no tau override (FixedPatternMasking doesn’t use `fixed_tau`)
- `results/results0117/default_random_50hz_vq16384/eval_youtube_10s_ple/`
  - tau from: `.../dtp_stats_ple_ft/summary.json` (best.fixed_tau)

Each output directory contains:

- `metrics.json` (final summary; includes `cfg_overrides`)
- `audio_metrics.json`
- `pred_16k/` (reconstructed audio at 16 kHz WAV)
- `gt_16k/` (GT audio at 16 kHz WAV)
- `manifest.jsonl` (per-sample paths used for the metrics stage)

### Notes
- The script auto-runs `python utils/update_legacy_config.py --path <run_dir>/hydra/config.yaml` (same logic as `run_results0117_ft_suite.sh`).
- The first metrics run may download HF model weights (`facebook/hubert-large-ls960-ft`) into cache (normal).


