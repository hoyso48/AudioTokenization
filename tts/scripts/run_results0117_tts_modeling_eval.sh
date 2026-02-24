#!/usr/bin/env bash
set -euo pipefail

# Fill these before running.
TRAIN_INPUT="/path/to/LibriTTS/train-clean-100"
VAL_INPUT="/path/to/LibriTTS/dev-clean"
SEED_EVAL_REPO="/path/to/seed-tts-eval"
SEED_META_EN="/path/to/seed_tts/en/meta.lst"
SEED_META_ZH="/path/to/seed_tts/zh/meta.lst"
VARSTOK_META="/path/to/varstok_eval/meta.lst"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1) VFR: default_PLE_50hz_vq16384
bash "${SCRIPT_DIR}/run_tts_modeling_and_eval.sh" \
  --codec-run-dir "/home/hoyso/projects/AudioTokenization/results/results0117/default_PLE_50hz_vq16384" \
  --variant vfr \
  --train-input "${TRAIN_INPUT}" \
  --val-input "${VAL_INPUT}" \
  --work-dir "/home/hoyso/projects/AudioTokenization/tts/experiments/results0117_PLE_vfr" \
  --seed-eval-repo "${SEED_EVAL_REPO}" \
  --seed-meta-en "${SEED_META_EN}" \
  --seed-meta-zh "${SEED_META_ZH}" \
  --varstok-meta "${VARSTOK_META}" \
  --train-epochs 100 \
  --train-batch 4 \
  --grad-accum 1 \
  --max-new-tokens 1024

# 2) Non-VFR: default_fixedpattern_50hz_vq65536
bash "${SCRIPT_DIR}/run_tts_modeling_and_eval.sh" \
  --codec-run-dir "/home/hoyso/projects/AudioTokenization/results/results0117/default_fixedpattern_50hz_vq65536" \
  --variant ffr \
  --train-input "${TRAIN_INPUT}" \
  --val-input "${VAL_INPUT}" \
  --work-dir "/home/hoyso/projects/AudioTokenization/tts/experiments/results0117_fixedpattern_ffr" \
  --seed-eval-repo "${SEED_EVAL_REPO}" \
  --seed-meta-en "${SEED_META_EN}" \
  --seed-meta-zh "${SEED_META_ZH}" \
  --varstok-meta "${VARSTOK_META}" \
  --train-epochs 100 \
  --train-batch 4 \
  --grad-accum 1 \
  --max-new-tokens 1024
