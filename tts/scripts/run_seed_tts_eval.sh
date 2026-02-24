#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  cat <<'USAGE'
Usage:
  run_seed_tts_eval.sh <seed_eval_repo> <meta_lst> <synth_dir> <lang:en|zh> <output_dir> [wavlm_ckpt]

Example:
  run_seed_tts_eval.sh \
    /path/to/seed-tts-eval \
    /path/to/en/meta.lst \
    /path/to/synth_en \
    en \
    /path/to/eval_out \
    /path/to/wavlm_large_finetune.pth
USAGE
  exit 1
fi

SEED_EVAL_REPO="$1"
META_LST="$2"
SYNTH_DIR="$3"
LANG="$4"
OUTPUT_DIR="$5"
WAVLM_CKPT="${6:-}"

mkdir -p "${OUTPUT_DIR}"

if [[ ! -d "${SEED_EVAL_REPO}" ]]; then
  echo "[ERROR] seed eval repo not found: ${SEED_EVAL_REPO}"
  exit 1
fi
if [[ ! -f "${META_LST}" ]]; then
  echo "[ERROR] meta file not found: ${META_LST}"
  exit 1
fi
if [[ ! -d "${SYNTH_DIR}" ]]; then
  echo "[ERROR] synth dir not found: ${SYNTH_DIR}"
  exit 1
fi

if [[ -z "${WAVLM_CKPT}" ]]; then
  WAVLM_CKPT="/home/hoyso/projects/AudioTokenization/eval/wavlm_large_finetune.pth"
fi

pushd "${SEED_EVAL_REPO}" >/dev/null

echo "[SEED-TTS] Running WER..."
bash cal_wer.sh "${META_LST}" "${SYNTH_DIR}" "${LANG}" | tee "${OUTPUT_DIR}/seed_tts_wer_${LANG}.log"

echo "[SEED-TTS] Running SIM..."
bash cal_sim.sh "${META_LST}" "${SYNTH_DIR}" "${WAVLM_CKPT}" | tee "${OUTPUT_DIR}/seed_tts_sim_${LANG}.log"

popd >/dev/null

echo "[SEED-TTS] Done. Logs saved under ${OUTPUT_DIR}"
