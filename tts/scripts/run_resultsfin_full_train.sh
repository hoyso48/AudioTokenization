#!/usr/bin/env bash
set -euo pipefail

# Prefilled full LibriTTS(585h) training recipe for resultsfin codec runs.

LIBRITTS_ROOT="/home/hoyso/datasets/LibriTTS/LibriTTS"
WORK_BASE="/home/hoyso/projects/atk/AudioTokenization/tts/experiments/resultsfin_full_585h"

FFR_CODEC_RUN="/home/hoyso/projects/atk/AudioTokenization/results/resultsfin/default-transformer-bs64-600k-80hz-FixedPattern40hz-vq65536-2stage-ropebase10kposids-window128-lneps1e2-layerscale1-qknorm-nodropout"
VFR_CODEC_RUN="/home/hoyso/projects/atk/AudioTokenization/results/resultsfin/default-transformer-bs64-600k-80hz-PLEBatchTopK40hz-vq16384-2stage-ropebase10kposids-window128-lneps1e2-layerscale1-qknorm-nodropout"

TRAIN_FILELIST="${WORK_BASE}/data/libritts_train_585h.txt"
VAL_INPUT="${LIBRITTS_ROOT}/dev-clean"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${WORK_BASE}/data"

echo "[1/3] Building 585h train file list"
python - "${LIBRITTS_ROOT}" "${TRAIN_FILELIST}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
out_path = Path(sys.argv[2]).resolve()
subsets = ["train-clean-100", "train-clean-360", "train-other-500"]

files = []
for subset in subsets:
    subset_dir = root / subset
    if not subset_dir.is_dir():
        raise FileNotFoundError(f"Missing subset directory: {subset_dir}")
    files.extend(sorted(str(p.resolve()) for p in subset_dir.rglob("*.wav")))
    files.extend(sorted(str(p.resolve()) for p in subset_dir.rglob("*.flac")))

if not files:
    raise RuntimeError("No train audio files found for LibriTTS 585h subsets.")

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("\n".join(files) + "\n", encoding="utf-8")
print(f"wrote {len(files)} paths to {out_path}")
PY

echo "[2/3] Train VFR AR-TTS (PLE tokenizer)"
bash "${SCRIPT_DIR}/run_tts_modeling_train_only.sh" \
  --codec-run-dir "${VFR_CODEC_RUN}" \
  --variant vfr \
  --train-input "${TRAIN_FILELIST}" \
  --val-input "${VAL_INPUT}" \
  --work-dir "${WORK_BASE}/vfr" \
  --speech-vocab-size 16384 \
  --tokenizer-type phoneme \
  --prompt-seconds 3.0 \
  --train-epochs 100 \
  --train-batch 4 \
  --grad-accum 1 \
  --max-batch-tokens 6000 \
  --max-batch-samples 16 \
  --dynamic-batch-measure target \
  --dynamic-bucket-size 256

echo "[3/3] Train FFR AR-TTS (FixedPattern tokenizer)"
bash "${SCRIPT_DIR}/run_tts_modeling_train_only.sh" \
  --codec-run-dir "${FFR_CODEC_RUN}" \
  --variant ffr \
  --train-input "${TRAIN_FILELIST}" \
  --val-input "${VAL_INPUT}" \
  --work-dir "${WORK_BASE}/ffr" \
  --speech-vocab-size 65536 \
  --tokenizer-type phoneme \
  --prompt-seconds 3.0 \
  --train-epochs 100 \
  --train-batch 4 \
  --grad-accum 1 \
  --max-batch-tokens 6000 \
  --max-batch-samples 16 \
  --dynamic-batch-measure target \
  --dynamic-bucket-size 256

echo "[DONE] resultsfin full 585h training pipelines finished"
