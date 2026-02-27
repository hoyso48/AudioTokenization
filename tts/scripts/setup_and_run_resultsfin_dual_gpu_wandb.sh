#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  setup_and_run_resultsfin_dual_gpu_wandb.sh \
    --wandb-project <project>

Optional:
    --env-name <name>                         (default: speech_eval)
    --setup-env                               (create/install env; default: on)
    --no-setup-env                            (skip env setup)
    --libritts-root <path>                    (default: /home/hoyso/datasets/LibriTTS)
    --download-libritts                       (download missing subsets)
    --work-base <path>                        (default: /home/hoyso/projects/atk/AudioTokenization/tts/experiments/resultsfin_dual_gpu_585h)
    --wandb-entity <entity>                   (optional)
    --wandb-mode <online|offline|disabled>    (default: online)
    --run-tag <name>                          (default: resultsfin_585h)
    --train-epochs <float>                    (default: 100)
    --train-batch <int>                       (default: 4)
    --grad-accum <int>                        (default: 1)
    --max-batch-tokens <int>                  (default: 6000)
    --max-batch-samples <int>                 (default: 16)
    --dynamic-batch-measure <target|total>    (default: target)
    --dynamic-bucket-size <int>               (default: 256)
    --max-train-files <int>                   (optional debug limit)
    --max-val-files <int>                     (optional debug limit)

This script performs:
  1) Environment setup (optional)
  2) LibriTTS subset check/download (train-clean-100/360/train-other-500/dev-clean)
  3) 585h train filelist creation
  4) Concurrent training launch:
     - VFR on GPU 0
     - FFR on GPU 1
  5) W&B logging enabled for both jobs
USAGE
}

ENV_NAME="speech_eval"
SETUP_ENV="1"
DOWNLOAD_LIBRITTS="0"
LIBRITTS_ROOT="/home/hoyso/datasets/LibriTTS"
WORK_BASE="/home/hoyso/projects/atk/AudioTokenization/tts/experiments/resultsfin_dual_gpu_585h"
WANDB_PROJECT=""
WANDB_ENTITY=""
WANDB_MODE="online"
RUN_TAG="resultsfin_585h"
TRAIN_EPOCHS="100"
TRAIN_BATCH="4"
GRAD_ACCUM="1"
MAX_BATCH_TOKENS="6000"
MAX_BATCH_SAMPLES="16"
DYNAMIC_BATCH_MEASURE="target"
DYNAMIC_BUCKET_SIZE="256"
MAX_TRAIN_FILES=""
MAX_VAL_FILES=""

FFR_CODEC_RUN="/home/hoyso/projects/atk/AudioTokenization/results/resultsfin/default-transformer-bs64-600k-80hz-FixedPattern40hz-vq65536-2stage-ropebase10kposids-window128-lneps1e2-layerscale1-qknorm-nodropout"
VFR_CODEC_RUN="/home/hoyso/projects/atk/AudioTokenization/results/resultsfin/default-transformer-bs64-600k-80hz-PLEBatchTopK40hz-vq16384-2stage-ropebase10kposids-window128-lneps1e2-layerscale1-qknorm-nodropout"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name) ENV_NAME="$2"; shift 2 ;;
    --setup-env) SETUP_ENV="1"; shift 1 ;;
    --no-setup-env) SETUP_ENV="0"; shift 1 ;;
    --libritts-root) LIBRITTS_ROOT="$2"; shift 2 ;;
    --download-libritts) DOWNLOAD_LIBRITTS="1"; shift 1 ;;
    --work-base) WORK_BASE="$2"; shift 2 ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --wandb-entity) WANDB_ENTITY="$2"; shift 2 ;;
    --wandb-mode) WANDB_MODE="$2"; shift 2 ;;
    --run-tag) RUN_TAG="$2"; shift 2 ;;
    --train-epochs) TRAIN_EPOCHS="$2"; shift 2 ;;
    --train-batch) TRAIN_BATCH="$2"; shift 2 ;;
    --grad-accum) GRAD_ACCUM="$2"; shift 2 ;;
    --max-batch-tokens) MAX_BATCH_TOKENS="$2"; shift 2 ;;
    --max-batch-samples) MAX_BATCH_SAMPLES="$2"; shift 2 ;;
    --dynamic-batch-measure) DYNAMIC_BATCH_MEASURE="$2"; shift 2 ;;
    --dynamic-bucket-size) DYNAMIC_BUCKET_SIZE="$2"; shift 2 ;;
    --max-train-files) MAX_TRAIN_FILES="$2"; shift 2 ;;
    --max-val-files) MAX_VAL_FILES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "${WANDB_PROJECT}" ]]; then
  echo "[ERROR] --wandb-project is required"
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found in PATH"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBRITTS_DIR="${LIBRITTS_ROOT}/LibriTTS"
TRAIN_FILELIST="${WORK_BASE}/data/libritts_train_585h.txt"
VAL_INPUT="${LIBRITTS_DIR}/dev-clean"
VFR_WORK_DIR="${WORK_BASE}/vfr"
FFR_WORK_DIR="${WORK_BASE}/ffr"
VFR_LOG="${WORK_BASE}/logs/vfr_gpu0.log"
FFR_LOG="${WORK_BASE}/logs/ffr_gpu1.log"

mkdir -p "${WORK_BASE}/data" "${WORK_BASE}/logs"

run_with_heartbeat() {
  local heartbeat_tag="$1"
  shift
  "$@" &
  local cmd_pid=$!
  while kill -0 "${cmd_pid}" 2>/dev/null; do
    echo "[${heartbeat_tag}] still running... $(date '+%Y-%m-%d %H:%M:%S')"
    sleep 30
  done
  wait "${cmd_pid}"
}

if [[ "${SETUP_ENV}" == "1" ]]; then
  echo "[SETUP] Installing environment ${ENV_NAME}"
  bash "${SCRIPT_DIR}/setup_env_tts.sh" "${ENV_NAME}"
fi

missing_subsets=()
for subset in train-clean-100 train-clean-360 train-other-500 dev-clean; do
  if [[ ! -d "${LIBRITTS_DIR}/${subset}" ]]; then
    missing_subsets+=("${subset}")
  fi
done

if [[ ${#missing_subsets[@]} -gt 0 ]]; then
  if [[ "${DOWNLOAD_LIBRITTS}" != "1" ]]; then
    echo "[ERROR] Missing LibriTTS subsets: ${missing_subsets[*]}"
    echo "        Re-run with --download-libritts or prepare dataset at ${LIBRITTS_DIR}"
    exit 1
  fi
  echo "[DATA] Downloading missing LibriTTS subsets: ${missing_subsets[*]}"
  run_with_heartbeat "DATA" \
    conda run --no-capture-output -n "${ENV_NAME}" python -u "${SCRIPT_DIR}/download_libritts.py" \
      --root "${LIBRITTS_ROOT}" \
      --subsets train-clean-100 train-clean-360 train-other-500 dev-clean test-clean \
      --download
fi

echo "[DATA] Building 585h filelist at ${TRAIN_FILELIST}"
python - "${LIBRITTS_DIR}" "${TRAIN_FILELIST}" <<'PY'
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

COMMON_ARGS=(
  --train-input "${TRAIN_FILELIST}"
  --val-input "${VAL_INPUT}"
  --prompt-seconds 3.0
  --tokenizer-type phoneme
  --train-epochs "${TRAIN_EPOCHS}"
  --train-batch "${TRAIN_BATCH}"
  --grad-accum "${GRAD_ACCUM}"
  --max-batch-tokens "${MAX_BATCH_TOKENS}"
  --max-batch-samples "${MAX_BATCH_SAMPLES}"
  --dynamic-batch-measure "${DYNAMIC_BATCH_MEASURE}"
  --dynamic-bucket-size "${DYNAMIC_BUCKET_SIZE}"
  --report-to wandb
)

if [[ -n "${MAX_TRAIN_FILES}" ]]; then
  COMMON_ARGS+=(--max-train-files "${MAX_TRAIN_FILES}")
fi
if [[ -n "${MAX_VAL_FILES}" ]]; then
  COMMON_ARGS+=(--max-val-files "${MAX_VAL_FILES}")
fi

echo "[LAUNCH] Starting VFR on GPU 0 (W&B: ${WANDB_PROJECT})"
(
  export CUDA_VISIBLE_DEVICES=0
  export WANDB_PROJECT="${WANDB_PROJECT}"
  export WANDB_MODE="${WANDB_MODE}"
  export PYTHONUNBUFFERED=1
  if [[ -n "${WANDB_ENTITY}" ]]; then
    export WANDB_ENTITY="${WANDB_ENTITY}"
  fi
  conda run --no-capture-output -n "${ENV_NAME}" bash "${SCRIPT_DIR}/run_tts_modeling_train_only.sh" \
    --codec-run-dir "${VFR_CODEC_RUN}" \
    --variant vfr \
    --work-dir "${VFR_WORK_DIR}" \
    --speech-vocab-size 16384 \
    --run-name "${RUN_TAG}_vfr" \
    --device cuda \
    "${COMMON_ARGS[@]}"
) > "${VFR_LOG}" 2>&1 &
VFR_PID=$!

echo "[LAUNCH] Starting FFR on GPU 1 (W&B: ${WANDB_PROJECT})"
(
  export CUDA_VISIBLE_DEVICES=1
  export WANDB_PROJECT="${WANDB_PROJECT}"
  export WANDB_MODE="${WANDB_MODE}"
  export PYTHONUNBUFFERED=1
  if [[ -n "${WANDB_ENTITY}" ]]; then
    export WANDB_ENTITY="${WANDB_ENTITY}"
  fi
  conda run --no-capture-output -n "${ENV_NAME}" bash "${SCRIPT_DIR}/run_tts_modeling_train_only.sh" \
    --codec-run-dir "${FFR_CODEC_RUN}" \
    --variant ffr \
    --work-dir "${FFR_WORK_DIR}" \
    --speech-vocab-size 65536 \
    --run-name "${RUN_TAG}_ffr" \
    --device cuda \
    "${COMMON_ARGS[@]}"
) > "${FFR_LOG}" 2>&1 &
FFR_PID=$!

echo "[RUNNING] VFR PID=${VFR_PID}, log=${VFR_LOG}"
echo "[RUNNING] FFR PID=${FFR_PID}, log=${FFR_LOG}"
echo "[TIP] Follow logs: tail -f ${VFR_LOG} ${FFR_LOG}"

set +e
wait "${VFR_PID}"
VFR_STATUS=$?
wait "${FFR_PID}"
FFR_STATUS=$?
set -e

echo "[DONE] VFR exit=${VFR_STATUS}, FFR exit=${FFR_STATUS}"
if [[ ${VFR_STATUS} -ne 0 || ${FFR_STATUS} -ne 0 ]]; then
  echo "[ERROR] One or more training jobs failed. Check logs under ${WORK_BASE}/logs"
  exit 1
fi

echo "[SUCCESS] Both VFR/FFR training jobs completed"
