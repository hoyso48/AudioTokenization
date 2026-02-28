#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

usage() {
  cat <<'USAGE'
Usage:
  run_tts_modeling_train_only.sh \
    --codec-run-dir <path> \
    --variant <vfr|ffr> \
    --train-input <audio_dir_or_list> \
    --val-input <audio_dir_or_list> \
    --work-dir <path>

Optional:
    --codec-ckpt <path>
    --speech-vocab-size <int>                 (default: inferred from tokens)
    --max-train-files <int>
    --max-val-files <int>
    --prompt-seconds <float>                  (default: 3.0)
    --tokenizer-type <phoneme|char>           (default: phoneme)
    --train-epochs <float>                    (default: 100)
    --train-batch <int>                       (default: 4, eval/static fallback batch)
    --grad-accum <int>                        (default: 1)
    --no-dynamic-batching                     (disable dynamic token-batch sampler)
    --max-batch-tokens <int>                  (default: 6000)
    --max-batch-samples <int>                 (default: 16)
    --dynamic-batch-measure <target|total>    (default: target)
    --dynamic-bucket-size <int>               (default: 256)
    --report-to <name>                        (default: none)
    --run-name <name>                         (default: <variant>)
    --device <cpu|cuda>                       (default: cuda if available)

This script runs only the modeling part:
  1) extract codec tokens (train/val)
  2) build prompt-target examples
  3) build text tokenizer
  4) train AR-TTS
USAGE
}

CODEC_RUN_DIR=""
CODEC_CKPT=""
VARIANT=""
TRAIN_INPUT=""
VAL_INPUT=""
WORK_DIR=""
SPEECH_VOCAB_SIZE=""
MAX_TRAIN_FILES=""
MAX_VAL_FILES=""
PROMPT_SECONDS="3.0"
TOKENIZER_TYPE="phoneme"
TRAIN_EPOCHS="100"
TRAIN_BATCH="4"
GRAD_ACCUM="1"
DYNAMIC_BATCHING="1"
MAX_BATCH_TOKENS="6000"
MAX_BATCH_SAMPLES="16"
DYNAMIC_BATCH_MEASURE="target"
DYNAMIC_BUCKET_SIZE="256"
REPORT_TO="none"
RUN_NAME=""
DEVICE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --codec-run-dir) CODEC_RUN_DIR="$2"; shift 2 ;;
    --codec-ckpt) CODEC_CKPT="$2"; shift 2 ;;
    --variant) VARIANT="$2"; shift 2 ;;
    --train-input) TRAIN_INPUT="$2"; shift 2 ;;
    --val-input) VAL_INPUT="$2"; shift 2 ;;
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    --speech-vocab-size) SPEECH_VOCAB_SIZE="$2"; shift 2 ;;
    --max-train-files) MAX_TRAIN_FILES="$2"; shift 2 ;;
    --max-val-files) MAX_VAL_FILES="$2"; shift 2 ;;
    --prompt-seconds) PROMPT_SECONDS="$2"; shift 2 ;;
    --tokenizer-type) TOKENIZER_TYPE="$2"; shift 2 ;;
    --train-epochs) TRAIN_EPOCHS="$2"; shift 2 ;;
    --train-batch) TRAIN_BATCH="$2"; shift 2 ;;
    --grad-accum) GRAD_ACCUM="$2"; shift 2 ;;
    --no-dynamic-batching) DYNAMIC_BATCHING="0"; shift 1 ;;
    --max-batch-tokens) MAX_BATCH_TOKENS="$2"; shift 2 ;;
    --max-batch-samples) MAX_BATCH_SAMPLES="$2"; shift 2 ;;
    --dynamic-batch-measure) DYNAMIC_BATCH_MEASURE="$2"; shift 2 ;;
    --dynamic-bucket-size) DYNAMIC_BUCKET_SIZE="$2"; shift 2 ;;
    --report-to) REPORT_TO="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "${CODEC_RUN_DIR}" || -z "${VARIANT}" || -z "${TRAIN_INPUT}" || -z "${VAL_INPUT}" || -z "${WORK_DIR}" ]]; then
  usage
  exit 1
fi

if [[ "${VARIANT}" != "vfr" && "${VARIANT}" != "ffr" ]]; then
  echo "[ERROR] --variant must be vfr or ffr"
  exit 1
fi

if [[ "${TOKENIZER_TYPE}" != "phoneme" && "${TOKENIZER_TYPE}" != "char" ]]; then
  echo "[ERROR] --tokenizer-type must be phoneme or char"
  exit 1
fi

if [[ -z "${RUN_NAME}" ]]; then
  RUN_NAME="${VARIANT}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${DEVICE}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    DEVICE="cuda"
  else
    DEVICE="cpu"
  fi
fi

if [[ "${DEVICE}" == "cuda" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[ERROR] --device cuda requested but nvidia-smi is unavailable."
    echo "        Check NVIDIA driver/container GPU runtime (e.g., --gpus all)."
    exit 1
  fi

  python - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print("[ERROR] torch.cuda.is_available() is False.")
    print("        CUDA driver/runtime is not visible in this environment.")
    sys.exit(1)

try:
    _ = torch.tensor([1.0], device="cuda")
except Exception as exc:
    print(f"[ERROR] CUDA tensor allocation failed: {exc}")
    sys.exit(1)

print(f"[CHECK] CUDA ready (device_count={torch.cuda.device_count()}, torch_cuda={torch.version.cuda})")
PY
fi

mkdir -p "${WORK_DIR}" "${WORK_DIR}/data" "${WORK_DIR}/artifacts" "${WORK_DIR}/runs"

TRAIN_UTT_JSONL="${WORK_DIR}/data/utt_tokens_train.jsonl"
VAL_UTT_JSONL="${WORK_DIR}/data/utt_tokens_val.jsonl"
TRAIN_EX_JSONL="${WORK_DIR}/data/examples_train.jsonl"
VAL_EX_JSONL="${WORK_DIR}/data/examples_val.jsonl"
TOKENIZER_JSON="${WORK_DIR}/artifacts/text_tokenizer.json"
MODEL_OUT_DIR="${WORK_DIR}/runs/${VARIANT}"

TRAIN_LIMIT_ARGS=()
if [[ -n "${MAX_TRAIN_FILES}" ]]; then
  TRAIN_LIMIT_ARGS+=(--max_files "${MAX_TRAIN_FILES}")
fi

VAL_LIMIT_ARGS=()
if [[ -n "${MAX_VAL_FILES}" ]]; then
  VAL_LIMIT_ARGS+=(--max_files "${MAX_VAL_FILES}")
fi

CODEC_CKPT_ARGS=()
if [[ -n "${CODEC_CKPT}" ]]; then
  CODEC_CKPT_ARGS+=(--ckpt "${CODEC_CKPT}")
fi

echo "[1/5] Extracting train codec tokens"
python "${SCRIPT_DIR}/extract_codec_tokens.py" \
  --run_dir "${CODEC_RUN_DIR}" \
  "${CODEC_CKPT_ARGS[@]}" \
  --input "${TRAIN_INPUT}" \
  --output_jsonl "${TRAIN_UTT_JSONL}" \
  --device "${DEVICE}" \
  "${TRAIN_LIMIT_ARGS[@]}"

echo "[2/5] Extracting val codec tokens"
python "${SCRIPT_DIR}/extract_codec_tokens.py" \
  --run_dir "${CODEC_RUN_DIR}" \
  "${CODEC_CKPT_ARGS[@]}" \
  --input "${VAL_INPUT}" \
  --output_jsonl "${VAL_UTT_JSONL}" \
  --device "${DEVICE}" \
  "${VAL_LIMIT_ARGS[@]}"

echo "[3/5] Building train/val TTS examples"
python "${SCRIPT_DIR}/build_tts_examples.py" \
  --input_jsonl "${TRAIN_UTT_JSONL}" \
  --output_jsonl "${TRAIN_EX_JSONL}" \
  --prompt_seconds "${PROMPT_SECONDS}"

python "${SCRIPT_DIR}/build_tts_examples.py" \
  --input_jsonl "${VAL_UTT_JSONL}" \
  --output_jsonl "${VAL_EX_JSONL}" \
  --prompt_seconds "${PROMPT_SECONDS}"

echo "[4/5] Building text tokenizer (${TOKENIZER_TYPE})"
python "${SCRIPT_DIR}/prepare_text_tokenizer.py" \
  --input_jsonl "${TRAIN_EX_JSONL}" \
  --tokenizer_type "${TOKENIZER_TYPE}" \
  --output_path "${TOKENIZER_JSON}"

echo "[5/5] Training AR-TTS (${VARIANT})"
TRAIN_CMD=(
  python "${SCRIPT_DIR}/train_ar_tts.py"
  --train_jsonl "${TRAIN_EX_JSONL}"
  --val_jsonl "${VAL_EX_JSONL}"
  --tokenizer_path "${TOKENIZER_JSON}"
  --output_dir "${MODEL_OUT_DIR}"
  --num_train_epochs "${TRAIN_EPOCHS}"
  --per_device_train_batch_size "${TRAIN_BATCH}"
  --per_device_eval_batch_size "${TRAIN_BATCH}"
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  --report_to "${REPORT_TO}"
  --run_name "${RUN_NAME}"
)

if [[ "${DYNAMIC_BATCHING}" == "1" ]]; then
  TRAIN_CMD+=(
    --dynamic_batching
    --dynamic_batch_measure "${DYNAMIC_BATCH_MEASURE}"
    --max_batch_tokens "${MAX_BATCH_TOKENS}"
    --max_batch_samples "${MAX_BATCH_SAMPLES}"
    --dynamic_bucket_size "${DYNAMIC_BUCKET_SIZE}"
  )
fi

if [[ -n "${SPEECH_VOCAB_SIZE}" ]]; then
  TRAIN_CMD+=(--speech_vocab_size "${SPEECH_VOCAB_SIZE}")
fi

if [[ "${VARIANT}" == "vfr" ]]; then
  TRAIN_CMD+=(--use_vfr --max_span_len 512 --lambda_span 1.0)
fi

"${TRAIN_CMD[@]}"

echo "[DONE] Training pipeline finished"
echo "- Variant: ${VARIANT}"
echo "- Model dir: ${MODEL_OUT_DIR}"
echo "- Tokenizer: ${TOKENIZER_JSON}"
