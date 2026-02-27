#!/usr/bin/env bash
set -euo pipefail

# End-to-end pipeline:
# 1) extract codec tokens (train/val)
# 2) build prompt-target examples
# 3) build text tokenizer
# 4) train AR TTS (FFR or VFR)
# 5) synthesize benchmark wavs
# 6) run SEED-TTS eval + VARSTOK-style eval

usage() {
  cat <<'USAGE'
Usage:
  run_tts_modeling_and_eval.sh \
    --codec-run-dir <path> \
    --variant <vfr|ffr> \
    --train-input <audio_dir_or_list> \
    --val-input <audio_dir_or_list> \
    --work-dir <path> \
    --seed-eval-repo <path> \
    --seed-meta-en <path> \
    --varstok-meta <path>

Optional:
    --codec-ckpt <path>
    --speech-vocab-size <int>                 (default: inferred from tokens)
    --max-train-files <int>
    --max-val-files <int>
    --train-epochs <float>                    (default: 100, VARSTOK-style)
    --train-batch <int>                       (default: 4, eval/static fallback batch)
    --grad-accum <int>                        (default: 1)
    --no-dynamic-batching                     (disable dynamic token-batch sampler)
    --max-batch-tokens <int>                  (default: 6000)
    --max-batch-samples <int>                 (default: 16)
    --dynamic-batch-measure <target|total>    (default: target)
    --dynamic-bucket-size <int>               (default: 256)
    --max-new-tokens <int>                    (default: 1024)
    --temperature <float>                     (default: 0.0)
    --top-k <int>                             (default: 0)
    --seed-meta-zh <path>
    --device <cpu|cuda>                       (default: cuda if available)

Notes:
  - VARSTOK-style eval uses eval_varstok_style.py (WER/SIM/UTMOS)
  - SEED-TTS eval uses external seed-tts-eval scripts (WER/SIM)
USAGE
}

CODEC_RUN_DIR=""
CODEC_CKPT=""
VARIANT=""
TRAIN_INPUT=""
VAL_INPUT=""
WORK_DIR=""
SEED_EVAL_REPO=""
SEED_META_EN=""
SEED_META_ZH=""
VARSTOK_META=""
SPEECH_VOCAB_SIZE=""
MAX_TRAIN_FILES=""
MAX_VAL_FILES=""
TRAIN_EPOCHS="100"
TRAIN_BATCH="4"
GRAD_ACCUM="1"
DYNAMIC_BATCHING="1"
MAX_BATCH_TOKENS="6000"
MAX_BATCH_SAMPLES="16"
DYNAMIC_BATCH_MEASURE="target"
DYNAMIC_BUCKET_SIZE="256"
MAX_NEW_TOKENS="1024"
TEMPERATURE="0.0"
TOP_K="0"
DEVICE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --codec-run-dir) CODEC_RUN_DIR="$2"; shift 2 ;;
    --codec-ckpt) CODEC_CKPT="$2"; shift 2 ;;
    --variant) VARIANT="$2"; shift 2 ;;
    --train-input) TRAIN_INPUT="$2"; shift 2 ;;
    --val-input) VAL_INPUT="$2"; shift 2 ;;
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    --seed-eval-repo) SEED_EVAL_REPO="$2"; shift 2 ;;
    --seed-meta-en) SEED_META_EN="$2"; shift 2 ;;
    --seed-meta-zh) SEED_META_ZH="$2"; shift 2 ;;
    --varstok-meta) VARSTOK_META="$2"; shift 2 ;;
    --speech-vocab-size) SPEECH_VOCAB_SIZE="$2"; shift 2 ;;
    --max-train-files) MAX_TRAIN_FILES="$2"; shift 2 ;;
    --max-val-files) MAX_VAL_FILES="$2"; shift 2 ;;
    --train-epochs) TRAIN_EPOCHS="$2"; shift 2 ;;
    --train-batch) TRAIN_BATCH="$2"; shift 2 ;;
    --grad-accum) GRAD_ACCUM="$2"; shift 2 ;;
    --no-dynamic-batching) DYNAMIC_BATCHING="0"; shift 1 ;;
    --max-batch-tokens) MAX_BATCH_TOKENS="$2"; shift 2 ;;
    --max-batch-samples) MAX_BATCH_SAMPLES="$2"; shift 2 ;;
    --dynamic-batch-measure) DYNAMIC_BATCH_MEASURE="$2"; shift 2 ;;
    --dynamic-bucket-size) DYNAMIC_BUCKET_SIZE="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --top-k) TOP_K="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "${CODEC_RUN_DIR}" || -z "${VARIANT}" || -z "${TRAIN_INPUT}" || -z "${VAL_INPUT}" || -z "${WORK_DIR}" || -z "${SEED_EVAL_REPO}" || -z "${SEED_META_EN}" || -z "${VARSTOK_META}" ]]; then
  usage
  exit 1
fi

if [[ "${VARIANT}" != "vfr" && "${VARIANT}" != "ffr" ]]; then
  echo "[ERROR] --variant must be vfr or ffr"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TTS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -z "${DEVICE}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    DEVICE="cuda"
  else
    DEVICE="cpu"
  fi
fi

mkdir -p "${WORK_DIR}" "${WORK_DIR}/data" "${WORK_DIR}/artifacts" "${WORK_DIR}/runs" "${WORK_DIR}/eval"

TRAIN_UTT_JSONL="${WORK_DIR}/data/utt_tokens_train.jsonl"
VAL_UTT_JSONL="${WORK_DIR}/data/utt_tokens_val.jsonl"
TRAIN_EX_JSONL="${WORK_DIR}/data/examples_train.jsonl"
VAL_EX_JSONL="${WORK_DIR}/data/examples_val.jsonl"
TOKENIZER_JSON="${WORK_DIR}/artifacts/text_tokenizer.json"
MODEL_OUT_DIR="${WORK_DIR}/runs/${VARIANT}"

SEED_SYNTH_EN_DIR="${WORK_DIR}/eval/seed_en_synth"
SEED_SYNTH_ZH_DIR="${WORK_DIR}/eval/seed_zh_synth"
VARSTOK_SYNTH_DIR="${WORK_DIR}/eval/varstok_synth"

SEED_SYNTH_EN_MANIFEST="${WORK_DIR}/eval/seed_en_synthesis_manifest.jsonl"
SEED_SYNTH_ZH_MANIFEST="${WORK_DIR}/eval/seed_zh_synthesis_manifest.jsonl"
VARSTOK_SYNTH_MANIFEST="${WORK_DIR}/eval/varstok_synthesis_manifest.jsonl"

TTS_USE_VFR_ARGS=()
if [[ "${VARIANT}" == "vfr" ]]; then
  TTS_USE_VFR_ARGS+=(--use_vfr)
fi

CODEC_CKPT_ARGS=()
if [[ -n "${CODEC_CKPT}" ]]; then
  CODEC_CKPT_ARGS+=(--codec_ckpt "${CODEC_CKPT}")
fi

TRAIN_LIMIT_ARGS=()
if [[ -n "${MAX_TRAIN_FILES}" ]]; then
  TRAIN_LIMIT_ARGS+=(--max_files "${MAX_TRAIN_FILES}")
fi

VAL_LIMIT_ARGS=()
if [[ -n "${MAX_VAL_FILES}" ]]; then
  VAL_LIMIT_ARGS+=(--max_files "${MAX_VAL_FILES}")
fi

echo "[1/8] Extracting train codec tokens"
python "${SCRIPT_DIR}/extract_codec_tokens.py" \
  --run_dir "${CODEC_RUN_DIR}" \
  --input "${TRAIN_INPUT}" \
  --output_jsonl "${TRAIN_UTT_JSONL}" \
  --device "${DEVICE}" \
  "${TRAIN_LIMIT_ARGS[@]}"

echo "[2/8] Extracting val codec tokens"
python "${SCRIPT_DIR}/extract_codec_tokens.py" \
  --run_dir "${CODEC_RUN_DIR}" \
  --input "${VAL_INPUT}" \
  --output_jsonl "${VAL_UTT_JSONL}" \
  --device "${DEVICE}" \
  "${VAL_LIMIT_ARGS[@]}"

echo "[3/8] Building train/val TTS examples"
python "${SCRIPT_DIR}/build_tts_examples.py" \
  --input_jsonl "${TRAIN_UTT_JSONL}" \
  --output_jsonl "${TRAIN_EX_JSONL}" \
  --prompt_seconds 3.0
python "${SCRIPT_DIR}/build_tts_examples.py" \
  --input_jsonl "${VAL_UTT_JSONL}" \
  --output_jsonl "${VAL_EX_JSONL}" \
  --prompt_seconds 3.0

echo "[4/8] Building text tokenizer"
python "${SCRIPT_DIR}/prepare_text_tokenizer.py" \
  --input_jsonl "${TRAIN_EX_JSONL}" \
  --tokenizer_type phoneme \
  --output_path "${TOKENIZER_JSON}"

echo "[5/8] Training AR-TTS (${VARIANT})"
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

if [[ -z "${SPEECH_VOCAB_SIZE}" ]]; then
  MODEL_SETUP_PATH="${MODEL_OUT_DIR}/model_setup.json"
  SPEECH_VOCAB_SIZE=$(python - "${MODEL_SETUP_PATH}" <<'PY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
obj = json.loads(p.read_text())
print(int(obj["speech_vocab_size"]))
PY
)
fi

echo "[6/8] Synthesizing benchmark wavs"
python "${SCRIPT_DIR}/synthesize_from_meta.py" \
  --model_dir "${MODEL_OUT_DIR}" \
  --tokenizer_path "${TOKENIZER_JSON}" \
  --speech_vocab_size "${SPEECH_VOCAB_SIZE}" \
  --codec_run_dir "${CODEC_RUN_DIR}" \
  "${CODEC_CKPT_ARGS[@]}" \
  --meta_lst "${SEED_META_EN}" \
  --output_dir "${SEED_SYNTH_EN_DIR}" \
  --output_manifest "${SEED_SYNTH_EN_MANIFEST}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top_k "${TOP_K}" \
  --device "${DEVICE}" \
  "${TTS_USE_VFR_ARGS[@]}"

if [[ -n "${SEED_META_ZH}" ]]; then
  python "${SCRIPT_DIR}/synthesize_from_meta.py" \
    --model_dir "${MODEL_OUT_DIR}" \
    --tokenizer_path "${TOKENIZER_JSON}" \
    --speech_vocab_size "${SPEECH_VOCAB_SIZE}" \
    --codec_run_dir "${CODEC_RUN_DIR}" \
    "${CODEC_CKPT_ARGS[@]}" \
    --meta_lst "${SEED_META_ZH}" \
    --output_dir "${SEED_SYNTH_ZH_DIR}" \
    --output_manifest "${SEED_SYNTH_ZH_MANIFEST}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top_k "${TOP_K}" \
    --device "${DEVICE}" \
    "${TTS_USE_VFR_ARGS[@]}"
fi

python "${SCRIPT_DIR}/synthesize_from_meta.py" \
  --model_dir "${MODEL_OUT_DIR}" \
  --tokenizer_path "${TOKENIZER_JSON}" \
  --speech_vocab_size "${SPEECH_VOCAB_SIZE}" \
  --codec_run_dir "${CODEC_RUN_DIR}" \
  "${CODEC_CKPT_ARGS[@]}" \
  --meta_lst "${VARSTOK_META}" \
  --output_dir "${VARSTOK_SYNTH_DIR}" \
  --output_manifest "${VARSTOK_SYNTH_MANIFEST}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top_k "${TOP_K}" \
  --device "${DEVICE}" \
  "${TTS_USE_VFR_ARGS[@]}"

echo "[7/8] Running SEED-TTS objective eval"
bash "${SCRIPT_DIR}/run_seed_tts_eval.sh" \
  "${SEED_EVAL_REPO}" \
  "${SEED_META_EN}" \
  "${SEED_SYNTH_EN_DIR}" \
  en \
  "${WORK_DIR}/eval/seed_en"

if [[ -n "${SEED_META_ZH}" ]]; then
  bash "${SCRIPT_DIR}/run_seed_tts_eval.sh" \
    "${SEED_EVAL_REPO}" \
    "${SEED_META_ZH}" \
    "${SEED_SYNTH_ZH_DIR}" \
    zh \
    "${WORK_DIR}/eval/seed_zh"
fi

echo "[8/8] Running VARSTOK-style objective eval"
python "${SCRIPT_DIR}/eval_varstok_style.py" \
  --meta_lst "${VARSTOK_META}" \
  --synth_dir "${VARSTOK_SYNTH_DIR}" \
  --output_json "${WORK_DIR}/eval/varstok_style_metrics.json" \
  --language en \
  --device "${DEVICE}"

echo "[DONE] Pipeline finished"
echo "- Model: ${MODEL_OUT_DIR}"
echo "- Seed EN eval: ${WORK_DIR}/eval/seed_en"
echo "- VARSTOK-style metrics: ${WORK_DIR}/eval/varstok_style_metrics.json"
