#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRAIN_ENV="${TRAIN_ENV:-atk}"
EVAL_ENV="${EVAL_ENV:-speech_eval}"

RUN_ID="${RUN_ID:-dtmae_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$ROOT/outputs/$RUN_ID}"

TRAIN_CONFIG_PATH="${TRAIN_CONFIG_PATH:-config_default}"
TRAIN_CONFIG_NAME="${TRAIN_CONFIG_NAME:-}"

INPUT_LIST="${INPUT_LIST:-$ROOT/DTMAE/filelists/librispeech_test_clean.txt}"
EVAL_OUT="${EVAL_OUT:-}"
EVAL_STAGE="${EVAL_STAGE:-all}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LENGTH_MODE="${LENGTH_MODE:-pad}"
DEVICE="${DEVICE:-}"
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-}"

TARGET_AVG_R="${TARGET_AVG_R:-}"
TAU_MIN="${TAU_MIN:-0.001}"
TAU_MAX="${TAU_MAX:-1.0}"
TAU_STEP="${TAU_STEP:-0.001}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
STATS_SUBDIR="${STATS_SUBDIR:-dtp_stats_ft}"

TAU_FINETUNE=1
BOOTSTRAP_UPDATE_TEST_TIME=1
BOOTSTRAP_OVERRIDE_UPDATE_TEST_TIME=1
SEARCH_NO_RESUME=1
AUTO_EXPAND=0
AUTO_EXPAND_MAX_TAU="${AUTO_EXPAND_MAX_TAU:-100.0}"
DIRECTION_PROBE_STEP="${DIRECTION_PROBE_STEP:-16}"

CHECK_WAVLM=1
FORCE=0
SKIP_TRAIN=0

PYTHON_TRAIN="${PYTHON_TRAIN:-python}"
PYTHON_EVAL="${PYTHON_EVAL:-python}"

declare -a TRAIN_EXTRA_ARGS=()
declare -a EVAL_CFG_OVERRIDES=()
declare -a EVAL_EXTRA_ARGS=()
POSITIONAL_CONFIG_PATH_SET=0

usage() {
  cat <<'EOF'
Train in `TRAIN_ENV` (default: atk), then evaluate in speech_eval.

Usage:
  bash run_train_then_eval.sh [options]

Options:
  --train_env <name>                 Conda env for training (default: atk)
  --eval_env <name>                  Conda env for eval (default: speech_eval)
  --run_id <id>                      Run identifier (default: dtmae_YYYYmmdd_HHMMSS)
  --run_dir <path>                   Output run directory (default: <repo>/outputs/<run_id>)

  --train_config_path <path>         Hydra config path override for train.py (default: config_default)
  --config_path <path>               Alias of --train_config_path
  --config-path <path>               Alias of --train_config_path
  --train_config_name <name>         Hydra config name override (optional)

  --input <path>                     Eval input path (default: DTMAE/filelists/librispeech_test_clean.txt)
  --eval_out <path>                  Eval output directory (default: <run_dir>/eval_ft)
  --eval_stage <save|metrics|all>    Eval stage (default: all)
  --num_workers <int>                Eval dataloader workers (default: 4)
  --length_mode <pad|truncate>       Eval length mode (default: pad)
  --device <str>                     Eval/search device (optional)

  --train_cuda_visible_devices <ids> Set CUDA_VISIBLE_DEVICES for train only (e.g., 0 or 0,1)
  --cuda_visible_devices <ids>       Alias of --train_cuda_visible_devices

  --tau_finetune                     Enable target_r->fixed_tau search when supported (default)
  --no_tau_finetune                  Disable tau search and run eval directly
  --target_avg_r <float>             Target avg_r for tau search (default: model.resampler.dtp_params.r)
  --tau_min <float>                  Tau search min (default: 0.001)
  --tau_max <float>                  Tau search max (default: 1.0)
  --tau_step <float>                 Tau search step (default: 0.001)
  --max_samples <int>                Max samples per tau trial (optional)
  --stats_subdir <name>              Tau-search output subdir under run_dir (default: dtp_stats_ft)
  --bootstrap_update_test_time       Enable bootstrap tau warm-start (default)
  --no_bootstrap_update_test_time    Disable bootstrap
  --bootstrap_override_update_test_time
                                     Force update_test_time=True during bootstrap (default)
  --no_bootstrap_override_update_test_time
                                     Do not force update_test_time=True
  --auto_expand                      Enable dtp_stats_search --auto_expand
  --auto_expand_max_tau <float>      Max tau used by auto-expand (default: 100.0)
  --direction_probe_step <int>       Probe step for search direction inference (default: 16)
  --resume_search                    Reuse existing trials.jsonl
  --no_resume_search                 Do not reuse trials.jsonl (default)

  --train_arg <arg>                  Extra arg passed to train.py (repeatable)
  --cfg_override <dotlist>           Eval/search config override (repeatable)
  --eval_arg <arg>                   Extra arg passed to eval.py (repeatable)

  --no_check_wavlm                   Do not require eval/wavlm_large_finetune.pth
  --skip_train                       Skip training and run eval only on existing run_dir
  --force                            Re-run eval even if metrics already exists
  --python_train <bin>               Python for train inside conda run (default: python)
  --python_eval <bin>                Python for eval inside conda run (default: python)
  -h, --help                         Show this help

Examples:
  bash run_train_then_eval.sh

  # positional config_path shortcut
  bash run_train_then_eval.sh config_default2 --run_dir outputs/exp1

  bash run_train_then_eval.sh \
    --run_dir /workspace/AudioTokenization/outputs/exp1 \
    --train_cuda_visible_devices 0 \
    --cfg_override model.resampler.dtp_params.r=0.4 \
    --train_arg train.trainer.devices=1
EOF
}

require_conda() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda command not found in PATH." >&2
    exit 1
  fi
}

check_env_python() {
  local env_name="$1"
  local py_bin="$2"
  if ! conda run -n "$env_name" "$py_bin" -c "import sys; print(sys.executable)" >/dev/null 2>&1; then
    echo "[ERROR] Cannot run Python in conda env: $env_name" >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train_env)
      TRAIN_ENV="$2"
      shift 2
      ;;
    --eval_env)
      EVAL_ENV="$2"
      shift 2
      ;;
    --run_id)
      RUN_ID="$2"
      shift 2
      ;;
    --run_dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --train_config_path|--config_path|--config-path)
      TRAIN_CONFIG_PATH="$2"
      shift 2
      ;;
    --train_config_name)
      TRAIN_CONFIG_NAME="$2"
      shift 2
      ;;
    --input)
      INPUT_LIST="$2"
      shift 2
      ;;
    --eval_out)
      EVAL_OUT="$2"
      shift 2
      ;;
    --eval_stage|--stage)
      EVAL_STAGE="$2"
      shift 2
      ;;
    --num_workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --length_mode)
      LENGTH_MODE="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --train_cuda_visible_devices|--cuda_visible_devices)
      TRAIN_CUDA_VISIBLE_DEVICES="$2"
      shift 2
      ;;
    --tau_finetune)
      TAU_FINETUNE=1
      shift
      ;;
    --no_tau_finetune)
      TAU_FINETUNE=0
      shift
      ;;
    --target_avg_r)
      TARGET_AVG_R="$2"
      shift 2
      ;;
    --tau_min)
      TAU_MIN="$2"
      shift 2
      ;;
    --tau_max)
      TAU_MAX="$2"
      shift 2
      ;;
    --tau_step)
      TAU_STEP="$2"
      shift 2
      ;;
    --max_samples)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    --stats_subdir)
      STATS_SUBDIR="$2"
      shift 2
      ;;
    --bootstrap_update_test_time)
      BOOTSTRAP_UPDATE_TEST_TIME=1
      shift
      ;;
    --no_bootstrap_update_test_time)
      BOOTSTRAP_UPDATE_TEST_TIME=0
      shift
      ;;
    --bootstrap_override_update_test_time)
      BOOTSTRAP_OVERRIDE_UPDATE_TEST_TIME=1
      shift
      ;;
    --no_bootstrap_override_update_test_time)
      BOOTSTRAP_OVERRIDE_UPDATE_TEST_TIME=0
      shift
      ;;
    --auto_expand)
      AUTO_EXPAND=1
      shift
      ;;
    --auto_expand_max_tau)
      AUTO_EXPAND_MAX_TAU="$2"
      shift 2
      ;;
    --direction_probe_step)
      DIRECTION_PROBE_STEP="$2"
      shift 2
      ;;
    --resume_search)
      SEARCH_NO_RESUME=0
      shift
      ;;
    --no_resume_search)
      SEARCH_NO_RESUME=1
      shift
      ;;
    --train_arg)
      TRAIN_EXTRA_ARGS+=("$2")
      shift 2
      ;;
    --cfg_override)
      EVAL_CFG_OVERRIDES+=("$2")
      shift 2
      ;;
    --eval_arg)
      EVAL_EXTRA_ARGS+=("$2")
      shift 2
      ;;
    --no_check_wavlm)
      CHECK_WAVLM=0
      shift
      ;;
    --skip_train)
      SKIP_TRAIN=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --python_train)
      PYTHON_TRAIN="$2"
      shift 2
      ;;
    --python_eval)
      PYTHON_EVAL="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -* )
      echo "[ERROR] Unknown option: $1" >&2
      usage
      exit 1
      ;;
    *)
      if [[ "$POSITIONAL_CONFIG_PATH_SET" -eq 0 ]]; then
        TRAIN_CONFIG_PATH="$1"
        POSITIONAL_CONFIG_PATH_SET=1
        shift
      else
        echo "[ERROR] Unexpected positional argument: $1" >&2
        usage
        exit 1
      fi
      ;;
  esac
done

if [[ "$EVAL_STAGE" != "save" && "$EVAL_STAGE" != "metrics" && "$EVAL_STAGE" != "all" ]]; then
  echo "[ERROR] --eval_stage must be one of: save, metrics, all" >&2
  exit 1
fi

if [[ "$LENGTH_MODE" != "pad" && "$LENGTH_MODE" != "truncate" ]]; then
  echo "[ERROR] --length_mode must be one of: pad, truncate" >&2
  exit 1
fi

if [[ -z "$EVAL_OUT" ]]; then
  EVAL_OUT="$RUN_DIR/eval_ft"
fi

require_conda
check_env_python "$TRAIN_ENV" "$PYTHON_TRAIN"
check_env_python "$EVAL_ENV" "$PYTHON_EVAL"

echo "============================================================"
echo "[INFO] ROOT=$ROOT"
echo "[INFO] RUN_DIR=$RUN_DIR"
echo "[INFO] EVAL_OUT=$EVAL_OUT"
echo "[INFO] TRAIN_ENV=$TRAIN_ENV"
echo "[INFO] EVAL_ENV=$EVAL_ENV"
if [[ -n "$TRAIN_CUDA_VISIBLE_DEVICES" ]]; then
  echo "[INFO] TRAIN CUDA_VISIBLE_DEVICES=$TRAIN_CUDA_VISIBLE_DEVICES"
fi
echo "============================================================"

if [[ "$SKIP_TRAIN" -eq 0 ]]; then
  mkdir -p "$RUN_DIR"

  declare -a TRAIN_CMD=("$PYTHON_TRAIN" "train.py" "--config-path" "$TRAIN_CONFIG_PATH" "hydra.run.dir=$RUN_DIR")
  if [[ -n "$TRAIN_CONFIG_NAME" ]]; then
    TRAIN_CMD+=("--config-name" "$TRAIN_CONFIG_NAME")
  fi
  for arg in "${TRAIN_EXTRA_ARGS[@]}"; do
    TRAIN_CMD+=("$arg")
  done

  echo "[TRAIN] Starting training in env: $TRAIN_ENV"
  if [[ -n "$TRAIN_CUDA_VISIBLE_DEVICES" ]]; then
    (
      cd "$ROOT/DTMAE"
      CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES" conda run --no-capture-output -n "$TRAIN_ENV" "${TRAIN_CMD[@]}"
    )
  else
    (
      cd "$ROOT/DTMAE"
      conda run --no-capture-output -n "$TRAIN_ENV" "${TRAIN_CMD[@]}"
    )
  fi
  echo "[TRAIN] Finished"
else
  echo "[TRAIN] Skipped by --skip_train"
fi

if [[ ! -f "$RUN_DIR/hydra/config.yaml" ]]; then
  echo "[ERROR] Missing file: $RUN_DIR/hydra/config.yaml" >&2
  exit 1
fi
if [[ ! -f "$RUN_DIR/pl_log/last.ckpt" ]]; then
  echo "[ERROR] Missing file: $RUN_DIR/pl_log/last.ckpt" >&2
  exit 1
fi

declare -a RUN_EVAL_CMD=(
  "bash"
  "$ROOT/run_eval_only.sh"
  "--run_dir" "$RUN_DIR"
  "--input" "$INPUT_LIST"
  "--eval_out" "$EVAL_OUT"
  "--eval_stage" "$EVAL_STAGE"
  "--num_workers" "$NUM_WORKERS"
  "--length_mode" "$LENGTH_MODE"
  "--stats_subdir" "$STATS_SUBDIR"
  "--eval_env" "$EVAL_ENV"
  "--python_eval" "$PYTHON_EVAL"
  "--tau_min" "$TAU_MIN"
  "--tau_max" "$TAU_MAX"
  "--tau_step" "$TAU_STEP"
  "--direction_probe_step" "$DIRECTION_PROBE_STEP"
)

if [[ "$TAU_FINETUNE" -eq 0 ]]; then
  RUN_EVAL_CMD+=("--no_tau_finetune")
fi
if [[ -n "$TARGET_AVG_R" ]]; then
  RUN_EVAL_CMD+=("--target_avg_r" "$TARGET_AVG_R")
fi
if [[ -n "$MAX_SAMPLES" ]]; then
  RUN_EVAL_CMD+=("--max_samples" "$MAX_SAMPLES")
fi
if [[ -n "$DEVICE" ]]; then
  RUN_EVAL_CMD+=("--device" "$DEVICE")
fi
if [[ "$BOOTSTRAP_UPDATE_TEST_TIME" -eq 0 ]]; then
  RUN_EVAL_CMD+=("--no_bootstrap_update_test_time")
fi
if [[ "$BOOTSTRAP_OVERRIDE_UPDATE_TEST_TIME" -eq 0 ]]; then
  RUN_EVAL_CMD+=("--no_bootstrap_override_update_test_time")
fi
if [[ "$AUTO_EXPAND" -eq 1 ]]; then
  RUN_EVAL_CMD+=("--auto_expand" "--auto_expand_max_tau" "$AUTO_EXPAND_MAX_TAU")
fi
if [[ "$SEARCH_NO_RESUME" -eq 0 ]]; then
  RUN_EVAL_CMD+=("--resume_search")
fi
if [[ "$CHECK_WAVLM" -eq 0 ]]; then
  RUN_EVAL_CMD+=("--no_check_wavlm")
fi
if [[ "$FORCE" -eq 1 ]]; then
  RUN_EVAL_CMD+=("--force")
fi
for ov in "${EVAL_CFG_OVERRIDES[@]}"; do
  RUN_EVAL_CMD+=("--cfg_override" "$ov")
done
for arg in "${EVAL_EXTRA_ARGS[@]}"; do
  RUN_EVAL_CMD+=("--eval_arg" "$arg")
done

echo "[EVAL] Starting eval step via run_eval_only.sh"
(
  cd "$ROOT"
  "${RUN_EVAL_CMD[@]}"
)
echo "[EVAL] Finished"
echo "[DONE] Train->Eval pipeline completed."
