#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SBATCH_SCRIPT="$ROOT/slurm/run_train_eval_docker.sbatch"

JOB_NAME="audio_tokenizer"
GPUS="1"
CPUS_PER_TASK="16"
MEMORY="28G"
TIME_LIMIT="336:00:00"
OUTPUT_FILE="audio_tokenizer_%j.out"
PARTITION=""
QOS=""

WANDB_MODE="online" # online | offline | disabled
WANDB_API_KEY_FILE=""

declare -a FORWARD_ARGS=()

usage() {
  cat <<'EOF'
Submit train->eval Docker job to Slurm.

Usage:
  bash slurm/submit_train_eval.sh [submit-options] -- [run-options]

Submit options (wrapper-level):
  --job-name <name>            (default: audio_tokenizer)
  --gpus <int>                 (default: 1)
  --cpus <int>                 (default: 16)
  --mem <value>                (default: 28G)
  --time <HH:MM:SS>            (default: 336:00:00)
  --output <path>              (default: audio_tokenizer_%j.out)
  --partition <name>
  --qos <name>

  --wandb-mode <online|offline|disabled>    (default: online)
  --wandb-api-key-file <path>               Read WANDB token from file

Run options (forwarded to run_train_eval_docker.sbatch):
  --host_base_dir <path>                    Base path for AudioTokenization and datasets
  --image <docker_image>                    (default: hoyeol_atk:251116)
  --host_repo_dir <path>                    (default: <SLURM submit dir>)
  --host_data_dir <path>                    (default: <parent_of_repo>/datasets)
  --conda_envs_host <path>                  Optional persistent env path
  --conda_pkgs_host <path>                  Optional persistent pip/conda cache path
  --train_env <name> --eval_env <name>
  --setup_envs
  --run_dir_in_container <path>
  --train_config_path <path>
  --target_avg_r <float>
  --no_tau_finetune
  --skip_train
  --force
  --cfg_override <dotlist> (repeatable)
  --train_arg <arg> (repeatable)
  --eval_arg <arg> (repeatable)
  ... (all options accepted by slurm/run_train_eval_docker.sbatch)

Examples:
  export WANDB_API_KEY=...   # recommended

  # 1 GPU preset (run1 style)
  bash slurm/submit_train_eval.sh \
    --job-name dtmae-1g \
    --gpus 1 --cpus 16 --mem 28G \
    -- --train_env atk --eval_env speech_eval

  # 2 GPU preset (run2 style)
  bash slurm/submit_train_eval.sh \
    --job-name dtmae-2g \
    --gpus 2 --cpus 32 --mem 56G \
    -- \
    --train_env atk \
    --eval_env speech_eval \
    --train_arg train.trainer.devices=2

  # token file mode (no token in shell history)
  bash slurm/submit_train_eval.sh \
    --wandb-api-key-file ~/.secrets/wandb_api_key.txt \
    -- --train_env atk --eval_env speech_eval
EOF
}

trim_whitespace() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf "%s" "$s"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-name)
      JOB_NAME="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --cpus)
      CPUS_PER_TASK="$2"
      shift 2
      ;;
    --mem)
      MEMORY="$2"
      shift 2
      ;;
    --time)
      TIME_LIMIT="$2"
      shift 2
      ;;
    --output)
      OUTPUT_FILE="$2"
      shift 2
      ;;
    --partition)
      PARTITION="$2"
      shift 2
      ;;
    --qos)
      QOS="$2"
      shift 2
      ;;
    --wandb-mode)
      WANDB_MODE="$2"
      shift 2
      ;;
    --wandb-api-key-file)
      WANDB_API_KEY_FILE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        FORWARD_ARGS+=("$1")
        shift
      done
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
  echo "[ERROR] SBATCH script not found: $SBATCH_SCRIPT" >&2
  exit 1
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "[ERROR] sbatch command not found." >&2
  exit 1
fi

if [[ "$WANDB_MODE" != "online" && "$WANDB_MODE" != "offline" && "$WANDB_MODE" != "disabled" ]]; then
  echo "[ERROR] --wandb-mode must be one of: online, offline, disabled" >&2
  exit 1
fi

if [[ -n "$WANDB_API_KEY_FILE" ]]; then
  if [[ ! -f "$WANDB_API_KEY_FILE" ]]; then
    echo "[ERROR] WANDB API key file not found: $WANDB_API_KEY_FILE" >&2
    exit 1
  fi
  WANDB_API_KEY="$(trim_whitespace "$(<"$WANDB_API_KEY_FILE")")"
  if [[ -z "$WANDB_API_KEY" ]]; then
    echo "[ERROR] WANDB API key file is empty: $WANDB_API_KEY_FILE" >&2
    exit 1
  fi
  export WANDB_API_KEY
fi

if [[ "$WANDB_MODE" == "online" ]]; then
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "[ERROR] WANDB online mode requires WANDB_API_KEY env var or --wandb-api-key-file" >&2
    exit 1
  fi
fi

EXPORT_SPEC="ALL,WANDB_MODE=${WANDB_MODE}"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  EXPORT_SPEC+=" ,WANDB_API_KEY"
fi

# remove accidental spaces after comma from concatenation
EXPORT_SPEC="${EXPORT_SPEC// ,/,}"

SBATCH_CMD=(
  sbatch
  --job-name "$JOB_NAME"
  --ntasks 1
  --cpus-per-task "$CPUS_PER_TASK"
  --gres "gpu:$GPUS"
  --time "$TIME_LIMIT"
  --mem "$MEMORY"
  --output "$OUTPUT_FILE"
  --export "$EXPORT_SPEC"
)

if [[ -n "$PARTITION" ]]; then
  SBATCH_CMD+=(--partition "$PARTITION")
fi
if [[ -n "$QOS" ]]; then
  SBATCH_CMD+=(--qos "$QOS")
fi

SBATCH_CMD+=("$SBATCH_SCRIPT")
for arg in "${FORWARD_ARGS[@]}"; do
  SBATCH_CMD+=("$arg")
done

echo "[SUBMIT] ${SBATCH_CMD[*]}"
"${SBATCH_CMD[@]}"
