#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SBATCH_SCRIPT="$ROOT/slurm/setup_env_docker.sbatch"

JOB_NAME="atk-setup"
GPUS="1"
CPUS_PER_TASK="8"
MEMORY="32G"
TIME_LIMIT="04:00:00"
OUTPUT_FILE="atk_setup_%j.out"
PARTITION=""
QOS=""

WANDB_MODE="offline" # online | offline | disabled
WANDB_API_KEY_FILE=""

declare -a FORWARD_ARGS=()

usage() {
  cat <<'EOF'
Submit environment setup Docker job to Slurm.

Usage:
  bash slurm/submit_setup_env.sh [submit-options] -- [setup-options]

Submit options:
  --job-name <name>            (default: atk-setup)
  --gpus <int>                 (default: 1)
  --cpus <int>                 (default: 8)
  --mem <value>                (default: 32G)
  --time <HH:MM:SS>            (default: 04:00:00)
  --output <path>              (default: atk_setup_%j.out)
  --partition <name>
  --qos <name>

  --wandb-mode <online|offline|disabled>    (default: offline)
  --wandb-api-key-file <path>

Setup options (forwarded):
  --host_base_dir <path>                    Base path for AudioTokenization and datasets
  --image <docker_image>                    (default: hoyeol_atk:251116)
  --host_repo_dir <path>                    (default: <SLURM submit dir>)
  --host_data_dir <path>                    (default: <parent_of_repo>/datasets)
  --conda_envs_host <path>                  Optional persistent env path
  --conda_pkgs_host <path>                  Optional persistent pip/conda cache path
  --train_env <name> --eval_env <name>
  --flash_attn_version <ver>
  --force_reinstall_eval_torch
  --no_ffmpeg
  --no_fix_nccl
  --skip_train
  --skip_eval

Example:
  bash slurm/submit_setup_env.sh \
    --wandb-api-key-file ~/.secrets/wandb_api_key.txt \
    -- \
    --train_env atk \
    --eval_env speech_eval
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

if [[ "$WANDB_MODE" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "[ERROR] WANDB online mode requires WANDB_API_KEY env var or --wandb-api-key-file" >&2
  exit 1
fi

EXPORT_SPEC="ALL,WANDB_MODE=${WANDB_MODE}"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  EXPORT_SPEC+=" ,WANDB_API_KEY"
fi
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
