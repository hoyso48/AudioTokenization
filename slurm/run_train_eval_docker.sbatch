#!/usr/bin/env bash
#SBATCH --job-name=dtmae-train-eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=336:00:00
#SBATCH --mem=28G
#SBATCH --output=dtmae_train_eval_%j.out

# Preset guide:
# - 1 GPU (run1 스타일): defaults 그대로 사용
# - 2 GPU (run2 스타일): submit 시 override
#   sbatch --gres=gpu:2 --cpus-per-task=32 --mem=56G slurm/run_train_eval_docker.sbatch

set -euo pipefail

IMAGE="${IMAGE:-hoyeol_atk:251116}"
HOST_BASE_DIR="${HOST_BASE_DIR:-}"
PROJECT_ROOT_HOST="${PROJECT_ROOT_HOST:-}"
DATASETS_ROOT_HOST="${DATASETS_ROOT_HOST:-}"
PROJECT_ROOT_CONTAINER="${PROJECT_ROOT_CONTAINER:-/workspace/AudioTokenization}"
DATASETS_ROOT_CONTAINER="${DATASETS_ROOT_CONTAINER:-/workspace/datasets}"
CONDA_ENVS_HOST="${CONDA_ENVS_HOST:-}"
CONDA_PKGS_HOST="${CONDA_PKGS_HOST:-}"

TRAIN_ENV="${TRAIN_ENV:-atk}"
EVAL_ENV="${EVAL_ENV:-speech_eval}"

RUN_ID="${RUN_ID:-slurm_${SLURM_JOB_ID:-manual}}"
RUN_DIR_IN_CONTAINER="${RUN_DIR_IN_CONTAINER:-${PROJECT_ROOT_CONTAINER}/outputs/${RUN_ID}}"
EVAL_OUT_IN_CONTAINER="${EVAL_OUT_IN_CONTAINER:-${RUN_DIR_IN_CONTAINER}/eval_ft}"

TRAIN_CONFIG_PATH="${TRAIN_CONFIG_PATH:-config_default}"
TRAIN_CONFIG_NAME="${TRAIN_CONFIG_NAME:-}"
INPUT="${INPUT:-${PROJECT_ROOT_CONTAINER}/DTMAE/filelists/librispeech_test_clean.txt}"
EVAL_STAGE="${EVAL_STAGE:-all}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LENGTH_MODE="${LENGTH_MODE:-pad}"
DEVICE="${DEVICE:-}"

# Leave empty to follow model.resampler.dtp_params.r from run config.
TARGET_AVG_R="${TARGET_AVG_R:-}"
TAU_MIN="${TAU_MIN:-0.001}"
TAU_MAX="${TAU_MAX:-1.0}"
TAU_STEP="${TAU_STEP:-0.001}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
STATS_SUBDIR="${STATS_SUBDIR:-dtp_stats_ft}"

TAU_FINETUNE=1
SKIP_TRAIN=0
FORCE=0

SETUP_ENVS=0
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3}"

TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-}"
CONTAINER_SHM_SIZE="${CONTAINER_SHM_SIZE:-16G}"
CONTAINER_NAME_PREFIX="${CONTAINER_NAME_PREFIX:-atkrun}"

declare -a CFG_OVERRIDES=()
declare -a TRAIN_ARGS=()
declare -a EVAL_ARGS=()

usage() {
  cat <<'EOF'
Run train->eval in Docker under Slurm.

Options:
  --image <docker_image>                    (default: hoyeol_atk:251116)
  --host_base_dir <path>                    base path where AudioTokenization and datasets live
  --host_repo_dir <path>                    (default: <SLURM submit dir>)
  --host_data_dir <path>                    (default: <parent_of_repo>/datasets)
  --repo_dir_in_container <path>            (default: /workspace/AudioTokenization)
  --data_dir_in_container <path>            (default: /workspace/datasets)
  --conda_envs_host <path>                  Optional persistent host path mounted to /opt/conda/envs
  --conda_pkgs_host <path>                  Optional persistent host path mounted to /opt/conda/pkgs

  --train_env <name>                        (default: atk)
  --eval_env <name>                         (default: speech_eval)
  --run_id <id>                             (default: slurm_<jobid>)
  --run_dir_in_container <path>
  --eval_out_in_container <path>

  --train_config_path <path>                (default: config_default)
  --train_config_name <name>
  --input <path>
  --eval_stage <save|metrics|all>
  --num_workers <int>
  --length_mode <pad|truncate>
  --device <str>

  --target_avg_r <float>                    (default: from run config dtp_params.r)
  --tau_finetune | --no_tau_finetune
  --tau_min <float> --tau_max <float> --tau_step <float>
  --max_samples <int>
  --stats_subdir <name>

  --cfg_override <dotlist>                  (repeatable)
  --train_arg <arg>                         (repeatable)
  --eval_arg <arg>                          (repeatable)

  --setup_envs                              Run setup_conda_envs_minimal.sh first
  --flash_attn_version <ver>                (used with --setup_envs)
  --skip_train                              Skip train and run eval only
  --force                                   Re-run eval even if metrics exist
  --train_cuda_visible_devices <ids>        train-only CUDA_VISIBLE_DEVICES
  --container_shm_size <size>               (default: 16G)
  --container_name_prefix <name>            (default: atkrun)
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2 ;;
    --host_base_dir) HOST_BASE_DIR="$2"; shift 2 ;;
    --host_repo_dir) PROJECT_ROOT_HOST="$2"; shift 2 ;;
    --host_data_dir) DATASETS_ROOT_HOST="$2"; shift 2 ;;
    --repo_dir_in_container) PROJECT_ROOT_CONTAINER="$2"; shift 2 ;;
    --data_dir_in_container) DATASETS_ROOT_CONTAINER="$2"; shift 2 ;;
    --conda_envs_host) CONDA_ENVS_HOST="$2"; shift 2 ;;
    --conda_pkgs_host) CONDA_PKGS_HOST="$2"; shift 2 ;;
    --train_env) TRAIN_ENV="$2"; shift 2 ;;
    --eval_env) EVAL_ENV="$2"; shift 2 ;;
    --run_id)
      RUN_ID="$2"
      RUN_DIR_IN_CONTAINER="${PROJECT_ROOT_CONTAINER}/outputs/${RUN_ID}"
      EVAL_OUT_IN_CONTAINER="${RUN_DIR_IN_CONTAINER}/eval_ft"
      shift 2
      ;;
    --run_dir_in_container) RUN_DIR_IN_CONTAINER="$2"; shift 2 ;;
    --eval_out_in_container) EVAL_OUT_IN_CONTAINER="$2"; shift 2 ;;
    --train_config_path) TRAIN_CONFIG_PATH="$2"; shift 2 ;;
    --train_config_name) TRAIN_CONFIG_NAME="$2"; shift 2 ;;
    --input) INPUT="$2"; shift 2 ;;
    --eval_stage) EVAL_STAGE="$2"; shift 2 ;;
    --num_workers) NUM_WORKERS="$2"; shift 2 ;;
    --length_mode) LENGTH_MODE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --target_avg_r) TARGET_AVG_R="$2"; shift 2 ;;
    --tau_finetune) TAU_FINETUNE=1; shift ;;
    --no_tau_finetune) TAU_FINETUNE=0; shift ;;
    --tau_min) TAU_MIN="$2"; shift 2 ;;
    --tau_max) TAU_MAX="$2"; shift 2 ;;
    --tau_step) TAU_STEP="$2"; shift 2 ;;
    --max_samples) MAX_SAMPLES="$2"; shift 2 ;;
    --stats_subdir) STATS_SUBDIR="$2"; shift 2 ;;
    --cfg_override) CFG_OVERRIDES+=("$2"); shift 2 ;;
    --train_arg) TRAIN_ARGS+=("$2"); shift 2 ;;
    --eval_arg) EVAL_ARGS+=("$2"); shift 2 ;;
    --setup_envs) SETUP_ENVS=1; shift ;;
    --flash_attn_version) FLASH_ATTN_VERSION="$2"; shift 2 ;;
    --skip_train) SKIP_TRAIN=1; shift ;;
    --force) FORCE=1; shift ;;
    --train_cuda_visible_devices) TRAIN_CUDA_VISIBLE_DEVICES="$2"; shift 2 ;;
    --container_shm_size) CONTAINER_SHM_SIZE="$2"; shift 2 ;;
    --container_name_prefix) CONTAINER_NAME_PREFIX="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$PROJECT_ROOT_HOST" ]]; then
  if [[ -n "$HOST_BASE_DIR" ]]; then
    PROJECT_ROOT_HOST="$HOST_BASE_DIR/AudioTokenization"
  else
    PROJECT_ROOT_HOST="${SLURM_SUBMIT_DIR:-$PWD}"
  fi
fi

if [[ -z "$HOST_BASE_DIR" ]]; then
  HOST_BASE_DIR="$(dirname "$PROJECT_ROOT_HOST")"
fi

if [[ -z "$DATASETS_ROOT_HOST" ]]; then
  DATASETS_ROOT_HOST="$HOST_BASE_DIR/datasets"
fi

if [[ ! -d "$PROJECT_ROOT_HOST" ]]; then
  echo "[ERROR] host repo path not found: $PROJECT_ROOT_HOST" >&2
  exit 1
fi
if [[ -n "$DATASETS_ROOT_HOST" && ! -d "$DATASETS_ROOT_HOST" ]]; then
  echo "[ERROR] host data path not found: $DATASETS_ROOT_HOST" >&2
  exit 1
fi
if [[ -n "$CONDA_ENVS_HOST" ]]; then
  mkdir -p "$CONDA_ENVS_HOST"
fi
if [[ -n "$CONDA_PKGS_HOST" ]]; then
  mkdir -p "$CONDA_PKGS_HOST"
fi

if [[ -z "$TRAIN_CUDA_VISIBLE_DEVICES" ]]; then
  if [[ "${SLURM_GPUS_ON_NODE:-}" =~ ^[0-9]+$ ]] && [[ "${SLURM_GPUS_ON_NODE}" -gt 0 ]]; then
    TRAIN_CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((SLURM_GPUS_ON_NODE - 1)))"
  else
    TRAIN_CUDA_VISIBLE_DEVICES="0"
  fi
fi

GPU_LIST="${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
GPU_LIST="${GPU_LIST//,/\,}"
CONTAINER_NAME="${CONTAINER_NAME_PREFIX}_${SLURM_JOB_ID:-manual}"

DOCKER_BASE=(
  docker run --rm --gpus "device=${GPU_LIST}"
  --init -i
  --cpus "${SLURM_CPUS_PER_TASK:-16}"
  --memory "${SLURM_MEM_PER_NODE:-28672}m"
  --pid=host
  --name "$CONTAINER_NAME"
  --shm-size "$CONTAINER_SHM_SIZE"
  -e PYTHONUNBUFFERED=1
  -e WANDB_MODE
  -e HF_HOME=/workspace/hf_cache
  -e TORCH_HOME=/workspace/torch_cache
  -v "$PROJECT_ROOT_HOST:$PROJECT_ROOT_CONTAINER"
)

if [[ -n "$DATASETS_ROOT_HOST" ]]; then
  DOCKER_BASE+=( -v "$DATASETS_ROOT_HOST:$DATASETS_ROOT_CONTAINER" )
fi
if [[ -n "$CONDA_ENVS_HOST" ]]; then
  DOCKER_BASE+=( -v "$CONDA_ENVS_HOST:/opt/conda/envs" )
fi
if [[ -n "$CONDA_PKGS_HOST" ]]; then
  DOCKER_BASE+=( -v "$CONDA_PKGS_HOST:/opt/conda/pkgs" )
fi
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  DOCKER_BASE+=( -e WANDB_API_KEY )
fi

if [[ "$SETUP_ENVS" -eq 1 ]]; then
  echo "[SETUP] Running setup_conda_envs_minimal.sh in container"
  srun --unbuffered --cpu-bind=cores,verbose \
    "${DOCKER_BASE[@]}" \
    "$IMAGE" \
    bash "$PROJECT_ROOT_CONTAINER/setup_conda_envs_minimal.sh" \
      --train_env "$TRAIN_ENV" \
      --eval_env "$EVAL_ENV" \
      --flash_attn_version "$FLASH_ATTN_VERSION"
fi

RUN_CMD=(
  bash "$PROJECT_ROOT_CONTAINER/run_train_then_eval.sh"
  --train_env "$TRAIN_ENV"
  --eval_env "$EVAL_ENV"
  --train_config_path "$TRAIN_CONFIG_PATH"
  --input "$INPUT"
  --run_dir "$RUN_DIR_IN_CONTAINER"
  --eval_out "$EVAL_OUT_IN_CONTAINER"
  --eval_stage "$EVAL_STAGE"
  --num_workers "$NUM_WORKERS"
  --length_mode "$LENGTH_MODE"
  --train_cuda_visible_devices "$TRAIN_CUDA_VISIBLE_DEVICES"
  --tau_min "$TAU_MIN"
  --tau_max "$TAU_MAX"
  --tau_step "$TAU_STEP"
  --stats_subdir "$STATS_SUBDIR"
)

if [[ -n "$TRAIN_CONFIG_NAME" ]]; then RUN_CMD+=(--train_config_name "$TRAIN_CONFIG_NAME"); fi
if [[ -n "$TARGET_AVG_R" ]]; then RUN_CMD+=(--target_avg_r "$TARGET_AVG_R"); fi
if [[ "$TAU_FINETUNE" -eq 0 ]]; then RUN_CMD+=(--no_tau_finetune); fi
if [[ -n "$MAX_SAMPLES" ]]; then RUN_CMD+=(--max_samples "$MAX_SAMPLES"); fi
if [[ -n "$DEVICE" ]]; then RUN_CMD+=(--device "$DEVICE"); fi
if [[ "$SKIP_TRAIN" -eq 1 ]]; then RUN_CMD+=(--skip_train); fi
if [[ "$FORCE" -eq 1 ]]; then RUN_CMD+=(--force); fi
for ov in "${CFG_OVERRIDES[@]}"; do RUN_CMD+=(--cfg_override "$ov"); done
for arg in "${TRAIN_ARGS[@]}"; do RUN_CMD+=(--train_arg "$arg"); done
for arg in "${EVAL_ARGS[@]}"; do RUN_CMD+=(--eval_arg "$arg"); done

echo "[INFO] IMAGE=$IMAGE"
echo "[INFO] PROJECT_ROOT_HOST=$PROJECT_ROOT_HOST"
echo "[INFO] DATASETS_ROOT_HOST=$DATASETS_ROOT_HOST"
if [[ -n "$CONDA_ENVS_HOST" ]]; then
  echo "[INFO] Persistent conda envs: $CONDA_ENVS_HOST -> /opt/conda/envs"
fi
if [[ -n "$CONDA_PKGS_HOST" ]]; then
  echo "[INFO] Persistent conda pkgs: $CONDA_PKGS_HOST -> /opt/conda/pkgs"
fi
echo "[INFO] RUN_DIR_IN_CONTAINER=$RUN_DIR_IN_CONTAINER"
echo "[INFO] TRAIN_CUDA_VISIBLE_DEVICES=$TRAIN_CUDA_VISIBLE_DEVICES"
echo "[INFO] SLURM requested GPUs=${SLURM_GPUS_ON_NODE:-unknown}, docker device list=$GPU_LIST"

srun --unbuffered --cpu-bind=cores,verbose \
  "${DOCKER_BASE[@]}" \
  "$IMAGE" \
  "${RUN_CMD[@]}"
