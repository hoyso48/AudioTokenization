#!/usr/bin/env bash
#SBATCH --job-name=atk-setup
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --output=atk_setup_%j.out

set -euo pipefail

IMAGE="${IMAGE:-hoyeol_atk:251116}"
HOST_BASE_DIR="${HOST_BASE_DIR:-}"
PROJECT_ROOT_HOST="${PROJECT_ROOT_HOST:-}"
DATASETS_ROOT_HOST="${DATASETS_ROOT_HOST:-}"
PROJECT_ROOT_CONTAINER="/workspace/AudioTokenization"
DATASETS_ROOT_CONTAINER="/workspace/datasets"
CONDA_ENVS_HOST="${CONDA_ENVS_HOST:-}"
CONDA_PKGS_HOST="${CONDA_PKGS_HOST:-}"

TRAIN_ENV="atk"
EVAL_ENV="speech_eval"
FLASH_ATTN_VERSION="2.8.3"
FORCE_REINSTALL_EVAL_TORCH=0
NO_FFMPEG=0
NO_FIX_NCCL=0
SKIP_TRAIN=0
SKIP_EVAL=0

CONTAINER_SHM_SIZE="16G"
CONTAINER_NAME_PREFIX="atksetup"

usage() {
  cat <<'EOF'
Setup conda envs inside Docker for Slurm workloads.

Defaults:
  --image hoyeol_atk:251116
  --host_repo_dir <SLURM submit dir>
  --host_data_dir <parent_of_repo>/datasets

Override:
  --host_base_dir <path>                base path where AudioTokenization and datasets live
  --image <docker_image>
  --host_repo_dir <path>

Optional:
  --host_data_dir <path>
  --repo_dir_in_container <path>      (default: /workspace/AudioTokenization)
  --data_dir_in_container <path>      (default: /workspace/datasets)
  --conda_envs_host <path>            Optional persistent host path mounted to /opt/conda/envs
  --conda_pkgs_host <path>            Optional persistent host path mounted to /opt/conda/pkgs
  --train_env <name>                  (default: atk)
  --eval_env <name>                   (default: speech_eval)
  --flash_attn_version <ver>          (default: 2.8.3)
  --force_reinstall_eval_torch
  --no_ffmpeg
  --no_fix_nccl
  --skip_train
  --skip_eval
  --container_shm_size <size>         (default: 16G)
  --container_name_prefix <name>      (default: atksetup)
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      IMAGE="$2"
      shift 2
      ;;
    --host_base_dir)
      HOST_BASE_DIR="$2"
      shift 2
      ;;
    --host_repo_dir)
      PROJECT_ROOT_HOST="$2"
      shift 2
      ;;
    --host_data_dir)
      DATASETS_ROOT_HOST="$2"
      shift 2
      ;;
    --repo_dir_in_container)
      PROJECT_ROOT_CONTAINER="$2"
      shift 2
      ;;
    --data_dir_in_container)
      DATASETS_ROOT_CONTAINER="$2"
      shift 2
      ;;
    --conda_envs_host)
      CONDA_ENVS_HOST="$2"
      shift 2
      ;;
    --conda_pkgs_host)
      CONDA_PKGS_HOST="$2"
      shift 2
      ;;
    --train_env)
      TRAIN_ENV="$2"
      shift 2
      ;;
    --eval_env)
      EVAL_ENV="$2"
      shift 2
      ;;
    --flash_attn_version)
      FLASH_ATTN_VERSION="$2"
      shift 2
      ;;
    --force_reinstall_eval_torch)
      FORCE_REINSTALL_EVAL_TORCH=1
      shift
      ;;
    --no_ffmpeg)
      NO_FFMPEG=1
      shift
      ;;
    --no_fix_nccl)
      NO_FIX_NCCL=1
      shift
      ;;
    --skip_train)
      SKIP_TRAIN=1
      shift
      ;;
    --skip_eval)
      SKIP_EVAL=1
      shift
      ;;
    --container_shm_size)
      CONTAINER_SHM_SIZE="$2"
      shift 2
      ;;
    --container_name_prefix)
      CONTAINER_NAME_PREFIX="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1" >&2
      usage
      exit 1
      ;;
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

GPU_LIST="${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
GPU_LIST="${GPU_LIST//,/\,}"
CONTAINER_NAME="${CONTAINER_NAME_PREFIX}_${SLURM_JOB_ID:-manual}"

DOCKER_ARGS=(
  docker run --rm --gpus "device=${GPU_LIST}"
  --init -i
  --cpus "${SLURM_CPUS_PER_TASK:-8}"
  --memory "${SLURM_MEM_PER_NODE:-32768}m"
  --pid=host
  --name "$CONTAINER_NAME"
  --shm-size "$CONTAINER_SHM_SIZE"
  -e PYTHONUNBUFFERED=1
  -e WANDB_MODE
  -v "$PROJECT_ROOT_HOST:$PROJECT_ROOT_CONTAINER"
)

if [[ -n "$DATASETS_ROOT_HOST" ]]; then
  DOCKER_ARGS+=( -v "$DATASETS_ROOT_HOST:$DATASETS_ROOT_CONTAINER" )
fi
if [[ -n "$CONDA_ENVS_HOST" ]]; then
  DOCKER_ARGS+=( -v "$CONDA_ENVS_HOST:/opt/conda/envs" )
fi
if [[ -n "$CONDA_PKGS_HOST" ]]; then
  DOCKER_ARGS+=( -v "$CONDA_PKGS_HOST:/opt/conda/pkgs" )
fi

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  DOCKER_ARGS+=( -e WANDB_API_KEY )
fi

CMD=(
  bash "$PROJECT_ROOT_CONTAINER/setup_conda_envs_minimal.sh"
  --train_env "$TRAIN_ENV"
  --eval_env "$EVAL_ENV"
  --flash_attn_version "$FLASH_ATTN_VERSION"
)

if [[ "$FORCE_REINSTALL_EVAL_TORCH" -eq 1 ]]; then
  CMD+=(--force_reinstall_eval_torch)
fi
if [[ "$NO_FFMPEG" -eq 1 ]]; then
  CMD+=(--no_ffmpeg)
fi
if [[ "$NO_FIX_NCCL" -eq 1 ]]; then
  CMD+=(--no_fix_nccl)
fi
if [[ "$SKIP_TRAIN" -eq 1 ]]; then
  CMD+=(--skip_train)
fi
if [[ "$SKIP_EVAL" -eq 1 ]]; then
  CMD+=(--skip_eval)
fi

echo "[INFO] IMAGE=$IMAGE"
echo "[INFO] PROJECT_ROOT_HOST=$PROJECT_ROOT_HOST"
echo "[INFO] TRAIN_ENV=$TRAIN_ENV EVAL_ENV=$EVAL_ENV"
if [[ -n "$CONDA_ENVS_HOST" ]]; then
  echo "[INFO] Persistent conda envs: $CONDA_ENVS_HOST -> /opt/conda/envs"
fi
if [[ -n "$CONDA_PKGS_HOST" ]]; then
  echo "[INFO] Persistent conda pkgs: $CONDA_PKGS_HOST -> /opt/conda/pkgs"
fi

srun --unbuffered --cpu-bind=cores,verbose \
  "${DOCKER_ARGS[@]}" \
  "$IMAGE" \
  "${CMD[@]}"
