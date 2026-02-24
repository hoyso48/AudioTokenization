#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash semantic_eval/bootstrap_arch_semantic_eval.sh [options] [-- <extra run.py args>]

Self-contained helper (auto-skip enabled):
  1) Create/update dedicated virtualenv (skips when already prepared)
  2) Clone/update upstream ARCH repo (skips when already present)
  3) Download ARCH speech datasets (skips when all datasets already ready)
  4) Run semantic_eval/run.py

Stage semantics:
  - --stage all  : always run run.py with stage=all
  - --stage auto : choose build/eval/all automatically based on existing outputs

Options:
  --stage {build|eval|all|auto}      Pipeline stage (default: all)
  --run_dir PATH                      Codec run dir with hydra/config.yaml + pl_log/last.ckpt
  --data_root PATH                    Dataset root (default: /home/hoyso/projects/datasets)
  --output_dir PATH                   Output root (default: <repo>/semantic_eval/outputs/arch_speech)
  --env_dir PATH                      Virtualenv path (default: <repo>/.venv_semantic_eval)
  --python_bin BIN                    Python executable for venv creation (default: python3)
  --arch_repo PATH                    Upstream ARCH repo path (default: <semantic_eval>/third_party/ARCH)
  --datasets CSV                      Dataset subset (default: ravdess,emovo,audio_mnist,slurp)
  --skip_env_setup                    Hard skip creating/updating virtualenv
  --skip_arch_setup                   Hard skip cloning/updating ARCH repo
  --skip_download                     Hard skip dataset download step
  --force_env_setup                   Force env setup even if already completed
  --force_arch_setup                  Force ARCH clone/update even if already present
  --force_download                    Force dataset download script even if datasets are ready
  --force_run                         Force run.py execution even if outputs already exist
  -h, --help                          Show this message

Examples:
  bash semantic_eval/bootstrap_arch_semantic_eval.sh \
    --run_dir /path/to/run_dir \
    --stage all

  bash semantic_eval/bootstrap_arch_semantic_eval.sh \
    --run_dir /path/to/run_dir \
    --stage auto

  bash semantic_eval/bootstrap_arch_semantic_eval.sh \
    --stage build \
    --skip_env_setup \
    --skip_download
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

STAGE="all"
RUN_DIR=""
DATA_ROOT="/home/hoyso/projects/datasets"
OUTPUT_DIR="$SCRIPT_DIR/outputs/arch_speech"
ENV_DIR="$REPO_ROOT/.venv_semantic_eval"
PYTHON_BIN="python3"
ARCH_REPO="$SCRIPT_DIR/third_party/ARCH"
DATASETS="ravdess,emovo,audio_mnist,slurp"
SKIP_ENV_SETUP=0
SKIP_ARCH_SETUP=0
SKIP_DOWNLOAD=0
FORCE_ENV_SETUP=0
FORCE_ARCH_SETUP=0
FORCE_DOWNLOAD=0
FORCE_RUN=0
EXTRA_ARGS=()
FORCE_RECOMPUTE_EXISTING_IN_RUN=0

log() {
  printf "[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

hash_file() {
  local file_path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$file_path" | awk '{print $1}'
  else
    "$PYTHON_BIN" - "$file_path" <<'PY'
import hashlib
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
print(hashlib.sha256(path.read_bytes()).hexdigest())
PY
  fi
}

read_meta_value() {
  local key="$1"
  local meta_path="$2"
  if [[ ! -f "$meta_path" ]]; then
    return 1
  fi
  awk -F'=' -v k="$key" '$1==k {print substr($0, index($0,$2)); exit}' "$meta_path"
}

has_any_audio() {
  local dir="$1"
  [[ -d "$dir" ]] || return 1
  [[ -n "$(find "$dir" -type f \( -name "*.wav" -o -name "*.flac" \) -print -quit 2>/dev/null)" ]]
}

datasets_ready() {
  local root="$1"

  local ravdess_ok=0
  local emovo_ok=0
  local audio_mnist_ok=0
  local slurp_ok=0

  if has_any_audio "$root/ravdess"; then
    ravdess_ok=1
  fi

  if has_any_audio "$root/emovo/EMOVO" || has_any_audio "$root/emovo"; then
    emovo_ok=1
  fi

  if [[ -f "$root/audio_mnist/audioMNIST_meta.txt" ]] && has_any_audio "$root/audio_mnist/data"; then
    audio_mnist_ok=1
  fi

  if [[ -d "$root/slurp/slurp_real" ]] && [[ -f "$root/slurp/train.jsonl" ]] && [[ -f "$root/slurp/devel.jsonl" ]] && [[ -f "$root/slurp/test.jsonl" ]]; then
    slurp_ok=1
  fi

  [[ "$ravdess_ok" -eq 1 ]] && [[ "$emovo_ok" -eq 1 ]] && [[ "$audio_mnist_ok" -eq 1 ]] && [[ "$slurp_ok" -eq 1 ]]
}

build_summary_matches() {
  local summary_path="$OUTPUT_DIR/build_summary.json"
  [[ -f "$summary_path" ]] || return 1

  "${PY_CMD[@]}" - "$summary_path" "$DATASETS" <<'PY'
import json
import sys

summary_path, datasets_csv = sys.argv[1], sys.argv[2]
want = [x.strip().lower() for x in datasets_csv.split(',') if x.strip()]
try:
    data = json.load(open(summary_path, 'r', encoding='utf-8'))
except Exception:
    sys.exit(1)
have = sorted([str(k).strip().lower() for k in data.keys()])
sys.exit(0 if sorted(want) == have else 1)
PY
}

eval_summary_matches() {
  local summary_path="$OUTPUT_DIR/results/summary.json"
  [[ -f "$summary_path" ]] || return 1
  [[ -n "$RUN_DIR" ]] || return 1

  "${PY_CMD[@]}" - "$summary_path" "$RUN_DIR" "$DATASETS" <<'PY'
import json
import os
import sys

summary_path, run_dir, datasets_csv = sys.argv[1], sys.argv[2], sys.argv[3]
want_datasets = [x.strip().lower() for x in datasets_csv.split(',') if x.strip()]

try:
    data = json.load(open(summary_path, 'r', encoding='utf-8'))
except Exception:
    sys.exit(1)

saved_run_dir = os.path.realpath(str(data.get('run_dir', '')))
want_run_dir = os.path.realpath(run_dir)
if saved_run_dir != want_run_dir:
    sys.exit(1)

saved_datasets = [str(x).strip().lower() for x in data.get('datasets', [])]
if sorted(saved_datasets) != sorted(want_datasets):
    sys.exit(1)

sys.exit(0)
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)
      STAGE="$2"
      shift 2
      ;;
    --run_dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --data_root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --env_dir)
      ENV_DIR="$2"
      shift 2
      ;;
    --python_bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --arch_repo)
      ARCH_REPO="$2"
      shift 2
      ;;
    --datasets)
      DATASETS="$2"
      shift 2
      ;;
    --skip_env_setup)
      SKIP_ENV_SETUP=1
      shift
      ;;
    --skip_arch_setup)
      SKIP_ARCH_SETUP=1
      shift
      ;;
    --skip_download)
      SKIP_DOWNLOAD=1
      shift
      ;;
    --force_env_setup)
      FORCE_ENV_SETUP=1
      shift
      ;;
    --force_arch_setup)
      FORCE_ARCH_SETUP=1
      shift
      ;;
    --force_download)
      FORCE_DOWNLOAD=1
      shift
      ;;
    --force_run)
      FORCE_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

for arg in "${EXTRA_ARGS[@]}"; do
  if [[ "$arg" == "--force_recompute_existing" ]]; then
    FORCE_RECOMPUTE_EXISTING_IN_RUN=1
    break
  fi
done

if [[ "$STAGE" != "build" ]] && [[ -z "$RUN_DIR" ]]; then
  echo "--run_dir is required unless --stage build" >&2
  exit 1
fi

if [[ "$STAGE" != "build" ]] && [[ "$STAGE" != "eval" ]] && [[ "$STAGE" != "all" ]] && [[ "$STAGE" != "auto" ]]; then
  echo "Unsupported --stage: $STAGE" >&2
  exit 1
fi

PY_CMD=("$PYTHON_BIN")
ENV_META_PATH="$ENV_DIR/.semantic_eval_env_meta"
REQ_HASH="$(hash_file "$SCRIPT_DIR/requirements.txt")"

if [[ "$SKIP_ENV_SETUP" -eq 0 ]]; then
  local_env_ready=0
  if [[ -x "$ENV_DIR/bin/python" ]] && [[ -f "$ENV_META_PATH" ]]; then
    saved_hash="$(read_meta_value "REQUIREMENTS_HASH" "$ENV_META_PATH" || true)"
    if [[ "$saved_hash" == "$REQ_HASH" ]]; then
      local_env_ready=1
    fi
  fi

  if [[ "$FORCE_ENV_SETUP" -eq 1 ]] || [[ "$local_env_ready" -eq 0 ]]; then
    log "Preparing virtualenv: $ENV_DIR"
    "$PYTHON_BIN" -m venv "$ENV_DIR"
    PY_CMD=("$ENV_DIR/bin/python")
    "${PY_CMD[@]}" -m pip install --upgrade pip
    "${PY_CMD[@]}" -m pip install -r "$SCRIPT_DIR/requirements.txt"

    {
      printf "REQUIREMENTS_HASH=%s\n" "$REQ_HASH"
      printf "PYTHON_BIN=%s\n" "$PYTHON_BIN"
    } > "$ENV_META_PATH"
  else
    PY_CMD=("$ENV_DIR/bin/python")
    log "Skip env setup: already completed"
  fi
else
  if [[ -x "$ENV_DIR/bin/python" ]]; then
    PY_CMD=("$ENV_DIR/bin/python")
    log "Skip env setup: --skip_env_setup"
  else
    if command -v python >/dev/null 2>&1; then
      PY_CMD=("python")
      log "--skip_env_setup set and env_dir missing; using active python from PATH"
    else
      PY_CMD=("$PYTHON_BIN")
      log "--skip_env_setup set but env not found; using $PYTHON_BIN"
    fi
  fi
fi

if [[ "$SKIP_ARCH_SETUP" -eq 0 ]]; then
  if ! command -v git >/dev/null 2>&1; then
    echo "git is required to clone/update ARCH repo" >&2
    exit 1
  fi

  arch_ready=0
  if [[ -d "$ARCH_REPO/.git" ]] && [[ -f "$ARCH_REPO/arch_eval/__init__.py" ]]; then
    arch_ready=1
  fi

  if [[ "$FORCE_ARCH_SETUP" -eq 1 ]]; then
    arch_ready=0
  fi

  if [[ "$arch_ready" -eq 1 ]]; then
    log "Skip ARCH setup: already completed"
  else
    mkdir -p "$(dirname "$ARCH_REPO")"
    if [[ -d "$ARCH_REPO/.git" ]]; then
      log "Updating ARCH repo"
      git -C "$ARCH_REPO" pull --ff-only
    else
      log "Cloning ARCH repo"
      git clone --depth 1 "https://github.com/MorenoLaQuatra/ARCH.git" "$ARCH_REPO"
    fi
  fi
else
  log "Skip ARCH setup: --skip_arch_setup"
fi

if [[ "$SKIP_DOWNLOAD" -eq 0 ]]; then
  if [[ "$FORCE_DOWNLOAD" -eq 1 ]]; then
    log "Running dataset download (forced)"
    bash "$SCRIPT_DIR/download_arch_datasets.sh" --root "$DATA_ROOT"
  elif datasets_ready "$DATA_ROOT"; then
    log "Skip dataset download: all datasets already ready"
  else
    log "Running dataset download"
    bash "$SCRIPT_DIR/download_arch_datasets.sh" --root "$DATA_ROOT"
  fi
else
  log "Skip dataset download: --skip_download"
fi

RUN_STAGE="$STAGE"
if [[ "$STAGE" == "auto" ]] && [[ "$FORCE_RUN" -eq 0 ]]; then
  build_done=0
  eval_done=0

  if build_summary_matches; then
    build_done=1
  fi
  if eval_summary_matches; then
    eval_done=1
  fi

  if [[ "$FORCE_RECOMPUTE_EXISTING_IN_RUN" -eq 1 ]]; then
    eval_done=0
    log "Auto stage note: forcing eval due to run.py --force_recompute_existing"
  fi

  if [[ "$build_done" -eq 1 ]] && [[ "$eval_done" -eq 1 ]]; then
    log "Skip run.py: build+eval already completed"
    exit 0
  fi
  if [[ "$build_done" -eq 1 ]] && [[ "$eval_done" -eq 0 ]]; then
    RUN_STAGE="eval"
    log "Auto stage resolved to eval (build already done)"
  elif [[ "$build_done" -eq 0 ]] && [[ "$eval_done" -eq 1 ]]; then
    RUN_STAGE="build"
    log "Auto stage resolved to build (eval already done)"
  else
    RUN_STAGE="all"
    log "Auto stage resolved to all"
  fi
fi

CMD=("${PY_CMD[@]}" "$SCRIPT_DIR/run.py" --stage "$RUN_STAGE" --datasets "$DATASETS" --data_root "$DATA_ROOT" --output_dir "$OUTPUT_DIR" --arch_repo "$ARCH_REPO")
if [[ -n "$RUN_DIR" ]]; then
  CMD+=(--run_dir "$RUN_DIR")
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

log "Executing: ${CMD[*]}"
"${CMD[@]}"
