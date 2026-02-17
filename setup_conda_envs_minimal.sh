#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
RECREATE_ON_PY_MISMATCH=0

TRAIN_ENV="${TRAIN_ENV:-atk}"
EVAL_ENV="${EVAL_ENV:-speech_eval}"

INSTALL_TRAIN=1
INSTALL_EVAL=1

TRAIN_REQUIREMENTS="${TRAIN_REQUIREMENTS:-$ROOT/requirements.txt}"

INSTALL_FFMPEG=1
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3}"

FIX_NCCL=1

FORCE_REINSTALL_EVAL_TORCH=0

TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.9.0+cu128}"
TORCHAUDIO_SPEC="${TORCHAUDIO_SPEC:-torchaudio==2.9.0+cu128}"
TORCHVISION_SPEC="${TORCHVISION_SPEC:-torchvision==0.24.0+cu128}"
TORCHCODEC_SPEC="${TORCHCODEC_SPEC:-torchcodec==0.9.1}"
S3PRL_VERSION_FALLBACK="${S3PRL_VERSION_FALLBACK:-0.4.18}"

TORCH_INDEX_URL_SET=0
TORCH_SPEC_SET=0
TORCHAUDIO_SPEC_SET=0
TORCHVISION_SPEC_SET=0
TORCHCODEC_SPEC_SET=0

usage() {
  cat <<'EOF'
Create/update minimal conda environments for train->eval automation.

Usage:
  bash setup_conda_envs_minimal.sh [options]

Options:
  --python_version <ver>          Python version for newly created envs (default: 3.10)
  --recreate_on_python_mismatch   Recreate env automatically if existing Python version mismatches
                                  NOTE: eval/fairseq currently requires Python 3.10

  --train_env <name>              Train env name (default: atk)
  --eval_env <name>               Eval env name (default: speech_eval)

  --skip_train                    Skip train env setup
  --skip_eval                     Skip eval env setup

  --train_requirements <path>     Requirements for train env (default: <repo>/requirements.txt)

  --no_ffmpeg                     Skip conda ffmpeg install
  --flash_attn_version <ver>      flash-attn version (default: 2.8.3)

  --no_fix_nccl                   Skip nvidia-nccl-cu12 reinstall in train env

  --force_reinstall_eval_torch    Reinstall torch/torchaudio/torchvision in eval env

  --torch_index_url <url>         PyTorch wheel index URL
  --torch_spec <spec>             Torch package spec
  --torchaudio_spec <spec>        Torchaudio package spec
  --torchvision_spec <spec>       Torchvision package spec
  --torchcodec_spec <spec>        TorchCodec package spec

  # NOTE: unless explicitly overridden by options above, eval torch stack
  # specs are auto-synced from --train_requirements (default: requirements.txt)

  -h, --help                      Show this help

Examples:
  bash setup_conda_envs_minimal.sh
EOF
}

log() {
  echo "[setup-env] $*"
}

err() {
  echo "[setup-env][ERROR] $*" >&2
}

require_conda() {
  if ! command -v conda >/dev/null 2>&1; then
    err "conda command not found in PATH"
    exit 1
  fi
}

env_exists() {
  local env_name="$1"
  conda run -n "$env_name" python -V >/dev/null 2>&1
}

env_python_version() {
  local env_name="$1"
  conda run -n "$env_name" python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
}

extract_major_minor_version() {
  local raw="$1"
  if [[ "$raw" =~ ([0-9]+)\.([0-9]+) ]]; then
    printf "%s.%s" "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
    return 0
  fi
  return 1
}

sync_torch_specs_from_requirements() {
  local req_path="$1"
  local req_line=""
  local parsed_index_url=""
  local parsed_torch_spec=""
  local parsed_torchaudio_spec=""
  local parsed_torchvision_spec=""
  local parsed_torchcodec_spec=""

  if [[ ! -f "$req_path" ]]; then
    return
  fi

  while IFS= read -r req_line || [[ -n "$req_line" ]]; do
    req_line="${req_line%%#*}"
    if [[ "$req_line" =~ ^[[:space:]]*--extra-index-url[[:space:]]+([^[:space:]]+) ]]; then
      parsed_index_url="${BASH_REMATCH[1]}"
      continue
    fi
    if [[ "$req_line" =~ ^[[:space:]]*(torch==[^[:space:]]+) ]]; then
      parsed_torch_spec="${BASH_REMATCH[1]}"
      continue
    fi
    if [[ "$req_line" =~ ^[[:space:]]*(torchaudio==[^[:space:]]+) ]]; then
      parsed_torchaudio_spec="${BASH_REMATCH[1]}"
      continue
    fi
    if [[ "$req_line" =~ ^[[:space:]]*(torchvision==[^[:space:]]+) ]]; then
      parsed_torchvision_spec="${BASH_REMATCH[1]}"
      continue
    fi
    if [[ "$req_line" =~ ^[[:space:]]*(torchcodec==[^[:space:]]+) ]]; then
      parsed_torchcodec_spec="${BASH_REMATCH[1]}"
      continue
    fi
  done < "$req_path"

  if [[ "$TORCH_INDEX_URL_SET" -eq 0 && -n "$parsed_index_url" ]]; then
    TORCH_INDEX_URL="$parsed_index_url"
  fi
  if [[ "$TORCH_SPEC_SET" -eq 0 && -n "$parsed_torch_spec" ]]; then
    TORCH_SPEC="$parsed_torch_spec"
  fi
  if [[ "$TORCHAUDIO_SPEC_SET" -eq 0 && -n "$parsed_torchaudio_spec" ]]; then
    TORCHAUDIO_SPEC="$parsed_torchaudio_spec"
  fi
  if [[ "$TORCHVISION_SPEC_SET" -eq 0 && -n "$parsed_torchvision_spec" ]]; then
    TORCHVISION_SPEC="$parsed_torchvision_spec"
  fi
  if [[ "$TORCHCODEC_SPEC_SET" -eq 0 && -n "$parsed_torchcodec_spec" ]]; then
    TORCHCODEC_SPEC="$parsed_torchcodec_spec"
  fi
}

require_eval_python_compat() {
  local env_name="$1"
  local py_raw=""
  local py_ver=""

  py_raw="$(env_python_version "$env_name" 2>&1 || true)"
  py_ver="$(extract_major_minor_version "$py_raw" || true)"
  if [[ -z "$py_ver" ]]; then
    py_raw="$(conda run -n "$env_name" python -V 2>&1 || true)"
    py_ver="$(extract_major_minor_version "$py_raw" || true)"
  fi

  if [[ -z "$py_ver" ]]; then
    err "Unable to determine Python version for eval env '$env_name'."
    err "Raw output: $py_raw"
    exit 1
  fi

  if [[ "$py_ver" != "3.10" ]]; then
    err "Eval env '$env_name' uses Python $py_ver, but vendored fairseq currently requires Python 3.10."
    err "Recreate env with: --python_version 3.10 --recreate_on_python_mismatch"
    exit 1
  fi
}

ensure_env() {
  local env_name="$1"
  if env_exists "$env_name"; then
    local current_py
    local current_py_raw
    current_py_raw="$(env_python_version "$env_name" 2>&1 || true)"
    current_py="$(extract_major_minor_version "$current_py_raw" || true)"
    if [[ -z "$current_py" ]]; then
      current_py_raw="$(conda run -n "$env_name" python -V 2>&1 || true)"
      current_py="$(extract_major_minor_version "$current_py_raw" || true)"
    fi
    if [[ -z "$current_py" ]]; then
      err "Unable to determine Python version for conda env '$env_name'."
      err "Raw output: $current_py_raw"
      exit 1
    fi
    if [[ "$current_py" != "$PYTHON_VERSION" ]]; then
      if [[ "$RECREATE_ON_PY_MISMATCH" -eq 1 ]]; then
        log "Recreating $env_name due to Python mismatch ($current_py != $PYTHON_VERSION)"
        conda remove -n "$env_name" --all -y
      else
        err "Conda env '$env_name' uses Python $current_py, but this setup expects $PYTHON_VERSION."
        err "Fix option A: remove env and rerun setup"
        err "  conda remove -n $env_name --all -y"
        err "Fix option B: rerun with --recreate_on_python_mismatch"
        exit 1
      fi
    else
      log "Conda env exists: $env_name (python=$current_py)"
      return
    fi
  fi

  if env_exists "$env_name"; then
    log "Conda env exists: $env_name"
    return
  fi
  log "Creating conda env: $env_name (python=$PYTHON_VERSION)"
  conda create -n "$env_name" -y "python=$PYTHON_VERSION" pip
}

run_in_env() {
  local env_name="$1"
  shift
  conda run --no-capture-output -n "$env_name" "$@"
}

ensure_pip_base() {
  local env_name="$1"
  run_in_env "$env_name" python -m pip install --upgrade pip setuptools wheel
}

ensure_ffmpeg() {
  local env_name="$1"
  if [[ "$INSTALL_FFMPEG" -eq 1 ]]; then
    log "Installing ffmpeg in $env_name"
    conda install -n "$env_name" -y ffmpeg
  fi
}

ensure_eval_subrepos() {
  local fairseq_pkg="$ROOT/eval/fairseq/fairseq/__init__.py"
  local s3prl_pkg="$ROOT/eval/s3prl/s3prl/__init__.py"

  if [[ ! -f "$fairseq_pkg" ]]; then
    err "Missing fairseq sources: $fairseq_pkg"
    err "This repository now vendors eval/fairseq directly."
    err "Refresh your working tree from origin and restore eval/fairseq."
    exit 1
  fi
  if [[ ! -f "$s3prl_pkg" ]]; then
    err "Missing s3prl sources: $s3prl_pkg"
    err "Your repo is missing eval/s3prl sources. Re-clone this repository with full contents."
    exit 1
  fi
}

ensure_s3prl_version_file() {
  local version_file="$ROOT/eval/s3prl/s3prl/version.txt"
  if [[ -f "$version_file" ]]; then
    return
  fi
  log "Missing s3prl version.txt; creating fallback at $version_file"
  mkdir -p "$(dirname "$version_file")"
  printf "%s\n" "$S3PRL_VERSION_FALLBACK" > "$version_file"
}

flash_attn_kernels_ready() {
  local env_name="$1"
  run_in_env "$env_name" python - <<'PY'
try:
    from flash_attn import flash_attn_qkvpacked_func
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
except Exception:
    raise SystemExit(1)
raise SystemExit(0)
PY
}

ensure_flash_attn() {
  local env_name="$1"

  # flash-attn setup.py imports psutil during metadata/build steps.
  # Keep these lightweight build helpers present before attempting install.
  run_in_env "$env_name" python -m pip install --upgrade psutil ninja packaging

  if flash_attn_kernels_ready "$env_name" >/dev/null 2>&1; then
    log "flash-attn kernels already available in $env_name"
    return
  fi

  log "Installing flash-attn==$FLASH_ATTN_VERSION in $env_name"
  run_in_env "$env_name" python -m pip uninstall -y flash-attn flash_attn >/dev/null 2>&1 || true
  run_in_env "$env_name" python -m pip install --force-reinstall --no-cache-dir --no-deps "flash-attn==$FLASH_ATTN_VERSION" --no-build-isolation

  if ! flash_attn_kernels_ready "$env_name" >/dev/null 2>&1; then
    err "flash-attn kernels are unavailable in $env_name after reinstall."
    err "This usually means torch/flash-attn ABI mismatch. Reinstall torch stack first, then rerun setup."
    exit 1
  fi
}

install_eval_torch_stack_if_needed() {
  if [[ "$FORCE_REINSTALL_EVAL_TORCH" -eq 0 ]]; then
    if run_in_env "$EVAL_ENV" python -c "import torch, torchaudio, torchvision, torchcodec" >/dev/null 2>&1; then
      log "Torch stack already available in $EVAL_ENV"
      return
    fi
  fi

  log "Installing torch stack in $EVAL_ENV"
  run_in_env "$EVAL_ENV" python -m pip install \
    --extra-index-url "$TORCH_INDEX_URL" \
    "$TORCH_SPEC" "$TORCHAUDIO_SPEC" "$TORCHVISION_SPEC" "$TORCHCODEC_SPEC"
}

setup_train_env() {
  ensure_env "$TRAIN_ENV"
  ensure_pip_base "$TRAIN_ENV"
  ensure_ffmpeg "$TRAIN_ENV"

  log "Installing train requirements in $TRAIN_ENV from $TRAIN_REQUIREMENTS"
  run_in_env "$TRAIN_ENV" python -m pip install -r "$TRAIN_REQUIREMENTS"

  # Kept explicit because existing bootstrap flow uses it.
  run_in_env "$TRAIN_ENV" python -m pip install psutil

  if [[ "$FIX_NCCL" -eq 1 ]]; then
    log "Reinstalling nvidia-nccl-cu12 in $TRAIN_ENV"
    run_in_env "$TRAIN_ENV" python -m pip install --upgrade --force-reinstall "nvidia-nccl-cu12>=2.26.2"
  fi

  ensure_flash_attn "$TRAIN_ENV"

  log "Verifying train env imports"
  run_in_env "$TRAIN_ENV" python - <<'PY'
import torch
import torchaudio
import hydra
import pytorch_lightning
print("[verify-train] torch:", torch.__version__)
print("[verify-train] torchaudio:", torchaudio.__version__)
print("[verify-train] ok")
PY

  run_in_env "$TRAIN_ENV" python - <<'PY'
import flash_attn
print("[verify-train] flash_attn:", getattr(flash_attn, "__version__", "unknown"))
PY
}

setup_eval_env() {
  ensure_eval_subrepos
  ensure_s3prl_version_file

  ensure_env "$EVAL_ENV"
  require_eval_python_compat "$EVAL_ENV"
  ensure_pip_base "$EVAL_ENV"
  log "Removing conflicting eval packages (if present): fairseq catalyst"
  run_in_env "$EVAL_ENV" python -m pip uninstall -y fairseq catalyst >/dev/null 2>&1 || true
  ensure_ffmpeg "$EVAL_ENV"

  install_eval_torch_stack_if_needed

  log "Installing minimal eval dependencies in $EVAL_ENV"
  run_in_env "$EVAL_ENV" python -m pip install \
    "numpy==1.26.4" \
    "einops" \
    "omegaconf==2.3.0" \
    "hydra-core==1.3.2" \
    "tqdm" \
    "pytorch_lightning==2.5.1.post0" \
    "transformers==4.57.3" \
    "torchmetrics" \
    "pesq" \
    "pystoi" \
    "jiwer<4" \
    "mel-cepstral-distance" \
    "soundfile" \
    "requests" \
    "fire" \
    "packaging" \
    "PyYAML" \
    "filelock" \
    "protobuf>=4.21.1" \
    "regex" \
    "sacrebleu>=1.4.12" \
    "bitarray" \
    "scikit-learn" \
    "scipy" \
    "cffi" \
    "cython" \
    "sentencepiece" \
    "librosa" \
    "vector_quantize_pytorch" \
    "wandb"

  ensure_flash_attn "$EVAL_ENV"

  log "Verifying eval env imports"
  run_in_env "$EVAL_ENV" python - <<PY
import sys
from pathlib import Path

root = Path(r"$ROOT").resolve()
eval_root = root / "eval"
fairseq_root = eval_root / "fairseq"
s3prl_root = eval_root / "s3prl"
paths = [
    eval_root / "speaker_verification",
    eval_root,
    root,
    root / "DTMAE",
]
if (fairseq_root / "fairseq" / "__init__.py").is_file():
    paths.insert(0, fairseq_root)
if (s3prl_root / "s3prl" / "__init__.py").is_file():
    paths.insert(0, s3prl_root)
for p in paths:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import torch
import torchaudio
import numpy
import omegaconf
import tqdm
import pytorch_lightning
import transformers
import jiwer
import soundfile
import requests
import fire
from torchmetrics.audio import ShortTimeObjectiveIntelligibility
from mel_cepstral_distance import compare_audio_files
import fairseq
import s3prl
from verification import init_model
from UTMOS import UTMOSScore

print("[verify-eval] torch:", torch.__version__)
print("[verify-eval] torchaudio:", torchaudio.__version__)
print("[verify-eval] ok")
PY

  run_in_env "$EVAL_ENV" python - <<'PY'
import flash_attn
print("[verify-eval] flash_attn:", getattr(flash_attn, "__version__", "unknown"))
PY

  if [[ ! -f "$ROOT/eval/wavlm_large_finetune.pth" ]]; then
    log "WARNING: missing $ROOT/eval/wavlm_large_finetune.pth (speaker similarity will fail)."
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python_version)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --recreate_on_python_mismatch)
      RECREATE_ON_PY_MISMATCH=1
      shift
      ;;
    --train_env)
      TRAIN_ENV="$2"
      shift 2
      ;;
    --eval_env)
      EVAL_ENV="$2"
      shift 2
      ;;
    --skip_train)
      INSTALL_TRAIN=0
      shift
      ;;
    --skip_eval)
      INSTALL_EVAL=0
      shift
      ;;
    --train_requirements)
      TRAIN_REQUIREMENTS="$2"
      shift 2
      ;;
    --no_ffmpeg)
      INSTALL_FFMPEG=0
      shift
      ;;
    --flash_attn_version)
      FLASH_ATTN_VERSION="$2"
      shift 2
      ;;
    --no_fix_nccl)
      FIX_NCCL=0
      shift
      ;;
    --force_reinstall_eval_torch)
      FORCE_REINSTALL_EVAL_TORCH=1
      shift
      ;;
    --torch_index_url)
      TORCH_INDEX_URL="$2"
      TORCH_INDEX_URL_SET=1
      shift 2
      ;;
    --torch_spec)
      TORCH_SPEC="$2"
      TORCH_SPEC_SET=1
      shift 2
      ;;
    --torchaudio_spec)
      TORCHAUDIO_SPEC="$2"
      TORCHAUDIO_SPEC_SET=1
      shift 2
      ;;
    --torchvision_spec)
      TORCHVISION_SPEC="$2"
      TORCHVISION_SPEC_SET=1
      shift 2
      ;;
    --torchcodec_spec)
      TORCHCODEC_SPEC="$2"
      TORCHCODEC_SPEC_SET=1
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      err "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

require_conda

if [[ ! -f "$TRAIN_REQUIREMENTS" ]]; then
  err "Train requirements not found: $TRAIN_REQUIREMENTS"
  exit 1
fi

sync_torch_specs_from_requirements "$TRAIN_REQUIREMENTS"

if [[ "$INSTALL_TRAIN" -eq 0 && "$INSTALL_EVAL" -eq 0 ]]; then
  err "Nothing to do: both --skip_train and --skip_eval set"
  exit 1
fi

log "ROOT=$ROOT"
log "TRAIN_ENV=$TRAIN_ENV"
log "EVAL_ENV=$EVAL_ENV"
log "Torch stack specs:"
log "  TORCH_INDEX_URL=$TORCH_INDEX_URL"
log "  TORCH_SPEC=$TORCH_SPEC"
log "  TORCHAUDIO_SPEC=$TORCHAUDIO_SPEC"
log "  TORCHVISION_SPEC=$TORCHVISION_SPEC"
log "  TORCHCODEC_SPEC=$TORCHCODEC_SPEC"

if [[ "$INSTALL_TRAIN" -eq 1 ]]; then
  log "--- Setup train env start ---"
  setup_train_env
  log "--- Setup train env done ---"
fi

if [[ "$INSTALL_EVAL" -eq 1 ]]; then
  log "--- Setup eval env start ---"
  setup_eval_env
  log "--- Setup eval env done ---"
fi

log "All requested environment setup steps completed."
