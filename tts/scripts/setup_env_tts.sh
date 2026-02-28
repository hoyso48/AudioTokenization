#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-audiotok_tts}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found in PATH"
  exit 1
fi

echo "[INFO] Creating/using conda env: ${ENV_NAME} (python ${PYTHON_VERSION})"
conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" || true

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

pip install --upgrade pip
pip install -r "${ROOT_DIR}/requirements_tts.txt"

if ! command -v wget >/dev/null 2>&1; then
  echo "[ERROR] wget is required for NLTK resource bootstrap"
  exit 1
fi

NLTK_DATA_DIR="${NLTK_DATA:-${HOME}/nltk_data}"
mkdir -p "${NLTK_DATA_DIR}/corpora" "${NLTK_DATA_DIR}/taggers"

download_nltk_zip() {
  local package_group="$1"
  local package_name="$2"
  local out_dir="$3"
  local tmp_zip
  tmp_zip="/tmp/${package_name}.zip"

  local url="https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/${package_group}/${package_name}.zip"
  echo "[INFO] Downloading NLTK package ${package_name} via wget"
  wget -nv -O "${tmp_zip}" "${url}"

  python - "${tmp_zip}" "${out_dir}" <<'PY'
import sys
import zipfile
from pathlib import Path

zip_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(out_dir)
print(f"Extracted {zip_path.name} -> {out_dir}")
PY

  rm -f "${tmp_zip}"
}

download_nltk_zip "corpora" "cmudict" "${NLTK_DATA_DIR}/corpora"
download_nltk_zip "taggers" "averaged_perceptron_tagger_eng" "${NLTK_DATA_DIR}/taggers"

export NLTK_DATA="${NLTK_DATA_DIR}"
echo "[INFO] NLTK_DATA=${NLTK_DATA_DIR}"

echo "[INFO] Environment ready: ${ENV_NAME}"
echo "[INFO] For offline runs export:"
echo "  export HF_DATASETS_OFFLINE=1"
echo "  export TRANSFORMERS_OFFLINE=1"
