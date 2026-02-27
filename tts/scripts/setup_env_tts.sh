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

python - <<'PY'
import nltk
for pkg in ["cmudict", "averaged_perceptron_tagger_eng"]:
    nltk.download(pkg, quiet=False)
PY

echo "[INFO] Environment ready: ${ENV_NAME}"
echo "[INFO] For offline runs export:"
echo "  export HF_DATASETS_OFFLINE=1"
echo "  export TRANSFORMERS_OFFLINE=1"
