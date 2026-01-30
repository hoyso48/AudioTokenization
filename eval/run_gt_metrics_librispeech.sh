#!/usr/bin/env bash
set -euo pipefail

# Compute GT-only metrics (WER/UTMOS + GT-vs-GT STOI/PESQ) for LibriSpeech filelists.
#
# Usage:
#   cd /home/hoyso/projects/AudioTokenization
#   bash eval/run_gt_metrics_librispeech.sh
#
# Outputs:
#   eval/gt_metrics/librispeech_test_clean/metrics.json (+ per_file.jsonl)
#   eval/gt_metrics/librispeech_test_clean_filtered_4s10s/metrics.json (+ per_file.jsonl)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# User requested env activation in the runner.
# Conda's activate/deactivate scripts can reference unset variables; temporarily disable 'nounset'.
set +u
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate speech_eval
set -u

PYTHON_BIN="${PYTHON_BIN:-python}"

INPUT_A="/home/hoyso/projects/AudioTokenization/DTMAE/filelists/librispeech_test_clean.txt"
INPUT_B="/home/hoyso/projects/AudioTokenization/DTMAE/filelists/librispeech_test_clean_filtered_4s10s.txt"

OUT_ROOT="/home/hoyso/projects/AudioTokenization/eval/gt_metrics"

mkdir -p "$OUT_ROOT"

echo "=== [GT METRICS] $INPUT_A ==="
"$PYTHON_BIN" eval/gt_metrics.py \
  --input "$INPUT_A" \
  --output_dir "$OUT_ROOT/librispeech_test_clean"

echo
echo "=== [GT METRICS] $INPUT_B ==="
"$PYTHON_BIN" eval/gt_metrics.py \
  --input "$INPUT_B" \
  --output_dir "$OUT_ROOT/librispeech_test_clean_filtered_4s10s"

echo
echo "Done."


