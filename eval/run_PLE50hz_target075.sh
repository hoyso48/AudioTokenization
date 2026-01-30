#!/usr/bin/env bash
set -euo pipefail

# Run a single evaluation on results/results1203/PLE50hz with target_avg_r=0.75.
#
# It does:
# - dtp_stats_search.py to find fixed_tau for target_avg_r=0.75
# - eval.py with that fixed_tau
#
# Prerequisite:
#   conda activate speech_eval
#
# Run:
#   cd /home/hoyso/projects/AudioTokenization
#   bash eval/run_PLE50hz_target075.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

RUN_DIR="results/results1203/PLE50hz"
INPUT="DTMAE/filelists/librispeech_test_clean.txt"

TARGET_AVG_R="0.75"
TAU_STEP="0.001"
TAU_MIN="0.001"
TAU_MAX="1.0"

STATS_OUT="${RUN_DIR}/dtp_stats_ft_r075"
EVAL_OUT="${RUN_DIR}/eval_ft_r075"

require_env() {
  if ! "$PYTHON_BIN" -c "import torch, torchaudio, omegaconf" >/dev/null 2>&1; then
    echo "[ERROR] Python deps not found. Did you run: conda activate speech_eval ?" >&2
    exit 1
  fi
}

ensure_new_config() {
  local cfg_path="$RUN_DIR/hydra/config.yaml"
  if [[ ! -f "$cfg_path" ]]; then
    echo "[ERROR] Missing config: $cfg_path" >&2
    exit 1
  fi
  echo "=== [CONFIG] update_legacy_config.py on $cfg_path ==="
  "$PYTHON_BIN" utils/update_legacy_config.py --path "$cfg_path"
}

json_get_best_tau() {
  local summary_json="$1"
  "$PYTHON_BIN" - "$summary_json" <<'PY'
import json, sys
path = sys.argv[1]
with open(path, "r") as f:
    d = json.load(f)
print(d["best"]["fixed_tau"])
PY
}

require_env
ensure_new_config

if [[ -f "$EVAL_OUT/metrics.json" ]]; then
  echo "=== [SKIP] metrics.json already exists at $EVAL_OUT/metrics.json ==="
  exit 0
fi

echo
echo "=== [SEARCH] run_dir=$RUN_DIR out=$STATS_OUT target_avg_r=$TARGET_AVG_R ==="
"$PYTHON_BIN" eval/dtp_stats_search.py \
  --input "$INPUT" \
  --run_dir "$RUN_DIR" \
  --output_dir "$STATS_OUT" \
  --target_avg_r "$TARGET_AVG_R" \
  --tau_min "$TAU_MIN" \
  --tau_max "$TAU_MAX" \
  --tau_step "$TAU_STEP" \
  --bootstrap_update_test_time \
  --bootstrap_override_update_test_time \
  --no_resume

SUMMARY_JSON="$STATS_OUT/summary.json"
if [[ ! -f "$SUMMARY_JSON" ]]; then
  echo "[ERROR] Missing summary.json at: $SUMMARY_JSON" >&2
  exit 1
fi

TAU="$(json_get_best_tau "$SUMMARY_JSON")"
echo "=== [FOUND] fixed_tau=$TAU (from $SUMMARY_JSON) ==="

echo
echo "=== [EVAL] run_dir=$RUN_DIR out=$EVAL_OUT fixed_tau=$TAU ==="
"$PYTHON_BIN" eval/eval.py \
  --input "$INPUT" \
  --run_dir "$RUN_DIR" \
  --output_dir "$EVAL_OUT" \
  --cfg_override "model.resampler.dtp_params.fixed_tau=${TAU}"

echo
echo "PLE50hz target_avg_r=0.75 finished."


