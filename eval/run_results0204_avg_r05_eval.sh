#!/usr/bin/env bash
set -euo pipefail

# Single-run eval helper for results0204:
# - Ensure legacy hydra/config.yaml is converted to the new model.quantizer style.
# - Search fixed_tau that matches a target avg_r (default: 0.5).
# - Evaluate on LibriSpeech test-clean with the found fixed_tau.
#
# Prerequisite:
#   conda activate speech_eval
#
# Run from repo root:
#   cd /home/hoyso/projects/AudioTokenization
#   bash eval/run_results0204_avg_r05_eval.sh
#
# Or override run dir:
#   bash eval/run_results0204_avg_r05_eval.sh /path/to/run_dir

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_DIR_DEFAULT="/home/hoyso/projects/AudioTokenization/results/results0204/default-100hz-PLE50hz-vq16384-mls"
RUN_DIR="${1:-$RUN_DIR_DEFAULT}"

INPUT="${INPUT:-DTMAE/filelists/librispeech_test_clean.txt}"
TARGET_AVG_R="${TARGET_AVG_R:-0.5}"

# Tau search parameters.
TAU_STEP="${TAU_STEP:-0.001}"
TAU_MIN="${TAU_MIN:-0.001}"
TAU_MAX="${TAU_MAX:-1.0}"

# Optional: limit sequences per tau evaluation (speeds up search).
MAX_SAMPLES="${MAX_SAMPLES:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"

require_env() {
  if ! "$PYTHON_BIN" -c "import torch, torchaudio, omegaconf" >/dev/null 2>&1; then
    echo "[ERROR] Python deps not found. Did you run: conda activate speech_eval ?" >&2
    echo "        (torch/torchaudio/omegaconf must import successfully)" >&2
    exit 1
  fi
}

ensure_run_dir() {
  local run_dir="$1"
  local cfg_path="$run_dir/hydra/config.yaml"
  local ckpt_path="$run_dir/pl_log/last.ckpt"
  if [[ ! -f "$cfg_path" ]]; then
    echo "[ERROR] Missing config: $cfg_path" >&2
    exit 1
  fi
  if [[ ! -f "$ckpt_path" ]]; then
    echo "[ERROR] Missing checkpoint: $ckpt_path" >&2
    exit 1
  fi
}

ensure_new_config() {
  # Convert legacy hydra/config.yaml (codec_decoder VQ fields) to new model.quantizer style.
  # This is required because eval scripts expect cfg.model.quantizer.
  local run_dir="$1"
  local cfg_path="$run_dir/hydra/config.yaml"
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
tau = d["best"]["fixed_tau"]
print(tau)
PY
}

maybe_flag_max_samples() {
  if [[ -n "${MAX_SAMPLES}" ]]; then
    echo "--max_samples" "${MAX_SAMPLES}"
  fi
}

require_env
ensure_run_dir "$RUN_DIR"

ensure_new_config "$RUN_DIR"

STATS_OUT="${STATS_OUT:-$RUN_DIR/dtp_stats_avg_r${TARGET_AVG_R}}"
EVAL_OUT="${EVAL_OUT:-$RUN_DIR/eval_avg_r${TARGET_AVG_R}}"

SUMMARY_JSON="$STATS_OUT/summary.json"

if [[ ! -f "$SUMMARY_JSON" ]]; then
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
    --no_resume \
    $(maybe_flag_max_samples)
else
  echo "=== [SKIP] Found existing tau search summary at: $SUMMARY_JSON ==="
fi

if [[ ! -f "$SUMMARY_JSON" ]]; then
  echo "[ERROR] Missing summary.json at: $SUMMARY_JSON" >&2
  exit 1
fi

TAU="$(json_get_best_tau "$SUMMARY_JSON")"
echo "=== [FOUND] fixed_tau=$TAU (from $SUMMARY_JSON) ==="

if [[ -f "$EVAL_OUT/metrics.json" ]]; then
  echo "=== [SKIP] metrics.json already exists at $EVAL_OUT/metrics.json ==="
  exit 0
fi

echo
echo "=== [EVAL] run_dir=$RUN_DIR out=$EVAL_OUT fixed_tau=$TAU input=$INPUT ==="
"$PYTHON_BIN" eval/eval.py \
  --input "$INPUT" \
  --run_dir "$RUN_DIR" \
  --output_dir "$EVAL_OUT" \
  --cfg_override "model.resampler.dtp_params.fixed_tau=${TAU}"

echo
echo "Done."


