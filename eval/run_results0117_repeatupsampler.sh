#!/usr/bin/env bash
set -euo pipefail

# Evaluate only:
#   results/results0117/default_PLE_50hz_vq163843_repeatupsampler
#
# Runs both:
# - Unfiltered LibriSpeech test-clean (DTMAE/filelists/librispeech_test_clean.txt)
# - LSfiltered subset (4-10s) built via torchaudio.info metadata
#
# Prerequisite:
#   conda activate speech_eval
#
# Run:
#   cd /home/hoyso/projects/AudioTokenization
#   bash eval/run_results0117_repeatupsampler.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

RUN_DIR="results/results0117/default_PLE_50hz_vq163843_repeatupsampler"

INPUT_RAW="DTMAE/filelists/librispeech_test_clean.txt"
INPUT_FILTERED="DTMAE/filelists/librispeech_test_clean_filtered_4s10s.txt"

FILTER_MIN_SEC="4"
FILTER_MAX_SEC="10"

TARGET_AVG_R="0.5"
TAU_STEP="0.001"
TAU_MIN="0.001"
TAU_MAX="1.0"

require_env() {
  if ! "$PYTHON_BIN" -c "import torch, torchaudio, omegaconf" >/dev/null 2>&1; then
    echo "[ERROR] Python deps not found. Did you run: conda activate speech_eval ?" >&2
    echo "        (torch/torchaudio/omegaconf must import successfully)" >&2
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

ensure_filtered_input() {
  echo "=== [FILTER] Building $INPUT_FILTERED from $INPUT_RAW (dur ${FILTER_MIN_SEC}-${FILTER_MAX_SEC}s) ==="
  "$PYTHON_BIN" eval/filter_filelist_by_duration.py \
    --input_list "$INPUT_RAW" \
    --output_list "$INPUT_FILTERED" \
    --min_sec "$FILTER_MIN_SEC" \
    --max_sec "$FILTER_MAX_SEC" \
    --overwrite
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

should_skip_eval() {
  local eval_out="$1"
  if [[ -f "$eval_out/metrics.json" ]]; then
    echo "=== [SKIP] metrics.json already exists at $eval_out/metrics.json ==="
    return 0
  fi
  return 1
}

run_search_and_eval() {
  local input_list="$1"; shift
  local stats_out="$1"; shift
  local eval_out="$1"; shift

  if should_skip_eval "$eval_out"; then
    return 0
  fi

  ensure_new_config

  echo
  echo "=== [SEARCH] run_dir=$RUN_DIR out=$stats_out target_avg_r=$TARGET_AVG_R input=$input_list ==="
  "$PYTHON_BIN" eval/dtp_stats_search.py \
    --input "$input_list" \
    --run_dir "$RUN_DIR" \
    --output_dir "$stats_out" \
    --target_avg_r "$TARGET_AVG_R" \
    --tau_min "$TAU_MIN" \
    --tau_max "$TAU_MAX" \
    --tau_step "$TAU_STEP" \
    --bootstrap_update_test_time \
    --bootstrap_override_update_test_time \
    --no_resume

  local summary_json="$stats_out/summary.json"
  if [[ ! -f "$summary_json" ]]; then
    echo "[ERROR] Missing summary.json at: $summary_json" >&2
    exit 1
  fi

  local tau
  tau="$(json_get_best_tau "$summary_json")"
  echo "=== [FOUND] fixed_tau=$tau (from $summary_json) ==="

  echo
  echo "=== [EVAL] run_dir=$RUN_DIR out=$eval_out fixed_tau=$tau input=$input_list ==="
  "$PYTHON_BIN" eval/eval.py \
    --input "$input_list" \
    --run_dir "$RUN_DIR" \
    --output_dir "$eval_out" \
    --cfg_override "model.resampler.dtp_params.fixed_tau=${tau}"
}

require_env

# Unfiltered
run_search_and_eval \
  "$INPUT_RAW" \
  "$RUN_DIR/dtp_stats_ft" \
  "$RUN_DIR/eval_ft"

# Filtered (4-10s)
ensure_filtered_input
run_search_and_eval \
  "$INPUT_FILTERED" \
  "$RUN_DIR/dtp_stats_ft_LSfiltered" \
  "$RUN_DIR/eval_ft_LSfiltered"

echo
echo "RepeatUpsampler run finished."


