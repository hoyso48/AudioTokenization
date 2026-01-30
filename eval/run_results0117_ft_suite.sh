#!/usr/bin/env bash
set -euo pipefail

# Fine-tuning eval suite runner for results0117 experiments.
#
# What it does:
# - For specified run dirs, searches fixed_tau (target_avg_r=0.5) using eval/dtp_stats_search.py
# - Then runs eval/eval.py using that fixed_tau (via --cfg_override)
#
# Prerequisite:
#   conda activate speech_eval
# Run from repo root:
#   cd /home/hoyso/projects/AudioTokenization
#   bash eval/run_results0117_ft_suite.sh
#
# Outputs:
# - tau search:  <run_dir>/dtp_stats_ft*/summary.json + trials.jsonl
# - eval:        <run_dir>/eval_ft*/metrics.json (+ other eval artifacts)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

INPUT="DTMAE/filelists/librispeech_test_clean.txt"
TARGET_AVG_R="0.5"
TAU_STEP="0.001"
TAU_MIN="0.001"
TAU_MAX="1.0"

PYTHON_BIN="${PYTHON_BIN:-python}"

require_env() {
  if ! "$PYTHON_BIN" -c "import torch, torchaudio, omegaconf" >/dev/null 2>&1; then
    echo "[ERROR] Python deps not found. Did you run: conda activate speech_eval ?" >&2
    echo "        (torch/torchaudio/omegaconf must import successfully)" >&2
    exit 1
  fi
}

ensure_new_config() {
  # Convert legacy hydra/config.yaml (codec_decoder VQ fields) to new model.quantizer style.
  # This is required because eval scripts expect cfg.model.quantizer.
  local run_dir="$1"
  local cfg_path="$run_dir/hydra/config.yaml"
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
tau = d["best"]["fixed_tau"]
print(tau)
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
  local run_dir="$1"; shift
  local stats_out="$1"; shift
  local eval_out="$1"; shift

  local -a overrides=()
  while [[ $# -gt 0 ]]; do
    overrides+=("$1")
    shift
  done

  if should_skip_eval "$eval_out"; then
    return 0
  fi

  ensure_new_config "$run_dir"

  local -a cfg_args=()
  for ov in "${overrides[@]}"; do
    cfg_args+=(--cfg_override "$ov")
  done

  echo
  echo "=== [SEARCH] run_dir=$run_dir out=$stats_out target_avg_r=$TARGET_AVG_R ==="
  "$PYTHON_BIN" eval/dtp_stats_search.py \
    --input "$INPUT" \
    --run_dir "$run_dir" \
    --output_dir "$stats_out" \
    --target_avg_r "$TARGET_AVG_R" \
    --tau_min "$TAU_MIN" \
    --tau_max "$TAU_MAX" \
    --tau_step "$TAU_STEP" \
    --bootstrap_update_test_time \
    --bootstrap_override_update_test_time \
    --no_resume \
    "${cfg_args[@]}"

  local summary_json="$stats_out/summary.json"
  if [[ ! -f "$summary_json" ]]; then
    echo "[ERROR] Missing summary.json at: $summary_json" >&2
    exit 1
  fi

  local tau
  tau="$(json_get_best_tau "$summary_json")"
  echo "=== [FOUND] fixed_tau=$tau (from $summary_json) ==="

  echo
  echo "=== [EVAL] run_dir=$run_dir out=$eval_out fixed_tau=$tau ==="
  "$PYTHON_BIN" eval/eval.py \
    --input "$INPUT" \
    --run_dir "$run_dir" \
    --output_dir "$eval_out" \
    "${cfg_args[@]}" \
    --cfg_override "model.resampler.dtp_params.fixed_tau=${tau}"
}

require_env

# ------------------------------------------------------------------------------
# 1) default_PLE_50hz_vq16384
# 2) default_PLEms4_50hz_vq16384
# 3) default_PLE_25hz_vq16384
#    -> search tau (target_avg_r=0.5), then eval with fixed_tau override.
# ------------------------------------------------------------------------------
run_search_and_eval \
  "results/results0117/default_PLE_50hz_vq16384" \
  "results/results0117/default_PLE_50hz_vq16384/dtp_stats_ft" \
  "results/results0117/default_PLE_50hz_vq16384/eval_ft"

run_search_and_eval \
  "results/results0117/default_PLEms4_50hz_vq16384" \
  "results/results0117/default_PLEms4_50hz_vq16384/dtp_stats_ft" \
  "results/results0117/default_PLEms4_50hz_vq16384/eval_ft"

run_search_and_eval \
  "results/results0117/default_PLE_25hz_vq16384" \
  "results/results0117/default_PLE_25hz_vq16384/dtp_stats_ft" \
  "results/results0117/default_PLE_25hz_vq16384/eval_ft"

# ------------------------------------------------------------------------------
# 4) default_fixedpattern_50hz_vq65536
#    -> eval only (no overrides)
# ------------------------------------------------------------------------------
echo
echo "=== [EVAL] run_dir=results/results0117/default_fixedpattern_50hz_vq65536 out=.../eval_ft (no overrides) ==="
ensure_new_config "results/results0117/default_fixedpattern_50hz_vq65536"
if [[ -f "results/results0117/default_fixedpattern_50hz_vq65536/eval_ft/metrics.json" ]]; then
  echo "=== [SKIP] metrics.json already exists at results/results0117/default_fixedpattern_50hz_vq65536/eval_ft/metrics.json ==="
else
  "$PYTHON_BIN" eval/eval.py \
    --input "$INPUT" \
    --run_dir "results/results0117/default_fixedpattern_50hz_vq65536" \
    --output_dir "results/results0117/default_fixedpattern_50hz_vq65536/eval_ft"
fi

# ------------------------------------------------------------------------------
# 5) default_random_50hz_vq16384
#    For 6 combinations:
#    - max_s: (ms4 override vs default)
#    - dtp_cls: BatchGreedy, BatchTopK, PLE(default/no override)
#    Each: search tau to satisfy target_avg_r, then eval with fixed_tau.
# ------------------------------------------------------------------------------

RUN5="results/results0117/default_random_50hz_vq16384"

for ms_mode in "default" "ms4"; do
  ms_suffix=""
  ms_override=""
  if [[ "$ms_mode" == "ms4" ]]; then
    ms_suffix="_ms4"
    ms_override="model.resampler.dtp_params.max_s=4"
  fi

  # PLE (no dtp_cls override)
  if [[ -n "$ms_override" ]]; then
    run_search_and_eval \
      "$RUN5" \
      "$RUN5/dtp_stats_ple_ft${ms_suffix}" \
      "$RUN5/eval_ple_ft${ms_suffix}" \
      "$ms_override"
  else
    run_search_and_eval \
      "$RUN5" \
      "$RUN5/dtp_stats_ple_ft" \
      "$RUN5/eval_ple_ft"
  fi

  # BatchTopK
  if [[ -n "$ms_override" ]]; then
    run_search_and_eval \
      "$RUN5" \
      "$RUN5/dtp_stats_topk_ft${ms_suffix}" \
      "$RUN5/eval_topk_ft${ms_suffix}" \
      "model.resampler.dtp_cls=BatchTopK" \
      "$ms_override"
  else
    run_search_and_eval \
      "$RUN5" \
      "$RUN5/dtp_stats_topk_ft" \
      "$RUN5/eval_topk_ft" \
      "model.resampler.dtp_cls=BatchTopK"
  fi

  # BatchGreedy
  if [[ -n "$ms_override" ]]; then
    run_search_and_eval \
      "$RUN5" \
      "$RUN5/dtp_stats_greedy_ft${ms_suffix}" \
      "$RUN5/eval_greedy_ft${ms_suffix}" \
      "model.resampler.dtp_cls=BatchGreedy" \
      "$ms_override"
  else
    run_search_and_eval \
      "$RUN5" \
      "$RUN5/dtp_stats_greedy_ft" \
      "$RUN5/eval_greedy_ft" \
      "model.resampler.dtp_cls=BatchGreedy"
  fi
done

echo
echo "All jobs finished."


