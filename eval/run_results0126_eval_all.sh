#!/usr/bin/env bash
set -euo pipefail

# Eval suite runner for all experiments under results/results0126.
#
# What it does (per run dir):
# - Ensure legacy hydra/config.yaml is converted to the new model.quantizer style.
# - If model.resampler.use_dtp=true:
#     - Search fixed_tau that matches the run's configured target r (model.resampler.dtp_params.r)
#     - Run eval/eval.py with --cfg_override model.resampler.dtp_params.fixed_tau=<best_tau>
# - Otherwise:
#     - Run eval/eval.py without tau search.
#
# Prerequisite:
#   conda activate speech_eval
#
# Run from repo root:
#   cd /home/hoyso/projects/AudioTokenization
#   bash eval/run_results0126_eval_all.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RESULTS_DIR="results/results0126"
INPUT="${INPUT:-DTMAE/filelists/librispeech_test_clean.txt}"

# Tau search parameters (only used when use_dtp=true).
TAU_STEP="${TAU_STEP:-0.001}"
TAU_MIN="${TAU_MIN:-0.001}"
TAU_MAX="${TAU_MAX:-1.0}"

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

cfg_get_use_dtp() {
  local cfg_path="$1"
  "$PYTHON_BIN" - "$cfg_path" <<'PY'
import sys
from omegaconf import OmegaConf

cfg_path = sys.argv[1]
cfg = OmegaConf.load(cfg_path)
use_dtp = bool(OmegaConf.select(cfg, "model.resampler.use_dtp", default=False))
print("1" if use_dtp else "0")
PY
}

cfg_get_target_avg_r() {
  local cfg_path="$1"
  "$PYTHON_BIN" - "$cfg_path" <<'PY'
import sys
from omegaconf import OmegaConf

cfg_path = sys.argv[1]
cfg = OmegaConf.load(cfg_path)
r = OmegaConf.select(cfg, "model.resampler.dtp_params.r", default=None)
if r is None:
    raise SystemExit(f"[ERROR] Missing cfg field: model.resampler.dtp_params.r in {cfg_path}")
r = float(r)
if not (0.0 <= r < 1.0):
    raise SystemExit(f"[ERROR] Invalid dtp_params.r={r} (must be in [0, 1)) in {cfg_path}")
print(r)
PY
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

run_eval_only() {
  local run_dir="$1"
  local eval_out="$2"

  if should_skip_eval "$eval_out"; then
    return 0
  fi

  ensure_new_config "$run_dir"

  echo
  echo "=== [EVAL] run_dir=$run_dir out=$eval_out (no tau search) ==="
  "$PYTHON_BIN" eval/eval.py \
    --input "$INPUT" \
    --run_dir "$run_dir" \
    --output_dir "$eval_out"
}

run_search_and_eval() {
  local run_dir="$1"
  local stats_out="$2"
  local eval_out="$3"

  if should_skip_eval "$eval_out"; then
    return 0
  fi

  ensure_new_config "$run_dir"

  local cfg_path="$run_dir/hydra/config.yaml"
  local target_avg_r
  target_avg_r="$(cfg_get_target_avg_r "$cfg_path")"

  echo
  echo "=== [SEARCH] run_dir=$run_dir out=$stats_out target_avg_r=$target_avg_r ==="
  "$PYTHON_BIN" eval/dtp_stats_search.py \
    --input "$INPUT" \
    --run_dir "$run_dir" \
    --output_dir "$stats_out" \
    --target_avg_r "$target_avg_r" \
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
  echo "=== [EVAL] run_dir=$run_dir out=$eval_out fixed_tau=$tau ==="
  "$PYTHON_BIN" eval/eval.py \
    --input "$INPUT" \
    --run_dir "$run_dir" \
    --output_dir "$eval_out" \
    --cfg_override "model.resampler.dtp_params.fixed_tau=${tau}"
}

require_env

if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "[ERROR] Results directory not found: $RESULTS_DIR" >&2
  exit 1
fi

mapfile -t RUN_DIRS < <(find "$RESULTS_DIR" -mindepth 1 -maxdepth 1 -type d -print | sort)
if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
  echo "[ERROR] No run directories found under: $RESULTS_DIR" >&2
  exit 1
fi

echo "Found ${#RUN_DIRS[@]} run dirs under: $RESULTS_DIR"

for run_dir in "${RUN_DIRS[@]}"; do
  cfg_path="$run_dir/hydra/config.yaml"
  ckpt_path="$run_dir/pl_log/last.ckpt"
  if [[ ! -f "$cfg_path" ]]; then
    echo "=== [WARN] Skipping (missing config): $cfg_path ===" >&2
    continue
  fi
  if [[ ! -f "$ckpt_path" ]]; then
    echo "=== [WARN] Skipping (missing checkpoint): $ckpt_path ===" >&2
    continue
  fi

  eval_out="$run_dir/eval"
  stats_out="$run_dir/dtp_stats"

  use_dtp="$(cfg_get_use_dtp "$cfg_path")"
  if [[ "$use_dtp" == "1" ]]; then
    run_search_and_eval "$run_dir" "$stats_out" "$eval_out"
  else
    run_eval_only "$run_dir" "$eval_out"
  fi
done

echo
echo "All jobs finished."


