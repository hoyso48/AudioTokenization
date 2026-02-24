#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

INPUT="DTMAE/filelists/librispeech_test_clean.txt"
TARGET_AVG_R="0.5"
TAU_MIN="0.001"
TAU_MAX="1.0"
TAU_STEP="0.001"
MAX_SAMPLES=""

NUM_WORKERS="4"
LENGTH_MODE="pad"
DEVICE=""

EVAL_STAGE="all"
EVAL_SUBDIR="eval_ft"
STATS_SUBDIR="dtp_stats_ft"
NAME_SUFFIX=""
KEEP_AUDIO=0

TAU_FINETUNE=1
BOOTSTRAP_UPDATE_TEST_TIME=1
BOOTSTRAP_ONLY=0
BOOTSTRAP_OVERRIDE_UPDATE_TEST_TIME=1
BOOTSTRAP_ITERS="1"
SEARCH_NO_RESUME=1
AUTO_EXPAND=0
AUTO_EXPAND_MAX_TAU="100.0"
METRICS="all"
THROUGHPUT_WARMUP_ITEMS="5"
AUTO_INSTALL_METRIC_DEPS=1
UTMOSV2_SPEC="${UTMOSV2_SPEC:-git+https://github.com/sarulab-speech/UTMOSv2.git@v1.2.1}"
UTMOSV2_SPEC_FALLBACK="${UTMOSV2_SPEC_FALLBACK:-utmosv2}"

FORCE=0

PYTHON_BIN="${PYTHON_BIN:-python}"

declare -a RUN_DIRS=()
declare -a RUN_LIST_FILES=()
declare -a CFG_OVERRIDES=()

usage() {
  cat <<'EOF'
Generalized eval suite runner.

What it does per run_dir:
1) Always runs utils/update_legacy_config.py on hydra/config.yaml.
2) If DTP is enabled and the selected dtp_cls supports fixed_tau, then:
   - (optional) searches tau via eval/dtp_stats_search.py
   - evaluates via eval/eval.py with --cfg_override model.resampler.dtp_params.fixed_tau=<best_tau>
3) Otherwise, runs eval/eval.py directly without tau search.

Usage:
  bash eval/run_results_ft_suite.sh [options] <run_dir> [<run_dir> ...]

Options:
  --run_dir <path>                 Add one run directory (repeatable)
  --run_list <file.txt>            File with one run_dir per line (# comments allowed)
  --cfg_override <dotlist>         Extra override for both search/eval (repeatable)

  --input <path>                   Input dir/file/filelist for eval scripts
  --target_avg_r <float>           Target avg_r for tau search (default: 0.5)
  --tau_min <float>                Tau search min (default: 0.001)
  --tau_max <float>                Tau search max (default: 1.0)
  --tau_step <float>               Tau search step (default: 0.001)
  --max_samples <int>              Max samples per tau trial (optional)

  --tau_finetune                   Enable tau search when supported (default)
  --no_tau_finetune                Disable tau search; eval only

  --bootstrap_update_test_time     Enable bootstrap tau warm-start (default)
  --no_bootstrap_update_test_time  Disable bootstrap
  --bootstrap_only                 Use bootstrap tau directly
  --no_bootstrap_only              Continue with binary search after bootstrap (default)
  --bootstrap_override_update_test_time
                                    Force update_test_time=True during bootstrap (default)
  --no_bootstrap_override_update_test_time
                                    Do not force update_test_time=True
  --bootstrap_iters <int>           Number of bootstrap passes (default: 1)

  --auto_expand                    Enable dtp_stats_search --auto_expand
  --auto_expand_max_tau <float>    Max tau used by auto-expand (default: 100.0)

  --resume_search                  Reuse existing trials.jsonl
  --no_resume_search               Do not reuse trials.jsonl (default)

  --eval_stage <save|metrics|all>  Eval stage (default: all)
  --metrics <list|all>             Metrics for eval.py (default: all)
  --throughput_warmup_items <int>  Exclude first N iterations from throughput (default: 5)
  --auto_install_metric_deps       Auto-install missing metric deps like UTMOSv2 (default)
  --no_auto_install_metric_deps    Disable auto-install and fail with guidance
  --utmosv2_spec <spec>            UTMOSv2 spec used when auto-installing (default: git+https://github.com/sarulab-speech/UTMOSv2.git@v1.2.1)
  --utmosv2_spec_fallback <spec>   Fallback UTMOSv2 spec if primary install fails
  --eval_subdir <name>             Eval output subdir under run_dir (default: eval_ft)
  --stats_subdir <name>            Tau-search output subdir (default: dtp_stats_ft)
  --name_suffix <suffix>           Suffix appended to both subdir names
  --keep_audio                     Keep eval.py generated audio dirs
  --force                          Ignore existing metrics.json and re-run eval

  --device <str>                   Device for eval/search scripts (optional)
  --num_workers <int>              Dataloader workers (default: 4)
  --length_mode <pad|truncate>     Length handling mode (default: pad)
  --python_bin <path>              Python executable (default: $PYTHON_BIN)
  -h, --help                       Show this help

Examples:
  bash eval/run_results_ft_suite.sh \
    results/default_PLE_50hz_vq16384 \
    results/default_fixedpattern_50hz_vq65536

  bash eval/run_results_ft_suite.sh \
    --run_list eval/run_dirs.txt \
    --cfg_override model.resampler.dtp_cls=BatchTopK \
    --cfg_override model.resampler.dtp_params.max_s=4 \
    --name_suffix _topk_ms4
EOF
}

trim_whitespace() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf "%s" "$s"
}

append_run_dirs_from_list() {
  local list_file="$1"
  if [[ ! -f "$list_file" ]]; then
    echo "[ERROR] --run_list file not found: $list_file" >&2
    exit 1
  fi

  local raw line
  while IFS= read -r raw || [[ -n "$raw" ]]; do
    line="${raw%%#*}"
    line="$(trim_whitespace "$line")"
    if [[ -n "$line" ]]; then
      RUN_DIRS+=("$line")
    fi
  done < "$list_file"
}

require_env() {
  if ! "$PYTHON_BIN" -c "import torch, torchaudio, omegaconf" >/dev/null 2>&1; then
    echo "[ERROR] Python deps not found. Did you run: conda activate speech_eval ?" >&2
    echo "        (torch/torchaudio/omegaconf must import successfully)" >&2
    exit 1
  fi
}

ensure_new_config() {
  local run_dir="$1"
  local cfg_path="$run_dir/hydra/config.yaml"
  if [[ ! -f "$cfg_path" ]]; then
    echo "[ERROR] Missing config: $cfg_path" >&2
    exit 1
  fi
  echo "=== [CONFIG] update_legacy_config.py on $cfg_path ==="
  "$PYTHON_BIN" utils/update_legacy_config.py --path "$cfg_path"
}

probe_dtp_capability() {
  local run_dir="$1"
  shift

  "$PYTHON_BIN" - "$run_dir" "$@" <<'PY'
import inspect
import sys
from pathlib import Path

from omegaconf import OmegaConf

run_dir = Path(sys.argv[1]).resolve()
overrides = sys.argv[2:]

cfg_path = run_dir / "hydra" / "config.yaml"
cfg = OmegaConf.load(str(cfg_path))
if overrides:
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

resampler = getattr(getattr(cfg, "model", None), "resampler", None)
use_dtp = bool(getattr(resampler, "use_dtp", False)) if resampler is not None else False
dtp_cls = str(getattr(resampler, "dtp_cls", "")) if resampler is not None else ""

supports_fixed_tau = False
supports_update_test_time = False

if use_dtp and dtp_cls:
    project_root = Path.cwd().resolve()
    dtmae_root = project_root / "DTMAE"
    for p in (project_root, dtmae_root):
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)

    try:
        import dtp.ops as dtp_ops

        cls = getattr(dtp_ops, dtp_cls, None)
        if cls is not None:
            sig = inspect.signature(cls.__init__)
            params = sig.parameters
            supports_fixed_tau = "fixed_tau" in params
            supports_update_test_time = "update_test_time" in params
    except Exception:
        pass

print(f"{int(use_dtp)}|{dtp_cls}|{int(supports_fixed_tau)}|{int(supports_update_test_time)}")
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
  local metrics_json="$eval_out/metrics.json"
  if [[ ! -f "$metrics_json" ]]; then
    return 1
  fi

  if "$PYTHON_BIN" - "$metrics_json" "$METRICS" <<'PY'
import json
import sys

metrics_path = sys.argv[1]
metrics_arg = (sys.argv[2] or "").strip().lower()

aliases = {
    "spk": "speaker_similarity",
    "spk_sim": "speaker_similarity",
    "speaker": "speaker_similarity",
    "utmosv2": "utmos_v2",
}

if not metrics_arg or metrics_arg == "all":
    requested = [
        "stoi",
        "pesq_wb",
        "pesq_nb",
        "si_snr",
        "si_sdr",
        "speaker_similarity",
        "mcd",
        "wer",
        "wer_sentence_avg",
        "utmos",
        "utmos_v2",
    ]
else:
    requested = []
    for tok in metrics_arg.split(","):
        key = aliases.get(tok.strip(), tok.strip())
        if key:
            requested.append(key)

if "wer_sentence_avg" in requested and "wer" not in requested:
    requested.append("wer")

requested = list(dict.fromkeys(requested))

with open(metrics_path, "r") as f:
    data = json.load(f)

if not isinstance(data, dict):
    sys.exit(1)

for k in requested:
    if k not in data:
        sys.exit(1)

sys.exit(0)
PY
  then
    echo "=== [SKIP] requested metrics already exist at $metrics_json ==="
    return 0
  fi
  return 1
}

metrics_contains() {
  local needle="$1"
  local metrics_lc
  local -a items
  metrics_lc="$(printf "%s" "$METRICS" | tr '[:upper:]' '[:lower:]')"
  if [[ -z "$metrics_lc" || "$metrics_lc" == "all" ]]; then
    return 0
  fi

  local item trimmed
  IFS=',' read -r -a items <<< "$metrics_lc"
  for item in "${items[@]}"; do
    trimmed="$(trim_whitespace "$item")"
    if [[ "$trimmed" == "utmosv2" ]]; then
      trimmed="utmos_v2"
    fi
    if [[ "$trimmed" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

ensure_metric_runtime_deps() {
  if [[ "$EVAL_STAGE" == "save" ]]; then
    return
  fi

  if ! metrics_contains "utmos_v2"; then
    return
  fi

  if "$PYTHON_BIN" -c "import utmosv2" >/dev/null 2>&1; then
    return
  fi

  if [[ "$AUTO_INSTALL_METRIC_DEPS" -ne 1 ]]; then
    echo "[ERROR] utmosv2 is required for metric 'utmos_v2' but is not installed." >&2
    echo "        Install with: $PYTHON_BIN -m pip install '$UTMOSV2_SPEC'" >&2
    exit 2
  fi

  echo "[INFO] Missing utmosv2 detected; installing into current Python env..."
  if ! "$PYTHON_BIN" -m pip install "$UTMOSV2_SPEC"; then
    if [[ "$UTMOSV2_SPEC" != "$UTMOSV2_SPEC_FALLBACK" ]]; then
      echo "[WARN] Primary UTMOSv2 install failed; retrying fallback: $UTMOSV2_SPEC_FALLBACK"
      "$PYTHON_BIN" -m pip install "$UTMOSV2_SPEC_FALLBACK"
    else
      echo "[ERROR] Failed to install UTMOSv2 spec: $UTMOSV2_SPEC" >&2
      exit 2
    fi
  fi

  if ! "$PYTHON_BIN" -c "import utmosv2" >/dev/null 2>&1; then
    echo "[ERROR] utmosv2 import still fails after install." >&2
    exit 2
  fi
}

run_tau_search() {
  local run_dir="$1"
  local stats_out="$2"
  local supports_update_test_time="$3"

  local -a args=()
  args+=(--input "$INPUT")
  args+=(--run_dir "$run_dir")
  args+=(--output_dir "$stats_out")
  args+=(--target_avg_r "$TARGET_AVG_R")
  args+=(--tau_min "$TAU_MIN")
  args+=(--tau_max "$TAU_MAX")
  args+=(--tau_step "$TAU_STEP")
  args+=(--length_mode "$LENGTH_MODE")
  args+=(--num_workers "$NUM_WORKERS")

  if [[ -n "$MAX_SAMPLES" ]]; then
    args+=(--max_samples "$MAX_SAMPLES")
  fi
  if [[ -n "$DEVICE" ]]; then
    args+=(--device "$DEVICE")
  fi

  if [[ "$AUTO_EXPAND" -eq 1 ]]; then
    args+=(--auto_expand)
    args+=(--auto_expand_max_tau "$AUTO_EXPAND_MAX_TAU")
  fi

  if [[ "$SEARCH_NO_RESUME" -eq 1 ]]; then
    args+=(--no_resume)
  fi

  if [[ "$BOOTSTRAP_UPDATE_TEST_TIME" -eq 1 && "$supports_update_test_time" -eq 1 ]]; then
    args+=(--bootstrap_update_test_time)
    args+=(--bootstrap_iters "$BOOTSTRAP_ITERS")
    if [[ "$BOOTSTRAP_ONLY" -eq 1 ]]; then
      args+=(--bootstrap_only)
    else
      args+=(--no_bootstrap_only)
    fi
    if [[ "$BOOTSTRAP_OVERRIDE_UPDATE_TEST_TIME" -eq 1 ]]; then
      args+=(--bootstrap_override_update_test_time)
    fi
  fi

  for ov in "${CFG_OVERRIDES[@]}"; do
    args+=(--cfg_override "$ov")
  done

  echo
  echo "=== [SEARCH] run_dir=$run_dir out=$stats_out target_avg_r=$TARGET_AVG_R ==="
  "$PYTHON_BIN" eval/dtp_stats_search.py "${args[@]}"
}

run_eval() {
  local run_dir="$1"
  local eval_out="$2"
  local fixed_tau="${3:-}"

  ensure_metric_runtime_deps

  local -a args=()
  args+=(--input "$INPUT")
  args+=(--run_dir "$run_dir")
  args+=(--output_dir "$eval_out")
  args+=(--stage "$EVAL_STAGE")
  args+=(--metrics "$METRICS")
  args+=(--throughput_warmup_items "$THROUGHPUT_WARMUP_ITEMS")
  args+=(--length_mode "$LENGTH_MODE")
  args+=(--num_workers "$NUM_WORKERS")

  if [[ -n "$DEVICE" ]]; then
    args+=(--device "$DEVICE")
  fi
  if [[ "$KEEP_AUDIO" -eq 1 ]]; then
    args+=(--keep_audio)
  fi

  for ov in "${CFG_OVERRIDES[@]}"; do
    args+=(--cfg_override "$ov")
  done

  if [[ -n "$fixed_tau" ]]; then
    args+=(--cfg_override "model.resampler.dtp_params.fixed_tau=${fixed_tau}")
    echo
    echo "=== [EVAL] run_dir=$run_dir out=$eval_out fixed_tau=$fixed_tau ==="
  else
    echo
    echo "=== [EVAL] run_dir=$run_dir out=$eval_out (no fixed_tau override) ==="
  fi

  "$PYTHON_BIN" eval/eval.py "${args[@]}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run_dir)
      RUN_DIRS+=("$2")
      shift 2
      ;;
    --run_list)
      RUN_LIST_FILES+=("$2")
      shift 2
      ;;
    --cfg_override)
      CFG_OVERRIDES+=("$2")
      shift 2
      ;;
    --input)
      INPUT="$2"
      shift 2
      ;;
    --target_avg_r)
      TARGET_AVG_R="$2"
      shift 2
      ;;
    --tau_min)
      TAU_MIN="$2"
      shift 2
      ;;
    --tau_max)
      TAU_MAX="$2"
      shift 2
      ;;
    --tau_step)
      TAU_STEP="$2"
      shift 2
      ;;
    --max_samples)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    --tau_finetune)
      TAU_FINETUNE=1
      shift
      ;;
    --no_tau_finetune)
      TAU_FINETUNE=0
      shift
      ;;
    --bootstrap_update_test_time)
      BOOTSTRAP_UPDATE_TEST_TIME=1
      shift
      ;;
    --no_bootstrap_update_test_time)
      BOOTSTRAP_UPDATE_TEST_TIME=0
      shift
      ;;
    --bootstrap_only)
      BOOTSTRAP_ONLY=1
      shift
      ;;
    --no_bootstrap_only)
      BOOTSTRAP_ONLY=0
      shift
      ;;
    --bootstrap_override_update_test_time)
      BOOTSTRAP_OVERRIDE_UPDATE_TEST_TIME=1
      shift
      ;;
    --no_bootstrap_override_update_test_time)
      BOOTSTRAP_OVERRIDE_UPDATE_TEST_TIME=0
      shift
      ;;
    --bootstrap_iters)
      BOOTSTRAP_ITERS="$2"
      shift 2
      ;;
    --auto_expand)
      AUTO_EXPAND=1
      shift
      ;;
    --auto_expand_max_tau)
      AUTO_EXPAND_MAX_TAU="$2"
      shift 2
      ;;
    --resume_search)
      SEARCH_NO_RESUME=0
      shift
      ;;
    --no_resume_search)
      SEARCH_NO_RESUME=1
      shift
      ;;
    --eval_stage)
      EVAL_STAGE="$2"
      shift 2
      ;;
    --metrics)
      METRICS="$2"
      shift 2
      ;;
    --throughput_warmup_items)
      THROUGHPUT_WARMUP_ITEMS="$2"
      shift 2
      ;;
    --auto_install_metric_deps)
      AUTO_INSTALL_METRIC_DEPS=1
      shift
      ;;
    --no_auto_install_metric_deps)
      AUTO_INSTALL_METRIC_DEPS=0
      shift
      ;;
    --utmosv2_spec)
      UTMOSV2_SPEC="$2"
      shift 2
      ;;
    --utmosv2_spec_fallback)
      UTMOSV2_SPEC_FALLBACK="$2"
      shift 2
      ;;
    --eval_subdir)
      EVAL_SUBDIR="$2"
      shift 2
      ;;
    --stats_subdir)
      STATS_SUBDIR="$2"
      shift 2
      ;;
    --name_suffix)
      NAME_SUFFIX="$2"
      shift 2
      ;;
    --keep_audio)
      KEEP_AUDIO=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --num_workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --length_mode)
      LENGTH_MODE="$2"
      shift 2
      ;;
    --python_bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        RUN_DIRS+=("$1")
        shift
      done
      ;;
    -* )
      echo "[ERROR] Unknown option: $1" >&2
      usage
      exit 1
      ;;
    *)
      RUN_DIRS+=("$1")
      shift
      ;;
  esac
done

for list_file in "${RUN_LIST_FILES[@]}"; do
  append_run_dirs_from_list "$list_file"
done

if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
  echo "[ERROR] No run directories provided." >&2
  usage
  exit 1
fi

require_env

total_runs=${#RUN_DIRS[@]}
done_runs=0
skipped_runs=0
tau_runs=0
eval_only_runs=0

for run_dir in "${RUN_DIRS[@]}"; do
  run_dir="${run_dir%/}"
  if [[ ! -d "$run_dir" ]]; then
    echo "[ERROR] run_dir not found: $run_dir" >&2
    exit 1
  fi

  eval_out="$run_dir/${EVAL_SUBDIR}${NAME_SUFFIX}"
  stats_out="$run_dir/${STATS_SUBDIR}${NAME_SUFFIX}"

  echo
  echo "============================================================"
  echo "[RUN] $run_dir"
  echo "[OUT] eval=$eval_out"
  echo "[OUT] stats=$stats_out"

  if [[ "$FORCE" -eq 0 ]] && should_skip_eval "$eval_out"; then
    skipped_runs=$((skipped_runs + 1))
    done_runs=$((done_runs + 1))
    continue
  fi

  ensure_new_config "$run_dir"

  probe_line="$(probe_dtp_capability "$run_dir" "${CFG_OVERRIDES[@]}")"
  IFS='|' read -r use_dtp dtp_cls supports_fixed_tau supports_update_test_time <<< "$probe_line"

  do_tau_search=0
  if [[ "$TAU_FINETUNE" -eq 1 && "$use_dtp" -eq 1 && "$supports_fixed_tau" -eq 1 ]]; then
    do_tau_search=1
  fi

  summary_json="$stats_out/summary.json"

  if [[ "$do_tau_search" -eq 1 ]]; then
    if [[ "$FORCE" -eq 0 && -f "$summary_json" ]]; then
      tau="$(json_get_best_tau "$summary_json")"
      echo "=== [REUSE] fixed_tau=$tau (from existing $summary_json) ==="
    else
      run_tau_search "$run_dir" "$stats_out" "$supports_update_test_time"
      if [[ ! -f "$summary_json" ]]; then
        echo "[ERROR] Missing summary.json at: $summary_json" >&2
        exit 1
      fi
      tau="$(json_get_best_tau "$summary_json")"
      echo "=== [FOUND] fixed_tau=$tau (from $summary_json) ==="
    fi

    run_eval "$run_dir" "$eval_out" "$tau"
    tau_runs=$((tau_runs + 1))
  else
    if [[ "$TAU_FINETUNE" -eq 0 ]]; then
      echo "=== [INFO] Tau search disabled by --no_tau_finetune ==="
    elif [[ "$use_dtp" -ne 1 ]]; then
      echo "=== [INFO] use_dtp=False -> eval only ==="
    elif [[ "$supports_fixed_tau" -ne 1 ]]; then
      echo "=== [INFO] dtp_cls=${dtp_cls:-unknown} does not expose fixed_tau -> eval only ==="
    else
      echo "=== [INFO] Tau search skipped for run_dir=$run_dir ==="
    fi

    run_eval "$run_dir" "$eval_out"
    eval_only_runs=$((eval_only_runs + 1))
  fi

  done_runs=$((done_runs + 1))
done

echo
echo "All jobs finished."
echo "- total:   $total_runs"
echo "- done:    $done_runs"
echo "- skipped: $skipped_runs"
echo "- tau+eval:$tau_runs"
echo "- eval-only:$eval_only_runs"
