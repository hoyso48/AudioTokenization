#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EVAL_ENV="${EVAL_ENV:-speech_eval}"
PYTHON_EVAL="${PYTHON_EVAL:-python}"
CHECK_PYTHON_BIN="${CHECK_PYTHON_BIN:-python3}"

INPUT_LIST="${INPUT_LIST:-$ROOT/DTMAE/filelists/librispeech_test_clean.txt}"
TARGET_AVG_R="${TARGET_AVG_R:-}"
TAU_MIN="${TAU_MIN:-0.001}"
TAU_MAX="${TAU_MAX:-1.0}"
TAU_STEP="${TAU_STEP:-0.001}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

NUM_WORKERS="${NUM_WORKERS:-4}"
LENGTH_MODE="${LENGTH_MODE:-pad}"
DEVICE="${DEVICE:-}"

EVAL_STAGE="${EVAL_STAGE:-all}"
EVAL_SUBDIR="${EVAL_SUBDIR:-eval_avg_bps{bps}}"
STATS_SUBDIR="${STATS_SUBDIR:-dtp_stats_avg_bps{bps}}"

CHECK_WAVLM=1
FORCE=0

TAU_FINETUNE=1
BOOTSTRAP_UPDATE_TEST_TIME=1
BOOTSTRAP_ONLY=0
BOOTSTRAP_OVERRIDE_UPDATE_TEST_TIME=1
BOOTSTRAP_ITERS="1"
SEARCH_NO_RESUME=1
AUTO_EXPAND=0
AUTO_EXPAND_MAX_TAU="${AUTO_EXPAND_MAX_TAU:-100.0}"
DIRECTION_PROBE_STEP="${DIRECTION_PROBE_STEP:-16}"
METRICS="${METRICS:-all}"
THROUGHPUT_WARMUP_ITEMS="${THROUGHPUT_WARMUP_ITEMS:-5}"
AUTO_INSTALL_METRIC_DEPS=1
UTMOSV2_SPEC="${UTMOSV2_SPEC:-git+https://github.com/sarulab-speech/UTMOSv2.git@v1.2.1}"
UTMOSV2_SPEC_FALLBACK="${UTMOSV2_SPEC_FALLBACK:-utmosv2}"

EVAL_OUT_OVERRIDE=""
EXTERNAL_OUTPUTS_ROOT=""
EXTERNAL_EVAL_ROOT="${EXTERNAL_EVAL_ROOT:-$ROOT/external_codecs/eval_results/waveform_level}"
EXTERNAL_GT_CACHE_DIR=""
EXTERNAL_TRANSCRIPT_KEY=""

declare -a RUN_DIRS=()
declare -a RUN_LIST_FILES=()
declare -a EXTERNAL_VARIANT_DIRS=()
declare -a EVAL_CFG_OVERRIDES=()
declare -a EVAL_EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Run eval for training runs or external reconstructed waveforms.

Usage:
  bash run_eval_only.sh [options] <run_dir> [<run_dir> ...]

Options:
  --run_dir <path>                 Add one run directory (repeatable)
  --run_list <file.txt>            File with one run_dir per line (# comments allowed)
  --external_outputs_root <path>   Root with <codec>/<variant> reconstructed wavs
  --external_variant_dir <path>    One external variant dir (repeatable)
  --external_eval_root <path>      Output root for external metrics (default: external_codecs/eval_results/waveform_level)
  --external_gt_cache_dir <path>   Shared 16k GT wav cache for external metrics (default: <external_eval_root>/__gt_16k_cache)
  --external_transcript_key <path> Transcript key (.txt/.csv/.jsonl) for WER in external mode

  --input <path>                   Eval input path (default: DTMAE/filelists/librispeech_test_clean.txt)
  --eval_stage <save|metrics|all>  Eval stage (default: all)
  --eval_subdir <name>             Eval output subdir/template (default: eval_avg_bps{bps})
  --stats_subdir <name>            Tau-search subdir/template (default: dtp_stats_avg_bps{bps})
                                   Templates support {bps} and {r} placeholders.
  --eval_out <path>                Explicit eval output path (single run only)
  --num_workers <int>              Eval dataloader workers (default: 4)
  --length_mode <pad|truncate>     Eval length mode (default: pad)
  --device <str>                   Eval/search device (optional)

  --tau_finetune                   Enable target_r->fixed_tau search when supported (default)
  --no_tau_finetune                Disable tau search and run eval directly
  --target_avg_r <float>           Target avg_r for tau search (default: model.resampler.dtp_params.r)
  --tau_min <float>                Tau search min (default: 0.001)
  --tau_max <float>                Tau search max (default: 1.0)
  --tau_step <float>               Tau search step (default: 0.001)
  --max_samples <int>              Max samples per tau trial (optional)
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
  --direction_probe_step <int>     Probe step for search direction inference (default: 16)
  --resume_search                  Reuse existing trials.jsonl
  --no_resume_search               Do not reuse trials.jsonl (default)

  --metrics <list|all>             Metrics for eval.py (default: all)
  --throughput_warmup_items <int>  Exclude first N iterations from throughput (default: 5)
  --auto_install_metric_deps       Auto-install missing metric deps like UTMOSv2 (default)
  --no_auto_install_metric_deps    Disable auto-install and fail with guidance
  --utmosv2_spec <spec>            UTMOSv2 spec used when auto-installing (default: git+https://github.com/sarulab-speech/UTMOSv2.git@v1.2.1)
  --utmosv2_spec_fallback <spec>   Fallback UTMOSv2 spec if primary install fails

  --eval_env <name>                Conda env for eval/search (default: speech_eval)
  --python_eval <bin>              Python binary inside eval env (default: python)
  --cfg_override <dotlist>         Eval/search config override (repeatable)
  --eval_arg <arg>                 Extra arg passed to eval.py (repeatable)

  --no_check_wavlm                 Skip precheck for eval/wavlm_large_finetune.pth
  --force                          Re-run eval even when metrics.json already exists
  -h, --help                       Show this help

Examples:
  bash run_eval_only.sh outputs/exp1

  bash run_eval_only.sh \
    --run_dir outputs/exp1 \
    --run_dir outputs/exp2 \
    --target_avg_r 0.5

  bash run_eval_only.sh \
    --run_list run_dirs.txt \
    --cfg_override model.resampler.dtp_params.r=0.4 \
    --eval_arg --keep_audio

  bash run_eval_only.sh \
    --external_outputs_root external_codecs/outputs \
    --eval_stage metrics
EOF
}

require_conda() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda command not found in PATH." >&2
    exit 1
  fi
}

check_env_python() {
  local env_name="$1"
  local py_bin="$2"
  if ! conda run -n "$env_name" "$py_bin" -c "import sys; print(sys.executable)" >/dev/null 2>&1; then
    echo "[ERROR] Cannot run Python in conda env: $env_name" >&2
    exit 1
  fi
}

resolve_check_python_bin() {
  if command -v "$CHECK_PYTHON_BIN" >/dev/null 2>&1; then
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    CHECK_PYTHON_BIN="python3"
    return
  fi
  if command -v python >/dev/null 2>&1; then
    CHECK_PYTHON_BIN="python"
    return
  fi
  echo "[ERROR] No local python interpreter found for metrics key checks." >&2
  exit 1
}

trim_whitespace() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf "%s" "$s"
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

metrics_already_covered() {
  local metrics_json="$1"
  if [[ ! -f "$metrics_json" ]]; then
    return 1
  fi

  if "$CHECK_PYTHON_BIN" - "$metrics_json" "$METRICS" <<'PY'
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
    return 0
  fi
  return 1
}

ensure_metric_runtime_deps() {
  if [[ "$EVAL_STAGE" == "save" ]]; then
    return
  fi

  if ! metrics_contains "utmos_v2"; then
    return
  fi

  if conda run --no-capture-output -n "$EVAL_ENV" "$PYTHON_EVAL" -c "import utmosv2" >/dev/null 2>&1; then
    return
  fi

  if [[ "$AUTO_INSTALL_METRIC_DEPS" -ne 1 ]]; then
    echo "[ERROR] utmosv2 is required for metric 'utmos_v2' but is not installed in env '$EVAL_ENV'." >&2
    echo "        Install with: conda run -n $EVAL_ENV $PYTHON_EVAL -m pip install '$UTMOSV2_SPEC'" >&2
    exit 2
  fi

  echo "[INFO] Missing utmosv2 detected; installing into env '$EVAL_ENV'..."
  if ! conda run --no-capture-output -n "$EVAL_ENV" "$PYTHON_EVAL" -m pip install "$UTMOSV2_SPEC"; then
    if [[ "$UTMOSV2_SPEC" != "$UTMOSV2_SPEC_FALLBACK" ]]; then
      echo "[WARN] Primary UTMOSv2 install failed; retrying fallback: $UTMOSV2_SPEC_FALLBACK"
      conda run --no-capture-output -n "$EVAL_ENV" "$PYTHON_EVAL" -m pip install "$UTMOSV2_SPEC_FALLBACK"
    else
      echo "[ERROR] Failed to install UTMOSv2 spec: $UTMOSV2_SPEC" >&2
      exit 2
    fi
  fi

  if ! conda run --no-capture-output -n "$EVAL_ENV" "$PYTHON_EVAL" -c "import utmosv2" >/dev/null 2>&1; then
    echo "[ERROR] utmosv2 import still fails after install in env '$EVAL_ENV'." >&2
    exit 2
  fi
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

ensure_new_config() {
  local run_dir="$1"
  local cfg_path="$run_dir/hydra/config.yaml"
  if [[ ! -f "$cfg_path" ]]; then
    echo "[ERROR] Missing config: $cfg_path" >&2
    exit 1
  fi
  echo "=== [CONFIG] update_legacy_config.py on $cfg_path ==="
  conda run --no-capture-output -n "$EVAL_ENV" "$PYTHON_EVAL" "$ROOT/utils/update_legacy_config.py" --path "$cfg_path"
}

probe_dtp_capability() {
  local run_dir="$1"
  shift

  conda run --no-capture-output -n "$EVAL_ENV" "$PYTHON_EVAL" - "$run_dir" "$@" <<'PY'
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
  conda run --no-capture-output -n "$EVAL_ENV" "$PYTHON_EVAL" - "$summary_json" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r") as f:
    d = json.load(f)
print(d["best"]["fixed_tau"])
PY
}

resolve_target_avg_r_from_cfg() {
  local run_dir="$1"
  shift

  conda run --no-capture-output -n "$EVAL_ENV" "$PYTHON_EVAL" - "$run_dir" "$@" <<'PY'
import sys
from pathlib import Path

from omegaconf import OmegaConf

run_dir = Path(sys.argv[1]).resolve()
overrides = sys.argv[2:]

cfg_path = run_dir / "hydra" / "config.yaml"
cfg = OmegaConf.load(str(cfg_path))
if overrides:
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

target = None
try:
    target = float(cfg.model.resampler.dtp_params.r)
except Exception:
    target = 0.5

print(target)
PY
}

render_subdir_template() {
  local template="$1"
  local bps="$2"
  local target_r="$3"

  local rendered="$template"
  rendered="${rendered//\{bps\}/$bps}"
  rendered="${rendered//\{r\}/$target_r}"
  printf "%s" "$rendered"
}

resolve_target_bps_from_cfg() {
  local run_dir="$1"
  local target_avg_r="$2"
  shift 2

  conda run --no-capture-output -n "$EVAL_ENV" "$PYTHON_EVAL" - "$run_dir" "$target_avg_r" "$@" <<'PY'
import math
import sys
from pathlib import Path

from omegaconf import OmegaConf


def get_nested(cfg, path, default=None):
    cur = cfg
    for key in path:
        if cur is None:
            return default
        try:
            cur = cur[key]
        except Exception:
            cur = getattr(cur, key, None)
        if cur is None:
            return default
    return cur


def fmt(v: float) -> str:
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.6f}".rstrip("0").rstrip(".")


run_dir = Path(sys.argv[1]).resolve()
target_avg_r = float(sys.argv[2])
overrides = sys.argv[3:]

cfg_path = run_dir / "hydra" / "config.yaml"
cfg = OmegaConf.load(str(cfg_path))
if overrides:
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

sample_rate = float(
    get_nested(cfg, ["dataset", "sample_rate"], None)
    or get_nested(cfg, ["preprocess", "audio", "sr"], 16000)
)
hop_length = float(get_nested(cfg, ["model", "codec_encoder", "hop_length"], 80))
f_l1 = sample_rate / hop_length / 2.0

quant_params = get_nested(cfg, ["model", "quantizer", "params"], None)
if quant_params is None:
    raise RuntimeError("Missing model.quantizer.params in config")

codebook_size = get_nested(quant_params, ["codebook_size"], None)
if codebook_size is not None:
    codebook_size = float(codebook_size)
else:
    levels = get_nested(quant_params, ["levels"], None)
    if not levels:
        raise RuntimeError("Cannot infer codebook size: need codebook_size or levels")
    codebook_size = 1.0
    for lv in levels:
        codebook_size *= float(lv)

bits_per_token = int(math.ceil(math.log2(codebook_size)))

dtp_cls = str(get_nested(cfg, ["model", "resampler", "dtp_cls"], ""))
dtp_cls_lc = dtp_cls.lower()
is_fixed_pattern = dtp_cls_lc == "fixedpattern" or dtp_cls_lc.endswith(".fixedpattern")

f_tok = f_l1 * (1.0 - target_avg_r)
b_content = f_tok * bits_per_token
b_pos = 0.0 if is_fixed_pattern else f_l1
b_total = b_content + b_pos
bps_suffix = int(round(b_total))

print(
    "|".join(
        [
            str(bps_suffix),
            fmt(b_total),
            fmt(b_content),
            fmt(b_pos),
            fmt(f_l1),
            str(bits_per_token),
        ]
    )
)
PY
}

run_tau_search() {
  local run_dir="$1"
  local stats_out="$2"
  local supports_update_test_time="$3"
  local target_avg_r="$4"

  local -a args=()
  args+=(--input "$INPUT_LIST")
  args+=(--run_dir "$run_dir")
  args+=(--output_dir "$stats_out")
  args+=(--target_avg_r "$target_avg_r")
  args+=(--tau_min "$TAU_MIN")
  args+=(--tau_max "$TAU_MAX")
  args+=(--tau_step "$TAU_STEP")
  args+=(--length_mode "$LENGTH_MODE")
  args+=(--num_workers "$NUM_WORKERS")
  args+=(--direction_probe_step "$DIRECTION_PROBE_STEP")

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
  for ov in "${EVAL_CFG_OVERRIDES[@]}"; do
    args+=(--cfg_override "$ov")
  done

  echo "=== [SEARCH] run_dir=$run_dir out=$stats_out target_avg_r=$target_avg_r ==="
  conda run --no-capture-output -n "$EVAL_ENV" "$PYTHON_EVAL" "$ROOT/eval/dtp_stats_search.py" "${args[@]}"
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
    --external_outputs_root)
      EXTERNAL_OUTPUTS_ROOT="$2"
      shift 2
      ;;
    --external_variant_dir)
      EXTERNAL_VARIANT_DIRS+=("$2")
      shift 2
      ;;
    --external_eval_root)
      EXTERNAL_EVAL_ROOT="$2"
      shift 2
      ;;
    --external_gt_cache_dir)
      EXTERNAL_GT_CACHE_DIR="$2"
      shift 2
      ;;
    --external_transcript_key)
      EXTERNAL_TRANSCRIPT_KEY="$2"
      shift 2
      ;;
    --input)
      INPUT_LIST="$2"
      shift 2
      ;;
    --eval_stage|--stage)
      EVAL_STAGE="$2"
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
    --eval_out)
      EVAL_OUT_OVERRIDE="$2"
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
    --device)
      DEVICE="$2"
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
    --direction_probe_step)
      DIRECTION_PROBE_STEP="$2"
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
    --eval_env)
      EVAL_ENV="$2"
      shift 2
      ;;
    --python_eval)
      PYTHON_EVAL="$2"
      shift 2
      ;;
    --cfg_override)
      EVAL_CFG_OVERRIDES+=("$2")
      shift 2
      ;;
    --eval_arg)
      EVAL_EXTRA_ARGS+=("$2")
      shift 2
      ;;
    --no_check_wavlm)
      CHECK_WAVLM=0
      shift
      ;;
    --force)
      FORCE=1
      shift
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

EXTERNAL_MODE=0
if [[ -n "$EXTERNAL_OUTPUTS_ROOT" || "${#EXTERNAL_VARIANT_DIRS[@]}" -gt 0 ]]; then
  EXTERNAL_MODE=1
fi

if [[ "$EXTERNAL_MODE" -eq 0 && "${#RUN_DIRS[@]}" -eq 0 ]]; then
  echo "[ERROR] No run directories provided." >&2
  usage
  exit 1
fi

if [[ "$EVAL_STAGE" != "save" && "$EVAL_STAGE" != "metrics" && "$EVAL_STAGE" != "all" ]]; then
  echo "[ERROR] --eval_stage must be one of: save, metrics, all" >&2
  exit 1
fi

if [[ "$EXTERNAL_MODE" -eq 1 ]]; then
  if [[ "$EVAL_STAGE" == "save" ]]; then
    echo "[ERROR] external mode does not support --eval_stage save." >&2
    exit 1
  fi
  if [[ "$EVAL_STAGE" == "all" ]]; then
    echo "[INFO] external mode uses waveform-level metrics only; overriding --eval_stage all -> metrics"
    EVAL_STAGE="metrics"
  fi
  if [[ "${#RUN_DIRS[@]}" -gt 0 ]]; then
    echo "[ERROR] Do not mix run_dir mode with external mode options." >&2
    exit 1
  fi
fi

if [[ "$LENGTH_MODE" != "pad" && "$LENGTH_MODE" != "truncate" ]]; then
  echo "[ERROR] --length_mode must be one of: pad, truncate" >&2
  exit 1
fi

if [[ "$EXTERNAL_MODE" -eq 0 && -n "$EVAL_OUT_OVERRIDE" && "${#RUN_DIRS[@]}" -ne 1 ]]; then
  echo "[ERROR] --eval_out can only be used with exactly one run_dir." >&2
  exit 1
fi

if [[ "$EXTERNAL_MODE" -eq 1 && -n "$EVAL_OUT_OVERRIDE" ]]; then
  echo "[ERROR] --eval_out is not used in external mode. Use --external_eval_root." >&2
  exit 1
fi

require_conda
check_env_python "$EVAL_ENV" "$PYTHON_EVAL"
resolve_check_python_bin

if [[ "$CHECK_WAVLM" -eq 1 && "$EVAL_STAGE" != "save" ]] && metrics_contains "speaker_similarity"; then
  if [[ ! -f "$ROOT/eval/wavlm_large_finetune.pth" ]]; then
    echo "[ERROR] Missing required speaker checkpoint: $ROOT/eval/wavlm_large_finetune.pth" >&2
    echo "        Add the file (or pass --no_check_wavlm)." >&2
    exit 2
  fi
fi

if [[ "$EXTERNAL_MODE" -eq 1 ]]; then
  declare -a ALL_EXTERNAL_VARIANTS=()

  if [[ -n "$EXTERNAL_OUTPUTS_ROOT" ]]; then
    if [[ ! -d "$EXTERNAL_OUTPUTS_ROOT" ]]; then
      echo "[ERROR] external outputs root not found: $EXTERNAL_OUTPUTS_ROOT" >&2
      exit 1
    fi
    for codec_dir in "$EXTERNAL_OUTPUTS_ROOT"/*; do
      [[ -d "$codec_dir" ]] || continue
      for variant_dir in "$codec_dir"/*; do
        [[ -d "$variant_dir" ]] || continue
        ALL_EXTERNAL_VARIANTS+=("$variant_dir")
      done
    done
  fi

  for variant_dir in "${EXTERNAL_VARIANT_DIRS[@]}"; do
    ALL_EXTERNAL_VARIANTS+=("$variant_dir")
  done

  if [[ "${#ALL_EXTERNAL_VARIANTS[@]}" -eq 0 ]]; then
    echo "[ERROR] No external variant directories found." >&2
    exit 1
  fi

  declare -A seen_variants=()
  declare -a EXTERNAL_VARIANTS=()
  for variant_dir in "${ALL_EXTERNAL_VARIANTS[@]}"; do
    variant_dir="${variant_dir%/}"
    if [[ ! -d "$variant_dir" ]]; then
      echo "[ERROR] external variant dir not found: $variant_dir" >&2
      exit 1
    fi
    if [[ -n "${seen_variants[$variant_dir]:-}" ]]; then
      continue
    fi
    seen_variants[$variant_dir]=1
    EXTERNAL_VARIANTS+=("$variant_dir")
  done

  ensure_metric_runtime_deps

  if [[ -z "$EXTERNAL_GT_CACHE_DIR" ]]; then
    EXTERNAL_GT_CACHE_DIR="$EXTERNAL_EVAL_ROOT/__gt_16k_cache"
  fi

  if [[ -n "$EXTERNAL_TRANSCRIPT_KEY" && ! -f "$EXTERNAL_TRANSCRIPT_KEY" ]]; then
    echo "[ERROR] external transcript key not found: $EXTERNAL_TRANSCRIPT_KEY" >&2
    exit 1
  fi

  total_runs=${#EXTERNAL_VARIANTS[@]}
  done_runs=0
  skipped_runs=0

  for variant_dir in "${EXTERNAL_VARIANTS[@]}"; do
    codec_name="$(basename "$(dirname "$variant_dir")")"
    variant_name="$(basename "$variant_dir")"
    eval_out="$EXTERNAL_EVAL_ROOT/$codec_name/$variant_name"
    manifest_path="$eval_out/manifest.jsonl"

    echo "============================================================"
    echo "[EXTERNAL] codec=$codec_name variant=$variant_name"
    echo "[SRC] $variant_dir"
    echo "[OUT] $eval_out"

    if [[ "$FORCE" -eq 0 ]] && metrics_already_covered "$eval_out/metrics.json"; then
      echo "[SKIP] requested metrics already exist: $eval_out/metrics.json"
      skipped_runs=$((skipped_runs + 1))
      done_runs=$((done_runs + 1))
      continue
    fi

    declare -a EVAL_CMD=(
      "$PYTHON_EVAL"
      "$ROOT/eval/eval.py"
      "--input" "$INPUT_LIST"
      "--output_dir" "$eval_out"
      "--manifest" "$manifest_path"
      "--stage" "metrics"
      "--pred_root" "$variant_dir"
      "--gt_cache_dir" "$EXTERNAL_GT_CACHE_DIR"
      "--metrics" "$METRICS"
      "--throughput_warmup_items" "$THROUGHPUT_WARMUP_ITEMS"
      "--length_mode" "$LENGTH_MODE"
      "--num_workers" "$NUM_WORKERS"
    )

    if [[ -n "$DEVICE" ]]; then
      EVAL_CMD+=("--device" "$DEVICE")
    fi
    if [[ -n "$EXTERNAL_TRANSCRIPT_KEY" ]]; then
      EVAL_CMD+=("--transcript_key" "$EXTERNAL_TRANSCRIPT_KEY")
    fi
    for ov in "${EVAL_CFG_OVERRIDES[@]}"; do
      EVAL_CMD+=("--cfg_override" "$ov")
    done
    for arg in "${EVAL_EXTRA_ARGS[@]}"; do
      EVAL_CMD+=("$arg")
    done

    conda run --no-capture-output -n "$EVAL_ENV" "${EVAL_CMD[@]}"
    done_runs=$((done_runs + 1))
  done

  echo
  echo "All external waveform eval jobs finished."
  echo "- total:   $total_runs"
  echo "- done:    $done_runs"
  echo "- skipped: $skipped_runs"
  exit 0
fi

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

  if [[ ! -f "$run_dir/hydra/config.yaml" ]]; then
    echo "[ERROR] Missing config file: $run_dir/hydra/config.yaml" >&2
    exit 1
  fi
  if [[ ! -f "$run_dir/pl_log/last.ckpt" ]]; then
    echo "[ERROR] Missing checkpoint file: $run_dir/pl_log/last.ckpt" >&2
    exit 1
  fi

  ensure_new_config "$run_dir"

  probe_line="$(probe_dtp_capability "$run_dir" "${EVAL_CFG_OVERRIDES[@]}")"
  IFS='|' read -r use_dtp dtp_cls supports_fixed_tau supports_update_test_time <<< "$probe_line"

  dtp_cls_lc="$(printf "%s" "$dtp_cls" | tr '[:upper:]' '[:lower:]')"
  is_fixed_pattern=0
  if [[ "$dtp_cls_lc" == "fixedpattern" || "$dtp_cls_lc" == *.fixedpattern ]]; then
    is_fixed_pattern=1
  fi

  resolved_target_avg_r="$TARGET_AVG_R"
  if [[ -z "$resolved_target_avg_r" ]]; then
    resolved_target_avg_r="$(resolve_target_avg_r_from_cfg "$run_dir" "${EVAL_CFG_OVERRIDES[@]}")"
  fi

  target_bps_line="$(resolve_target_bps_from_cfg "$run_dir" "$resolved_target_avg_r" "${EVAL_CFG_OVERRIDES[@]}")"
  IFS='|' read -r target_bps_suffix target_total_bps target_content_bps target_pos_bps target_f_l1 target_bits_per_token <<< "$target_bps_line"

  rendered_eval_subdir="$(render_subdir_template "$EVAL_SUBDIR" "$target_bps_suffix" "$resolved_target_avg_r")"
  rendered_stats_subdir="$(render_subdir_template "$STATS_SUBDIR" "$target_bps_suffix" "$resolved_target_avg_r")"

  if [[ -n "$EVAL_OUT_OVERRIDE" ]]; then
    eval_out="$EVAL_OUT_OVERRIDE"
  else
    eval_out="$run_dir/$rendered_eval_subdir"
  fi
  stats_out="$run_dir/$rendered_stats_subdir"

  echo "============================================================"
  echo "[RUN] $run_dir"
  echo "[OUT] eval=$eval_out"
  echo "[OUT] stats=$stats_out"
  echo "[TARGET] avg_r=$resolved_target_avg_r -> total_bps=$target_total_bps (content=$target_content_bps, pos=$target_pos_bps, f_L1=$target_f_l1, b=$target_bits_per_token)"

  fixed_tau=""
  fixed_pattern_r_override=""
  do_tau_search=0
  if [[ "$TAU_FINETUNE" -eq 1 && "$use_dtp" -eq 1 && "$supports_fixed_tau" -eq 1 && "$is_fixed_pattern" -eq 0 ]]; then
    do_tau_search=1
  fi

  if [[ "$use_dtp" -eq 1 && "$is_fixed_pattern" -eq 1 ]]; then
    if [[ -n "$TARGET_AVG_R" ]]; then
      fixed_pattern_r_override="$TARGET_AVG_R"
      echo "=== [INFO] dtp_cls=FixedPattern -> skip tau search and set r=${fixed_pattern_r_override} ==="
    else
      echo "=== [INFO] dtp_cls=FixedPattern -> skip tau search (tau has no effect) ==="
    fi
  fi

  summary_json="$stats_out/summary.json"
  if [[ "$FORCE" -eq 0 ]] && metrics_already_covered "$eval_out/metrics.json"; then
    if [[ "$do_tau_search" -eq 1 && ! -f "$summary_json" ]]; then
      echo "[INFO] metrics exists but tau summary missing -> rerun tau search+eval"
    else
      echo "[SKIP] requested metrics already exist: $eval_out/metrics.json"
      if [[ "$do_tau_search" -eq 1 ]]; then
        echo "[SKIP] tau summary exists: $summary_json"
      fi
      skipped_runs=$((skipped_runs + 1))
      done_runs=$((done_runs + 1))
      continue
    fi
  fi

  if [[ "$do_tau_search" -eq 1 ]]; then
    if [[ "$FORCE" -eq 0 && -f "$summary_json" ]]; then
      fixed_tau="$(json_get_best_tau "$summary_json")"
      echo "=== [REUSE] fixed_tau=$fixed_tau (from existing $summary_json) ==="
    else
      run_tau_search "$run_dir" "$stats_out" "$supports_update_test_time" "$resolved_target_avg_r"

      if [[ ! -f "$summary_json" ]]; then
        echo "[ERROR] Missing summary.json at: $summary_json" >&2
        exit 1
      fi
      fixed_tau="$(json_get_best_tau "$summary_json")"
      echo "=== [FOUND] fixed_tau=$fixed_tau (from $summary_json) ==="
    fi
    tau_runs=$((tau_runs + 1))
  else
    if [[ "$TAU_FINETUNE" -eq 0 ]]; then
      echo "=== [INFO] Tau search disabled by --no_tau_finetune ==="
    elif [[ "$use_dtp" -ne 1 ]]; then
      echo "=== [INFO] use_dtp=False -> eval only ==="
    elif [[ "$is_fixed_pattern" -eq 1 ]]; then
      echo "=== [INFO] dtp_cls=FixedPattern -> eval only ==="
    elif [[ "$supports_fixed_tau" -ne 1 ]]; then
      echo "=== [INFO] dtp_cls=${dtp_cls:-unknown} does not expose fixed_tau -> eval only ==="
    else
      echo "=== [INFO] Tau search skipped for run_dir=$run_dir ==="
    fi
    eval_only_runs=$((eval_only_runs + 1))
  fi

  declare -a EVAL_CMD=(
    "$PYTHON_EVAL"
    "$ROOT/eval/eval.py"
    "--input" "$INPUT_LIST"
    "--run_dir" "$run_dir"
    "--output_dir" "$eval_out"
    "--stage" "$EVAL_STAGE"
    "--metrics" "$METRICS"
    "--throughput_warmup_items" "$THROUGHPUT_WARMUP_ITEMS"
    "--length_mode" "$LENGTH_MODE"
    "--num_workers" "$NUM_WORKERS"
  )

  if [[ -n "$DEVICE" ]]; then
    EVAL_CMD+=("--device" "$DEVICE")
  fi
  for ov in "${EVAL_CFG_OVERRIDES[@]}"; do
    EVAL_CMD+=("--cfg_override" "$ov")
  done
  if [[ -n "$fixed_pattern_r_override" ]]; then
    EVAL_CMD+=("--cfg_override" "model.resampler.dtp_params.r=${fixed_pattern_r_override}")
  fi
  if [[ -n "$fixed_tau" ]]; then
    EVAL_CMD+=("--cfg_override" "model.resampler.dtp_params.fixed_tau=${fixed_tau}")
  fi
  for arg in "${EVAL_EXTRA_ARGS[@]}"; do
    EVAL_CMD+=("$arg")
  done

  echo "[EVAL] run_dir=$run_dir output=$eval_out"
  ensure_metric_runtime_deps
  conda run --no-capture-output -n "$EVAL_ENV" "${EVAL_CMD[@]}"

  done_runs=$((done_runs + 1))
done

echo
echo "All eval jobs finished."
echo "- total:    $total_runs"
echo "- done:     $done_runs"
echo "- skipped:  $skipped_runs"
echo "- tau+eval: $tau_runs"
echo "- eval-only:$eval_only_runs"
