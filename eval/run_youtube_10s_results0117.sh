#!/usr/bin/env bash
set -euo pipefail

# Download 10-second YouTube audio clips (MP3, titled filenames), resample to 16 kHz mono WAV,
# then run eval/eval.py on results0117 run dirs using pre-computed best.fixed_tau from dtp_stats.
#
# Prerequisite:
#   conda activate speech_eval   (or any env where torch/torchaudio/omegaconf/transformers work)
#
# Run from repo root:
#   cd /home/hoyso/projects/AudioTokenization
#   bash eval/run_youtube_10s_results0117.sh
#
# Outputs:
# - clips:
#   eval/youtube_10s/{raw_mp3,mp3_10s,wav_16k}/
#   eval/youtube_10s/manifest.jsonl
# - eval per run_dir:
#   <run_dir>/eval_youtube_10s/metrics.json (+ audio_metrics.json, reconstructed wavs, manifest.jsonl, ...)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
YTDLP_BIN="${YTDLP_BIN:-yt-dlp}"
FFMPEG_BIN="${FFMPEG_BIN:-ffmpeg}"
CONDA_BIN="${CONDA_BIN:-conda}"

YTDLP_PLAYER_CLIENT="${YTDLP_PLAYER_CLIENT:-android}"

CLIP_ROOT="eval/youtube_10s"
RAW_MP3_DIR="$CLIP_ROOT/raw_mp3"
MP3_10S_DIR="$CLIP_ROOT/mp3_10s"
WAV_16K_DIR="$CLIP_ROOT/wav_16k"
MANIFEST_JSONL="$CLIP_ROOT/manifest.jsonl"

require_env() {
  if ! "$PYTHON_BIN" -c "import torch, torchaudio, omegaconf" >/dev/null 2>&1; then
    echo "[ERROR] Python deps not found. Did you activate the correct env?" >&2
    echo "        (torch/torchaudio/omegaconf must import successfully)" >&2
    exit 1
  fi
}

require_tools() {
  if ! command -v "$FFMPEG_BIN" >/dev/null 2>&1; then
    echo "[ERROR] ffmpeg not found. Install it (e.g. conda-forge ffmpeg) and retry." >&2
    exit 1
  fi

  if command -v "$YTDLP_BIN" >/dev/null 2>&1; then
    return 0
  fi

  echo "[WARN] yt-dlp not found. Installing via pip..."
  if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    "$PYTHON_BIN" -m pip install -U yt-dlp
  else
    python3 -m pip install --user -U yt-dlp
  fi

  if ! command -v "$YTDLP_BIN" >/dev/null 2>&1; then
    echo "[ERROR] yt-dlp install succeeded but yt-dlp still not on PATH." >&2
    echo "        Try: export PATH=\$HOME/.local/bin:\$PATH" >&2
    exit 1
  fi
}

ensure_js_runtime() {
  # yt-dlp increasingly requires a JS runtime for signature extraction on some videos.
  # Prefer node if available; if not, try installing nodejs via conda-forge.
  if command -v node >/dev/null 2>&1; then
    return 0
  fi

  if ! command -v "$CONDA_BIN" >/dev/null 2>&1; then
    echo "[ERROR] node is missing and conda is not available to install it." >&2
    echo "        Install nodejs and retry (or set YTDLP_BIN to a yt-dlp with a JS runtime configured)." >&2
    exit 1
  fi

  echo "[WARN] node is not found; installing nodejs via conda-forge..."
  "$CONDA_BIN" install -y -c conda-forge nodejs

  if ! command -v node >/dev/null 2>&1; then
    echo "[ERROR] nodejs install finished but node is still not on PATH." >&2
    exit 1
  fi
}

ensure_new_config() {
  # Convert legacy hydra/config.yaml (codec_decoder VQ fields) to new model.quantizer style.
  local run_dir="$1"
  local cfg_path="$run_dir/hydra/config.yaml"
  if [[ ! -f "$cfg_path" ]]; then
    echo "[ERROR] Missing config: $cfg_path" >&2
    exit 1
  fi
  echo "=== [CONFIG] update_legacy_config.py on $cfg_path ==="
  "$PYTHON_BIN" utils/update_legacy_config.py --path "$cfg_path"
}

time_to_seconds() {
  # Accept "SS", "MM:SS", or "HH:MM:SS" and print integer seconds.
  local t="$1"
  local IFS=":"
  read -r -a parts <<<"$t"
  if [[ "${#parts[@]}" -eq 1 ]]; then
    echo "${parts[0]}"
    return 0
  fi
  if [[ "${#parts[@]}" -eq 2 ]]; then
    echo $((10#${parts[0]} * 60 + 10#${parts[1]}))
    return 0
  fi
  if [[ "${#parts[@]}" -eq 3 ]]; then
    echo $((10#${parts[0]} * 3600 + 10#${parts[1]} * 60 + 10#${parts[2]}))
    return 0
  fi
  echo "[ERROR] Bad timestamp: $t" >&2
  exit 1
}

yt_get_meta_tsv() {
  # Prints: "<title>\t<id>"
  local url="$1"
  "$YTDLP_BIN" \
    --no-playlist \
    --skip-download \
    --js-runtimes node \
    --extractor-args "youtube:player_client=$YTDLP_PLAYER_CLIENT" \
    --print "%(title)s\t%(id)s" \
    "$url" | head -n 1
}

yt_download_mp3() {
  # Downloads MP3 (full audio) and prints the resulting filepath.
  local url="$1"
  mkdir -p "$RAW_MP3_DIR"
  "$YTDLP_BIN" \
    --no-playlist \
    --js-runtimes node \
    --extractor-args "youtube:player_client=$YTDLP_PLAYER_CLIENT" \
    --extract-audio \
    --audio-format mp3 \
    --audio-quality 0 \
    --restrict-filenames \
    --no-progress \
    -o "$RAW_MP3_DIR/%(title)s__%(id)s.%(ext)s" \
    --print after_move:filepath \
    "$url" | tail -n 1
}

make_clips() {
  mkdir -p "$RAW_MP3_DIR" "$MP3_10S_DIR" "$WAV_16K_DIR"
  mkdir -p "$(dirname "$MANIFEST_JSONL")"
  : >"$MANIFEST_JSONL"

  # (url, start_time) pairs
  local -a URLS=(
    "https://youtu.be/9TxtTF_dUHQ?si=4kPSZ5HNpA5iYxHZ"
    "https://youtu.be/jNQXAC9IVRw?si=k7sNX3mM8lYMW3cU"
    "https://youtu.be/sP5ElraFHHE?si=1DUetO19MwoFLKW3"
    "https://youtu.be/gfHEOL-sDy4?si=vZAtCdpFv1TrHWjx"
    "https://youtu.be/g4xoe5Ccuzc?si=v9JdeTYoZk-Y5P9P"
    "https://youtu.be/YWq_CGT0QPY?si=G2UmlikXbSM-kzTs"
  )
  local -a STARTS=("2:05" "0" "44" "26" "0" "1:30")

  if [[ "${#URLS[@]}" -ne "${#STARTS[@]}" ]]; then
    echo "[ERROR] URL/START array length mismatch." >&2
    exit 1
  fi

  echo "=== [CLIPS] Preparing ${#URLS[@]} clips (10s) into $CLIP_ROOT ==="

  local -a failed=()
  local i
  for ((i = 0; i < ${#URLS[@]}; i++)); do
    local url="${URLS[$i]}"
    local start="${STARTS[$i]}"
    local start_sec
    start_sec="$(time_to_seconds "$start")"
    local start_tag="${start//:/-}"
    local idx
    idx="$(printf "%02d" $((i + 1)))"

    echo
    echo "=== [CLIP $idx] url=$url start=$start (${start_sec}s) ==="

    local meta
    if ! meta="$(yt_get_meta_tsv "$url")"; then
      echo "[ERROR] Failed to fetch metadata via yt-dlp for: $url" >&2
      failed+=("$idx:meta:$url")
      continue
    fi
    local title yt_id
    title="$(printf "%s" "$meta" | cut -f1)"
    yt_id="$(printf "%s" "$meta" | cut -f2)"

    local src_mp3
    if ! src_mp3="$(yt_download_mp3 "$url")"; then
      echo "[ERROR] Failed to download audio via yt-dlp for: $url" >&2
      failed+=("$idx:download:$url")
      continue
    fi
    if [[ ! -f "$src_mp3" ]]; then
      echo "[ERROR] Download reported: $src_mp3 but file not found." >&2
      failed+=("$idx:missing_file:$url")
      continue
    fi

    local base
    base="$(basename "$src_mp3" .mp3)"
    local out_base="${idx}__${base}__ss-${start_tag}__dur-10"
    local out_mp3="$MP3_10S_DIR/${out_base}.mp3"
    local out_wav="$WAV_16K_DIR/${out_base}.wav"

    if [[ ! -f "$out_mp3" ]]; then
      echo "=== [CROP] $out_mp3 ==="
      "$FFMPEG_BIN" -hide_banner -loglevel error -y \
        -ss "$start" -t 10 \
        -i "$src_mp3" \
        -vn -c:a libmp3lame -q:a 2 \
        "$out_mp3"
    else
      echo "=== [SKIP] mp3_10s exists: $out_mp3 ==="
    fi

    if [[ ! -f "$out_wav" ]]; then
      echo "=== [RESAMPLE] $out_wav (mono 16k) ==="
      "$FFMPEG_BIN" -hide_banner -loglevel error -y \
        -i "$out_mp3" \
        -ac 1 -ar 16000 -c:a pcm_s16le \
        "$out_wav"
    else
      echo "=== [SKIP] wav_16k exists: $out_wav ==="
    fi

    "$PYTHON_BIN" - "$MANIFEST_JSONL" "$url" "$start" "$start_sec" "$title" "$yt_id" "$src_mp3" "$out_mp3" "$out_wav" <<'PY'
import json, sys
manifest_path, url, start, start_sec, title, yt_id, src_mp3, out_mp3, out_wav = sys.argv[1:]
rec = {
    "url": url,
    "start": start,
    "start_seconds": int(float(start_sec)),
    "duration_seconds": 10,
    "title": title,
    "youtube_id": yt_id,
    "raw_mp3": src_mp3,
    "clip_mp3": out_mp3,
    "clip_wav_16k": out_wav,
}
with open(manifest_path, "a") as f:
    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
PY
  done

  if [[ "${#failed[@]}" -gt 0 ]]; then
    echo
    echo "[ERROR] Some clips failed. Fix the download issue and re-run."
    printf " - %s\n" "${failed[@]}"
    exit 1
  fi

  echo
  echo "=== [DONE] Clips ready:"
  echo " - wav inputs: $WAV_16K_DIR"
  echo " - manifest:   $MANIFEST_JSONL"
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

run_eval() {
  local run_dir="$1"
  local eval_out="$2"
  local tau_summary_json="${3:-}"

  ensure_new_config "$run_dir"

  if [[ -f "$eval_out/metrics.json" ]]; then
    echo "=== [SKIP] metrics.json already exists at $eval_out/metrics.json ==="
    return 0
  fi

  local -a cfg_args=()
  if [[ -n "$tau_summary_json" ]]; then
    if [[ ! -f "$tau_summary_json" ]]; then
      echo "[ERROR] Missing dtp_stats summary.json: $tau_summary_json" >&2
      exit 1
    fi
    local tau
    tau="$(json_get_best_tau "$tau_summary_json")"
    echo "=== [TAU] best.fixed_tau=$tau (from $tau_summary_json) ==="
    cfg_args+=(--cfg_override "model.resampler.dtp_params.fixed_tau=${tau}")
  else
    echo "=== [TAU] (none) no fixed_tau override for this run ==="
  fi

  echo
  echo "=== [EVAL] run_dir=$run_dir out=$eval_out input=$WAV_16K_DIR ==="
  "$PYTHON_BIN" eval/eval.py \
    --input "$WAV_16K_DIR" \
    --run_dir "$run_dir" \
    --output_dir "$eval_out" \
    --stage all \
    --length_mode pad \
    --num_workers 2 \
    "${cfg_args[@]}"
}

main() {
  require_env
  require_tools
  ensure_js_runtime

  make_clips

  # ---------------------------------------------------------------------------
  # Evaluate requested run dirs (results0117)
  # ---------------------------------------------------------------------------
  run_eval \
    "results/results0117/default_PLE_25hz_vq16384" \
    "results/results0117/default_PLE_25hz_vq16384/eval_youtube_10s" \
    "results/results0117/default_PLE_25hz_vq16384/dtp_stats_ft/summary.json"

  run_eval \
    "results/results0117/default_PLE_50hz_vq16384" \
    "results/results0117/default_PLE_50hz_vq16384/eval_youtube_10s" \
    "results/results0117/default_PLE_50hz_vq16384/dtp_stats_ft/summary.json"

  run_eval \
    "results/results0117/default_fixedpattern_50hz_vq65536" \
    "results/results0117/default_fixedpattern_50hz_vq65536/eval_youtube_10s"

  # Uses precomputed PLE dtp_stats for the random run.
  run_eval \
    "results/results0117/default_random_50hz_vq16384" \
    "results/results0117/default_random_50hz_vq16384/eval_youtube_10s_ple" \
    "results/results0117/default_random_50hz_vq16384/dtp_stats_ple_ft/summary.json"

  echo
  echo "=== [ALL DONE] ==="
}

main "$@"


