#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash download_arch_datasets.sh [--root DATA_ROOT]

Downloads required ARCH datasets for semantic_eval:
  - ravdess
  - emovo
  - audio_mnist
  - slurp

Examples:
  bash download_arch_datasets.sh
  DATA_ROOT=/mnt/data bash download_arch_datasets.sh --root /mnt/data

Notes:
  - Skip already completed datasets if output paths already exist.
  - Downloaded paths must be under: <DATA_ROOT>/<dataset>.
EOF
}

DATA_ROOT="${DATA_ROOT:-/home/hoyso/projects/datasets}"
RAVDESS_URL="https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
EMOVO_GDRIVE_ID="1SUtaKeA-LYnKaD3qv87Y5wYgihJiNJAo"
SLURP_REAL_URL="https://zenodo.org/record/4274930/files/slurp_real.tar.gz?download=1"
SLURP_SYNTH_URL="https://zenodo.org/record/4274930/files/slurp_synth.tar.gz?download=1"

log() {
  printf "[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

need_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Required command not found: $cmd" >&2
    return 1
  fi
}

has_any_audio() {
  local dir="$1"
  if [[ ! -d "$dir" ]]; then
    return 1
  fi
  find "$dir" -type f \( -name "*.wav" -o -name "*.flac" \) | head -n 1 | grep -q .
}

download_ravdess() {
  local target="$DATA_ROOT/ravdess"
  if has_any_audio "$target"; then
    log "ravdess already exists at $target; skip"
    return 0
  fi

  log "downloading ravdess speech zip"
  mkdir -p "$target"
  local tmp
  tmp="$(mktemp)"
  wget -c "$RAVDESS_URL" -O "$tmp"
  unzip -q "$tmp" -d "$target"
  rm -f "$tmp"
}

download_emovo() {
  local target="$DATA_ROOT/emovo/EMOVO"
  if has_any_audio "$target"; then
    log "emovo already exists at $DATA_ROOT/emovo; skip"
    return 0
  fi

  log "downloading emovo from Google Drive"
  mkdir -p "$DATA_ROOT/emovo"

  local tmp
  tmp="$(mktemp /tmp/emovo.XXXXXX.zip)"

  if command -v gdown >/dev/null 2>&1; then
    gdown --id "$EMOVO_GDRIVE_ID" -O "$tmp"
  elif python3 -m gdown --help >/dev/null 2>&1; then
    python3 -m gdown --id "$EMOVO_GDRIVE_ID" -O "$tmp"
  else
    echo "Cannot download emovo: gdown is required (pip install gdown)." >&2
    rm -f "$tmp"
    return 1
  fi

  unzip -q "$tmp" -d "$DATA_ROOT/emovo"
  rm -f "$tmp"
}

download_audio_mnist() {
  local target="$DATA_ROOT/audio_mnist"
  if [[ -f "$target/audioMNIST_meta.txt" ]] && [[ -d "$target/data" ]] && has_any_audio "$target/data"; then
    log "audio_mnist already exists at $target; skip"
    return 0
  fi

  if ! command -v git >/dev/null 2>&1; then
    echo "Cannot download audio_mnist: git is required." >&2
    return 1
  fi

  log "downloading audio_mnist from soerenab/AudioMNIST"
  local tmp_dir="$(mktemp -d)"
  git clone --depth 1 https://github.com/soerenab/AudioMNIST "$tmp_dir"
  mkdir -p "$target"
  if [[ -d "$tmp_dir/data" ]]; then
    cp -r "$tmp_dir/data" "$target/"
  fi
  if [[ -f "$tmp_dir/audioMNIST_meta.txt" ]]; then
    cp "$tmp_dir/audioMNIST_meta.txt" "$target/"
  fi
  rm -rf "$tmp_dir"
}

download_slurp() {
  local target="$DATA_ROOT/slurp"
  if [[ -d "$target/slurp_real" ]] && [[ -d "$target/slurp_synth" ]] && [[ -f "$target/train.jsonl" ]] && [[ -f "$target/devel.jsonl" ]] && [[ -f "$target/test.jsonl" ]]; then
    log "slurp already exists at $target; skip"
    return 0
  fi

  log "downloading slurp"
  mkdir -p "$target"

  local tmp_real
  local tmp_synth
  tmp_real="$(mktemp /tmp/slurp_real.XXXXXX.tar.gz)"
  tmp_synth="$(mktemp /tmp/slurp_synth.XXXXXX.tar.gz)"

  wget -c "$SLURP_REAL_URL" -O "$tmp_real"
  wget -c "$SLURP_SYNTH_URL" -O "$tmp_synth"
  tar -xzf "$tmp_real" -C "$target"
  tar -xzf "$tmp_synth" -C "$target"

  wget -c "https://raw.githubusercontent.com/pswietojanski/slurp/master/dataset/slurp/train.jsonl" -O "$target/train.jsonl"
  wget -c "https://raw.githubusercontent.com/pswietojanski/slurp/master/dataset/slurp/devel.jsonl" -O "$target/devel.jsonl"
  wget -c "https://raw.githubusercontent.com/pswietojanski/slurp/master/dataset/slurp/test.jsonl" -O "$target/test.jsonl"

  rm -f "$tmp_real" "$tmp_synth"
}

main() {
  need_cmd wget
  need_cmd find
  need_cmd unzip
  need_cmd tar

  mkdir -p "$DATA_ROOT"

  log "DATA_ROOT=$DATA_ROOT"

  download_ravdess
  download_emovo
  download_audio_mnist
  download_slurp

  log "all required datasets ready"
  log "layout summary:"
  for d in ravdess emovo audio_mnist slurp; do
    printf "  - %s: %s\n" "$d" "$DATA_ROOT/$d"
  done
}

parse_args() {
  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      -h|--help)
        usage
        exit 0
        ;;
      -r|--root)
        if [[ $# -lt 2 ]]; then
          echo "Missing argument for --root" >&2
          usage
          exit 1
        fi
        DATA_ROOT="$2"
        shift 2
        ;;
      *)
        DATA_ROOT="$1"
        shift
        ;;
    esac
  done
}

parse_args "$@"
main
