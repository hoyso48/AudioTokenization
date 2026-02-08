#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_OUTPUTS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/outputs"

outputs_dir="$DEFAULT_OUTPUTS_DIR"
match_mode="exact"

usage() {
  cat <<'EOF'
Usage:
  find_outputs_by_name.sh [--outputs-dir DIR] [--contains] NAME [NAME ...]

Options:
  --outputs-dir DIR   outputs root (absolute or relative path)
  --contains          substring match (default: exact match)
  -h, --help          show this help
EOF
}

resolve_dir() {
  local input_dir="$1"

  # Absolute path
  if [[ "$input_dir" = /* ]]; then
    if [[ -d "$input_dir" ]]; then
      (cd "$input_dir" && pwd)
      return 0
    fi
    return 1
  fi

  # Relative to current working directory
  if [[ -d "$input_dir" ]]; then
    (cd "$input_dir" && pwd)
    return 0
  fi

  # Relative to repo root (script_dir/..)
  local repo_rel="$SCRIPT_DIR/../$input_dir"
  if [[ -d "$repo_rel" ]]; then
    (cd "$repo_rel" && pwd)
    return 0
  fi

  return 1
}

if [[ $# -eq 0 ]]; then
  usage
  exit 2
fi

names=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --outputs-dir)
      if [[ $# -lt 2 ]]; then
        echo "[error] --outputs-dir requires a value" >&2
        exit 2
      fi
      outputs_dir="$2"
      shift 2
      ;;
    --contains)
      match_mode="contains"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      echo "[error] unknown option: $1" >&2
      exit 2
      ;;
    *)
      names+=("$1")
      shift
      ;;
  esac
done

if [[ ${#names[@]} -eq 0 ]]; then
  echo "[error] at least one NAME is required" >&2
  exit 2
fi

resolved_outputs_dir="$(resolve_dir "$outputs_dir" || true)"
if [[ -z "$resolved_outputs_dir" ]]; then
  echo "[error] outputs directory not found: $outputs_dir" >&2
  echo "        tried absolute path, cwd-relative, and repo-relative paths" >&2
  exit 2
fi
outputs_dir="$resolved_outputs_dir"

fmt_size() {
  local bytes="$1"
  if command -v numfmt >/dev/null 2>&1; then
    numfmt --to=iec-i --suffix=B "$bytes"
  else
    printf "%s B" "$bytes"
  fi
}

fmt_mtime_epoch() {
  local path="$1"
  stat -c "%Y" "$path"
}

fmt_mtime_str() {
  local path="$1"
  stat -c "%y" "$path" | cut -d'.' -f1
}

extract_top_level_name() {
  local cfg="$1"
  awk '
    /^[^[:space:]][^:]*:[[:space:]]*/ {
      if ($0 ~ /^name:[[:space:]]*/) {
        sub(/^name:[[:space:]]*/, "", $0)
        print
        exit
      }
    }
  ' "$cfg"
}

find_best_ckpt() {
  local run_dir="$1"
  local best_path=""
  local best_step="-1"
  local best_mtime="-1"
  local best_size="0"
  local count="0"

  shopt -s nullglob globstar
  local ckpt
  for ckpt in "$run_dir"/**/*.ckpt; do
    [[ -f "$ckpt" ]] || continue
    count=$((count + 1))

    local base step mtime size
    base="$(basename "$ckpt")"
    step="-1"
    if [[ "$base" =~ step=([0-9]+) ]]; then
      step="${BASH_REMATCH[1]}"
    fi
    mtime="$(fmt_mtime_epoch "$ckpt")"
    size="$(stat -c "%s" "$ckpt")"

    if (( step > best_step )) || { (( step == best_step )) && (( mtime > best_mtime )); }; then
      best_step="$step"
      best_mtime="$mtime"
      best_path="$ckpt"
      best_size="$size"
    fi
  done
  shopt -u globstar

  printf "%s|%s|%s|%s|%s\n" "$count" "$best_path" "$best_step" "$best_mtime" "$best_size"
}

print_match() {
  local idx="$1"
  local run_dir="$2"
  local cfg="$3"
  local cfg_size="$4"
  local cfg_mtime_str="$5"
  local ckpt_count="$6"
  local best_ckpt="$7"
  local best_step="$8"
  local best_size="$9"
  local best_mtime_str="${10}"

  echo "  [$idx] run_dir        : $run_dir"
  echo "      config         : $cfg"
  echo "      config_info    : size=$(fmt_size "$cfg_size"), mtime=$cfg_mtime_str"
  echo "      ckpt_count     : $ckpt_count"
  if [[ -z "$best_ckpt" ]]; then
    echo "      best_ckpt      : (none)"
  else
    echo "      best_ckpt      : $best_ckpt"
    echo "      best_ckpt_info : step=$best_step, size=$(fmt_size "$best_size"), mtime=$best_mtime_str"
  fi
}

echo "[info] outputs_dir: $outputs_dir"
echo "[info] match_mode : $match_mode"

any_found=0
for target in "${names[@]}"; do
  echo
  printf '=%.0s' {1..90}
  echo
  echo "[query] name: $target"

  shopt -s nullglob
  config_paths=("$outputs_dir"/*/*/hydra/config.yaml)

  matches=()
  for cfg in "${config_paths[@]}"; do
    [[ -f "$cfg" ]] || continue
    name_val="$(extract_top_level_name "$cfg" || true)"
    [[ -n "$name_val" ]] || continue

    is_match=0
    if [[ "$match_mode" == "contains" ]]; then
      [[ "$name_val" == *"$target"* ]] && is_match=1
    else
      [[ "$name_val" == "$target" ]] && is_match=1
    fi

    (( is_match == 1 )) || continue

    run_dir="$(dirname "$(dirname "$cfg")")"
    cfg_size="$(stat -c "%s" "$cfg")"
    cfg_mtime_epoch="$(fmt_mtime_epoch "$cfg")"
    cfg_mtime_str="$(fmt_mtime_str "$cfg")"

    best_info="$(find_best_ckpt "$run_dir")"
    IFS='|' read -r ckpt_count best_ckpt best_step best_mtime_epoch best_size <<< "$best_info"

    best_mtime_str=""
    if [[ -n "$best_ckpt" ]]; then
      best_mtime_str="$(fmt_mtime_str "$best_ckpt")"
    fi

    matches+=("$run_dir|$cfg|$cfg_size|$cfg_mtime_epoch|$cfg_mtime_str|$ckpt_count|$best_ckpt|$best_step|$best_size|$best_mtime_epoch|$best_mtime_str")
  done

  if [[ ${#matches[@]} -eq 0 ]]; then
    echo "[result] no matching config.yaml found"
    continue
  fi

  any_found=1
  echo "[result] found ${#matches[@]} matching run(s)"

  idx=1
  selected_idx=0
  selected_step=-1
  selected_mtime=-1

  for entry in "${matches[@]}"; do
    IFS='|' read -r run_dir cfg cfg_size cfg_mtime_epoch cfg_mtime_str ckpt_count best_ckpt best_step best_size best_mtime_epoch best_mtime_str <<< "$entry"

    print_match "$idx" "$run_dir" "$cfg" "$cfg_size" "$cfg_mtime_str" "$ckpt_count" "$best_ckpt" "$best_step" "$best_size" "$best_mtime_str"

    cmp_step="$best_step"
    cmp_mtime="$best_mtime_epoch"
    if [[ -z "$best_ckpt" ]]; then
      cmp_step=-1
      cmp_mtime="$cfg_mtime_epoch"
    fi

    if (( cmp_step > selected_step )) || { (( cmp_step == selected_step )) && (( cmp_mtime > selected_mtime )); }; then
      selected_step="$cmp_step"
      selected_mtime="$cmp_mtime"
      selected_idx="$idx"
    fi

    idx=$((idx + 1))
  done

  echo
  echo "[selected] choose one run (highest checkpoint step)"
  selected_entry="${matches[$((selected_idx - 1))]}"
  IFS='|' read -r run_dir cfg cfg_size _cfg_mtime_epoch cfg_mtime_str ckpt_count best_ckpt best_step best_size _best_mtime_epoch best_mtime_str <<< "$selected_entry"
  print_match 1 "$run_dir" "$cfg" "$cfg_size" "$cfg_mtime_str" "$ckpt_count" "$best_ckpt" "$best_step" "$best_size" "$best_mtime_str"

  if [[ ${#matches[@]} -gt 1 ]]; then
    echo "[note] multiple matches existed; full list shown above."
  fi
done

if (( any_found == 0 )); then
  exit 1
fi
