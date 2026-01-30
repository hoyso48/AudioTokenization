#!/usr/bin/env python3
"""
Filter an audio filelist by duration.

Typical usage for LibriSpeech test-clean:

  conda activate speech_eval
  cd /home/hoyso/projects/AudioTokenization
  python eval/filter_filelist_by_duration.py \
    --input_list DTMAE/filelists/librispeech_test_clean.txt \
    --output_list DTMAE/filelists/librispeech_test_clean_filtered_4s10s.txt \
    --min_sec 4 --max_sec 10 \
    --expected_count 1088

The input filelist can contain absolute paths (recommended) or relative paths.
We use torchaudio.info() to avoid loading full audio into memory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Optional


def read_lines(path: Path) -> List[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def duration_seconds(path: Path) -> float:
    import torchaudio  # local import (requires speech_eval env)

    info = torchaudio.info(str(path))
    if info.sample_rate <= 0:
        raise RuntimeError(f"Invalid sample_rate for {path}: {info.sample_rate}")
    if info.num_frames < 0:
        raise RuntimeError(f"Invalid num_frames for {path}: {info.num_frames}")
    return float(info.num_frames) / float(info.sample_rate)


def filter_paths(
    paths: List[str],
    min_sec: float,
    max_sec: float,
) -> Tuple[List[str], List[Tuple[str, float]]]:
    kept: List[str] = []
    stats: List[Tuple[str, float]] = []
    for p in paths:
        pp = Path(p)
        if not pp.is_absolute():
            pp = Path.cwd() / pp
        if not pp.exists():
            raise FileNotFoundError(f"Missing audio file: {pp} (from entry '{p}')")
        dur = duration_seconds(pp)
        if (dur >= min_sec) and (dur <= max_sec):
            kept.append(str(pp.resolve()))
            stats.append((str(pp.resolve()), dur))
    return kept, stats


def suggest_max_sec_for_expected_count(
    stats: List[Tuple[str, float]],
    *,
    min_sec: float,
    expected_count: int,
    resolution: float = 0.01,
) -> Optional[float]:
    """
    Given durations for kept items (>=min_sec), suggest a max_sec (<=) that yields expected_count.
    This is only for debugging when user-provided expected_count doesn't match.
    """
    if expected_count <= 0:
        return None
    durs = sorted(d for _, d in stats if d >= min_sec)
    if expected_count > len(durs):
        return None
    # Find threshold duration at rank expected_count (1-indexed), then round up to resolution.
    thresh = durs[expected_count - 1]
    return round(float(thresh) / resolution) * resolution


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Filter a filelist by audio duration (seconds).")
    p.add_argument("--input_list", type=str, required=True)
    p.add_argument("--output_list", type=str, required=True)
    p.add_argument("--min_sec", type=float, required=True)
    p.add_argument("--max_sec", type=float, required=True)
    p.add_argument("--expected_count", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    in_path = Path(args.input_list).expanduser().resolve()
    out_path = Path(args.output_list).expanduser().resolve()

    if not in_path.is_file():
        raise FileNotFoundError(f"input_list not found: {in_path}")
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"output_list already exists: {out_path} (use --overwrite)")
    if args.min_sec < 0 or args.max_sec <= 0 or args.max_sec < args.min_sec:
        raise ValueError("Invalid min_sec/max_sec range.")

    paths = read_lines(in_path)
    kept, stats = filter_paths(paths, min_sec=float(args.min_sec), max_sec=float(args.max_sec))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(kept) + ("\n" if kept else ""))

    count = len(kept)
    print(f"input_count={len(paths)} kept_count={count} min_sec={args.min_sec} max_sec={args.max_sec}")
    if args.expected_count is not None and count != int(args.expected_count):
        hint = suggest_max_sec_for_expected_count(
            stats,
            min_sec=float(args.min_sec),
            expected_count=int(args.expected_count),
        )
        hint_msg = f" (hint: with min_sec={args.min_sec}, max_sec≈{hint} yields {args.expected_count})" if hint else ""
        raise RuntimeError(f"Filtered count mismatch: expected {args.expected_count}, got {count}.{hint_msg}")


if __name__ == "__main__":
    main()


