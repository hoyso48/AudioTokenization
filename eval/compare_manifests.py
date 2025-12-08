#!/usr/bin/env python3
"""
Sample-level metric comparator for DTMAE experiments.

Given one or more manifest.jsonl files (containing gt/pred paths and per-utterance
metrics), this script selects the top-K and bottom-K examples for a requested metric
per anchor system, then exports the same utterances for every system so each directory
contains cross-system GT/pred audio plus spectrogram visualizations.

Example
-------
python compare_manifests.py \
    --system fixed=/home/.../fixedpatternmasking50hz-vq65536/eval/manifest.jsonl \
    --system ple_ms4=/home/.../PLE50hz-sampleprob1/eval_ft_ms4/manifest.jsonl \
    --metric pesq_wb \
    --k 3 \
    --out-dir /home/hoyso/projects/AudioTokenization/eval/sample_inspection
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np  # noqa: E402

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    _HAS_MATPLOTLIB = True
except Exception:
    matplotlib = None
    plt = None
    _HAS_MATPLOTLIB = False

try:
    from PIL import Image
except Exception:
    Image = None


try:  # Prefer soundfile for simplicity
    import soundfile as sf

    _AUDIO_BACKEND = "soundfile"
except Exception:  # pragma: no cover - fallback paths
    sf = None
    try:
        from scipy.io import wavfile

        _AUDIO_BACKEND = "scipy"
    except Exception:
        wavfile = None
        _AUDIO_BACKEND = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="System label and manifest.jsonl path (repeatable).",
    )
    parser.add_argument(
        "--metric",
        required=True,
        help="Metric key to rank by (e.g., pesq_wb, pesq_nb, stoi, mcd, wer, utmos).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of best/worst samples to export per system.",
    )
    parser.add_argument(
        "--lower-is-better",
        action="store_true",
        help="Flip the objective for metrics like MCD or WER.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("sample_inspection"),
        help="Destination directory for copied assets.",
    )
    parser.add_argument(
        "--n-fft",
        type=int,
        default=1024,
        help="FFT size for the spectrogram visualization.",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=256,
        help="Hop size for the spectrogram visualization.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure DPI for spectrogram PNGs.",
    )
    return parser.parse_args()


def load_audio(path: Path) -> Tuple[np.ndarray, int]:
    if _AUDIO_BACKEND == "soundfile":
        audio, sr = sf.read(path, always_2d=False)
    elif _AUDIO_BACKEND == "scipy":
        if wavfile is None:
            raise RuntimeError("scipy.io.wavfile is unavailable.")
        sr, audio = wavfile.read(path)
    else:  # pragma: no cover - guardrail
        raise RuntimeError(
            "No audio backend available. Install 'soundfile' or 'scipy'."
        )

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if np.issubdtype(audio.dtype, np.integer):
        max_val = np.iinfo(audio.dtype).max
        audio = audio.astype(np.float32) / float(max_val)
    else:
        audio = audio.astype(np.float32)

    return audio, sr


def save_spectrogram(
    audio: np.ndarray,
    sr: int,
    out_path: Path,
    n_fft: int,
    hop_length: int,
    title: str,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 3))
    _, _, _, im = ax.specgram(
        audio,
        NFFT=n_fft,
        Fs=sr,
        noverlap=n_fft - hop_length,
        cmap="magma",
    )
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Power (dB)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def ensure_audio_backend() -> None:
    if _AUDIO_BACKEND is None:
        raise RuntimeError(
            "Unable to read audio. Please install 'soundfile' (pip install soundfile)."
        )


def _key_from_record(record: Dict) -> str:
    orig = record.get("orig_path")
    if orig:
        return orig
    # Fallback to filename if absolute path missing.
    if "gt_16k_path" in record:
        return Path(record["gt_16k_path"]).stem
    if "pred_16k_path" in record:
        return Path(record["pred_16k_path"]).stem
    raise ValueError("Unable to derive unique key for record.")


def _sanitize_label(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def read_manifest(manifest_path: Path) -> List[Dict]:
    entries: List[Dict] = []
    with manifest_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["_utt_id"] = Path(record.get("orig_path", "")).stem
            record["_utt_key"] = _key_from_record(record)
            entries.append(record)
    if not entries:
        raise ValueError(f"No entries found in {manifest_path}")
    return entries


def select_best_worst(
    entries: List[Dict],
    metric: str,
    k: int,
    maximize: bool,
) -> Tuple[List[Dict], List[Dict]]:
    filtered = [e for e in entries if metric in e and e[metric] is not None]
    if not filtered:
        raise ValueError(f"Metric '{metric}' not present in manifest entries.")

    k = min(k, len(filtered))
    best = sorted(filtered, key=lambda x: x[metric], reverse=maximize)[:k]
    worst = sorted(filtered, key=lambda x: x[metric], reverse=not maximize)[:k]
    return best, worst


def copy_audio(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def dump_system_assets(
    sample: Dict,
    metric: str,
    system_dir: Path,
    system_name: str,
    args: argparse.Namespace,
) -> Dict:
    system_dir.mkdir(parents=True, exist_ok=True)

    gt_src = Path(sample["gt_16k_path"])
    pred_src = Path(sample["pred_16k_path"])
    gt_dst = system_dir / "gt.wav"
    pred_dst = system_dir / "pred.wav"
    copy_audio(gt_src, gt_dst)
    copy_audio(pred_src, pred_dst)

    ensure_audio_backend()
    gt_audio, sr = load_audio(gt_src)
    pred_audio, _ = load_audio(pred_src)
    save_spectrogram(
        gt_audio,
        sr,
        system_dir / "gt_spec.png",
        args.n_fft,
        args.hop_length,
        f"{system_name} • GT",
        args.dpi,
    )
    save_spectrogram(
        pred_audio,
        sr,
        system_dir / "pred_spec.png",
        args.n_fft,
        args.hop_length,
        f"{system_name} • Pred",
        args.dpi,
    )

    utt_id = sample["_utt_id"]
    meta = {
        "metric": metric,
        "score": float(sample[metric]),
        "system": system_name,
        "utt_id": utt_id,
        "utt_key": sample["_utt_key"],
        "orig_path": sample.get("orig_path"),
        "gt_path": str(gt_src),
        "pred_path": str(pred_src),
        "transcript_path": sample.get("transcript_path"),
        "gt_text": sample.get("gt_text"),
        "asr_text": sample.get("asr_text"),
    }
    with (system_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def main() -> None:
    args = parse_args()
    maximize = not args.lower_is_better
    metric = args.metric

    systems: Dict[str, Path] = {}
    for spec in args.system:
        if "=" not in spec:
            raise ValueError(f"--system expects NAME=PATH, got '{spec}'")
        name, path = spec.split("=", 1)
        name = name.strip()
        manifest_path = Path(path).expanduser()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        systems[name] = manifest_path

    manifest_entries: Dict[str, List[Dict]] = {}
    entry_lookup: Dict[str, Dict[str, Dict]] = {}
    for name, manifest_path in systems.items():
        entries = read_manifest(manifest_path)
        manifest_entries[name] = entries
        lookup = {record["_utt_key"]: record for record in entries}
        entry_lookup[name] = lookup

    metric_root = args.out_dir / metric
    best_root = metric_root / "best"
    worst_root = metric_root / "worst"
    for root in (best_root, worst_root):
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)

    summary = {
        "metric": metric,
        "higher_is_better": maximize,
        "k": args.k,
        "systems": {k: str(v) for k, v in systems.items()},
        "selections": [],
    }

    for anchor_name, entries in manifest_entries.items():
        best, worst = select_best_worst(entries, metric, args.k, maximize)

        anchor_root = (
            args.out_dir
            / metric
            / anchor_name.replace("/", "_").replace(" ", "_")
        )

        safe_anchor = _sanitize_label(anchor_name)
        for split_name, subset in (("best", best), ("worst", worst)):
            for rank, sample in enumerate(subset, 1):
                utt_key = sample["_utt_key"]
                anchor_score = float(sample[metric])
                utt_id = sample["_utt_id"]
                folder = f"{safe_anchor}_{rank:02d}_{anchor_score:.4f}_{utt_id}"
                parent = best_root if split_name == "best" else worst_root
                sample_dir = parent / folder
                if sample_dir.exists():
                    shutil.rmtree(sample_dir)
                sample_dir.mkdir(parents=True, exist_ok=True)

                selection_entry = {
                    "anchor_system": anchor_name,
                    "split": split_name,
                    "rank": rank,
                    "anchor_score": anchor_score,
                    "utt_id": utt_id,
                    "utt_key": utt_key,
                    "sample_dir": str(sample_dir),
                    "systems": {},
                }

                for system_name, lookup in entry_lookup.items():
                    record = lookup.get(utt_key)
                    if record is None:
                        continue
                    system_dir = sample_dir / _sanitize_label(system_name)
                    meta = dump_system_assets(
                        record, metric, system_dir, system_name, args
                    )
                    selection_entry["systems"][system_name] = meta

                with (sample_dir / "selection.json").open("w") as f:
                    json.dump(selection_entry, f, indent=2, ensure_ascii=False)
                summary["selections"].append(selection_entry)

    metric_root.mkdir(parents=True, exist_ok=True)
    summary_path = metric_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()

