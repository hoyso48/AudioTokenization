#!/usr/bin/env python3
"""
Benchmark DTMAE Hugging Face *streaming* throughput (samples/sec) for parler-tts/mls_eng.

This measures the DTMAE data pipeline cost:
  HF streaming fetch -> (optional) decode -> resample -> crop/pad -> DataLoader batch

It reads the same YAML config structure used by DTMAE (e.g. config_base_mls/dataset/mls_eng_streaming.yaml)
and applies:
  - dataloader.num_workers / pin_memory / persistent_workers / prefetch_factor / timeout / multiprocessing_context
  - hf_streaming_mls_eng.* spec
  - train.batch_size / train.shuffle / train.min_audio_length
  - dataset-level sample_rate / multiple_of
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader


def _add_dtmae_to_syspath() -> str:
    # This script lives in DTMAE/scripts/. Add DTMAE/ to sys.path so we can import hf_streaming_dataset.py.
    dtmae_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if dtmae_dir not in sys.path:
        sys.path.insert(0, dtmae_dir)
    return dtmae_dir


def _read_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "Missing dependency: pyyaml\n"
            "Install via:\n"
            "  python -m pip install pyyaml\n"
        ) from e
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping, got: {type(data)}")
    return data


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark DTMAE MLS-Eng HF streaming throughput (samples/sec)."
    )
    p.add_argument(
        "--config-yaml",
        type=str,
        required=True,
        help="Path to DTMAE dataset YAML (e.g. config_base_mls/dataset/mls_eng_streaming.yaml).",
    )
    p.add_argument("--phase", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument(
        "--warmup-batches",
        type=int,
        default=2,
        help="Warm up N batches (not measured). Set 0 to disable.",
    )
    p.add_argument(
        "--max-batches",
        type=int,
        default=100,
        help="Measure up to N batches (0 disables batch limit).",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Measure up to N samples (0 disables sample limit).",
    )
    p.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="Stop the measured phase after this many seconds (0 disables).",
    )
    p.add_argument(
        "--report-every",
        type=int,
        default=20,
        help="Print progress every N batches (0 disables).",
    )
    p.add_argument(
        "--override-num-workers",
        type=int,
        default=-1,
        help="If set >=0, overrides dataloader.num_workers from YAML.",
    )
    return p.parse_args()


def _get_nested(d: Dict[str, Any], path: str, default: Any) -> Any:
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _collate_wav_only(bs: list[Dict[str, Any]]) -> Dict[str, Any]:
    wavs = [b["wav"] for b in bs]
    return {"wav": torch.stack(wavs)}


def _collate_wav_transcript(bs: list[Dict[str, Any]]) -> Dict[str, Any]:
    wavs = [b["wav"] for b in bs]
    return {
        "wav": torch.stack(wavs),
        "utt_id": [str(b.get("utt_id", "")) for b in bs],
        "transcript": [str(b.get("transcript", "")) for b in bs],
    }


def _build_dataloader_from_yaml(y: Dict[str, Any], phase: str, override_num_workers: int) -> Tuple[DataLoader, Dict[str, Any]]:
    dtmae_dir = _add_dtmae_to_syspath()
    from hf_streaming_dataset import MlsEngStreamingDataset, MlsEngStreamingSpec  # noqa: E402

    phase_cfg = _get_nested(y, phase, None)
    if not isinstance(phase_cfg, dict):
        raise ValueError(f"Missing or invalid '{phase}' section in YAML.")

    backend = str(phase_cfg.get("backend", "filelist"))
    if backend != "hf_streaming_mls_eng":
        raise ValueError(
            f"Expected {phase}.backend == 'hf_streaming_mls_eng' for this benchmark, got: {backend}"
        )

    spec_cfg = _get_nested(y, "hf_streaming_mls_eng", None)
    if not isinstance(spec_cfg, dict):
        raise ValueError("Missing or invalid 'hf_streaming_mls_eng' section in YAML.")

    transcript_cfg = _get_nested(y, "transcript", {}) or {}
    if not isinstance(transcript_cfg, dict):
        transcript_cfg = {}

    include_transcript = bool(transcript_cfg.get("enable", False))
    lowercase_transcript = bool(transcript_cfg.get("lowercase", True))

    sample_rate = int(y.get("sample_rate", 16000))
    multiple_of = int(y.get("multiple_of", 320))
    batch_size = int(phase_cfg.get("batch_size", 32))

    cache_dir = spec_cfg.get("cache_dir", None)
    if cache_dir is not None:
        cache_dir = str(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)

    spec = MlsEngStreamingSpec(
        hf_repo_id=str(spec_cfg.get("hf_repo_id", "parler-tts/mls_eng")),
        split=str(spec_cfg.get("split", "train")),
        shuffle=bool(phase_cfg.get("shuffle", True)),
        shuffle_buffer_size=int(spec_cfg.get("shuffle_buffer_size", 2000)),
        seed=int(spec_cfg.get("seed", 1024)),
        sample_rate=sample_rate,
        min_audio_length=int(phase_cfg.get("min_audio_length", 64000)),
        multiple_of=multiple_of,
        lowercase_transcript=lowercase_transcript,
        include_transcript=include_transcript,
        cache_dir=cache_dir,
        download_max_retries=int(spec_cfg.get("download_max_retries", 50)),
        download_timeout=int(spec_cfg.get("download_timeout", 60)),
    )

    ds = MlsEngStreamingDataset(phase=phase, spec=spec)
    collate_fn = _collate_wav_transcript if include_transcript else _collate_wav_only

    dl_cfg = _get_nested(y, "dataloader", {}) or {}
    if not isinstance(dl_cfg, dict):
        dl_cfg = {}

    num_workers = int(dl_cfg.get("num_workers", 0))
    if override_num_workers >= 0:
        num_workers = int(override_num_workers)

    pin_memory = bool(dl_cfg.get("pin_memory", True))
    persistent_workers = bool(dl_cfg.get("persistent_workers", num_workers > 0))
    if num_workers == 0:
        persistent_workers = False

    timeout = dl_cfg.get("timeout", None)
    timeout_val: float = float(timeout) if timeout is not None else 0.0

    prefetch_factor = dl_cfg.get("prefetch_factor", None)
    prefetch_factor_val: Optional[int] = int(prefetch_factor) if prefetch_factor is not None else None

    multiprocessing_context = dl_cfg.get("multiprocessing_context", None)
    multiprocessing_context_val: Optional[str] = str(multiprocessing_context) if multiprocessing_context else None

    dl_kwargs: Dict[str, Any] = dict(
        dataset=ds,
        batch_size=batch_size,
        shuffle=False,  # IterableDataset: DataLoader shuffle must be False
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        timeout=timeout_val,
    )
    if num_workers > 0 and prefetch_factor_val is not None:
        dl_kwargs["prefetch_factor"] = prefetch_factor_val
    if num_workers > 0 and multiprocessing_context_val is not None:
        dl_kwargs["multiprocessing_context"] = multiprocessing_context_val

    meta = {
        "dtmae_dir": dtmae_dir,
        "phase": phase,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "timeout": timeout_val,
        "prefetch_factor": prefetch_factor_val,
        "multiprocessing_context": multiprocessing_context_val,
        "include_transcript": include_transcript,
        "spec": {
            "hf_repo_id": spec.hf_repo_id,
            "split": spec.split,
            "shuffle": spec.shuffle,
            "shuffle_buffer_size": spec.shuffle_buffer_size,
            "seed": spec.seed,
            "sample_rate": spec.sample_rate,
            "min_audio_length": spec.min_audio_length,
            "multiple_of": spec.multiple_of,
            "cache_dir": spec.cache_dir,
            "download_max_retries": spec.download_max_retries,
            "download_timeout": spec.download_timeout,
        },
    }
    return DataLoader(**dl_kwargs), meta


def main() -> None:
    args = _parse_args()
    cfg_path = os.path.abspath(args.config_yaml)
    y = _read_yaml(cfg_path)

    if args.warmup_batches < 0:
        raise SystemExit("--warmup-batches must be >= 0")
    if args.max_batches < 0:
        raise SystemExit("--max-batches must be >= 0")
    if args.max_samples < 0:
        raise SystemExit("--max-samples must be >= 0")
    if args.max_seconds < 0:
        raise SystemExit("--max-seconds must be >= 0")
    if args.report_every < 0:
        raise SystemExit("--report-every must be >= 0")

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    dl, meta = _build_dataloader_from_yaml(y, phase=args.phase, override_num_workers=int(args.override_num_workers))

    print("\n=== DTMAE HF streaming benchmark (MLS-Eng) ===")
    print(f"config_yaml         : {cfg_path}")
    print(f"phase               : {meta['phase']}")
    print(f"batch_size          : {meta['batch_size']}")
    print(f"num_workers         : {meta['num_workers']}")
    print(f"pin_memory          : {meta['pin_memory']}")
    print(f"persistent_workers  : {meta['persistent_workers']}")
    print(f"timeout             : {meta['timeout']}")
    print(f"prefetch_factor     : {meta['prefetch_factor']}")
    print(f"multiprocessing_ctx : {meta['multiprocessing_context']}")
    print(f"include_transcript  : {meta['include_transcript']}")
    print(f"hf_repo_id          : {meta['spec']['hf_repo_id']}")
    print(f"split               : {meta['spec']['split']}")
    print(f"shuffle             : {meta['spec']['shuffle']}")
    print(f"shuffle_buffer_size : {meta['spec']['shuffle_buffer_size']}")
    print(f"cache_dir           : {meta['spec']['cache_dir']}")
    print(f"sample_rate         : {meta['spec']['sample_rate']}")
    print(f"min_audio_length    : {meta['spec']['min_audio_length']}")
    print(f"multiple_of         : {meta['spec']['multiple_of']}")

    it = iter(dl)

    # Warmup (not measured)
    if args.warmup_batches:
        t_warm0 = time.perf_counter()
        for i in range(int(args.warmup_batches)):
            batch = next(it)
            _ = batch["wav"].shape
        t_warm1 = time.perf_counter()
        print(f"\nwarmup_batches      : {args.warmup_batches} ({(t_warm1 - t_warm0):.3f}s)")

    # Measured phase
    measured_batches = 0
    measured_samples = 0
    t0 = time.perf_counter()

    first_batch_latency: Optional[float] = None

    while True:
        if args.max_batches and measured_batches >= int(args.max_batches):
            break
        if args.max_samples and measured_samples >= int(args.max_samples):
            break
        if args.max_seconds and (time.perf_counter() - t0) >= float(args.max_seconds):
            break

        try:
            batch = next(it)
        except StopIteration:
            break

        if first_batch_latency is None:
            first_batch_latency = time.perf_counter() - t0

        bsz = int(batch["wav"].shape[0])
        measured_batches += 1
        measured_samples += bsz

        if args.report_every and (measured_batches % int(args.report_every) == 0):
            dt = time.perf_counter() - t0
            sps = measured_samples / dt if dt > 0 else float("inf")
            bps = measured_batches / dt if dt > 0 else float("inf")
            print(
                f"[progress] batches={measured_batches} samples={measured_samples} "
                f"elapsed={dt:.3f}s batches/s={bps:.2f} samples/s={sps:.2f}"
            )

    elapsed = time.perf_counter() - t0
    samples_per_sec = measured_samples / elapsed if elapsed > 0 else float("inf")
    batches_per_sec = measured_batches / elapsed if elapsed > 0 else float("inf")

    print("\n--- result ---")
    print(f"measured_batches    : {measured_batches}")
    print(f"measured_samples    : {measured_samples}")
    print(f"elapsed_sec         : {elapsed:.6f}")
    print(f"first_batch_latency : {first_batch_latency if first_batch_latency is not None else 'n/a'}")
    print(f"batches/sec         : {batches_per_sec:.3f}")
    print(f"samples/sec         : {samples_per_sec:.3f}")


if __name__ == "__main__":
    main()


