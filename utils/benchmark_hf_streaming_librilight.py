#!/usr/bin/env python3
"""
Benchmark Hugging Face *streaming* throughput (samples/sec) for Libri-Light.

Dataset reference (dataset script):
  https://huggingface.co/datasets/HugoLaurencon/libri_light/blob/main/libri_light.py
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import queue
import sys
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class BenchmarkResult:
    dataset: str
    config: str
    split: str
    streaming: bool
    decode_audio: bool
    datasets_decode_audio: bool
    workers: int
    worker_backend: str
    queue_size: int
    warmup_samples: int
    measured_samples: int
    elapsed_sec: float
    samples_per_sec: float
    python: str
    platform: str
    datasets_version: str
    start_time_unix: float


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Benchmark Hugging Face datasets streaming throughput for "
            "HugoLaurencon/libri_light (samples/sec)."
        )
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="HugoLaurencon/libri_light",
        help="Hugging Face dataset repo id.",
    )
    p.add_argument(
        "--config",
        type=str,
        default="small",
        help="Libri-Light config name (e.g., small/medium/large).",
    )
    p.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split name to iterate (typically 'train').",
    )
    p.add_argument(
        "--warmup-samples",
        type=int,
        default=10,
        help="Number of initial samples to iterate but not measure (warms cache/HTTP).",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Number of samples to measure.",
    )
    p.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="Optional wall-clock limit for the measured phase (0 disables).",
    )
    p.add_argument(
        "--report-every",
        type=int,
        default=25,
        help="Print a progress report every N measured samples (0 disables).",
    )
    p.add_argument(
        "--decode-audio",
        action="store_true",
        help=(
            "Force audio decoding by materializing the audio array. "
            "This measures end-to-end cost (download + decode), not just metadata."
        ),
    )
    p.add_argument(
        "--datasets-decode-audio",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If set, let `datasets` decode the Audio column automatically. "
            "Default is False to avoid decoding work inside the dataset iterator."
        ),
    )
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Number of worker threads/processes for parallel consume/decode. "
            "0 disables parallelism (serial)."
        ),
    )
    p.add_argument(
        "--worker-backend",
        type=str,
        choices=["thread", "process"],
        default="thread",
        help="Parallel backend when --workers>0.",
    )
    p.add_argument(
        "--queue-size",
        type=int,
        default=64,
        help="Prefetch queue size for the producer/consumer pipeline.",
    )
    p.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow execution of dataset scripts. Required for this dataset and the "
            "local streaming fallback script."
        ),
    )
    p.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional path to write the benchmark result JSON.",
    )
    return p.parse_args()


def _force_audio_decode(example: Dict[str, Any]) -> None:
    """
    Ensure we pay the cost of audio decoding (if available).

    For datasets.Audio, example['audio'] can be:
      - a dict with keys like {'array', 'sampling_rate', 'path'}
      - or a path-like placeholder depending on datasets version/formatting
    """
    audio = example.get("audio", None)
    if audio is None:
        return

    if isinstance(audio, dict):
        arr = audio.get("array", None)
        if arr is None:
            audio_bytes = audio.get("bytes", None)
            if audio_bytes is None:
                return
            # Decode bytes ourselves (useful when datasets Audio decode is disabled).
            import io

            import soundfile as sf  # dependency already in repo

            with io.BytesIO(audio_bytes) as bio:
                data, _sr = sf.read(bio, dtype="float32", always_2d=False)
            _ = getattr(data, "shape", None)
            return
        _ = getattr(arr, "shape", None)  # touch
        return

    # If audio is not a dict, we still "touch" it to avoid being optimized away.
    _ = str(audio)


def _get_datasets() -> Any:
    try:
        import datasets  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "Missing dependency: datasets\n"
            "Install via:\n"
            "  pip install datasets\n"
            "Or:\n"
            "  pip install -r /home/hoyso/projects/AudioTokenization/requirements.txt"
        ) from e
    return datasets


def _local_streaming_script_path() -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "hf_datasets",
        "libri_light_streaming.py",
    )


def _supports_trust_remote_code(datasets_mod: Any) -> bool:
    try:
        sig = inspect.signature(datasets_mod.load_dataset)
    except (TypeError, ValueError):
        return False
    return "trust_remote_code" in sig.parameters


def _load_dataset_streaming(
    datasets_mod: Any,
    dataset_or_path: str,
    config: str,
    split: str,
    trust_remote_code: bool,
) -> Any:
    kwargs: Dict[str, Any] = {
        "name": config,
        "split": split,
        "streaming": True,
    }
    if _supports_trust_remote_code(datasets_mod):
        kwargs["trust_remote_code"] = bool(trust_remote_code)
    return datasets_mod.load_dataset(dataset_or_path, **kwargs)


def _load_streaming_dataset(
    datasets_mod: Any,
    dataset_id: str,
    config: str,
    split: str,
    trust_remote_code: bool,
) -> Any:
    """
    Load a streaming dataset.

    If the upstream Libri-Light script fails in streaming mode due to TAR extraction,
    we fall back to a local streaming-friendly script (iter_archive-based).
    """
    try:
        return _load_dataset_streaming(
            datasets_mod,
            dataset_or_path=dataset_id,
            config=config,
            split=split,
            trust_remote_code=trust_remote_code,
        )
    except ValueError as e:
        msg = str(e)
        if "trust_remote_code=True" in msg:
            if not _supports_trust_remote_code(datasets_mod):
                raise SystemExit(
                    "Your installed `datasets` does not support `trust_remote_code`, "
                    "so script-based datasets cannot be loaded.\n"
                    "Fix:\n"
                    "  conda activate hoyso_ml\n"
                    "  python -m pip install --upgrade \"datasets<4\"\n"
                ) from e
            if not trust_remote_code:
                raise SystemExit(
                    "This dataset requires executing a dataset script.\n"
                    "Rerun with:\n"
                    "  --trust-remote-code\n"
                ) from e
            # Retry explicitly (some versions only accept True, not default False).
            return _load_dataset_streaming(
                datasets_mod,
                dataset_or_path=dataset_id,
                config=config,
                split=split,
                trust_remote_code=True,
            )
        raise
    except NotImplementedError as e:
        msg = str(e)
        if "Extraction protocol for TAR archives" in msg:
            local_script = _local_streaming_script_path()
            if os.path.exists(local_script):
                print(
                    "[info] Upstream Libri-Light script isn't streaming-friendly "
                    "(TAR extraction not supported in streaming)."
                )
                print(f"[info] Falling back to local script: {local_script}")
                return _load_dataset_streaming(
                    datasets_mod,
                    dataset_or_path=local_script,
                    config=config,
                    split=split,
                    trust_remote_code=trust_remote_code,
                )
        raise


def _try_disable_datasets_audio_decode(ds: Any, datasets_mod: Any) -> Any:
    """
    Best-effort: cast the `audio` column to Audio(decode=False) so decoding doesn't
    happen in the producer thread/process.
    """
    if not hasattr(ds, "cast_column"):
        return ds
    if not hasattr(datasets_mod, "Audio"):
        return ds

    try:
        audio_sig = inspect.signature(datasets_mod.Audio)
    except (TypeError, ValueError):
        return ds
    if "decode" not in audio_sig.parameters:
        return ds

    try:
        return ds.cast_column("audio", datasets_mod.Audio(sampling_rate=16_000, decode=False))
    except (KeyError, TypeError, ValueError) as e:
        print(f"[warn] Could not disable datasets audio decode: {e}", file=sys.stderr)
        return ds


def _bench_serial(
    it: Any,
    decode_audio: bool,
    max_samples: int,
    max_seconds: float,
    report_every: int,
) -> Tuple[int, float, float]:
    t0 = time.perf_counter()
    measured = 0
    while measured < max_samples:
        if max_seconds > 0 and (time.perf_counter() - t0) >= max_seconds:
            break

        try:
            ex = next(it)
        except StopIteration:
            break

        if decode_audio:
            _force_audio_decode(ex)

        measured += 1

        if report_every and (measured % report_every == 0):
            dt = time.perf_counter() - t0
            sps = measured / dt if dt > 0 else float("inf")
            print(f"[progress] measured={measured} elapsed={dt:.3f}s sps={sps:.2f}")

    elapsed = time.perf_counter() - t0
    sps = (measured / elapsed) if elapsed > 0 else float("inf")
    return measured, elapsed, sps


def _bench_threaded(
    it: Any,
    decode_audio: bool,
    max_samples: int,
    max_seconds: float,
    report_every: int,
    workers: int,
    queue_size: int,
) -> Tuple[int, float, float]:
    if workers <= 0:
        raise ValueError("workers must be > 0 for threaded benchmark")
    if queue_size <= 0:
        raise ValueError("queue_size must be > 0")

    q: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(maxsize=queue_size)
    processed = {"n": 0}
    processed_lock = threading.Lock()
    stop_sentinel: Optional[Dict[str, Any]] = None

    def worker_loop() -> None:
        while True:
            item = q.get()
            if item is stop_sentinel:
                return
            if decode_audio:
                _force_audio_decode(item)
            with processed_lock:
                processed["n"] += 1

    threads = [threading.Thread(target=worker_loop, daemon=True) for _ in range(workers)]
    for t in threads:
        t.start()

    t0 = time.perf_counter()
    produced = 0
    while produced < max_samples:
        if max_seconds > 0 and (time.perf_counter() - t0) >= max_seconds:
            break
        try:
            ex = next(it)
        except StopIteration:
            break
        q.put(ex)
        produced += 1

        if report_every:
            with processed_lock:
                cur = processed["n"]
            if cur and (cur % report_every == 0):
                dt = time.perf_counter() - t0
                sps = cur / dt if dt > 0 else float("inf")
                print(f"[progress] processed={cur} elapsed={dt:.3f}s sps={sps:.2f}")

    # signal stop and wait for workers to drain
    for _ in range(workers):
        q.put(stop_sentinel)
    for t in threads:
        t.join()

    elapsed = time.perf_counter() - t0
    with processed_lock:
        measured = processed["n"]
    sps = (measured / elapsed) if elapsed > 0 else float("inf")
    return measured, elapsed, sps


def _bench_process_worker_loop(q_: Any, processed_: Any, decode_: bool) -> None:
    while True:
        item = q_.get()
        if item is None:
            return
        if decode_:
            _force_audio_decode(item)
        with processed_.get_lock():
            processed_.value += 1


def _bench_process(
    it: Any,
    decode_audio: bool,
    max_samples: int,
    max_seconds: float,
    report_every: int,
    workers: int,
    queue_size: int,
) -> Tuple[int, float, float]:
    if workers <= 0:
        raise ValueError("workers must be > 0 for process benchmark")
    if queue_size <= 0:
        raise ValueError("queue_size must be > 0")

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    q: "mp.Queue[Optional[Dict[str, Any]]]" = ctx.Queue(maxsize=queue_size)
    processed = ctx.Value("i", 0)
    stop_sentinel: Optional[Dict[str, Any]] = None

    procs = [
        ctx.Process(
            target=_bench_process_worker_loop, args=(q, processed, decode_audio), daemon=True
        )
        for _ in range(workers)
    ]
    for p in procs:
        p.start()

    t0 = time.perf_counter()
    produced = 0
    while produced < max_samples:
        if max_seconds > 0 and (time.perf_counter() - t0) >= max_seconds:
            break
        try:
            ex = next(it)
        except StopIteration:
            break
        q.put(ex)
        produced += 1

        if report_every:
            with processed.get_lock():
                cur = processed.value
            if cur and (cur % report_every == 0):
                dt = time.perf_counter() - t0
                sps = cur / dt if dt > 0 else float("inf")
                print(f"[progress] processed={cur} elapsed={dt:.3f}s sps={sps:.2f}")

    for _ in range(workers):
        q.put(stop_sentinel)
    for p in procs:
        p.join()

    elapsed = time.perf_counter() - t0
    with processed.get_lock():
        measured = int(processed.value)
    sps = (measured / elapsed) if elapsed > 0 else float("inf")
    return measured, elapsed, sps


def main() -> None:
    args = _parse_args()
    if args.warmup_samples < 0 or args.max_samples <= 0:
        raise SystemExit("--warmup-samples must be >= 0 and --max-samples must be > 0")
    if args.report_every < 0:
        raise SystemExit("--report-every must be >= 0")
    if args.max_seconds < 0:
        raise SystemExit("--max-seconds must be >= 0")
    if args.workers < 0:
        raise SystemExit("--workers must be >= 0")
    if args.queue_size <= 0:
        raise SystemExit("--queue-size must be > 0")

    datasets = _get_datasets()

    # Encourage deterministic-ish iteration behavior.
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    start_unix = time.time()
    try:
        ds = _load_streaming_dataset(
            datasets,
            args.dataset,
            args.config,
            args.split,
            trust_remote_code=bool(args.trust_remote_code),
        )
    except RuntimeError as e:
        msg = str(e)
        if "Dataset scripts are no longer supported" in msg:
            raise SystemExit(
                "Your installed `datasets` version disables script-based datasets.\n"
                f"  dataset: {args.dataset}\n"
                "This repo is implemented via a dataset script (`libri_light.py`), so it won't load.\n\n"
                "Fix:\n"
                "  conda activate hoyso_ml\n"
                "  python -m pip install --upgrade \"datasets<4\"\n\n"
                "Then rerun this benchmark.\n"
            ) from e
        raise

    if not args.datasets_decode_audio:
        ds = _try_disable_datasets_audio_decode(ds, datasets)

    it = iter(ds)

    # Warm-up (not measured).
    for i in range(args.warmup_samples):
        try:
            ex = next(it)
        except StopIteration:
            raise SystemExit(
                f"Dataset exhausted during warmup after {i} samples. "
                f"Try smaller warmup or check split/config."
            )
        if args.decode_audio:
            _force_audio_decode(ex)

    if args.workers == 0:
        measured, elapsed, samples_per_sec = _bench_serial(
            it=it,
            decode_audio=bool(args.decode_audio),
            max_samples=int(args.max_samples),
            max_seconds=float(args.max_seconds),
            report_every=int(args.report_every),
        )
    else:
        if args.worker_backend == "thread":
            measured, elapsed, samples_per_sec = _bench_threaded(
                it=it,
                decode_audio=bool(args.decode_audio),
                max_samples=int(args.max_samples),
                max_seconds=float(args.max_seconds),
                report_every=int(args.report_every),
                workers=int(args.workers),
                queue_size=int(args.queue_size),
            )
        else:
            # process backend: avoid passing decoded arrays through mp queue
            if args.datasets_decode_audio:
                raise SystemExit(
                    "For --worker-backend process, run with --no-datasets-decode-audio "
                    "so audio is passed as bytes instead of large numpy arrays."
                )
            measured, elapsed, samples_per_sec = _bench_process(
                it=it,
                decode_audio=bool(args.decode_audio),
                max_samples=int(args.max_samples),
                max_seconds=float(args.max_seconds),
                report_every=int(args.report_every),
                workers=int(args.workers),
                queue_size=int(args.queue_size),
            )

    result = BenchmarkResult(
        dataset=args.dataset,
        config=args.config,
        split=args.split,
        streaming=True,
        decode_audio=bool(args.decode_audio),
        datasets_decode_audio=bool(args.datasets_decode_audio),
        workers=int(args.workers),
        worker_backend=str(args.worker_backend),
        queue_size=int(args.queue_size),
        warmup_samples=int(args.warmup_samples),
        measured_samples=int(measured),
        elapsed_sec=float(elapsed),
        samples_per_sec=float(samples_per_sec),
        python=sys.version.replace("\n", " "),
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        datasets_version=str(getattr(datasets, "__version__", "unknown")),
        start_time_unix=float(start_unix),
    )

    print("\n=== Hugging Face streaming benchmark (Libri-Light) ===")
    print(f"dataset        : {result.dataset}")
    print(f"config         : {result.config}")
    print(f"split          : {result.split}")
    print(f"streaming      : {result.streaming}")
    print(f"decode_audio   : {result.decode_audio}")
    print(f"ds_decode_audio: {result.datasets_decode_audio}")
    print(f"workers        : {result.workers} ({result.worker_backend})")
    print(f"queue_size     : {result.queue_size}")
    print(f"warmup_samples : {result.warmup_samples}")
    print(f"measured       : {result.measured_samples}")
    print(f"elapsed_sec    : {result.elapsed_sec:.6f}")
    print(f"samples/sec    : {result.samples_per_sec:.3f}")
    print(f"python         : {result.python}")
    print(f"platform       : {result.platform}")
    print(f"datasets       : {result.datasets_version}")

    if args.output_json:
        out_path = os.path.abspath(args.output_json)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2)
        print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()


