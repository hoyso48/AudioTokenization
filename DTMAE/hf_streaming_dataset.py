"""
Hugging Face streaming datasets for DTMAE.

Primary target:
- Libri-Light (large) via streaming TAR iteration (no full download/extract).

Notes:
- We rely on a local streaming-friendly dataset script:
    AudioTokenization/utils/hf_datasets/libri_light_streaming.py
  because the upstream Hugging Face dataset script is not streaming-compatible
  (it uses `download_and_extract` + glob, which fails in streaming mode).
"""

from __future__ import annotations

import io
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LibriLightStreamingSpec:
    """Config container for Libri-Light streaming training."""

    config_name: str = "large"  # small|medium|large
    split: str = "train"
    hf_repo_id: str = "HugoLaurencon/libri_light"
    shuffle: bool = True
    shuffle_buffer_size: int = 10_000
    seed: int = 1024
    sample_rate: int = 16_000
    min_audio_length: int = 64_000
    multiple_of: int = 320
    trust_remote_code: bool = True
    cache_dir: Optional[str] = None
    # Network robustness (applies to remote TAR streaming)
    download_max_retries: int = 50
    download_timeout: int = 60


@dataclass(frozen=True)
class MlsEngStreamingSpec:
    """
    Streaming spec for `parler-tts/mls_eng` (parquet-based, native HF streaming).

    Ref: https://huggingface.co/datasets/parler-tts/mls_eng
    """

    hf_repo_id: str = "parler-tts/mls_eng"
    split: str = "train"
    shuffle: bool = True
    # NOTE: In HF streaming mode, shuffle is implemented via a reservoir buffer that must warm up.
    # Large buffers can delay the first yielded example significantly on slow networks.
    shuffle_buffer_size: int = 2_000
    seed: int = 1024
    sample_rate: int = 16_000
    min_audio_length: int = 64_000
    multiple_of: int = 320
    lowercase_transcript: bool = True
    include_transcript: bool = False
    # Optional: set HF cache locations explicitly from config.
    # If provided, these will be exported to environment variables for this process.
    hf_home: Optional[str] = None
    hf_hub_cache: Optional[str] = None
    hf_datasets_cache: Optional[str] = None

    # If True, load parquet shards directly from the local HF hub snapshot directory:
    #   $HF_HOME/hub/datasets--<org>--<name>/snapshots/<revision>/data/*.parquet
    # This avoids any Hub access (useful for offline + avoids surprising re-downloads).
    use_local_snapshot: bool = False
    revision: Optional[str] = None  # commit hash; if None, resolve from refs/main

    # Passed through to datasets.load_dataset when using the Hub path.
    # NOTE: for parquet local snapshot mode, this is ignored.
    cache_dir: Optional[str] = None

    # Audio feature decoding:
    # - True: `datasets` decodes audio (opus->float array) and yields {"array", "sampling_rate", ...}
    # - False: yields {"bytes", "path"} (faster, but DTMAE expects decoded waveforms).
    audio_decode: bool = True

    # Performance: optionally preload the HF streaming iterable in the parent process.
    #
    # Why this matters:
    # - With DataLoader `multiprocessing_context=fork`, workers inherit the preloaded iterable,
    #   avoiding expensive per-worker initialization (e.g. resolving 1416 parquet shards).
    # - With `spawn`, the iterable is NOT picklable; we explicitly drop it during pickling so spawn
    #   still works, but without the preload speedup.
    preload_in_parent: bool = False
    download_max_retries: int = 50
    download_timeout: int = 60


def _local_librilight_streaming_script_path() -> str:
    # DTMAE/ is a sibling of utils/
    dtmae_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(dtmae_dir)
    return os.path.join(repo_root, "utils", "hf_datasets", "libri_light_streaming.py")


def _dist_rank_world() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _decode_flac_bytes_to_mono_float_tensor(audio_bytes: bytes) -> tuple[torch.Tensor, int]:
    """
    Decode FLAC bytes to mono float32 Tensor [T], and return (wav, sr).
    """
    import soundfile as sf

    with io.BytesIO(audio_bytes) as bio:
        wav_np, sr = sf.read(bio, dtype="float32", always_2d=False)
    wav = torch.from_numpy(wav_np)
    if wav.ndim == 2:
        # (T, C) or (C, T) depending on backend; normalize to mono.
        if wav.shape[0] < wav.shape[1]:
            wav = wav.transpose(0, 1)
        wav = wav.mean(dim=1)
    return wav.contiguous(), int(sr)


def _resample_if_needed(wav: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    if sr == target_sr:
        return wav
    import torchaudio

    return torchaudio.functional.resample(wav, sr, target_sr)


def _crop_or_pad_1d(
    wav: torch.Tensor,
    *,
    phase: str,
    min_audio_length: int,
    multiple_of: int,
) -> torch.Tensor:
    """
    Match the existing DTMAE logic:
    - if min_audio_length != -1: pad up to that length, then random crop (train) or head crop (val/test)
    - else: truncate to multiple_of.
    """
    length = int(wav.shape[0])
    if min_audio_length != -1:
        target_len = int(min_audio_length)
        if length < target_len:
            wav = F.pad(wav, (0, target_len - length))
            length = int(wav.shape[0])

        if phase == "train":
            start = int(torch.randint(low=0, high=length - target_len + 1, size=(1,)).item())
        else:
            start = 0

        target_len = (target_len // multiple_of) * multiple_of
        return wav[start : start + target_len]

    trimmed = (length // multiple_of) * multiple_of
    return wav[:trimmed]


def _audio_to_tensor_and_sr(audio: Dict[str, Any]) -> Tuple[torch.Tensor, int]:
    """
    Convert HF audio dict to (wav[T], sr).

    Handles either:
    - {"array": np.ndarray, "sampling_rate": int, ...}  (already decoded)
    - {"bytes": b"...", ...}                            (decode ourselves)
    """
    if "array" in audio and audio["array"] is not None:
        wav = torch.as_tensor(audio["array"]).to(torch.float32)
        sr = int(audio.get("sampling_rate", 16000))
        if wav.ndim == 2:
            # (C, T) or (T, C) -> mono [T]
            if wav.shape[0] < wav.shape[1]:
                # likely (C, T)
                wav = wav.mean(dim=0)
            else:
                wav = wav.mean(dim=1)
        return wav.contiguous(), sr

    if "bytes" in audio and audio["bytes"] is not None:
        return _decode_flac_bytes_to_mono_float_tensor(audio["bytes"])

    raise ValueError("Unsupported audio dict format (expected 'array' or 'bytes').")


def _maybe_apply_download_config(datasets_mod: Any, kwargs: Dict[str, Any], max_retries: int, timeout: int) -> None:
    if not hasattr(datasets_mod, "DownloadConfig"):
        return
    try:
        import inspect

        dc_sig = inspect.signature(datasets_mod.DownloadConfig)
        dc_kwargs: Dict[str, Any] = {}
        if "max_retries" in dc_sig.parameters:
            dc_kwargs["max_retries"] = int(max_retries)
        if "timeout" in dc_sig.parameters:
            dc_kwargs["timeout"] = int(timeout)
        if dc_kwargs:
            kwargs["download_config"] = datasets_mod.DownloadConfig(**dc_kwargs)
    except (TypeError, ValueError):
        return


def _maybe_set_env_if_provided(key: str, value: Optional[str]) -> None:
    if value is None:
        return
    value = str(value)
    if not value:
        return
    os.environ[key] = value


def _repo_id_to_hf_hub_dirname(repo_id: str, *, is_dataset: bool = True) -> str:
    """
    Hugging Face hub cache directory name convention.

    For datasets, the hub cache uses:
      datasets--org--name
    e.g. parler-tts/mls_eng -> datasets--parler-tts--mls_eng
    """
    prefix = "datasets--" if is_dataset else "models--"
    return prefix + repo_id.replace("/", "--")


def _resolve_hf_snapshot_dir(*, repo_id: str, hf_home: str, revision: Optional[str]) -> str:
    hub_root = os.path.join(hf_home, "hub")
    repo_dir = os.path.join(hub_root, _repo_id_to_hf_hub_dirname(repo_id, is_dataset=True))
    if revision is None:
        ref_path = os.path.join(repo_dir, "refs", "main")
        if not os.path.exists(ref_path):
            raise FileNotFoundError(
                f"Cannot resolve revision for {repo_id}: missing refs/main at {ref_path}. "
                "Set `revision` explicitly or ensure the dataset is cached."
            )
        with open(ref_path, "r", encoding="utf-8") as f:
            revision = f.read().strip()
    snap_dir = os.path.join(repo_dir, "snapshots", str(revision))
    if not os.path.isdir(snap_dir):
        raise FileNotFoundError(
            f"Local HF snapshot directory not found: {snap_dir}. "
            "Set `hf_home` correctly or disable `use_local_snapshot`."
        )
    return snap_dir


def _shard_iterable_dataset(ds: Any, rank: int, world: int, worker_id: int, num_workers: int) -> Any:
    """
    Preferred sharding for HF streaming IterableDataset.

    - split_by_node: shard by DDP rank
    - split_by_worker: shard by DataLoader worker
    Falls back to `.shard` if needed.
    """
    if hasattr(ds, "split_by_node"):
        ds = ds.split_by_node(rank=rank, world_size=world)
    if hasattr(ds, "split_by_worker"):
        ds = ds.split_by_worker(worker_id=worker_id, num_workers=num_workers)
        return ds

    num_shards = world * num_workers
    shard_index = rank * num_workers + worker_id
    if hasattr(ds, "shard"):
        return ds.shard(num_shards=num_shards, index=shard_index)
    return (ex for i, ex in enumerate(ds) if (i % num_shards) == shard_index)


class LibriLightStreamingDataset(IterableDataset):
    """
    IterableDataset that streams Libri-Light from the public fbaipublicfiles TAR.

    Supports:
    - DataLoader workers sharding
    - DDP rank sharding
    - Optional buffer shuffle (streaming-friendly)
    """

    def __init__(self, phase: str, spec: LibriLightStreamingSpec):
        super().__init__()
        self.phase = str(phase)
        self.spec = spec

        script_path = _local_librilight_streaming_script_path()
        if not os.path.exists(script_path):
            raise FileNotFoundError(
                f"Missing local Libri-Light streaming dataset script: {script_path}"
            )
        self._script_path = script_path

    def _load_hf_iterable(self) -> Any:
        import datasets

        kwargs: Dict[str, Any] = dict(
            name=self.spec.config_name,
            split=self.spec.split,
            streaming=True,
        )
        if self.spec.cache_dir is not None:
            kwargs["cache_dir"] = self.spec.cache_dir

        # Best-effort: tune retry/timeout for flaky connections
        _maybe_apply_download_config(
            datasets, kwargs, self.spec.download_max_retries, self.spec.download_timeout
        )

        # This is a local script; some datasets versions still gate script execution behind trust_remote_code.
        if hasattr(datasets, "load_dataset") and hasattr(datasets.load_dataset, "__code__"):
            if "trust_remote_code" in datasets.load_dataset.__code__.co_varnames:
                kwargs["trust_remote_code"] = bool(self.spec.trust_remote_code)

        t0 = time.monotonic()
        ds = datasets.load_dataset(self._script_path, **kwargs)
        t1 = time.monotonic()
        logger.info(
            "Loaded LibriLight streaming dataset in %.2fs (config=%s split=%s cache_dir=%s).",
            t1 - t0,
            self.spec.config_name,
            self.spec.split,
            self.spec.cache_dir,
        )
        if self.spec.shuffle:
            logger.info(
                "Enabling streaming shuffle for LibriLight (buffer_size=%d). Large buffers can delay the first batch.",
                int(self.spec.shuffle_buffer_size),
            )
            ds = ds.shuffle(buffer_size=int(self.spec.shuffle_buffer_size), seed=int(self.spec.seed))
        return ds

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        t_iter0 = time.monotonic()
        ds = self._load_hf_iterable()

        worker = get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        num_workers = 1 if worker is None else int(worker.num_workers)

        rank, world = _dist_rank_world()
        ds = _shard_iterable_dataset(ds, rank, world, worker_id, num_workers)

        for i, ex in enumerate(ds):
            if i == 0 and rank == 0 and worker_id == 0:
                logger.info(
                    "LibriLight first example fetched after %.2fs (world=%d num_workers=%d).",
                    time.monotonic() - t_iter0,
                    world,
                    num_workers,
                )
            audio = ex.get("audio")
            if not isinstance(audio, dict) or audio.get("bytes") is None:
                raise ValueError("Expected streaming audio bytes in example['audio']['bytes']")

            wav, sr = _audio_to_tensor_and_sr(audio)
            wav = _resample_if_needed(wav, sr, int(self.spec.sample_rate))
            wav = _crop_or_pad_1d(
                wav,
                phase=self.phase,
                min_audio_length=int(self.spec.min_audio_length),
                multiple_of=int(self.spec.multiple_of),
            )

            yield {
                "wav": wav,
                "id": ex.get("id", ""),
                "speaker_id": ex.get("speaker_id", -1),
            }


class MlsEngStreamingDataset(IterableDataset):
    """
    IterableDataset that streams MLS-English from Hugging Face (parquet-based).

    Ref: https://huggingface.co/datasets/parler-tts/mls_eng
    """

    def __init__(self, phase: str, spec: MlsEngStreamingSpec):
        super().__init__()
        self.phase = str(phase)
        self.spec = spec
        self._preloaded_hf_iterable: Any = None

        if bool(self.spec.preload_in_parent):
            # Safe for fork-based workers; for spawn, we strip this in __getstate__.
            t0 = time.monotonic()
            self._preloaded_hf_iterable = self._load_hf_iterable()
            t1 = time.monotonic()
            logger.info("Preloaded MLS-Eng iterable in parent process in %.2fs.", t1 - t0)

    def __getstate__(self) -> Dict[str, Any]:
        """
        Ensure this Dataset remains picklable (required by multiprocessing spawn).

        HF streaming iterables are not reliably picklable; drop them when pickling.
        """
        state = dict(self.__dict__)
        state["_preloaded_hf_iterable"] = None
        return state

    def _load_hf_iterable(self) -> Any:
        import datasets
        from datasets import Audio

        # Optional: set cache roots from config so the run is reproducible.
        _maybe_set_env_if_provided("HF_HOME", self.spec.hf_home)
        _maybe_set_env_if_provided("HF_HUB_CACHE", self.spec.hf_hub_cache)
        _maybe_set_env_if_provided("HF_DATASETS_CACHE", self.spec.hf_datasets_cache)

        t0 = time.monotonic()

        if self.spec.use_local_snapshot:
            hf_home = (
                self.spec.hf_home
                or os.environ.get("HF_HOME")
                or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
            )
            snapshot_dir = _resolve_hf_snapshot_dir(
                repo_id=str(self.spec.hf_repo_id),
                hf_home=str(hf_home),
                revision=self.spec.revision,
            )
            data_dir = os.path.join(snapshot_dir, "data")
            data_files = {
                "train": os.path.join(data_dir, "train-*.parquet"),
                "dev": os.path.join(data_dir, "dev-*.parquet"),
                "test": os.path.join(data_dir, "test-*.parquet"),
            }
            ds = datasets.load_dataset(
                "parquet",
                data_files=data_files,
                split=str(self.spec.split),
                streaming=True,
            )
            source = f"local_snapshot:{snapshot_dir}"
        else:
            kwargs: Dict[str, Any] = dict(
                split=self.spec.split,
                streaming=True,
            )
            if self.spec.cache_dir is not None:
                kwargs["cache_dir"] = self.spec.cache_dir
            _maybe_apply_download_config(
                datasets, kwargs, self.spec.download_max_retries, self.spec.download_timeout
            )
            ds = datasets.load_dataset(self.spec.hf_repo_id, **kwargs)
            source = f"hub:{self.spec.hf_repo_id}"

        # Ensure audio is in the expected format (array vs bytes) based on config.
        ds = ds.cast_column("audio", Audio(decode=bool(self.spec.audio_decode)))

        t1 = time.monotonic()
        logger.info(
            "Loaded MLS-Eng streaming dataset in %.2fs (source=%s split=%s).",
            t1 - t0,
            source,
            self.spec.split,
        )
        return ds

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        t_iter0 = time.monotonic()
        ds = self._preloaded_hf_iterable if self._preloaded_hf_iterable is not None else self._load_hf_iterable()

        worker = get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        num_workers = 1 if worker is None else int(worker.num_workers)

        rank, world = _dist_rank_world()
        ds = _shard_iterable_dataset(ds, rank, world, worker_id, num_workers)

        # IMPORTANT: shard *before* shuffle.
        # If we shuffle first and then shard, each worker/rank may need to warm up the shuffle buffer
        # and then discard most samples due to sharding, leading to very slow "first batch" latency
        # and even DataLoader timeouts in DDP runs.
        if self.spec.shuffle:
            logger.info(
                "Enabling streaming shuffle for MLS-Eng (buffer_size=%d) after sharding (world=%d num_workers=%d).",
                int(self.spec.shuffle_buffer_size),
                world,
                num_workers,
            )
            ds = ds.shuffle(buffer_size=int(self.spec.shuffle_buffer_size), seed=int(self.spec.seed))

        for i, ex in enumerate(ds):
            if i == 0 and rank == 0 and worker_id == 0:
                logger.info(
                    "MLS-Eng first example fetched after %.2fs (world=%d num_workers=%d).",
                    time.monotonic() - t_iter0,
                    world,
                    num_workers,
                )
            audio = ex.get("audio")
            if not isinstance(audio, dict):
                raise ValueError("Expected example['audio'] to be a dict.")

            wav, sr = _audio_to_tensor_and_sr(audio)
            wav = _resample_if_needed(wav, sr, int(self.spec.sample_rate))
            wav = _crop_or_pad_1d(
                wav,
                phase=self.phase,
                min_audio_length=int(self.spec.min_audio_length),
                multiple_of=int(self.spec.multiple_of),
            )

            out: Dict[str, Any] = {"wav": wav}
            if self.spec.include_transcript:
                transcript = ex.get("transcript", "")
                if transcript is None:
                    transcript = ""
                transcript = str(transcript)
                if self.spec.lowercase_transcript:
                    transcript = transcript.lower()
                out["transcript"] = transcript
                out["utt_id"] = str(ex.get("original_path", ex.get("id", "")))

            yield out


