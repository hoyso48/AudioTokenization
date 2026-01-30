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
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info


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
    shuffle_buffer_size: int = 20_000
    seed: int = 1024
    sample_rate: int = 16_000
    min_audio_length: int = 64_000
    multiple_of: int = 320
    lowercase_transcript: bool = True
    include_transcript: bool = False
    cache_dir: Optional[str] = None
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

        ds = datasets.load_dataset(self._script_path, **kwargs)
        if self.spec.shuffle:
            ds = ds.shuffle(buffer_size=int(self.spec.shuffle_buffer_size), seed=int(self.spec.seed))
        return ds

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        ds = self._load_hf_iterable()

        worker = get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        num_workers = 1 if worker is None else int(worker.num_workers)

        rank, world = _dist_rank_world()
        ds = _shard_iterable_dataset(ds, rank, world, worker_id, num_workers)

        for ex in ds:
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

    def _load_hf_iterable(self) -> Any:
        import datasets

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
        if self.spec.shuffle:
            ds = ds.shuffle(buffer_size=int(self.spec.shuffle_buffer_size), seed=int(self.spec.seed))
        return ds

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        ds = self._load_hf_iterable()

        worker = get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        num_workers = 1 if worker is None else int(worker.num_workers)

        rank, world = _dist_rank_world()
        ds = _shard_iterable_dataset(ds, rank, world, worker_id, num_workers)

        for ex in ds:
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


