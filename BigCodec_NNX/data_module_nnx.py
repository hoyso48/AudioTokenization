import os
from os.path import join
from typing import Iterable, List, Optional

import numpy as np
import librosa
import soundfile as sf
from grain.sources import RandomAccessDataSource
import grain
from multiprocessing import cpu_count
from hydra import utils as hydra_utils
import jax
import jax.numpy as jnp


def _read_filelist(file_path: str) -> List[str]:
    with open(file_path, 'r') as f:
        # Match CP: take first column before TAB if any
        return [line.strip().split('\t')[0] for line in f if line.strip()]


def _load_audio(file_audio: str, target_sample_rate: int, resample_res_type: Optional[str] = None) -> np.ndarray:
    info = sf.info(file_audio)
    sample_rate = info.samplerate
    with sf.SoundFile(file_audio, 'r') as f:
        waveform = f.read(dtype='float32')
        if waveform.ndim == 2:
            # take first channel to match CP single-channel
            waveform = waveform[:, 0]
        waveform = waveform.astype(np.float32)
    if target_sample_rate and target_sample_rate != sample_rate:
        try:
            if resample_res_type is not None:
                waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sample_rate, res_type=resample_res_type)
            else:
                waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sample_rate)
        except TypeError:
            # Older librosa without res_type kwarg
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sample_rate)
    return waveform


def _crop_or_pad(
    wav: np.ndarray,
    min_audio_length: int,
    multiple_of: int,
    phase: str,
) -> np.ndarray:
    length = wav.shape[0]
    if min_audio_length != -1:
        l = min_audio_length
        if length < l:
            pad = l - length
            wav = np.pad(wav, (0, pad), mode='constant')
            length = wav.shape[0]
        if phase == 'train':
            start = np.random.randint(0, max(1, length - l + 1))
        else:
            start = 0
        l = (l // multiple_of) * multiple_of
        wav = wav[start:start + l]
    else:
        l = (length // multiple_of) * multiple_of
        wav = wav[:l]
    return wav.astype(np.float32)


class FilelistAudioDataset(RandomAccessDataSource):
    """Grain RandomAccessDataSource that mirrors CP FSDataset semantics.

    Returns 1D float32 waveform per item (shape [T]).
    """

    def __init__(self, cfg, phase: str):
        self.cfg = cfg
        self.phase = phase
        self.phase_cfg = cfg.dataset.get(phase)
        self.sample_rate = cfg.dataset.sample_rate
        self.multiple_of = cfg.dataset.multiple_of
        self.min_audio_length = self.phase_cfg.min_audio_length

        ocwd = hydra_utils.get_original_cwd()
        filelist_path = join(ocwd, self.phase_cfg.filelist)
        self.filelist = _read_filelist(filelist_path)
        # self.root = cfg.preprocess.datasets.LibriSpeech.root

    def __len__(self) -> int:
        return len(self.filelist)

    def __getitem__(self, idx: int) -> np.ndarray:
        # rel_path = self.filelist[idx]
        # full_path = join(self.root, rel_path)
        full_path = self.filelist[idx]
        # Use high-quality resampler by default, can override via env
        resample_res_type = os.environ.get('LIBROSA_RESAMPLE_RES_TYPE', None)
        wav = _load_audio(full_path, self.sample_rate, resample_res_type=resample_res_type)
        wav = _crop_or_pad(wav, self.min_audio_length, self.multiple_of, self.phase)
        return wav


class DataModuleNNX:
    def __init__(self, cfg):
        self.cfg = cfg

    def _build_loader(self, phase: str) -> Iterable[dict]:
        ds = FilelistAudioDataset(self.cfg, phase)
        phase_cfg = self.cfg.dataset.get(phase)
        batch_size = phase_cfg.batch_size
        shuffle = phase_cfg.shuffle

        sampler = grain.samplers.IndexSampler(
            num_records=len(ds),
            num_epochs=1_000_000,  # effectively infinite for train; higher level controls steps
            shard_options=grain.sharding.NoSharding(),
            shuffle=shuffle,
            seed=42,
        )
        ops = [grain.transforms.Batch(batch_size, drop_remainder=True)]
        loader = grain.DataLoader(
            data_source=ds,
            operations=ops,
            sampler=sampler,
            # Tune worker processes for IO-bound audio: 4-8 is often enough
            worker_count=int(os.environ.get('GRAIN_WORKERS', max(1, min(cpu_count(), 8)))),
            # Increase read threads and prefetch buffer for better overlap
            read_options=grain.ReadOptions(
                num_threads=int(os.environ.get('GRAIN_READ_THREADS', 8)),
                prefetch_buffer_size=int(os.environ.get('GRAIN_PREFETCH', 400)),
            ),
        )

        def _iter():
            # Optional: small host-side prefetch to decouple iterator speed
            for batch in loader:
                # batch: np.ndarray [B, T]
                yield {"wav": jnp.asarray(batch)}

        return _iter()

    def train_dataloader(self) -> Iterable[dict]:
        return self._build_loader('train')

    def val_dataloader(self) -> Iterable[dict]:
        return self._build_loader('val')

    def test_dataloader(self) -> Iterable[dict]:
        return self._build_loader('test')

    def steps_per_epoch(self, phase: str) -> int:
        """Number of steps per epoch for a given phase, mirroring CP logic.

        Uses drop_remainder batching semantics to match Grain Batch transform.
        """
        ds = FilelistAudioDataset(self.cfg, phase)
        batch_size = self.cfg.dataset.get(phase).batch_size
        if batch_size <= 0:
            return 0
        return len(ds) // batch_size


