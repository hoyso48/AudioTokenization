import os
import json
from typing import Dict, Any, Tuple, List

import numpy as np

import torch

import jax
import jax.numpy as jnp
from flax import nnx

from AudioTokenization.BigCodec_NNX.common.spectral import (
    stft as JX_STFT,
    spectrogram as JX_SPECTROGRAM,
    _get_window as JX_GET_WINDOW,
    MelSpectrogram as JX_MelSpectrogram,
)
from AudioTokenization.CP.common.audio import stft as PT_STFT


def _abs_diff_stats(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    d = np.abs(a - b)
    return {
        'min_abs_diff': float(d.min()),
        'mean_abs_diff': float(d.mean()),
        'max_abs_diff': float(d.max()),
    }


def compare_stft(batch: int = 2, length: int = 4096, n_fft: int = 1024, hop: int = 256) -> Dict[str, Any]:
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((batch, length)).astype(np.float32)

    # Torch reference magnitude STFT via CP/common/audio.py
    x_t = torch.from_numpy(x_np)
    win = torch.hann_window(n_fft)
    mag_t = PT_STFT(x_t, n_fft, hop, n_fft, win, use_complex=False)  # (B, time, freq)
    mag_t = mag_t.permute(0, 2, 1).contiguous().cpu().numpy()  # (B, freq, time)

    # JAX complex STFT -> magnitude (replicate CP clamp behavior)
    x_j = jnp.asarray(x_np)
    win_j = JX_GET_WINDOW(n_fft, 'hann')
    stft_j = JX_STFT(
        waveform=x_j,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=win_j,
        center=True,
        pad_mode='reflect',
        normalized=False,
    )  # (B, freq, time) complex
    stft_c = np.asarray(stft_j)
    power_j = (stft_c.real ** 2 + stft_c.imag ** 2)
    power_j = np.clip(power_j, 1e-7, 1e3)
    mag_j = np.sqrt(power_j)

    return _abs_diff_stats(mag_t, mag_j)


def compare_spectrogram(batch: int = 2, length: int = 4096, n_fft: int = 1024, hop: int = 256) -> Dict[str, Any]:
    rng = np.random.default_rng(1)
    x_np = rng.standard_normal((batch, length)).astype(np.float32)

    # Torch manual spectrogram (power=1.0)
    x_t = torch.from_numpy(x_np)
    win = torch.hann_window(n_fft)
    x_t_pad = torch.nn.functional.pad(x_t, (n_fft // 2, n_fft // 2), mode='reflect')
    num_frames = (x_t_pad.shape[1] - n_fft) // hop + 1
    frames = torch.stack([x_t_pad[:, i * hop:i * hop + n_fft] for i in range(num_frames)], dim=1)
    frames = frames * win.to(frames)
    stft_t = torch.fft.rfft(frames, n=n_fft, dim=-1)
    spec_t = torch.abs(stft_t).movedim(-1, -2).contiguous().cpu().numpy()  # (B, freq, time)

    # JAX spectrogram (power=1.0)
    x_j = jnp.asarray(x_np)
    win_j = JX_GET_WINDOW(n_fft, 'hann')
    spec_j = JX_SPECTROGRAM(
        waveform=x_j,
        pad=0,
        window=win_j,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        power=1.0,
        normalized=False,
        center=True,
        pad_mode='reflect',
        onesided=True,
        return_complex=None,
    )
    spec_j = np.asarray(spec_j)

    return _abs_diff_stats(spec_t, spec_j)


def compare_melspec(
    sample_rate: int = 16000,
    n_fft_list: List[int] = (256, 512, 1024),
    n_mels_list: List[int] = (20, 40, 80),
    batch: int = 2,
    length: int = 8192,
) -> Dict[str, Any]:
    import torchaudio

    rng = np.random.default_rng(2)
    x_np = rng.standard_normal((batch, length)).astype(np.float32)
    x_t = torch.from_numpy(x_np)
    x_j = jnp.asarray(x_np)

    stats: Dict[str, Any] = {}
    for n_fft, n_mels in zip(n_fft_list, n_mels_list):
        hop = n_fft // 4
        # Torch: torchaudio MelSpectrogram (slaney)
        mel_t = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            n_mels=n_mels,
            power=1.0,
            center=True,
            norm='slaney',
            mel_scale='slaney',
        )(x_t)  # (B, n_mels, time)
        mel_t = mel_t.cpu().numpy()

        # JAX: MelSpectrogram module (slaney)
        mel_j_module = JX_MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            n_mels=n_mels,
            power=1.0,
            norm='slaney',
            mel_scale='slaney',
            rngs=nnx.Rngs(0),
        )
        mel_j = np.asarray(mel_j_module(x_j))  # (B, n_mels, time)

        stats[f'melspec_nfft{n_fft}_nmels{n_mels}'] = _abs_diff_stats(mel_t, mel_j)

    return stats


def main():
    # Force CPU for deterministic behavior if desired
    if 'JAX_PLATFORMS' not in os.environ:
        os.environ['JAX_PLATFORMS'] = 'cpu'

    results: Dict[str, Any] = {}
    results['stft_mag'] = compare_stft()
    results['spectrogram_pow1'] = compare_spectrogram()
    results['melspectrogram'] = compare_melspec()

    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()


