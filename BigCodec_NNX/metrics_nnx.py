import numpy as np
import jax
import jax.numpy as jnp
import librosa

try:
    from pesq import pesq as PESQ_FUNC
except Exception as _e:
    PESQ_FUNC = None
try:
    from pystoi.stoi import stoi as STOI_FUNC
except Exception as _e:
    STOI_FUNC = None

try:
    import torch
    import torchaudio
    _HAS_TORCHAUDIO = True
except Exception as _e:
    _HAS_TORCHAUDIO = False


def _batch_resample_np(wavs: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a batch of waveforms (B, T) using torchaudio if available, else librosa.

    Returns array shaped (B, T').
    """
    if _HAS_TORCHAUDIO:
        arr = np.asarray(wavs)
        if arr.ndim == 2:
            arr = arr[:, None, :]
        tensor = torch.from_numpy(arr)
        res = torchaudio.functional.resample(tensor, orig_sr, target_sr)
        res = res.numpy()
        if res.ndim == 3:
            res = res[:, 0, :]
        return res
    return np.stack([
        librosa.resample(wavs[i], orig_sr=orig_sr, target_sr=target_sr)
        for i in range(wavs.shape[0])
    ], axis=0)


class MeanMetric:
    """Simple mean aggregator for scalar metric values.

    Accumulates sum and count on host; compute() returns average.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._sum = 0.0
        self._count = 0

    def update(self, values) -> None:
        arr = np.asarray(values, dtype=np.float64)
        self._sum += float(arr.sum())
        self._count += int(arr.size)

    def compute(self) -> float:
        if self._count == 0:
            return 0.0
        return float(self._sum / self._count)


class JaxScalarMeanMetric(MeanMetric):
    """Generic wrapper to aggregate a JAX scalar-producing function over batches.

    The provided function should return a scalar jnp.ndarray when called.
    """

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def update_pair(self, *args, **kwargs) -> None:
        val = self._fn(*args, **kwargs)
        self.update(float(jax.device_get(val)))


def _ensure_bt(x):
    """Ensure waveform shape is (B, T). Accepts (B, 1, T), (B, T) or (T)."""
    x = jnp.asarray(x)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim == 3:
        x = x.squeeze(1)
    return x


def scale_invariant_signal_distortion_ratio_jax(
    preds: jnp.ndarray,
    target: jnp.ndarray,
    zero_mean: bool = False,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """JAX SI-SDR per-sample values (matches Torch functional implementation).

    Args:
        preds: (..., T)
        target: (..., T)
        zero_mean: If True, subtract mean from both signals (SI-SNR case)
        eps: Numerical stability constant

    Returns:
        jnp.ndarray with shape (...,) containing SI-SDR values per sample
    """
    preds = _ensure_bt(preds).astype(jnp.float32)
    target = _ensure_bt(target).astype(jnp.float32)

    if zero_mean:
        target = target - jnp.mean(target, axis=-1, keepdims=True)
        preds = preds - jnp.mean(preds, axis=-1, keepdims=True)

    alpha = (jnp.sum(preds * target, axis=-1, keepdims=True) + eps) / (
        jnp.sum(target * target, axis=-1, keepdims=True) + eps
    )
    target_scaled = alpha * target
    noise = target_scaled - preds
    val = (jnp.sum(target_scaled * target_scaled, axis=-1) + eps) / (
        jnp.sum(noise * noise, axis=-1) + eps
    )
    return 10.0 * jnp.log10(val)


def si_snr_jax(pred: jnp.ndarray, target: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Mean SI-SNR over batch (zero_mean=True)."""
    per_sample = scale_invariant_signal_distortion_ratio_jax(pred, target, zero_mean=True, eps=eps)
    return jnp.mean(per_sample)


def si_sdr_jax(pred: jnp.ndarray, target: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Mean SI-SDR over batch (zero_mean=False)."""
    per_sample = scale_invariant_signal_distortion_ratio_jax(pred, target, zero_mean=False, eps=eps)
    return jnp.mean(per_sample)


class CodebookStats:
    """Accumulates codebook usage statistics across a full validation set.

    Mirrors Torch lightning CodebookPerplexity/Utilization behavior by
    tracking total counts across the entire epoch.
    """

    def __init__(self, codebook_size: int):
        self.codebook_size = int(codebook_size)
        self.reset()

    def reset(self) -> None:
        self.counts = np.zeros(self.codebook_size, dtype=np.int64)
        self.total = 0

    def update(self, indices) -> None:
        if indices is None:
            return
        idx = np.asarray(indices).astype(np.int64).ravel()
        if idx.size == 0:
            return
        counts = np.bincount(idx, minlength=self.codebook_size)
        if counts.shape[0] > self.codebook_size:
            counts = counts[: self.codebook_size]
        self.counts += counts
        self.total += int(idx.size)

    def perplexity(self) -> float:
        if self.total <= 0:
            return 0.0
        probs = self.counts.astype(np.float64) / float(self.total)
        nonzero = probs > 0
        entropy = -np.sum(probs[nonzero] * np.log(probs[nonzero]))
        return float(np.exp(entropy))

    def utilization(self) -> float:
        used = (self.counts > 0).sum()
        return float(used) / float(self.codebook_size)


def codebook_metrics_from_indices(indices, codebook_size: int):
    agg = CodebookStats(codebook_size)
    agg.update(indices)
    return {
        'codebook_perplexity': agg.perplexity(),
        'codebook_utilization': agg.utilization(),
    }


def pesq_stoi_cpu(y_pred, y_true, sr: int):
    """Compute PESQ and STOI on CPU for a batch.

    Args:
        y_pred: Predicted waveforms, shape (B, T) or (B, 1, T)
        y_true: Ground truth waveforms, shape (B, T) or (B, 1, T)
        sr: Sample rate of inputs. Will be resampled to 16k if needed.

    Returns:
        Dict with optional 'pesq' and 'stoi' averages across batch.
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    if y_true.ndim == 3:
        y_true = y_true[:, 0, :]
    if y_pred.ndim == 3:
        y_pred = y_pred[:, 0, :]

    if sr != 16000:
        y_pred = _batch_resample_np(y_pred, sr, 16000)
        y_true = _batch_resample_np(y_true, sr, 16000)

    pesq_values = []
    stoi_values = []
    for i in range(y_true.shape[0]):
        ref = y_true[i].astype(np.float32)
        deg = y_pred[i].astype(np.float32)
        L = min(ref.shape[0], deg.shape[0])
        ref = ref[:L]
        deg = deg[:L]

        if PESQ_FUNC is None:
            raise ImportError("pesq package not available; install 'pesq' to compute PESQ")
        pesq_score = PESQ_FUNC(16000, ref, deg, 'wb')
        pesq_values.append(float(pesq_score))

        if STOI_FUNC is None:
            raise ImportError("pystoi package not available; install 'pystoi' to compute STOI")
        stoi_score = STOI_FUNC(ref, deg, 16000, extended=False)
        stoi_values.append(float(stoi_score))


    out = {}
    if len(pesq_values) > 0:
        out['pesq'] = float(np.mean(pesq_values))
    if len(stoi_values) > 0:
        out['stoi'] = float(np.mean(stoi_values))
    return out


def mel_loss_metric_jax(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    """Compute Multi-Resolution Mel Spectrogram loss as a validation metric."""
    from criterions.mel_loss_jax import MultiResolutionMelSpectrogramLoss
    if y_pred.ndim == 3:
        y_pred = y_pred.squeeze(1)
    if y_true.ndim == 3:
        y_true = y_true.squeeze(1)
    loss_fn = MultiResolutionMelSpectrogramLoss()
    return loss_fn(y_pred.astype(jnp.float32), y_true.astype(jnp.float32))


class SISDRMetric(MeanMetric):
    """Host-side aggregator for SI-SDR or SI-SNR (via zero_mean flag).

    Usage:
        m = SISDRMetric(zero_mean=False)  # SI-SDR
        m.update_pair(y_pred, y_true)
        val = m.compute()
    """

    def __init__(self, zero_mean: bool = False):
        super().__init__()
        self.zero_mean = bool(zero_mean)

    @staticmethod
    def _to_bt(x):
        arr = np.asarray(x)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim == 3:
            arr = arr[:, 0, :]
        return arr.astype(np.float32)

    def update_pair(self, preds, target) -> None:
        p = self._to_bt(preds)
        t = self._to_bt(target)
        if self.zero_mean:
            t = t - t.mean(axis=-1, keepdims=True)
            p = p - p.mean(axis=-1, keepdims=True)
        eps = np.finfo(np.float32).eps
        alpha = (np.sum(p * t, axis=-1, keepdims=True) + eps) / (np.sum(t * t, axis=-1, keepdims=True) + eps)
        t_scaled = alpha * t
        noise = t_scaled - p
        val = (np.sum(t_scaled * t_scaled, axis=-1) + eps) / (np.sum(noise * noise, axis=-1) + eps)
        sdr = 10.0 * np.log10(val)
        super().update(sdr)


class SISNRMetric(SISDRMetric):
    def __init__(self):
        super().__init__(zero_mean=True)


class SISNRJAXMetric(JaxScalarMeanMetric):
    def __init__(self):
        super().__init__(si_snr_jax)


class SISDRJAXMetric(JaxScalarMeanMetric):
    def __init__(self):
        super().__init__(si_sdr_jax)


class MelJAXMetric(JaxScalarMeanMetric):
    def __init__(self):
        super().__init__(mel_loss_metric_jax)


class AvgSimJAXMetric(MeanMetric):
    """Aggregates avg_sim from a device tensor per batch if provided."""

    def update_tensor(self, tensor: jnp.ndarray) -> None:
        if tensor is None:
            return
        val = jnp.mean(tensor)
        super().update(float(jax.device_get(val)))

class CodebookMetrics:
    def __init__(self, codebook_size: int):
        self.stats = CodebookStats(codebook_size)

    def reset(self) -> None:
        self.stats.reset()

    def update(self, indices) -> None:
        self.stats.update(indices)

    def compute(self) -> dict:
        return {
            'codebook_perplexity': self.stats.perplexity(),
            'codebook_utilization': self.stats.utilization(),
        }


class PESQMetric(MeanMetric):
    def __init__(self, sample_rate: int = 16000):
        super().__init__()
        self.sample_rate = int(sample_rate)

    def update_pair(self, y_pred, y_true, sr: int) -> None:
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        if y_true.ndim == 3:
            y_true = y_true[:, 0, :]
        if y_pred.ndim == 3:
            y_pred = y_pred[:, 0, :]
        if sr != 16000:
            y_pred = _batch_resample_np(y_pred, sr, 16000)
            y_true = _batch_resample_np(y_true, sr, 16000)
        vals = []
        for i in range(y_true.shape[0]):
            ref = y_true[i].astype(np.float32)
            deg = y_pred[i].astype(np.float32)
            L = min(ref.shape[0], deg.shape[0])
            ref = ref[:L]
            deg = deg[:L]
            if PESQ_FUNC is None:
                raise ImportError("pesq package not available; install 'pesq' to compute PESQ")
            vals.append(float(PESQ_FUNC(16000, ref, deg, 'wb')))
        super().update(np.asarray(vals, dtype=np.float32))


class STOIMetric(MeanMetric):
    def __init__(self, sample_rate: int = 16000):
        super().__init__()
        self.sample_rate = int(sample_rate)

    def update_pair(self, y_pred, y_true, sr: int) -> None:
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        if y_true.ndim == 3:
            y_true = y_true[:, 0, :]
        if y_pred.ndim == 3:
            y_pred = y_pred[:, 0, :]
        if sr != 16000:
            y_pred = _batch_resample_np(y_pred, sr, 16000)
            y_true = _batch_resample_np(y_true, sr, 16000)
        vals = []
        for i in range(y_true.shape[0]):
            ref = y_true[i].astype(np.float32)
            deg = y_pred[i].astype(np.float32)
            L = min(ref.shape[0], deg.shape[0])
            ref = ref[:L]
            deg = deg[:L]
            if STOI_FUNC is None:
                raise ImportError("pystoi package not available; install 'pystoi' to compute STOI")
            vals.append(float(STOI_FUNC(ref, deg, 16000, extended=False)))
        super().update(np.asarray(vals, dtype=np.float32))


