import typing
from typing import List, Optional, Union, Callable, Tuple
import jax
import jax.numpy as jnp
from flax import nnx
from common.spectral import MelSpectrogram

class L1Loss(nnx.Module):
    def __call__(self, x, y):
        return jnp.mean(jnp.abs(x - y))

class MultiResolutionMelSpectrogramLoss(nnx.Module):
    """다중 해상도 멜 스펙트로그램 손실"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: List[int] = [5, 10, 20, 40, 80, 160, 320],
        window_lengths: List[int] = [32, 64, 128, 256, 512, 1024, 2048],
        clamp_eps: float = 1e-5,
        pow: float = 1.0,
        mel_fmin: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        mel_fmax: List[Optional[float]] = [None, None, None, None, None, None, None],
        rngs: Optional[nnx.Rngs] = None
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.clamp_eps = clamp_eps
        self.pow = pow
        
        # 모든 멜 변환을 초기화 시점에 미리 생성
        self.mel_transforms = []
        
        for i, (n_mel, window_length) in enumerate(zip(n_mels, window_lengths)):
            f_max = mel_fmax[i] if i < len(mel_fmax) else None
            f_min = mel_fmin[i] if i < len(mel_fmin) else 0.0
            
            # nnx 스타일로 서브모듈 생성
            mel_transform = MelSpectrogram(
                n_fft=window_length,
                hop_length=window_length // 4,
                n_mels=n_mel,
                sample_rate=sample_rate,
                f_min=f_min,
                f_max=f_max,
                power=pow,
                norm='slaney',
                mel_scale='slaney',
                rngs=rngs
            )
            # nnx에서는 리스트 대신 직접 속성으로 할당
            self.mel_transforms.append(mel_transform)

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """두 오디오 신호 간의 다중 해상도 멜 스펙트로그램 손실 계산"""
        loss = 0.0
        
        for i, mel_transform in enumerate(self.mel_transforms):
            x_mel = mel_transform(x)
            y_mel = mel_transform(y)
            
            # 로그 멜 스펙트로그램 계산
            log_x_mel = jnp.log10(jnp.maximum(x_mel, self.clamp_eps) ** self.pow)
            log_y_mel = jnp.log10(jnp.maximum(y_mel, self.clamp_eps) ** self.pow)
            
            # L1 손실 계산
            loss += jnp.mean(jnp.abs(log_x_mel - log_y_mel))
            
        return loss












