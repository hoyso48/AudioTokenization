import typing
from typing import List, Optional, Union, Callable, Tuple
import jax
import jax.numpy as jnp
from flax import nnx
import math
from functools import partial

class Constant(nnx.Variable):
    pass

@partial(jax.jit, static_argnames=['window_length', 'window_type', 'fftbins'])
def _get_window(window_length: int, window_type: str = 'hann', fftbins: bool = True) -> jnp.ndarray:
    """
    librosa.get_window와 정확히 동일한 윈도우 함수 구현
    
    Parameters:
    -----------
    window_length : int
        윈도우 길이
    window_type : str
        윈도우 유형 ('hann', 'hamming' 등)
    fftbins : bool, default=True
        True: "periodic" 윈도우 생성 (FFT용)
        False: "symmetric" 윈도우 생성 (필터 설계용)
        
    librosa의 기본값은 fftbins=True (periodic window)
    """
    # fftbins -> sym 변환 (scipy.signal.get_window와 동일)
    sym = not fftbins
    
    if window_length < 1:
        return jnp.array([])
    
    if window_type == 'hann':
        if sym:
            # symmetric window (필터 설계용)
            n = jnp.arange(window_length)
            w = 0.5 - 0.5 * jnp.cos(2.0 * jnp.pi * n / (window_length - 1))
        else:
            # periodic window (FFT용 - librosa 기본값)
            n = jnp.arange(window_length + 1)
            w = 0.5 - 0.5 * jnp.cos(2.0 * jnp.pi * n / window_length)
            w = w[:-1]
        return w
    elif window_type == 'hamming':
        if sym:
            n = jnp.arange(window_length)
            w = 0.54 - 0.46 * jnp.cos(2.0 * jnp.pi * n / (window_length - 1))
        else:
            n = jnp.arange(window_length + 1)
            w = 0.54 - 0.46 * jnp.cos(2.0 * jnp.pi * n / window_length)
            w = w[:-1]
        return w
    else:
        raise ValueError(f"지원하지 않는 window_type: {window_type}")

@partial(jax.jit, static_argnames=['freq', 'mel_scale'])
def _hz_to_mel(freq: float, mel_scale: str = "htk") -> float:
    """헤르츠를 멜로 변환합니다.

    Args:
        freq (float): 헤르츠 단위의 주파수
        mel_scale (str, optional): 사용할 스케일: ``htk`` 또는 ``slaney``. (기본값: ``htk``)

    Returns:
        float: 멜 단위의 주파수
    """
    if mel_scale not in ["slaney", "htk"]:
        raise ValueError('mel_scale은 "htk" 또는 "slaney" 중 하나여야 합니다.')

    if mel_scale == "htk":
        return 2595.0 * math.log10(1.0 + (freq / 700.0))

    # 선형 부분 채우기
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (freq - f_min) / f_sp

    # 로그 스케일 부분 채우기
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0

    if freq >= min_log_hz:
        mels = min_log_mel + math.log(freq / min_log_hz) / logstep

    return mels

@partial(jax.jit, static_argnames=['mel_scale'])
def _mel_to_hz(mels, mel_scale: str = "htk"):
    """멜 빈 번호를 주파수로 변환합니다.

    Args:
        mels: 멜 주파수
        mel_scale (str, optional): 사용할 스케일: ``htk`` 또는 ``slaney``. (기본값: ``htk``)

    Returns:
        주파수(Hz)로 변환된 멜
    """
    if mel_scale not in ["slaney", "htk"]:
        raise ValueError('mel_scale은 "htk" 또는 "slaney" 중 하나여야 합니다.')

    if mel_scale == "htk":
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    # 선형 스케일 채우기
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # 비선형 스케일 부분
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0

    # JAX에서는 조건부 변경을 mask와 함께 수행
    log_t = mels >= min_log_mel
    freqs = jnp.where(log_t, min_log_hz * jnp.exp(logstep * (mels - min_log_mel)), freqs)

    return freqs


@jax.jit
def _create_triangular_filterbank(all_freqs, f_pts):
    """삼각형 필터 뱅크를 생성합니다.

    Args:
        all_freqs: 크기가 (`n_freqs`)인 STFT 주파수 포인트.
        f_pts: 크기가 (`n_filter`)인 필터 중간 포인트.

    Returns:
        크기가 (`n_freqs`, `n_filter`)인 필터 뱅크.
    """
    # Librosa에서 채택
    # 각 필터 중간 지점과 각 STFT 주파수 지점 간의 차이를 헤르츠 단위로 계산
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_filter + 1)
    slopes = jnp.expand_dims(f_pts, 0) - jnp.expand_dims(all_freqs, 1)  # (n_freqs, n_filter + 2)
    
    # 겹치는 삼각형 생성
    zero = jnp.array([0.0])
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_filter)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_filter)
    fb = jnp.maximum(zero, jnp.minimum(down_slopes, up_slopes))

    return fb

@partial(jax.jit, static_argnames=['n_freqs', 'f_min', 'f_max', 'n_mels', 'sample_rate', 'norm', 'mel_scale'])
def melscale_fbanks(
    n_freqs: int,
    f_min: float,
    f_max: float,
    n_mels: int,
    sample_rate: int,
    norm: str = None,
    mel_scale: str = "htk",
):
    """주파수 빈 변환 행렬을 생성합니다.

    Args:
        n_freqs (int): 강조/적용할 주파수 수
        f_min (float): 최소 주파수 (Hz)
        f_max (float): 최대 주파수 (Hz)
        n_mels (int): 멜 필터뱅크 수
        sample_rate (int): 오디오 파형의 샘플 레이트
        norm (str 또는 None, optional): "slaney"인 경우, 삼각형 멜 가중치를 멜 밴드의 너비로 나눕니다
            (영역 정규화). (기본값: ``None``)
        mel_scale (str, optional): 사용할 스케일: ``htk`` 또는 ``slaney``. (기본값: ``htk``)

    Returns:
        삼각형 필터 뱅크 (fb 행렬), 크기 (``n_freqs``, ``n_mels``)
    """
    if norm is not None and norm != "slaney":
        raise ValueError('norm은 None 또는 "slaney" 중 하나여야 합니다')

    # 주파수 빈
    all_freqs = jnp.linspace(0, sample_rate // 2, n_freqs)

    # 멜 주파수 빈 계산
    m_min = _hz_to_mel(f_min, mel_scale=mel_scale)
    m_max = _hz_to_mel(f_max, mel_scale=mel_scale)

    m_pts = jnp.linspace(m_min, m_max, n_mels + 2)
    f_pts = _mel_to_hz(m_pts, mel_scale=mel_scale)

    # 필터뱅크 생성
    fb = _create_triangular_filterbank(all_freqs, f_pts)

    if norm is not None and norm == "slaney":
        # Slaney 스타일 멜은 채널당 대략적으로 일정한 에너지를 갖도록 조정됨
        enorm = 2.0 / (f_pts[2:n_mels + 2] - f_pts[:n_mels])
        fb = fb * jnp.expand_dims(enorm, 0)

    return fb

@partial(jax.jit, static_argnames=['n_fft', 'hop_length', 'win_length', 'normalized', 'center', 'pad_mode'])
def stft(
    waveform: jnp.ndarray,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: jnp.ndarray,
    center: bool = True,
    pad_mode: str = "reflect",
    normalized: bool = False,
) -> jnp.ndarray:
    """JAX 최적화 STFT 구현 - moving_window 활용"""
    # waveform: (N, T) shape array
    
    # 윈도우가 제공되지 않은 경우 생성
    if window.shape[0] < n_fft:
        # 중앙 패딩으로 window 길이 확장
        pad_left = (n_fft - win_length) // 2
        pad_right = n_fft - win_length - pad_left
        window = jnp.pad(window, (pad_left, pad_right), mode='constant')
    
    # 가운데 패딩 적용
    if center:
        padding = [(0, 0), (n_fft // 2, n_fft // 2)]
        if pad_mode == 'reflect':
            # JAX에서 reflect 패딩
            waveform = jnp.pad(waveform, padding, mode='reflect')
        else:
            waveform = jnp.pad(waveform, padding, mode='constant')
    
    # 프레임 추출을 위한 준비
    batch_size, signal_length = waveform.shape

    frames = jax.lax.conv_general_dilated_patches(
        lhs=waveform[:, :, jnp.newaxis],
        filter_shape=(n_fft,),
        window_strides=(hop_length,),
        padding='VALID',
        dimension_numbers=('NHC', 'HIO', 'NHC')
    )
    # conv_general_dilated_patches의 출력 차원 순서는 JAX 버전에 따라 (B, n_fft, frames)
    # 혹은 (B, frames, n_fft)가 될 수 있으므로, 마지막 축이 항상 n_fft가 되도록 정규화한다.
    if frames.shape[-1] != n_fft:
        frames = frames.swapaxes(-1, -2)
    # 이제 frames: (batch, frames, n_fft)
    # Apply window over the last axis (n_fft)
    frames = frames * window[jnp.newaxis, jnp.newaxis, :]
    
    # FFT 계산
    stft_matrix = jnp.fft.rfft(frames, n=n_fft, axis=-1)
    
    # 정규화 적용
    if normalized:
        stft_matrix = stft_matrix / jnp.sqrt(jnp.sum(window**2))
    
    # (batch, frames, freq) -> (batch, freq, frames)
    return stft_matrix.swapaxes(-1, -2)

@partial(jax.jit, static_argnames=['pad', 'power', 'n_fft', 'hop_length', 'win_length', 'normalized', 'center', 'pad_mode', 'onesided', 'return_complex'])
def spectrogram(
    waveform: jnp.ndarray,
    pad: int,
    window: jnp.ndarray,
    n_fft: int,
    hop_length: int,
    win_length: int, 
    power: Optional[float],
    normalized: Union[bool, str],
    center: bool = True,
    pad_mode: str = "reflect",
    onesided: bool = True,
    return_complex: Optional[bool] = None,
) -> jnp.ndarray:
    """JAX로 구현된 스펙트로그램 또는 스펙트로그램 배치 생성 함수
    
    Args:
        waveform: (..., time) 차원의 오디오 텐서
        pad: 신호의 양쪽 패딩
        window: 각 프레임/윈도우에 적용되는 윈도우 텐서
        n_fft: FFT 크기
        hop_length: STFT 윈도우 간의 홉 길이
        win_length: 윈도우 크기
        power: 스펙트로그램의 지수(> 0이어야 함). 예: 1은 진폭, 2는 파워 등
               None이면 복소 스펙트럼 반환
        normalized: STFT 후 크기에 따라 정규화할지 여부. 문자열이면 'window'와 'frame_length' 중 선택
        center: 입력 신호에 양쪽 패딩을 추가할지 여부
        pad_mode: center가 True일 때 사용되는 패딩 방법
        onesided: 중복을 피하기 위해 결과의 절반만 반환할지 여부
        return_complex: 사용되지 않음 (호환성을 위해 유지)
        
    Returns:
        차원이 (..., freq, time)인 텐서, freq는 n_fft // 2 + 1
    """
    if return_complex is not None:
        print("'return_complex' 인자는 더 이상 사용되지 않으며 효과가 없습니다.")
    
    # 패딩 적용
    if pad > 0:
        padding = [(0, 0)] * (waveform.ndim - 1) + [(pad, pad)]
        waveform = jnp.pad(waveform, padding, mode='constant')
    
    # 정규화 설정
    frame_length_norm = False
    window_norm = False
    if normalized == True or normalized == "window":
        window_norm = True
    elif normalized == "frame_length":
        frame_length_norm = True
    
    # 원래 shape 저장
    orig_shape = waveform.shape
    # 배치 처리를 위해 shape 변경
    waveform = waveform.reshape(-1, orig_shape[-1])
    
    # STFT 계산 (이미 구현된 함수 사용)
    spec_f = stft(
        waveform=waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
        normalized=frame_length_norm
    )
    
    # Onesided 처리 (JAX의 RFFT는 이미 한쪽만 반환하므로 필요하지 않음)
    
    # 원래 shape으로 복원
    spec_f = spec_f.reshape(orig_shape[:-1] + spec_f.shape[-2:])
    
    # Window 정규화 적용
    if window_norm:
        spec_f = spec_f / jnp.sqrt(jnp.sum(window ** 2))
    
    # Power 매개변수에 따라 출력 형태 결정
    if power is not None:
        if power == 1.0:
            return jnp.abs(spec_f)
        return jnp.abs(spec_f) ** power
    
    return spec_f


class MelScale(nnx.Module):
    """일반 STFT를 삼각형 필터 뱅크를 사용한 멜 주파수 STFT로 변환합니다."""
    
    def __init__(
        self, 
        n_mels: int = 128,
        sample_rate: int = 16000,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        n_stft: int = 201,
        norm: Optional[str] = None,
        mel_scale: str = "htk",
        rngs: Optional[nnx.Rngs] = None
    ):
        super().__init__()
        # f_max 계산
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.f_min = f_min
        self.n_stft = n_stft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.norm = norm
        self.mel_scale = mel_scale
        
        if self.f_min > self.f_max:
            raise ValueError(f"f_min: {self.f_min}은 f_max: {self.f_max}보다 작거나 같아야 합니다")
            
        # 필터 뱅크 생성 - 초기화 시 생성
        self.fb = Constant(melscale_fbanks(
            self.n_stft, 
            self.f_min, 
            self.f_max, 
            self.n_mels, 
            self.sample_rate, 
            self.norm, 
            self.mel_scale
        ))
    
    def __call__(self, specgram: jnp.ndarray) -> jnp.ndarray:
        # 입력 specgram과 필터뱅크 간의 행렬 곱셈 수행
        mel_spec_t = jnp.matmul(specgram.swapaxes(-1, -2), self.fb.value).swapaxes(-1, -2)
        return mel_spec_t


class Spectrogram(nnx.Module):
    """오디오 신호에서 스펙트로그램을 생성합니다."""
    
    def __init__(
        self,
        n_fft: int = 400,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        pad: int = 0,
        window_type: str = 'hann',
        power: Optional[float] = 2.0,
        normalized: Union[bool, str] = False,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: bool = True,
        rngs: Optional[nnx.Rngs] = None
    ):
        super().__init__()
        self.n_fft = n_fft
        self.pad = pad
        self.power = power
        self.normalized = normalized
        self.center = center
        self.pad_mode = pad_mode
        self.onesided = onesided
        
        self.actual_win_length = win_length if win_length is not None else n_fft
        self.actual_hop_length = hop_length if hop_length is not None else self.actual_win_length // 2
        
        # 윈도우 함수 생성 - 초기화 시 생성
        self.window = Constant(_get_window(self.actual_win_length, window_type))

    def __call__(self, waveform: jnp.ndarray) -> jnp.ndarray:
        # spectrogram 함수 호출
        return spectrogram(
            waveform,
            self.pad,
            self.window.value,
            self.n_fft,
            self.actual_hop_length,
            self.actual_win_length,
            self.power,
            self.normalized,
            self.center,
            self.pad_mode,
            self.onesided,
        )

class MelSpectrogram(nnx.Module):
    """오디오 파형에서 멜 스펙트로그램을 계산하는 모듈"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        n_mels: int = 128,
        window_type: str = 'hann',
        power: float = 1.0,
        normalized: Union[bool, str] = False,
        center: bool = True,
        pad_mode: str = 'reflect',
        norm: Optional[str] = None,
        mel_scale: str = 'htk',
        rngs: Optional[nnx.Rngs] = None
    ):
        super().__init__()
        
        # win_length와 hop_length 처리
        actual_win_length = win_length if win_length is not None else n_fft
        actual_hop_length = hop_length if hop_length is not None else actual_win_length // 4
        
        # nnx에서는 서브모듈을 __init__에서 직접 초기화
        self.spectrogram_module = Spectrogram(
            n_fft=n_fft,
            win_length=actual_win_length,
            hop_length=actual_hop_length,
            window_type=window_type,
            power=power,
            normalized=normalized,
            center=center,
            pad_mode=pad_mode,
            rngs=rngs
        )
        
        # n_stft 계산
        n_stft = n_fft // 2 + 1
        
        self.mel_scale_module = MelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=f_max,
            n_stft=n_stft,
            norm=norm,
            mel_scale=mel_scale,
            rngs=rngs
        )

    def __call__(self, waveform: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            waveform (jnp.ndarray): 입력 오디오 파형 (batch_size, time)
            
        Returns:
            jnp.ndarray: 멜 스펙트로그램 (batch_size, n_mels, time)
        """
        specgram = self.spectrogram_module(waveform)
        mel_specgram = self.mel_scale_module(specgram)
        return mel_specgram