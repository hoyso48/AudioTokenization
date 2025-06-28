import torch
from torch import nn
from torch.nn import functional as F
from pathlib import Path
from typing import List, Tuple, Union, Optional
import os
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import numpy as np
import soundfile as sf
import sys


# --- load_libritts_item: Modified slightly to return fileid for path construction ---
# 원본 함수 시그니처 유지 노력, 단지 fileid 반환 추가
def load_libritts_item(
    fileid: str,
    path: str, # Subset base path, e.g., .../LibriSpeech/LibriSpeech/test-clean
    ext_audio: str,
    ext_original_txt: str, # Keep signature, though not used for indices
    ext_normalized_txt: str, # Keep signature, though not used for indices
    target_sample_rate: int = None,
    offset_mode: str = 'start',
    duration: Optional[float] = None,  # seconds
    pad_to_stride: int = None,
    # gain: float = -3.0, # Not used
) -> Tuple[Tensor, int, str, str, int, int, str]: # Original return signature + fileid
    try:
        # Try standard LibriTTS format first
        speaker_id, chapter_id, segment_id, utterance_id_part = fileid.split("_")
        # utterance_id = fileid # This line was in original, keep it? Redundant if fileid is used directly
    except ValueError:
        # Fallback for potential variations like LibriSpeech format
        try:
            speaker_id, chapter_id, utterance_id_part = fileid.split("-")
            # utterance_id = fileid
        except ValueError as e:
             # Handle cases where the split might fail differently
             parts = fileid.split('-')
             if len(parts) >= 3:
                 speaker_id, chapter_id = parts[0], parts[1]
                 # utterance_id = fileid
             else:
                 raise ValueError(f"Cannot parse speaker/chapter from fileid: {fileid}") from e

    file_audio_name = fileid + ext_audio
    # Construct path relative to the subset path 'path' provided
    file_audio = os.path.join(path, speaker_id, chapter_id, file_audio_name)

    if not os.path.exists(file_audio):
        raise FileNotFoundError(f"Audio file not found at: {file_audio}")

    # soundfile 방식 (original)
    info = sf.info(file_audio)
    sample_rate = info.samplerate
    total_frames = info.frames

    # Calculate frame offset and number of frames to load (original logic)
    if offset_mode == 'start':
        frame_offset = 0
    elif offset_mode == 'random':
        # Ensure target frames calculation handles duration=None
        target_frames = -1 if duration is None else int(duration * sample_rate)
        if duration is not None and total_frames > target_frames and target_frames > 0:
             # Prevent negative upper bound for randint if target_frames is 0 or negative
             upper_bound = total_frames - target_frames
             if upper_bound > 0:
                 frame_offset = np.random.randint(0, upper_bound + 1)
             else:
                 frame_offset = 0 # Start from beginning if calculation is problematic
        else:
            frame_offset = 0 # Start from beginning if duration is None, audio is shorter, or invalid duration
    else:
         raise ValueError("offset_mode must be 'start' or 'random'")

    num_frames = -1 if duration is None else int(duration * sample_rate)

    # Load audio with offset and duration (original logic)
    try:
        with sf.SoundFile(file_audio, 'r') as f:
            f.seek(frame_offset)
            frames_to_read = num_frames if num_frames != -1 else -1 # soundfile reads all if -1
            # Read as float32, ensure 2D (C, T) for consistency
            waveform = f.read(frames=frames_to_read, dtype='float32', always_2d=True).T
            waveform = torch.from_numpy(waveform) # Shape (C, T)
    except Exception as e:
        print(f"Error loading audio file {file_audio}: {e}")
        raise e

    # gain = np.random.uniform(-1, -6) if offset_mode == 'random' else -3 # Original gain logic (commented out as likely not needed for indices)
    # waveform, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate, [["norm", f"{gain:.2f}"]])

    # Pad if duration is specified and waveform is shorter (original logic)
    if duration is not None:
        target_length = int(duration * sample_rate)
        current_length = waveform.size(1)
        if current_length < target_length:
            padding_length = target_length - current_length
            # Original padding was (padding_length, 0) which pads at the beginning. Usually padding is at the end.
            # Let's stick to the original code's behavior unless specified otherwise.
            # waveform = torch.nn.functional.pad(waveform, (padding_length, 0), mode='constant', value=0)
            # Correcting to pad at the end which is more standard:
            waveform = torch.nn.functional.pad(waveform, (0, padding_length), mode='constant', value=0)
        # Ensure waveform is exactly duration length if specified (truncating if longer)
        waveform = waveform[:, :target_length]

    # Resample if needed (original logic)
    original_sample_rate = sample_rate # Store original SR before potential modification
    if target_sample_rate and target_sample_rate != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform.float()) # Ensure float for resampler
        sample_rate = target_sample_rate # Update sample rate

    # Pad to stride if needed (original logic)
    if pad_to_stride and waveform.size(1) % pad_to_stride != 0:
        padding_needed = pad_to_stride - waveform.size(1) % pad_to_stride
        waveform = torch.nn.functional.pad(waveform, (0, padding_needed), mode='constant', value=0)


    # Load text (keep logic even if not used directly, as per original function signature)
    # file_original_txt = os.path.join(path, speaker_id, chapter_id, fileid + ext_original_txt)
    # file_normalized_txt = os.path.join(path, speaker_id, chapter_id, fileid + ext_normalized_txt)

    # with open(file_original_txt) as f:
        # original_transcript = f.readline().strip()
    # with open(file_normalized_txt) as f:
        # normalized_transcript = f.readline().strip()

    # speaker_id, chapter_id, utterance_id are parsed above, but need int conversion per original return type
    speaker_id = int(speaker_id)
    chapter_id = int(chapter_id)
    # utterance_id needs parsing for the integer part if format is xxx_yyy_zzz_uuu
    # The original code had a potential bug here if format was speaker-chapter-utteranceid
    try:
       utterance_id_int = int(fileid.split('_')[-1]) # Assumes format xxx_yyy_zzz_uuu
    except:
       try:
           utterance_id_int = int(fileid.split('-')[-1]) # Assumes format xxx-yyy-uuu
       except:
           utterance_id_int = -1 # Fallback if ID cannot be parsed as int


    # Return signature matches original + fileid for path construction
    # Note: Original code didn't return file_audio path, we derive it again later if needed
    # Return waveform, target sample_rate, original transcript, normalized transcript, speaker ID (int), chapter ID (int), utterance ID (int), original sample rate, fileid
    # The original return tuple definition in the type hint was different from the actual implicit return in the calling code.
    # Let's return what the *dataset* needs: waveform, sample_rate, and info to reconstruct path (subset, fileid)
    # Returning the full original tuple + fileid might be too disruptive if downstream code expected fewer items.
    # Compromise: Return waveform, sample_rate (the potentially resampled one), and fileid.
    return (
        waveform.float(), # Ensure float
        sample_rate,
        fileid # Add fileid for constructing output path
        # Keep other returns if strictly needed by some *unmodified* downstream code.
        # If only index extraction is needed, waveform, sample_rate, fileid is sufficient.
        # Let's assume the original dataset __getitem__ adapted the output, so we provide essentials.
    )


# --- LibriTTSDataset: Minimal changes to support index extraction ---
class LibriTTSDataset(Dataset):
    """LibriTTS dataset - Minimally modified for index extraction."""

    _ext_original_txt = ".original.txt" # Keep signature
    _ext_normalized_txt = ".normalized.txt" # Keep signature

    def __init__(
        self,
        root: Union[str, Path],
        subsets: Union[str, List[str]],
        dataset_path: str="LibriSpeech/LibriSpeech",
        sample_rate: int = None,
        duration: Optional[float] = None,
        offset_mode: str = 'start',
        pad_to_stride: int = None,
        ext_audio: str = ".flac",
    ) -> None:
        # Keep original validation logic
        self.dataset_path = dataset_path
        self.ext_audio = ext_audio
        if offset_mode not in ['random', 'start']:
            raise ValueError("offset_mode must be either 'random' or 'start'")

        root = os.fspath(root)
        if isinstance(subsets, str):
            subsets = [subsets]

        valid_subsets = {
            "dev-clean", "dev-other", "test-clean", "test-other",
            "train-clean-100", "train-clean-360", "train-other-500"
        }
        for subset in subsets:
            if subset not in valid_subsets:
                 # Allow non-standard subsets but warn (as original might implicitly allow)
                 print(f"Warning: Subset '{subset}' not in standard {self.dataset_path} list: {valid_subsets}")
                 # raise ValueError(f"Invalid subset '{subset}'. Must be one of {valid_subsets}")

        self.sample_rate = sample_rate
        self.duration = duration
        self.offset_mode = offset_mode
        self.pad_to_stride = pad_to_stride

        # Walker stores subset name, subset base path, and fileid
        self._walker = []
        for subset in subsets:
            # Original path logic assumed root contained LibriSpeech/LibriSpeech
            subset_path = os.path.join(root, self.dataset_path, subset)
            if not os.path.isdir(subset_path):
                # Try alternative if root *is* LibriSpeech/LibriSpeech
                subset_path = os.path.join(root, subset)
                print(f"Trying alternative path: {subset_path}")
                if not os.path.isdir(subset_path):
                    raise RuntimeError(f"Dataset subset not found at expected locations relative to root '{root}': check structure.")

            print(f"Scanning subset: {subset_path}")
            # Original walk logic using Path.glob
            # Need to handle symbolic links potentially, rglob might be better
            # for p in Path(subset_path).glob(f"*/*/*{self._ext_audio}"):
            #    # Original walker stored (subset_path, stem)
            #    self._walker.append((subset_path, str(p.stem)))
            # Modified walker stores (subset_name, subset_base_path, fileid)
            for p in Path(subset_path).rglob(f"*{self.ext_audio}"): # Use rglob for recursive search
                 self._walker.append((subset, subset_path, str(p.stem)))


        if not self._walker:
             print(f"Warning: No audio files found for subsets {subsets} in root {root}. Check paths and file extensions.")


    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str]:
        """Load the n-th sample's waveform and metadata for index extraction."""
        subset, subset_path, fileid = self._walker[n]

        # Call the loader function - it now returns (waveform, sample_rate, fileid)
        waveform, sample_rate, _ = load_libritts_item( # Ignore the returned fileid here, we already have it
            fileid,
            subset_path, # Pass the base path for the subset
            self.ext_audio,
            self._ext_original_txt, # Pass required args even if unused
            self._ext_normalized_txt, # Pass required args even if unused
            self.sample_rate,
            self.offset_mode,
            duration=self.duration,
            pad_to_stride=self.pad_to_stride,
        )
        # Return only what's needed for index extraction: waveform, sample_rate, subset, fileid
        return waveform, sample_rate, subset, fileid

    def __len__(self) -> int:
        return len(self._walker)


import numpy as np
import librosa
import matplotlib.pyplot as plt
from IPython.display import Audio, display

def plot_spectrograms_and_audio(x_sample, recon_x, sr=16000, 
                               save_path=None, n_examples=10):
    """
    스펙트로그램과 오디오를 파일로 저장합니다.
    """
    if hasattr(x_sample, 'cpu'):
        x_sample = x_sample.cpu().float().numpy()
    if hasattr(recon_x, 'cpu'):
        recon_x = recon_x.cpu().float().numpy()
        
    n_examples = min(n_examples, len(x_sample), len(recon_x))
    
    # 저장 디렉토리 생성
    spec_dir = os.path.join(save_path, 'spec')
    audio_dir = os.path.join(save_path, 'audio')
    os.makedirs(spec_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    
    n_fft = 1024
    hop_length = 256
    
    for i in range(n_examples):
        # 스펙트로그램 저장
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
        
        orig_wave = x_sample[i][0]
        D_orig = librosa.amplitude_to_db(
            np.abs(librosa.stft(orig_wave, n_fft=n_fft, hop_length=hop_length)),
            ref=np.max
        )
        ax1.imshow(D_orig, origin='lower', aspect='auto', cmap='magma')
        ax1.set_title(f"Original {i+1}")
        ax1.axis('off')

        recon_wave = recon_x[i][0]
        D_recon = librosa.amplitude_to_db(
            np.abs(librosa.stft(recon_wave, n_fft=n_fft, hop_length=hop_length)),
            ref=np.max
        )
        ax2.imshow(D_recon, origin='lower', aspect='auto', cmap='magma')
        ax2.set_title(f"Reconstructed {i+1}")
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(spec_dir, f'spec_{i+1}.png'))
        plt.close()
        
        # 오디오 저장
        sf.write(os.path.join(audio_dir, f'orig_{i+1}.wav'), orig_wave, sr)
        sf.write(os.path.join(audio_dir, f'recon_{i+1}.wav'), recon_wave, sr)



import random
import os
import numpy as np
import math
import gc
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from collections import Counter

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all the statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Updates the meter with a new value.
        
        Args:
            val (float): The new value to update.
            n (int): The number of occurrences of this value (default is 1).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AverageMeterDict:
    """여러 개의 AverageMeter를 딕셔너리 형태로 관리하는 클래스"""
    def __init__(self):
        self.meters = {}

    def reset(self, name=None):
        """특정 미터 또는 모든 미터를 초기화합니다.
        
        Args:
            name (str, optional): 초기화할 미터의 이름. None이면 모든 미터 초기화
        """
        if name is None:
            # 모든 미터 초기화
            for meter in self.meters.values():
                meter.reset()
        else:
            # 특정 미터만 초기화
            if name in self.meters:
                self.meters[name].reset()

    def update(self, name, val, n=1):
        """특정 이름의 미터를 업데이트합니다. 없으면 새로 생성합니다.
        
        Args:
            name (str): 업데이트할 미터의 이름
            val (float): 새로운 값
            n (int): 해당 값의 발생 횟수 (기본값: 1)
        """
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(val, n)

    def get_average(self, name):
        """특정 미터의 평균값을 반환합니다.
        
        Args:
            name (str): 조회할 미터의 이름
            
        Returns:
            float: 해당 미터의 평균값
        """
        if name in self.meters:
            return self.meters[name].avg
        raise KeyError(f"Meter '{name}' is not found.")

    def get_all_averages(self):
        """모든 미터의 평균값을 딕셔너리 형태로 반환합니다.
        
        Returns:
            dict: 미터 이름을 키로, 평균값을 값으로 하는 딕셔너리
        """
        return {name: meter.avg for name, meter in self.meters.items()}
        
def get_cosine_decay_with_warmup(total_steps=1000, warmup_steps=100, max_lr=1e-3, min_lr=1e-7):
    
    def get_lr(step):

        if step < warmup_steps:
            # Linear warmup
            return max_lr * step / warmup_steps
        else:
            # Cosine decay
            cosine_decay = 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
            return min_lr + (max_lr - min_lr) * cosine_decay
        
    return get_lr

class LRScheduler:
    def __init__(self, optimizer, lr_fn):
        self.current_step = 0
        self.optimizer = optimizer
        self.lr_fn = lr_fn
    
    def step(self):
        lr = self.lr_fn(self.current_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.current_step += 1
        return lr



from pesq import pesq_batch
from pystoi import stoi

class PESQ:
    def __init__(self, in_sr=16000, sr=16000, on_error=1, mode='wb'):
        self.in_sr = in_sr
        self.sr = sr
        self.on_error = on_error
        self.mode = mode
        if in_sr != sr:
            self.resampler = torchaudio.transforms.Resample(in_sr, sr)
        else:
            self.resampler = None
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0
    
    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0
        
    def update(self, x, y):
        if self.resampler:
            x = self.resampler(x.float().cpu())
            y = self.resampler(y.float().cpu())
        x = x[:,0].float().cpu().numpy()
        y = y[:,0].float().cpu().numpy()
        min_len = min(x.shape[1], y.shape[1])
        x = x[:,:min_len]
        y = y[:,:min_len]
        n = x.shape[0]
        val = np.mean(pesq_batch(fs=self.sr, ref=x, deg=y, on_error=self.on_error, mode=self.mode))
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def compute(self):
        return self.avg

class STOI:
    def __init__(self, in_sr=16000, sr=16000):
        self.in_sr = in_sr
        self.sr = sr
        if in_sr != sr:
            self.resampler = torchaudio.transforms.Resample(in_sr, sr)
        else:
            self.resampler = None
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0
    
    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0
        
    def update(self, x, y):
        if self.resampler:
            x = self.resampler(x.float().cpu())
            y = self.resampler(y.float().cpu())
        x = x[:,0].float().cpu().numpy()
        y = y[:,0].float().cpu().numpy()
        min_len = min(x.shape[1], y.shape[1])
        x = x[:,:min_len]
        y = y[:,:min_len]
        n = x.shape[0]
        val = 0
        for ref, deg in zip(x,y):
            val += stoi(x=ref, y=deg, fs_sig=self.sr, extended=False) / n
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def compute(self):
        return self.avg

import os
import pytorch_lightning as pl
import hydra
import librosa
import soundfile as sf
import torch
import numpy as np
from os.path import join, exists, dirname, basename
from pytorch_lightning import seed_everything
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader
from data_module import DataModule
from lightning_module import CodecLightningModule
from tqdm import tqdm
from glob import glob
from time import time
from omegaconf import OmegaConf
class BigCodecModel(nn.Module):
    def __init__(self, ckpt_path, config_path):
        super(BigCodecModel, self).__init__()
        
        # 설정 직접 로드
        cfg = OmegaConf.load(config_path)
        
        # Lightning 모듈 생성
        self.lm = CodecLightningModule(cfg=cfg)
        
        # checkpoint 로드
        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)['state_dict']
        
        # 가중치 로드
        self.lm.load_state_dict(state_dict, strict=True)
        self.codebook_size = cfg.model.codec_decoder.codebook_size
        # self.lm.eval()
        
        # # GPU로 이동 (필요한 경우)
        # if torch.cuda.is_available():
        #     self.lm = self.lm.cuda()
        
    def forward(self, x):
        vq_emb = self.lm.model['CodecEnc'](x)
        vq_post_emb, vq_code, _ = self.lm.model['generator'](vq_emb, vq=True)
        recon = self.lm.model['generator'](vq_post_emb, vq=False)
        return {'x_rec': recon, 'indices': vq_code, 'loss': {}}


import librosa
import matplotlib.pyplot as plt
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio, ScaleInvariantSignalDistortionRatio, SignalDistortionRatio, SignalNoiseRatio
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality

def calculate_perplexity(counter, codebook_size):
    """
    코드북 사용 통계로부터 정규화된 perplexity를 계산합니다.
    
    Args:
        counter (Counter): 코드북 인덱스 사용 빈도수
        codebook_size (int): 코드북 크기
        
    Returns:
        float: 계산된 정규화된 perplexity 값 (0~1 사이)
    """
    # 총 코드 수
    total_counts = sum(counter.values())
    if total_counts == 0:
        return 0.0
    
    # 확률 분포 계산
    probs = np.zeros(codebook_size)
    for idx, count in counter.items():
        if idx < codebook_size:  # 유효한 인덱스만 처리
            probs[idx] = count / total_counts
    
    # 0이 아닌 확률에 대해서만 엔트로피 계산
    nonzero_probs = probs[probs > 0]
    entropy = -np.sum(nonzero_probs * np.log(nonzero_probs))
    
    # 코드북 크기로 정규화
    max_entropy = np.log(codebook_size)
    normalized_entropy = entropy / max_entropy
    normalized_perplexity = np.exp(normalized_entropy)
    
    # 원래 perplexity도 계산 (참고용)
    perplexity = np.exp(entropy)
    
    return normalized_perplexity, perplexity

def train(model, 
          train_loader=None, 
          val_loader=None,
          eval_loader=None,
          sample_rate=24000,
          val_freq=1,
          visualize_freq=1,
          epochs=5, 
          device='cuda', 
          seed=42,
          weight_decay=0.01,
          lr=2e-4,
          bf16=True,
          torch_compile=False,
          clip_grad=1.0,
          save_path=None):
    seed_everything(seed=seed)
    torch.cuda.empty_cache()
    gc.collect()

    model = model.to(device)
    if torch_compile:
        model.encoder = torch.compile(model.encoder)
        model.decoder = torch.compile(model.decoder)
    if hasattr(model, 'codebook_size'):
        codebook_size = model.codebook_size
    elif hasattr(model, 'quantizer'):
        codebook_size = model.quantizer.codebook_size
    else:
        codebook_size = 1 #No codebook
    print('codebook_size', codebook_size)

    optimizer = torch.optim.AdamW([
        {'params': [param for param in model.parameters() if param.ndim>=2], 'weight_decay': weight_decay},
        {'params': [param for param in model.parameters() if param.ndim<2], 'weight_decay': 0.0}
    ], lr=lr)
    
    accum_metrics = AverageMeterDict()
    si_snr = ScaleInvariantSignalNoiseRatio().to(device)
    si_sdr = ScaleInvariantSignalDistortionRatio().to(device)
    stoi = STOI(in_sr=sample_rate, sr=sample_rate)
    pesq = PESQ(in_sr=sample_rate, sr=16000)
    total_steps = len(train_loader) * epochs if train_loader is not None else 1
    lr_fn = get_cosine_decay_with_warmup(total_steps=total_steps, warmup_steps=0, max_lr=lr, min_lr=1e-7)
    scheduler = LRScheduler(optimizer, lr_fn)

    for epoch in range(1, epochs+1):
        if train_loader is not None:
            model.train()
            accum_metrics.reset()
            # si_snr.reset()
            # si_sdr.reset()
            # stoi.reset()
            # pesq.reset()
            train_codebook_counter = Counter()  # 각 에폭마다 초기화

            pbar = tqdm(train_loader, desc=f'TRAIN epoch {epoch}', total=len(train_loader))
            for data in pbar:
                # print(data)
                x = data[0].to(device)
                with torch.amp.autocast(device_type=device, dtype=torch.bfloat16, enabled=bf16):
                    output = model(x)
                    loss = sum(output['loss'].values())

                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                lr = scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                
                # Codebook usage 모니터링
                if 'indices' in output:
                    indices = output['indices'].flatten().cpu().numpy()
                    train_codebook_counter.update(indices)

                for name, val in output['loss'].items():
                    accum_metrics.update(name, val.detach().cpu().item(), len(x))

                # si_snr.update(output['x_rec'].float(), x.float())
                # stoi.update(output['x_rec'].float(), x.float())
                # pesq.update(output['x_rec'], x)
                

                metrics_dict = accum_metrics.get_all_averages()
                metrics_dict.update({
                    'lr': f'{lr:.6f}',
                    'grad_norm': f'{norm:.4f}',
                    'codebook_usage': f'{len(train_codebook_counter)/codebook_size:.3f}',
                    # 'si_snr': f'{si_snr.compute().cpu().item():.4f}',
                    # 'stoi': f'{stoi.compute():.4f}',
                    # 'pesq': f'{pesq.compute():.4f}'
                })
                pbar.set_postfix(metrics_dict)
        
        if val_loader is not None and (epoch % val_freq == 0 or epoch == epochs):
            # Validation
            model.eval()
            accum_metrics.reset()
            si_snr.reset()
            si_sdr.reset()
            stoi.reset()
            pesq.reset()
            test_codebook_counter = Counter()

            for data in val_loader:
                x = data[0].to(device)
                x = F.pad(x, (0, (200 - (x.shape[2] % 200))))
                with torch.no_grad():
                    output = model(x)
                        
                    if 'indices' in output:
                        indices = output['indices'].flatten().cpu().numpy()
                        test_codebook_counter.update(indices)
                        
                    for name, val in output['loss'].items():
                        accum_metrics.update(name, val.detach().cpu().item(), len(x))

                    si_snr.update(output['x_rec'].float(), x.float())
                    stoi.update(output['x_rec'].float(), x.float())
                    pesq.update(output['x_rec'].float(), x.float())

            # Perplexity 계산
            norm_perplexity, raw_perplexity = calculate_perplexity(test_codebook_counter, codebook_size)
            
            metrics_dict = accum_metrics.get_all_averages()
            metrics_dict['codebook_usage'] = len(test_codebook_counter)/codebook_size
            metrics_dict['norm_perplexity'] = norm_perplexity
            metrics_dict['raw_perplexity'] = raw_perplexity
            metrics_dict['si_snr'] = si_snr.compute().cpu().item()
            metrics_dict['si_sdr'] = si_sdr.compute().cpu().item()
            metrics_dict['stoi'] = stoi.compute()
            metrics_dict['pesq'] = pesq.compute()
            val_result = ' '.join([f'val_{name} {val:.4f}' for name, val in metrics_dict.items()])
            print(f'Epoch{epoch}: {val_result}')

        if val_loader is not None and (epoch % visualize_freq == 0 or epoch == epochs):
            model.eval()
            torch.manual_seed(seed)
            for data in val_loader:
                x_sample = data[0][:10].to(device)
                x_sample = F.pad(x_sample, (0, (200 - (x_sample.shape[2] % 200))))
            
            with torch.no_grad():
                output = model(x_sample)
                recon_x = output['x_rec']
            
            # 스펙트로그램과 오디오 플레이어 표시
            plot_spectrograms_and_audio(x_sample, recon_x, sr=sample_rate, save_path=save_path)

    if eval_loader is not None:
        model.eval()
        accum_metrics.reset()
        si_snr.reset()
        si_sdr.reset()
        stoi.reset()
        pesq.reset()
        test_codebook_counter = Counter()

        for data in eval_loader:
            x = data[0].to(device)
            with torch.no_grad():
                output = model(x)
                    
                if 'indices' in output:
                    indices = output['indices'].flatten().cpu().numpy()
                    test_codebook_counter.update(indices)
                    
                for name, val in output['loss'].items():
                    accum_metrics.update(name, val.detach().cpu().item(), len(x))

                si_snr.update(output['x_rec'].float(), x.float())
                si_sdr.update(output['x_rec'].float(), x.float())
                stoi.update(output['x_rec'].float(), x.float())
                pesq.update(output['x_rec'].float(), x.float())

        # Perplexity 계산
        norm_perplexity, raw_perplexity = calculate_perplexity(test_codebook_counter, codebook_size)
        
        metrics_dict = accum_metrics.get_all_averages()
        metrics_dict['codebook_usage'] = len(test_codebook_counter)/codebook_size
        metrics_dict['norm_perplexity'] = norm_perplexity
        metrics_dict['raw_perplexity'] = raw_perplexity
        metrics_dict['si_snr'] = si_snr.compute().cpu().item()
        metrics_dict['si_sdr'] = si_sdr.compute().cpu().item()
        metrics_dict['stoi'] = stoi.compute()
        metrics_dict['pesq'] = pesq.compute()
        val_result = ' '.join([f'eval_{name} {val:.4f}' for name, val in metrics_dict.items()])
        print(f'Epoch{epoch}: {val_result}')

    # 학습 완료 후 코드북 사용 히스토그램 시각화
    plt.figure(figsize=(15, 5))
    indices = sorted(test_codebook_counter.keys())
    counts = [test_codebook_counter[i] for i in indices]
    # print(indices, counts)
    plt.bar(indices, counts)
    plt.title('Codebook Usage Distribution (Test Set)')
    plt.xlabel('Codebook Index')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    return model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default="../../datasets", help="Path to the dataset")
    parser.add_argument('--dataset_path', type=str, default="LibriTTS", help="Dataset directory name")
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--output_folder', type=str, default='outputs')
    parser.add_argument('--duration', type=float, default=1)
    parser.add_argument('--ext_audio', type=str, default='.wav')
    parser.add_argument('--sample_rate', type=int, default=16000)
    args = parser.parse_args()
    print(args)
    # 로그 파일 설정
    out_path = os.path.join(args.save_path, args.output_folder)
    os.makedirs(out_path, exist_ok=True)
    log_path = os.path.join(out_path, 'log.txt')
    
    # stdout을 파일과 콘솔 모두에 출력하도록 설정
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w')
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
            
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    sys.stdout = Logger(log_path)
    
    # duration 처리
    duration = None if args.duration == -1 else args.duration
    
    config_path = f'{args.save_path}/hydra/config.yaml'
    ckpt_path = f'{args.save_path}/pl_log/last.ckpt'
    model = BigCodecModel(ckpt_path, config_path)

    test_dataset = LibriTTSDataset(
        root=args.dataset_root,
        subsets=["dev-clean", "dev-other"],
        dataset_path=args.dataset_path,
        sample_rate=args.sample_rate,
        duration=duration,
        ext_audio=args.ext_audio,
        offset_mode="start",
        pad_to_stride=None,
    )

    print(f"테스트 데이터셋 크기: {len(test_dataset)}")
    test_loader = DataLoader(test_dataset, batch_size=32 if duration is not None else 1, shuffle=False, 
                           drop_last=False, num_workers=4, pin_memory=True)
    
    _ = train(model, None, test_loader, None, 
             sample_rate=args.sample_rate, 
             val_freq=1, 
             visualize_freq=1, 
             epochs=1,
             save_path=out_path)