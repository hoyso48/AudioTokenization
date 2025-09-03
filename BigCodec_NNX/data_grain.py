from pathlib import Path
from typing import List, Tuple, Union, Optional
import os
import numpy as np
import soundfile as sf
# import torch
import jax
from grain.sources import RandomAccessDataSource
import librosa

def load_audio(file_audio,     
    target_sample_rate: int = None,
    offset_mode: str = 'start',
    duration: Optional[float] = None,  # seconds
    ):
    # # torchaudio 방식
    # # Get total frames first
    # info = torchaudio.info(file_audio)
    # total_frames = info.num_frames
    # sample_rate = info.sample_rate
    # 
    # # Calculate frame offset and number of frames to load
    # frame_offset = int(offset * total_frames)
    # num_frames = -1 if duration is None else int(duration * sample_rate)
    # 
    # # Load audio with offset and duration
    # waveform, sample_rate = torchaudio.load(
    #     file_audio,
    #     frame_offset=frame_offset,
    #     num_frames=num_frames,
    #     normalize=True
    # )
    # waveform = torch.from_numpy(waveform)

    # # librosa 방식
    # # Load audio with offset and duration
    # duration_samples = None if duration is None else int(duration * target_sample_rate)
    # offset_samples = 0 if offset == 0 else int(offset * duration_samples) if duration else None
    # waveform, sample_rate = librosa.load(
    #     file_audio,
    #     sr=target_sample_rate,
    #     offset=offset,
    #     duration=duration,
    # )
    # waveform = torch.from_numpy(waveform).unsqueeze(0)

    # soundfile 방식
    info = sf.info(file_audio)
    sample_rate = info.samplerate
    total_frames = info.frames
    
    # Calculate frame offset and number of frames to load
    if offset_mode == 'start':
        frame_offset = 0
    elif offset_mode == 'random':
        if total_frames - int(duration * sample_rate) <= 0:
            frame_offset = 0
        else:
            frame_offset = np.random.randint(0, total_frames - int(duration * sample_rate))
    num_frames = -1 if duration is None else int(duration * sample_rate)
    
    # Load audio with offset and duration
    with sf.SoundFile(file_audio, 'r') as f:
        f.seek(frame_offset)
        frames_to_read = num_frames if num_frames != -1 else -1
        waveform = f.read(frames=frames_to_read)
        waveform = waveform.astype(np.float32)[None, :]
        # waveform = torch.from_numpy(waveform).float().unsqueeze(0)

    # gain = np.random.uniform(-1, -6) if offset_mode == 'random' else -3
    # waveform, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate, [["norm", f"{gain:.2f}"]])

    # Pad if duration is specified and waveform is shorter
    if duration is not None:
        target_length = int(duration * sample_rate)
        current_length = waveform.shape[1]
        if current_length < target_length:
            padding_length = target_length - current_length
            # waveform = torch.nn.functional.pad(waveform, (padding_length,0), mode='constant', value=0)
            waveform = np.pad(waveform, ((0, 0), (0, padding_length)), mode='constant', constant_values=0)
    # Resample if needed
    if target_sample_rate and target_sample_rate != sample_rate:
        # resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        # waveform = resampler(torch.from_numpy(waveform)).numpy()
        # sample_rate = target_sample_rate
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sample_rate)
        sample_rate = target_sample_rate

    return (
        waveform,
        sample_rate,
    )

def load_libritts_item(
    fileid: str,
    path: str,
    ext_audio: str,
    ext_original_txt: str,
    ext_normalized_txt: str,
    target_sample_rate: int = None,
    offset_mode: str = 'start',
    duration: Optional[float] = None,  # seconds
    # gain: float = -3.0,
) -> Tuple[np.ndarray, int, str, str, int, int, str]:
    speaker_id, chapter_id, segment_id, utterance_id = fileid.split("_")
    utterance_id = fileid
    file_audio = utterance_id + ext_audio
    file_audio = os.path.join(path, speaker_id, chapter_id, file_audio)

    return load_audio(file_audio, target_sample_rate, offset_mode, duration)

def load_librispeech_item(
    fileid: str,
    path: str,
    ext_audio: str,
    ext_original_txt: str,
    ext_normalized_txt: str,
    target_sample_rate: int = None,
    offset_mode: str = 'start',
    duration: Optional[float] = None,  # seconds
    # gain: float = -3.0,
) -> Tuple[np.ndarray, int, str, str, int, int, str]:
    speaker_id, chapter_id, segment_id = fileid.split("-")
    file_audio = fileid + ext_audio
    file_audio = os.path.join(path, speaker_id, chapter_id, file_audio)

    return load_audio(file_audio, target_sample_rate, offset_mode, duration)
    
class LibriTTSDataset(RandomAccessDataSource):
    """LibriTTS dataset with customizable audio duration and offset modes.
    
    Args:
        root (str or Path): Path to the directory where the dataset is found
        subsets (str or List[str]): Subset(s) to use
        sample_rate (int, optional): Target sample rate
        duration (float, optional): Fixed duration in seconds for all audio clips
        offset_mode (str): Either 'random' or 'start'. If 'random', picks random offset for each sample
    """
    
    _ext_original_txt = ".original.txt"
    _ext_normalized_txt = ".normalized.txt"
    _ext_audio = ".wav"
    
    def __init__(
        self,
        root: Union[str, Path],
        subsets: Union[str, List[str]],
        sample_rate: int = None,
        duration: Optional[float] = None,
        offset_mode: str = 'start',
        dataset_type: str = 'libritts',
    ) -> None:
        if offset_mode not in ['random', 'start']:
            raise ValueError("offset_mode must be either 'random' or 'start'")
        if dataset_type not in ['libritts', 'librispeech']:
            raise ValueError("dataset_type must be either 'libritts' or 'librispeech'")

        self.dataset_type = dataset_type
        self._ext_audio = '.wav' if dataset_type == 'libritts' else '.flac'
        # Convert single subset to list
        if isinstance(subsets, str):
            subsets = [subsets]
            
        # Validate subsets
        valid_subsets = {
            "dev-clean", "dev-other", "test-clean", "test-other",
            "train-clean-100", "train-clean-360", "train-other-500"
        }
        for subset in subsets:
            if subset not in valid_subsets:
                raise ValueError(f"Invalid subset '{subset}'. Must be one of {valid_subsets}")
        
        self.sample_rate = sample_rate
        self.duration = duration
        self.offset_mode = offset_mode
        
        # Collect all file paths
        self._walker = []
        root = os.fspath(root)
        for subset in subsets:
            path = os.path.join(root, subset)
            if not os.path.isdir(path):
                raise RuntimeError(f"Dataset not found at {path}")
            
            self._walker.extend([
                (path, str(p.stem)) 
                for p in Path(path).glob(f"*/*/*{self._ext_audio}")
            ])

    def __getitem__(self, n: int) -> Tuple[np.ndarray, int, str, str, int, int, str]:
        """Load the n-th sample from the dataset."""
        path, fileid = self._walker[n]
        if self.dataset_type == 'libritts':
            output = load_libritts_item(
                fileid,
                path,
                self._ext_audio,
                self._ext_original_txt,
                self._ext_normalized_txt,
                self.sample_rate,
                self.offset_mode,
                duration=self.duration,
                # gain=np.random.uniform(-1, -6) if self.offset_mode == 'random' else -3
                )
        else:
            output = load_librispeech_item(
                fileid,
                path,
                self._ext_audio,
                self._ext_original_txt,
                self._ext_normalized_txt,
                self.sample_rate,
                self.offset_mode,
                duration=self.duration,
            )
        return output

    def __len__(self) -> int:
        return len(self._walker)

# # PyTorch 텐서를 NumPy 배열로 변환하는 함수 (No change needed)
# def numpy_collate(batch):
#     return jax.tree_util.tree_map(np.asarray, default_collate(batch))

# # JAX와 호환되는 데이터로더 클래스 (No change needed)
# class NumpyLoader(DataLoader):
#     def __init__(self, dataset, batch_size=1,
#                 shuffle=False, sampler=None,
#                 batch_sampler=None, num_workers=0,
#                 pin_memory=False, drop_last=False,
#                 timeout=0, worker_init_fn=None):
#         super().__init__(dataset,
#             batch_size=batch_size,
#             shuffle=shuffle,
#             sampler=sampler,
#             batch_sampler=batch_sampler,
#             num_workers=num_workers,
#             collate_fn=numpy_collate,
#             pin_memory=pin_memory,
#             drop_last=drop_last,
#             timeout=timeout,
#             worker_init_fn=worker_init_fn)