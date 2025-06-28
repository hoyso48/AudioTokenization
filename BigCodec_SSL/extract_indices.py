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
from tqdm import tqdm
import gc
import random
# collections.Counter 는 인덱스 분석에 필요할 수 있으나, 현재 저장 로직에는 불필요하여 주석 처리 또는 제거 가능
# from collections import Counter

# --- Original Utility functions (Keep if needed, remove if unused) ---
# seed_everything 은 재현성을 위해 유지합니다.
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True # 성능 저하 가능성으로 주석 처리

# AverageMeter, AverageMeterDict, LRScheduler, get_cosine_decay_with_warmup 등 학습 관련 유틸리티는 제거합니다.
# PESQ, STOI 등 평가 관련 클래스도 제거합니다.
# plot_spectrograms_and_audio 함수도 제거합니다.


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


# --- BigCodecModel: Use original definition ---
# Import necessary modules from original context if they are in separate files
# Assuming CodecLightningModule is defined/imported correctly
import pytorch_lightning as pl
from omegaconf import OmegaConf
# If CodecLightningModule is in lightning_module.py:
# from lightning_module import CodecLightningModule

class BigCodecModel(nn.Module):
    def __init__(self, ckpt_path, config_path):
        super(BigCodecModel, self).__init__()

        # 설정 직접 로드 (original)
        cfg = OmegaConf.load(config_path)

        # Lightning 모듈 생성 (original)
        # Ensure CodecLightningModule class is available in the scope
        # Replace with actual import if needed: from lightning_module import CodecLightningModule
        # If CodecLightningModule is not defined, this will fail. Assuming it's available.
        try:
            # Try to import dynamically if not already imported
            if 'CodecLightningModule' not in globals():
                 from lightning_module import CodecLightningModule
            self.lm = CodecLightningModule(cfg=cfg)
        except ImportError:
             print("ERROR: Could not find or import 'CodecLightningModule'. Make sure it's defined or importable.")
             raise
        except NameError:
             print("ERROR: 'CodecLightningModule' is not defined. Make sure it's defined or imported.")
             raise


        # checkpoint 로드 (original)
        print(f"Loading state dict from: {ckpt_path}")
        # state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
        # Robust loading for different checkpoint formats
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)#torch.load(ckpt_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint: # Handle checkpoints saved directly from model
             state_dict = checkpoint['model']
        else:
             state_dict = checkpoint # Assume the whole file is the state_dict

        # 가중치 로드 (original)
        try:
            self.lm.load_state_dict(state_dict, strict=True) # Try strict loading first
            print("State dict loaded strictly.")
        except RuntimeError as e:
             print(f"Strict state_dict loading failed: {e}. Attempting non-strict loading.")
             self.lm.load_state_dict(state_dict, strict=False) # Fallback to non-strict

        # Get codebook size from config (original)
        try:
             # Adjust path based on actual config structure
             # Example: could be cfg.model.quantizer.codebook_size or similar
             self.codebook_size = cfg.model.generator.vq.codebook_size # Adjust this path based on your config.yaml!
             # Or maybe: self.codebook_size = cfg.model.quantizer.vq.codebook_size
             # Or: self.codebook_size = cfg.model.codec_decoder.codebook_size # From original code
             print(f"Determined codebook size: {self.codebook_size}")
        except AttributeError as e:
             print(f"Warning: Could not automatically determine codebook_size from config path cfg.model....codebook_size: {e}")
             print("Please verify the path in config.yaml and update the code if necessary.")
             # Try accessing from loaded model if possible (might be architecture dependent)
             if hasattr(self.lm, 'model') and 'generator' in self.lm.model and hasattr(self.lm.model['generator'], 'vq'):
                 self.codebook_size = self.lm.model['generator'].vq.num_codes # Example access
             else:
                  print("Could not access codebook size from model either. Setting placeholder 1.")
                  self.codebook_size = 1 # Placeholder


        # Set model to evaluation mode
        self.lm.eval()

    @torch.no_grad() # Ensure no gradients are computed
    def forward(self, x):
        # Forward pass using the structure assumed in the original BigCodecModel
        # This might need adjustment based on the *actual* structure of your CodecLightningModule
        try:
             # Try the structure from the original provided code first
             vq_emb = self.lm.model['CodecEnc'](x)
             # Assume generator returns multiple values, code is the second
             _, vq_code, _ = self.lm.model['generator'](vq_emb, vq=True)
        except (KeyError, AttributeError, TypeError) as e:
             print(f"Warning: Forward pass using original structure failed: {e}")
             print("Attempting alternative structure (e.g., self.lm.encode, self.lm.quantize)...")
             # Adapt this fallback based on your specific CodecLightningModule implementation
             try:
                  encoded_features = self.lm.encode(x) # Example: if lm has encode method
                  # Example: if lm has quantize method that returns codes
                  _, vq_code, _, _, _ = self.lm.quantize(encoded_features)
                  # Adjust indices based on what quantize returns
             except AttributeError as e2:
                  print(f"Alternative forward pass failed: {e2}")
                  print("ERROR: Could not determine how to get indices from the loaded model.")
                  raise NotImplementedError("Model forward pass for index extraction needs adjustment.") from e2

        # Return only indices
        return {'indices': vq_code}


# --- Main execution block: Minimal changes, use original args where possible ---
if __name__ == "__main__":
    import argparse

    # Use original parser setup and add only --subsets
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default="../../datasets", help="Path to the dataset")
    parser.add_argument('--save_path', type=str, required=True, help="Path containing model checkpoint and hydra config")
    parser.add_argument('--output_folder', type=str, default='extracted_indices', help="Folder within --save_path to save index files")
    parser.add_argument('--duration', type=float, default=None, help="Duration of audio segments in seconds (None for full files). Original default was 1.") # Adjusted default to None
    parser.add_argument('--sample_rate', type=int, default=16000, help="Target sample rate for audio loading")
    parser.add_argument('--dataset_path', type=str, default="LibriTTS", help="Path to the dataset")
    parser.add_argument('--ext_audio', type=str, default=".flac", help="Audio file extension")
    # --- NEW ARGUMENT ---
    parser.add_argument('--subsets', type=str, nargs='+', required=True, help="List of dataset subsets to process (e.g., test-clean dev-other)")
    # --- Removed arguments: dataset_root, batch_size, num_workers, device, seed, pad_to_stride ---

    args = parser.parse_args()
    print(args)

    # --- Use hardcoded or derived values instead of removed args ---
    seed = 42
    dataset_root = args.dataset_root # Original hardcoded path
    batch_size = 1 # Force batch size 1 for individual saving
    num_workers = 4 # Original hardcoded value
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pad_to_stride = None # Original value used in main block for test_loader

    seed_everything(seed)
    print(f"Using Device: {device}")
    print(f"Using Dataset Root: {dataset_root}")
    print(f"Using Num Workers: {num_workers}")
    print(f"Using Batch Size: {batch_size}")


    # --- Prepare paths (similar to previous version) ---
    if not os.path.isdir(args.save_path):
        print(f"Error: Model save path does not exist: {args.save_path}")
        sys.exit(1)

    output_dir = os.path.join(args.save_path, args.output_folder)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory for indices: {output_dir}")

    # Log file setup (optional, kept from original request structure)
    log_path = os.path.join(output_dir, f'log_extraction_{"_".join(args.subsets)}.txt')
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
    # Redirect stdout only if logging is desired
    # sys.stdout = Logger(log_path)
    # print(f"Logging to: {log_path}")


    # --- Load Model (similar to previous version) ---
    config_path = os.path.join(args.save_path, 'hydra/config.yaml')
    ckpt_path_options = [
        os.path.join(args.save_path, 'pl_log/last.ckpt'),
        os.path.join(args.save_path, 'checkpoints/last.ckpt'), # Common structure
        os.path.join(args.save_path, 'pl_log/checkpoints/last.ckpt'),
        os.path.join(args.save_path, 'last.ckpt'),
    ]
    ckpt_path = None
    for option in ckpt_path_options:
        if os.path.exists(option):
            ckpt_path = option
            break

    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    if ckpt_path is None:
        print(f"Error: Checkpoint file not found in expected locations: {ckpt_path_options}")
        sys.exit(1)

    print(f"Loading model config from: {config_path}")
    print(f"Loading model checkpoint from: {ckpt_path}")
    model = BigCodecModel(ckpt_path, config_path)
    model = model.to(device)
    model.eval()

    # --- Prepare Dataset and DataLoader ---
    print(f"Loading dataset from root: {dataset_root}")
    print(f"Processing subsets: {args.subsets}")
    dataset = LibriTTSDataset(
        root=dataset_root,
        subsets=args.subsets,
        dataset_path=args.dataset_path,
        ext_audio=args.ext_audio,
        sample_rate=args.sample_rate,
        duration=args.duration,
        offset_mode="start", # Use 'start' for consistency
        pad_to_stride=pad_to_stride, # Use None as per original test setup
    )

    if len(dataset) == 0:
         print("Error: Dataset is empty. Check dataset root, subset names, and file structure.")
         sys.exit(1)
    print(f"Dataset size: {len(dataset)}")

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size, # Must be 1
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True, # Keep pin_memory
        drop_last=False
    )

    # --- Extraction Loop (same as previous correct version) ---
    print("Starting index extraction...")
    pbar = tqdm(data_loader, desc="Extracting Indices", total=len(data_loader))
    count = 0
    error_count = 0

    for batch in pbar:
        try:
            # Data loader returns: waveform, sample_rate, subset, fileid
            waveform, _, subset, fileid = batch # Unpack batch (size 1)

            # Get the single element from the batch lists/tensors
            waveform = waveform[0].to(device)
            subset = subset[0]
            fileid = fileid[0]

            # --- Get Indices ---
            with torch.no_grad():
                # Add batch dimension back for model input
                output = model(waveform.unsqueeze(0))

            if 'indices' not in output or output['indices'] is None:
                 print(f"Warning: No indices found for file: {fileid}. Skipping.")
                 error_count += 1
                 continue

            indices = output['indices'] # Shape (N_q, B, N) or (B, N) -> with B=1 -> (N_q, 1, N) or (1, N)

            # --- Process Indices ---
            indices = indices.squeeze(1) # Remove batch dim -> (N_q, N) or (N,)

            if indices.ndim == 2: # Shape (N_q, N)
                indices = indices.permute(1, 0) # Transpose to (N, N_q)
            elif indices.ndim == 1: # Shape (N,) - already correct
                pass
            else:
                 print(f"Warning: Unexpected indices dimension: {indices.ndim} for file: {fileid}. Skipping.")
                 error_count += 1
                 continue

            # Convert to numpy int16
            indices_np = indices.cpu().numpy().astype(np.int16)

            # --- Determine Output Path and Save ---
            try:
                # Parse fileid to get structure (handle both separators)
                if "_" in fileid:
                    parts = fileid.split("_")
                    speaker_id, chapter_id = parts[0], parts[1]
                elif "-" in fileid:
                    parts = fileid.split("-")
                    speaker_id, chapter_id = parts[0], parts[1]
                else:
                     # Fallback if no separator found
                     print(f"Warning: Could not determine speaker/chapter from fileid '{fileid}'. Using 'unknown'.")
                     speaker_id, chapter_id = "unknown", "unknown"

            except IndexError:
                 print(f"Warning: Could not parse speaker/chapter from fileid '{fileid}'. Using 'unknown'.")
                 speaker_id, chapter_id = "unknown", "unknown"


            relative_dir = os.path.join(subset, speaker_id, chapter_id)
            output_subdir = os.path.join(output_dir, relative_dir)
            os.makedirs(output_subdir, exist_ok=True)

            output_filename = f"{fileid}.npy"
            output_filepath = os.path.join(output_subdir, output_filename)

            # Save the numpy array
            np.save(output_filepath, indices_np)
            count += 1
            pbar.set_postfix({"Saved": count, "Errors": error_count, "Last File": fileid})

        except FileNotFoundError as e:
            print(f"\nSkipping item due to FileNotFoundError: {e}")
            error_count += 1
            pbar.set_postfix({"Saved": count, "Errors": error_count})
        except Exception as e:
            print(f"\nError processing batch item (fileid might be {fileid if 'fileid' in locals() else 'unknown'}): {e}")
            import traceback
            traceback.print_exc()
            error_count += 1
            pbar.set_postfix({"Saved": count, "Errors": error_count})


    print(f"\nExtraction complete.")
    print(f"Successfully saved {count} index files.")
    if error_count > 0:
        print(f"Encountered {error_count} errors.")
    print(f"Indices saved in: {output_dir}")

    # Cleanup
    del model
    del dataset
    del data_loader
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    # Restore stdout if it was redirected
    # if isinstance(sys.stdout, Logger):
    #      sys.stdout.log.close()
    #      sys.stdout = sys.stdout.terminal