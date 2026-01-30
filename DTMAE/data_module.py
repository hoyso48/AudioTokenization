import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import pytorch_lightning as pl
import random
import librosa
from os.path import basename, exists, join
from torch.utils.data import Dataset, DataLoader
import hydra
import utils
import torchaudio
from transformers import AutoFeatureExtractor
from torchaudio.transforms import Resample
from tqdm import tqdm

from hf_streaming_dataset import (
    LibriLightStreamingDataset,
    LibriLightStreamingSpec,
    MlsEngStreamingDataset,
    MlsEngStreamingSpec,
)

class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        ocwd = hydra.utils.get_original_cwd()
        self.ocwd = ocwd

    def get_loader(self, phase):
        phase_cfg = self.cfg.dataset.get(phase)
        batch_size = phase_cfg.batch_size
        backend = str(getattr(phase_cfg, "backend", "filelist"))
        dataloader_cfg = getattr(self.cfg.dataset, "dataloader", None)

        if backend == "hf_streaming_librilight":
            spec_cfg = getattr(self.cfg.dataset, "hf_streaming_librilight", None)
            if spec_cfg is None:
                raise ValueError(
                    "dataset.train.backend is hf_streaming_librilight but "
                    "dataset.hf_streaming_librilight config is missing."
                )
            spec = LibriLightStreamingSpec(
                config_name=str(getattr(spec_cfg, "config_name", "large")),
                split=str(getattr(spec_cfg, "split", "train")),
                hf_repo_id=str(getattr(spec_cfg, "hf_repo_id", "HugoLaurencon/libri_light")),
                shuffle=bool(getattr(phase_cfg, "shuffle", True)),
                shuffle_buffer_size=int(getattr(spec_cfg, "shuffle_buffer_size", 10_000)),
                seed=int(getattr(spec_cfg, "seed", 1024)),
                sample_rate=int(getattr(self.cfg.dataset, "sample_rate", 16000)),
                min_audio_length=int(getattr(phase_cfg, "min_audio_length", 64000)),
                multiple_of=int(getattr(self.cfg.dataset, "multiple_of", 320)),
                trust_remote_code=bool(getattr(spec_cfg, "trust_remote_code", True)),
                cache_dir=getattr(spec_cfg, "cache_dir", None),
                download_max_retries=int(getattr(spec_cfg, "download_max_retries", 50)),
                download_timeout=int(getattr(spec_cfg, "download_timeout", 60)),
            )
            ds = LibriLightStreamingDataset(phase=phase, spec=spec)
            collate_fn = _collate_wav_only
            # IterableDataset: DataLoader shuffle must be False
            shuffle = False
        elif backend == "hf_streaming_mls_eng":
            spec_cfg = getattr(self.cfg.dataset, "hf_streaming_mls_eng", None)
            if spec_cfg is None:
                raise ValueError(
                    "dataset.train.backend is hf_streaming_mls_eng but "
                    "dataset.hf_streaming_mls_eng config is missing."
                )
            transcript_cfg = getattr(self.cfg.dataset, "transcript", None)
            include_transcript = bool(getattr(transcript_cfg, "enable", False)) if transcript_cfg else False
            lowercase = bool(getattr(transcript_cfg, "lowercase", True)) if transcript_cfg else True

            spec = MlsEngStreamingSpec(
                hf_repo_id=str(getattr(spec_cfg, "hf_repo_id", "parler-tts/mls_eng")),
                split=str(getattr(spec_cfg, "split", "train")),
                shuffle=bool(getattr(phase_cfg, "shuffle", True)),
                shuffle_buffer_size=int(getattr(spec_cfg, "shuffle_buffer_size", 20_000)),
                seed=int(getattr(spec_cfg, "seed", 1024)),
                sample_rate=int(getattr(self.cfg.dataset, "sample_rate", 16000)),
                min_audio_length=int(getattr(phase_cfg, "min_audio_length", 64000)),
                multiple_of=int(getattr(self.cfg.dataset, "multiple_of", 320)),
                lowercase_transcript=lowercase,
                include_transcript=include_transcript,
                cache_dir=getattr(spec_cfg, "cache_dir", None),
                download_max_retries=int(getattr(spec_cfg, "download_max_retries", 50)),
                download_timeout=int(getattr(spec_cfg, "download_timeout", 60)),
            )
            ds = MlsEngStreamingDataset(phase=phase, spec=spec)
            collate_fn = _collate_wav_transcript if include_transcript else _collate_wav_only
            shuffle = False
        elif backend == "filelist":
            ds = FSDataset(phase, self.cfg)
            collate_fn = ds.collate_fn
            shuffle = bool(getattr(phase_cfg, "shuffle", False))
        else:
            raise ValueError(f"Unsupported dataset backend: {backend}")

        num_workers = int(getattr(dataloader_cfg, "num_workers", 28)) if dataloader_cfg else 28
        pin_memory = bool(getattr(dataloader_cfg, "pin_memory", True)) if dataloader_cfg else True
        persistent_workers = (
            bool(getattr(dataloader_cfg, "persistent_workers", True)) if dataloader_cfg else True
        )
        if num_workers == 0:
            persistent_workers = False

        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        return dl

    def train_dataloader(self):
        return self.get_loader('train')

    def val_dataloader(self):
        return self.get_loader('val')

    def test_dataloader(self):
        return self.get_loader('test')

class FSDataset(Dataset):
    """Dataset batching wav, mel 
    and other acoustic features

    Args:
        phase: train, val, test
        cfg: hydra config
    """
    def __init__(self, phase, cfg):
        self.phase = phase
        self.cfg = cfg
        self.phase_cfg = cfg.dataset.get(phase)
        self.ocwd = hydra.utils.get_original_cwd()
        self.transcript_cfg = getattr(cfg.dataset, 'transcript', None)
        self.use_transcript = self.transcript_cfg is not None and getattr(self.transcript_cfg, 'enable', False)
        self.lower_transcript = getattr(self.transcript_cfg, 'lowercase', True) if self.transcript_cfg else False
        
        self.sr = cfg.preprocess.audio.sr
        
        # self.filelist = utils.read_filelist(join(self.ocwd, self.phase_cfg.filelist))
        self.filelist = self.get_filelist(join(self.ocwd, self.phase_cfg.filelist))
        self.min_audio_length = self.phase_cfg.min_audio_length
        self.multiple_of = self.cfg.dataset.multiple_of
        # if self.cfg.train.use_semantic:
        #     self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

    def __len__(self):
        return len(self.filelist)

    def load_wav(self, path):
        wav, sr = librosa.load(path, sr=self.sr)
        return wav

    def get_filelist(self, fpath):
        with open(fpath, 'r') as f:
            # flist = [l.strip() for l in f if l.strip()]
            flist = [l.strip().split('\t')[0] for l in f if l.strip()]
        return flist

    def __getitem__(self, idx):
        # (  wavpath,fid) = self.filelist[idx]
        wavpath  = self.filelist[idx]
        if os.path.isabs(wavpath):
            wavpath_full = wavpath
        else:
            wavpath_full = join(self.cfg.preprocess.datasets.LibriSpeech.root, wavpath)
        # wav = self.load_wav(wavpath)
        # wav = torch.from_numpy(wav)
 
        wav, sr = torchaudio.load(wavpath_full)
                 
        if sr != self.cfg.dataset.sample_rate:
            wav = Resample(sr, self.cfg.dataset.sample_rate)(wav)
        wav = wav[0,:]
        length = wav.shape[0]
        # length = wav.shape[1]
        if self.min_audio_length != -1:
            l = self.min_audio_length
            if length < l:
                wav = F.pad(wav, (0, l - length))
                length = wav.shape[0]
            if self.phase == 'train':
                i = random.randint(0, length-l)
            else:
                i = 0
            l = (l // self.multiple_of) * self.multiple_of
            wav = wav[i:i+l]
        else:
            l = (length // self.multiple_of) * self.multiple_of
            wav = wav[:l]

        out = {
            'wav': wav,
            # 'paths': wavpath_full
        }
        if self.use_transcript:
            utt_id, transcript = self.load_transcript(wavpath_full)
            out['utt_id'] = utt_id
            out['transcript'] = transcript

        # if self.cfg.train.use_semantic:
        #     wav_pad = F.pad(wav, (160, 160))
        #     feat = self.feature_extractor(wav_pad, sampling_rate=16000, return_tensors="pt") .data['input_features']
        #     out['feat'] = feat
        
        return out
    
    def load_transcript(self, wavpath_full):
        fname = os.path.splitext(os.path.basename(wavpath_full))[0]
        parts = fname.split('-')
        if len(parts) < 3:
            raise ValueError(f'Unexpected filename format for LibriSpeech sample: {fname}')
        transcript_path = join(os.path.dirname(wavpath_full), f'{parts[0]}-{parts[1]}.trans.txt')
        if not exists(transcript_path):
            raise FileNotFoundError(f'Transcript file missing: {transcript_path}')

        target_prefix = f'{fname} '
        transcript = None
        with open(transcript_path, 'r') as f:
            for line in f:
                if line.startswith(target_prefix):
                    transcript = line[len(target_prefix):].strip()
                    break
        if transcript is None:
            raise ValueError(f'No transcript line found for {fname} in {transcript_path}')
        if self.lower_transcript:
            transcript = transcript.lower()
        return fname, transcript
    
    def collate_fn(self, bs):
 
        wavs = [b['wav'] for b in bs]
        wavs = torch.stack(wavs)
        out = {
            'wav': wavs,
            # 'paths': [b['paths'] for b in bs]
        }
        if self.use_transcript:
            out['utt_id'] = [b['utt_id'] for b in bs]
            out['transcript'] = [b['transcript'] for b in bs]
        # if self.cfg.train.use_semantic:
        #     feats = [b['feat'] for b in bs]
        #     feats = torch.stack(feats)
        #     out['feats'] = feats
        return out


def _collate_wav_only(bs):
    wavs = [b["wav"] for b in bs]
    wavs = torch.stack(wavs)
    return {"wav": wavs}


def _collate_wav_transcript(bs):
    wavs = [b["wav"] for b in bs]
    wavs = torch.stack(wavs)
    return {
        "wav": wavs,
        "utt_id": [b.get("utt_id", "") for b in bs],
        "transcript": [b.get("transcript", "") for b in bs],
    }

