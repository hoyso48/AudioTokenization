import os
import re
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import pytorch_lightning as pl
import random
import librosa
from os.path import basename, exists, join
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import hydra
import utils
from transformers import AutoFeatureExtractor
from torchaudio.transforms import Resample
from tqdm import tqdm

class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        ocwd = hydra.utils.get_original_cwd()
        self.ocwd = ocwd

    def get_loader(self, phase):
        phase_cfg = self.cfg.dataset.get(phase)
        batch_size = phase_cfg.batch_size
        ds = FSDataset(phase, self.cfg)
        # ds = FSDataset_add_STFT(phase, self.cfg)
        dl = DataLoader(ds, 
                        batch_size=batch_size,
                        shuffle=phase_cfg.shuffle,
                        num_workers=28,
                        collate_fn=ds.collate_fn,
                        pin_memory=True,
                        persistent_workers=True)

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
        
        self.sr = cfg.preprocess.audio.sr
        
        # self.filelist = utils.read_filelist(join(self.ocwd, self.phase_cfg.filelist))
        self.filelist = self.get_filelist(join(self.ocwd, self.phase_cfg.filelist))
        self.min_audio_length = self.phase_cfg.min_audio_length
        self.multiple_of = self.cfg.dataset.multiple_of
        # if self.cfg.train.use_semantic:
        #     self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

        bow_cfg = self.cfg.dataset.get('bow', {}) or {}
        self.use_bow = bool(bow_cfg.get('enable', False))
        self.bow_cfg = bow_cfg
        self.bow_vocab_size = bow_cfg.get('vocab_size')
        self.bow_labels = {}
        self.bow_missing = 0
        self.bow_meta = None

        if self.use_bow:
            label_path = bow_cfg.get('label_path')
            meta_path = bow_cfg.get('meta_path')
            if meta_path:
                meta_full = meta_path if os.path.isabs(meta_path) else join(self.ocwd, meta_path)
                self.bow_meta = self.load_bow_meta(meta_full)
                if self.bow_vocab_size is None:
                    self.bow_vocab_size = self.bow_meta.get('vocab_size')
                split_key = Path(self.phase_cfg.filelist).stem
                label_path = self.bow_meta.get('label_paths', {}).get(split_key, label_path)

            if label_path is None:
                raise ValueError("bow.enable=True but no label_path provided (set bow.label_path or bow.meta_path).")
            if not os.path.isabs(label_path):
                label_path = join(self.ocwd, label_path)

            if self.bow_vocab_size is None:
                raise ValueError("bow.vocab_size is required when bow is enabled (or provide meta_path with vocab_size).")

            self.bow_labels = self.load_bow_labels(label_path)

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

    def load_bow_meta(self, meta_path):
        with open(meta_path, 'r') as f:
            return json.load(f)

    def load_bow_labels(self, label_path):
        bow = {}
        with open(label_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                audio = entry.get('audio')
                if audio is None:
                    continue
                bow[str(Path(audio).resolve())] = entry.get('token_ids', [])
        return bow

    def __getitem__(self, idx):
        # (  wavpath,fid) = self.filelist[idx]
        wavpath  = self.filelist[idx]
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

        if self.use_bow:
            bow_vec = torch.zeros(self.bow_vocab_size, dtype=torch.float32)
            bow_entry = self.bow_labels.get(str(Path(wavpath_full).resolve()))
            if bow_entry is not None:
                bow_idx = torch.as_tensor(bow_entry, dtype=torch.long)
                if bow_idx.numel() > 0:
                    valid_mask = bow_idx < self.bow_vocab_size
                    bow_vec[bow_idx[valid_mask]] = 1.0
            else:
                self.bow_missing += 1
            out['bow'] = bow_vec

        # if self.cfg.train.use_semantic:
        #     wav_pad = F.pad(wav, (160, 160))
        #     feat = self.feature_extractor(wav_pad, sampling_rate=16000, return_tensors="pt") .data['input_features']
        #     out['feat'] = feat
        
        return out
    
    def collate_fn(self, bs):
 
        wavs = [b['wav'] for b in bs]
        wavs = torch.stack(wavs)
        out = {
            'wav': wavs,
            # 'paths': [b['paths'] for b in bs]
        }
        if self.use_bow:
            bows = [b['bow'] for b in bs]
            out['bow'] = torch.stack(bows)
        # if self.cfg.train.use_semantic:
        #     feats = [b['feat'] for b in bs]
        #     feats = torch.stack(feats)
        #     out['feats'] = feats
        return out


