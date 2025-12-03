import sys
import os
import math
from collections import defaultdict
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Union
from omegaconf import OmegaConf

# Add DTMAE to path to allow imports of vq, module, etc.
DTMAE_PATH = "/home/hoyso/projects/AudioTokenization/DTMAE"
if DTMAE_PATH not in sys.path:
    sys.path.append(DTMAE_PATH)

try:
    from lightning_module import CodecLightningModule
except ImportError as e:
    print(f"[DTMAE Upstream] Warning: Could not import CodecLightningModule: {e}")
    print(f"[DTMAE Upstream] sys.path: {sys.path}")

class UpstreamExpert(nn.Module):
    def __init__(self, ckpt: str = None, model_config: str = None, **kwargs):
        super().__init__()
        self.name = "[DTMAE Upstream]"
        
        if ckpt is None:
             raise ValueError("DTMAE requires a checkpoint path (ckpt).")
        
        print(f"[DTMAE Upstream] Loading model from {ckpt}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu":
            print("[DTMAE Upstream] CUDA not available. Running on CPU may crash because FlashAttention expects GPU tensors.")
        
        cfg = None
        if model_config is not None:
             print(f"[DTMAE Upstream] Loading config from {model_config}")
             try:
                 cfg = OmegaConf.load(model_config)
             except Exception as e:
                 print(f"[DTMAE Upstream] Error loading config file: {e}")
                 # Fallback: might rely on embedded hparams
        
        # Load model from checkpoint
        try:
            if cfg is not None:
                # Pass cfg to override or supply the config expected by __init__
                self.model = CodecLightningModule.load_from_checkpoint(ckpt, cfg=cfg, map_location=self.device)
            else:
                # Rely on embedded hparams if available, or default behavior
                self.model = CodecLightningModule.load_from_checkpoint(ckpt, map_location=self.device)
        except Exception as e:
            print(f"[DTMAE Upstream] Error loading checkpoint: {e}")
            print("[DTMAE Upstream] Tip: Ensure you provided --build_model.upstream_model_config if your model requires it.")
            raise e

        self.model.to(self.device)
        self.model.eval()
        dataset_cfg = getattr(self.model.cfg, "dataset", None)
        if dataset_cfg is not None and hasattr(dataset_cfg, "multiple_of"):
            self.multiple_of = dataset_cfg.multiple_of
        else:
            self.multiple_of = 1
        
        # Downsample rate
        # Config says Level 1 @ 100Hz, Level 2 @ 50Hz.
        # Input 16000Hz / 50Hz = 320.
        self._downsample_rate = 320

    def get_downsample_rates(self, key: str) -> int:
        return self._downsample_rate

    def forward(self, wavs: List[torch.Tensor]) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        wavs: List of 1D tensors (audio samples)
        """
        self.model.eval()
        
        # Check device from model
        device = self.device
        
        feats = [None] * len(wavs)
        valid_lengths = [None] * len(wavs)

        use_autocast = device.type == "cuda"
        if use_autocast:
            bf16_supported = torch.cuda.is_bf16_supported()
            autocast_dtype = torch.bfloat16 if bf16_supported else torch.float16
            autocast_cm = torch.autocast(device_type="cuda", dtype=autocast_dtype)
        else:
            autocast_cm = nullcontext()

        grouped = defaultdict(list)
        orig_lengths = []
        for idx, wav in enumerate(wavs):
            if wav.dim() == 1:
                w = wav.unsqueeze(0)
            elif wav.dim() == 2:
                w = wav[:1]
            else:
                w = wav
            length = w.size(-1)
            orig_lengths.append(length)
            pad_needed = (-length) % max(self.multiple_of, 1)
            if pad_needed > 0:
                w = F.pad(w, (0, pad_needed))
            grouped[w.size(-1)].append((idx, w))

        with torch.no_grad():
            for padded_len, items in grouped.items():
                batch = torch.stack([w for _, w in items], dim=0).to(device)
                with autocast_cm:
                    vq_emb = self.model.encoder(batch, level=1)

                    if self.model.use_dtp:
                        dtp_out = self.model.dtp(vq_emb)
                        if len(dtp_out) == 4:
                            mask, _, _, _ = dtp_out
                        else:
                            mask, _, _ = dtp_out
                        vq_emb, position_ids, cu_seqlens, max_seqlen = self.model.downsampler(
                            vq_emb, mask
                        )
                    else:
                        vq_emb = self.model.downsampler(vq_emb)
                        position_ids = cu_seqlens = max_seqlen = None

                    vq_emb = self.model.encoder(
                        vq_emb,
                        position_ids=position_ids,
                        cu_seqlens=cu_seqlens,
                        max_seqlen=max_seqlen,
                        level=2,
                    )

                for (sample_idx, _), sample_feat in zip(items, vq_emb):
                    valid_len = math.floor((orig_lengths[sample_idx] - 1) / self._downsample_rate) + 1
                    feats[sample_idx] = sample_feat[:valid_len].contiguous()
                    valid_lengths[sample_idx] = valid_len

        max_len = max(valid_lengths)
        hidden_dim = feats[0].size(-1)
        batch_hidden = feats[0].new_zeros(len(feats), max_len, hidden_dim)
        for idx, feat in enumerate(feats):
            length = valid_lengths[idx]
            batch_hidden[idx, :length] = feat

        return {
            "last_hidden_state": batch_hidden,
            "hidden_states": [batch_hidden],
        }
