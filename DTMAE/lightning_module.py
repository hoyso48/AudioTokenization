import os
import random
import inspect
import hydra
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import torchmetrics
from torchmetrics.text import CharErrorRate, WordErrorRate
import torchaudio

try:
    import wandb
except Exception:
    wandb = None
from vq import TransformerEncoderSTFT, TransformerDecoderISTFT
from module import HiFiGANMultiPeriodDiscriminator, SpecDiscriminator
from criterions import GANLoss, MultiResolutionMelSpectrogramLoss
from common.schedulers import WarmupLR
from transformers import LlamaForCausalLM, LlamaConfig
from transformers import AutoModel
from vq.module import Downsample,Upsample,Upsample1D,ConvDownsample,ConvUpsample
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
from torchmetrics.audio import ShortTimeObjectiveIntelligibility, PerceptualEvaluationSpeechQuality, ScaleInvariantSignalNoiseRatio, ScaleInvariantSignalDistortionRatio
from torchmetrics.aggregation import MeanMetric
from pesq import NoUtterancesError
import dtp.ops
import dtp.resampler

class CodebookPerplexity(torchmetrics.Metric):
    def __init__(self, codebook_size, **kwargs):
        super().__init__(**kwargs)
        self.codebook_size = codebook_size
        self.add_state("codebook_counts", default=torch.zeros(codebook_size), dist_reduce_fx="sum")
        self.add_state("total_counts", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, indices: torch.Tensor) -> None:
        one_hot = F.one_hot(indices.long().flatten(), num_classes=self.codebook_size).sum(dim=0)
        self.codebook_counts += one_hot
        self.total_counts += one_hot.sum()

    def compute(self) -> torch.Tensor:
        if self.total_counts == 0:
            return torch.tensor(0.0, device=self.total_counts.device)

        probs = self.codebook_counts / self.total_counts
        
        nonzero_probs = probs[probs > 0]
        entropy = -torch.sum(nonzero_probs * torch.log(nonzero_probs))
        
        # Raw perplexity
        perplexity = torch.exp(entropy)
        
        return perplexity
    
class CodebookUtilization(torchmetrics.Metric):
    """
    Calculates the percentage of the codebook that has been utilized.

    This version is optimized to only track which codes have been used,
    rather than their full counts, making it more memory and
    computationally efficient.
    """
    def __init__(self, codebook_size, **kwargs):
        super().__init__(**kwargs)
        self.codebook_size = codebook_size
        self.add_state("used_codes", default=torch.zeros(codebook_size, dtype=torch.bool), dist_reduce_fx="max")

    def update(self, indices: torch.Tensor) -> None:
        """Marks the codes present in the input indices tensor as 'used'."""
        self.used_codes[indices.flatten()] = True

    def compute(self) -> torch.Tensor:
        """Computes the final utilization ratio."""
        used_count = torch.sum(self.used_codes)
        return used_count / self.codebook_size

class CodecLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # self.ocwd = hydra.utils.get_original_cwd()
        self.construct_model()
        self.construct_criteria()
        self.val_metrics = self.construct_metrics(prefix='val_')
        self.test_metrics = self.construct_metrics(prefix='test_')
        self.construct_asr_probe()
        self.save_hyperparameters()
        self.automatic_optimization = False

    @staticmethod
    def _set_trainable(module, trainable):
        module.requires_grad_(bool(trainable))

    def construct_model(self):
        enccfg = self.cfg.model.codec_encoder
        quantizer_cfg = self.cfg.model.quantizer
        encoder_trainable = bool(getattr(enccfg, "trainable", True))
        decoder_trainable = bool(getattr(self.cfg.model.codec_decoder, "trainable", True))
        quantizer_trainable = bool(getattr(quantizer_cfg, "trainable", True))
        self.encoder = TransformerEncoderSTFT(
            hop_length=enccfg.hop_length,
            n_fft=enccfg.n_fft,
            window_size=enccfg.window_size,
            dim=enccfg.dim,
            n_layers_level1=enccfg.n_layers_level1,
            n_layers_level2=enccfg.n_layers_level2,
            r=enccfg.r,
            n_head=enccfg.n_head,
            ffn_mult=enccfg.ffn_mult,
            dropout=enccfg.dropout,
            max_position_embeddings=enccfg.max_position_embeddings,
            base=enccfg.base,
            causal=enccfg.causal,
            out_channels=enccfg.out_channels,
            norm_eps=enccfg.norm_eps,
            attn_window_size=enccfg.attn_window_size,
            layerscale_gamma_init=float(getattr(enccfg, "layerscale_gamma_init", 1.0)),
            trainable=encoder_trainable,
        )
        self._set_trainable(self.encoder, encoder_trainable)

        deccfg = self.cfg.model.codec_decoder
        self.decoder = TransformerDecoderISTFT(
                in_channels=deccfg.in_channels,
                hop_length=deccfg.hop_length,
                n_fft=deccfg.n_fft,
                window_size=deccfg.window_size,
                dim=deccfg.dim,
                n_layers_level1=deccfg.n_layers_level1,
                n_layers_level2=deccfg.n_layers_level2,
                r=deccfg.r,
                n_head=deccfg.n_head,
                ffn_mult=deccfg.ffn_mult,
                dropout=deccfg.dropout,
                max_position_embeddings=deccfg.max_position_embeddings,
                base=deccfg.base,
                causal=deccfg.causal,
                # Extensible quantizer config (like resampler pattern)
                quantizer_cls=quantizer_cfg.cls,
                quantizer_params=dict(quantizer_cfg.params),
                trainable=decoder_trainable,
                quantizer_trainable=quantizer_trainable,
                # Legacy quantizer params (kept for reference):
                # fsq=deccfg.fsq,
                # fsq_levels=deccfg.fsq_levels,
                # vq_num_quantizers=deccfg.vq_num_quantizers,
                # vq_commit_weight=deccfg.vq_commit_weight,
                # vq_weight_init=deccfg.vq_weight_init,
                # vq_full_commit_loss=deccfg.vq_full_commit_loss,
                # codebook_size=deccfg.codebook_size,
                # codebook_dim=deccfg.codebook_dim,
                norm_eps=deccfg.norm_eps,
                attn_window_size=deccfg.attn_window_size,
                layerscale_gamma_init=float(getattr(deccfg, "layerscale_gamma_init", 1.0)),
        )
        self._set_trainable(self.decoder, decoder_trainable)
        if hasattr(self.decoder, 'quantizer'):
            self._set_trainable(self.decoder.quantizer, quantizer_trainable)

        mpdcfg = self.cfg.model.mpd
        self.discriminator = HiFiGANMultiPeriodDiscriminator(
                    periods=mpdcfg.periods,
                    max_downsample_channels=mpdcfg.max_downsample_channels,
                    channels=mpdcfg.channels,
                    channel_increasing_factor=mpdcfg.channel_increasing_factor,
                    use_weight_norm=mpdcfg.use_weight_norm,
                )
        mstftcfg = self.cfg.model.mstft
        self.spec_discriminator = SpecDiscriminator(
                    stft_params=mstftcfg.stft_params,
                    in_channels=mstftcfg.in_channels,
                    out_channels=mstftcfg.out_channels,
                    kernel_sizes=mstftcfg.kernel_sizes,
                    channels=mstftcfg.channels,
                    max_downsample_channels=mstftcfg.max_downsample_channels,
                    downsample_scales=mstftcfg.downsample_scales,
                    use_weight_norm=mstftcfg.use_weight_norm,
                )

        resamplercfg = self.cfg.model.resampler
        self.use_dtp = resamplercfg.use_dtp

        if self.use_dtp:
            self.dtp = getattr(dtp.ops, resamplercfg.dtp_cls)(**resamplercfg.dtp_params)

        downsampler_cls = getattr(dtp.resampler, resamplercfg.downsampler_cls)
        upsampler_cls = getattr(dtp.resampler, resamplercfg.upsampler_cls)
        downsampler_params = dict(getattr(resamplercfg, "downsampler_params", {}))
        upsampler_params = dict(getattr(resamplercfg, "upsampler_params", {}))

        self.downsampler = downsampler_cls(**downsampler_params)
        self.upsampler = upsampler_cls(**upsampler_params)
        self.fixed_pattern_masking = None
        self.upsampler_uses_mask = "mask" in inspect.signature(self.upsampler.forward).parameters

        if (not self.use_dtp) and self.upsampler_uses_mask:
            fixed_mask_params = dict(getattr(resamplercfg, "dtp_params", {}))
            self.fixed_pattern_masking = dtp.resampler.FixedPatternMasking(**fixed_mask_params)

    def construct_asr_probe(self):
        # CTC probe is optional and should not backprop into encoder
        self.use_asr_probe = bool(getattr(self.cfg.train, "use_asr_probe", False)) and getattr(self.cfg.dataset, "transcript", None) and getattr(self.cfg.dataset.transcript, "enable", False)
        if not self.use_asr_probe:
            self.asr_head = None
            return
        feature_dim = self.cfg.model.codec_decoder.in_channels
        # Simple lowercase character set with space and apostrophe; index 0 is blank
        vocab_list = ["<blank>"] + ["'"] + [" "] + [chr(i) for i in range(ord('a'), ord('z') + 1)]
        self.asr_vocab = vocab_list
        self.blank_id = 0
        self.asr_token_map = {ch: idx for idx, ch in enumerate(self.asr_vocab)}
        self.asr_head = nn.Linear(feature_dim, len(self.asr_vocab))
        self.ctc_loss = nn.CTCLoss(blank=self.blank_id, zero_infinity=True)
        # Metrics for validation/test
        self.val_cer = CharErrorRate()
        self.val_wer = WordErrorRate()
        self.test_cer = CharErrorRate()
        self.test_wer = WordErrorRate()

    def construct_criteria(self):
        cfg = self.cfg.train
        self.criteria = nn.ModuleDict()
        if cfg.use_mel_loss:
            self.criteria['mel_loss'] = MultiResolutionMelSpectrogramLoss(sample_rate=self.cfg.dataset.sample_rate)
        if cfg.use_stft_loss:
            self.criteria['stft_loss'] = MultiResolutionSTFTLoss(
                fft_sizes=cfg.stft_loss_params.fft_sizes,
                hop_sizes=cfg.stft_loss_params.hop_sizes,
                win_sizes=cfg.stft_loss_params.win_lengths
            )
        if cfg.use_feat_match_loss:
            self.criteria['fm_loss'] = nn.L1Loss()
        self.criteria['gan_loss'] = GANLoss()
        self.criteria['l1_loss'] = nn.L1Loss()
        self.criteria['l2_loss'] = nn.MSELoss()
        print(self.criteria)

    def construct_metrics(self, prefix=''):
        metrics = {}
        metrics['stoi'] = ShortTimeObjectiveIntelligibility(fs=16000,extended=False)
        metrics['pesq'] = PerceptualEvaluationSpeechQuality(fs=16000,mode='wb')
        metrics['si_snr'] = ScaleInvariantSignalNoiseRatio()
        metrics['si_sdr'] = ScaleInvariantSignalDistortionRatio()
        # Calculate codebook_size based on quantizer type
        quantizer_params = self.cfg.model.quantizer.params
        if 'codebook_size' in quantizer_params:
            # ResidualVQ, FSQ, SimVQ
            codebook_size = quantizer_params.codebook_size
        elif 'inference_levels' in quantizer_params:
            # DitheredFSQ: inference_levels can be int or list
            inf_levels = quantizer_params.inference_levels
            if isinstance(inf_levels, (list, tuple)) or hasattr(inf_levels, '__iter__') and not isinstance(inf_levels, (int, str)):
                # List: product of all levels
                codebook_size = 1
                for L in inf_levels:
                    codebook_size *= L
            else:
                # Int: level^codebook_dim
                codebook_size = int(inf_levels) ** quantizer_params.codebook_dim
        elif 'train_levels' in quantizer_params and 'codebook_dim' in quantizer_params:
            # DitheredFSQ (inference_levels defaults to max(train_levels))
            codebook_size = max(quantizer_params.train_levels) ** quantizer_params.codebook_dim
        elif 'levels' in quantizer_params:
            # TAAEDitheredFSQ: explicit per-dimension levels
            codebook_size = 1
            for L in quantizer_params.levels:
                codebook_size *= L
        else:
            codebook_size = 16384  # fallback default
        metrics['codebook_perplexity'] = CodebookPerplexity(codebook_size=codebook_size)
        metrics['codebook_utilization'] = CodebookUtilization(codebook_size=codebook_size)
        if self.use_dtp:
            metrics['avg_r'] = MeanMetric()
        return torchmetrics.MetricCollection(prefix=prefix, metrics=metrics)
    
    @torch.compile
    def forward(self, batch):
        wav = batch['wav']
        vq_emb = self.encoder(wav.unsqueeze(1), level=1)

        if self.use_dtp:
            mask, avg_r, tau_used = self.dtp(vq_emb)
            downsample_out = self.downsampler(vq_emb, mask)
            if isinstance(downsample_out, tuple):
                if len(downsample_out) == 5:
                    vq_emb, position_ids, cu_seqlens, max_seqlen, mask = downsample_out
                elif len(downsample_out) == 4:
                    vq_emb, position_ids, cu_seqlens, max_seqlen = downsample_out
                else:
                    raise ValueError("downsampler must return tensor, 4-tuple, or 5-tuple")
            else:
                vq_emb = downsample_out
                position_ids = cu_seqlens = max_seqlen = None
        else:
            avg_r = tau_used = None
            if self.fixed_pattern_masking is not None:
                mask = self.fixed_pattern_masking(vq_emb)
            else:
                mask = None
            vq_emb = self.downsampler(vq_emb)
            position_ids = cu_seqlens = max_seqlen = None

        vq_emb = self.encoder(vq_emb, position_ids=position_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, level=2)

        vq_post_emb, vq_code, vq_loss = self.decoder(vq_emb, vq=True)

        vq_post_emb = self.decoder(vq_post_emb, vq=False, position_ids=position_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, level=2)

        if mask is not None:
            vq_post_emb = self.upsampler(vq_post_emb, mask=mask)
        else:
            vq_post_emb = self.upsampler(vq_post_emb)

        y_ = self.decoder(vq_post_emb, vq=False, level=1) # [B, 1, T]
        y = wav.unsqueeze(1)
        
        output = {
            'gt_wav': y,
            'gen_wav': y_,
            'vq_loss': vq_loss,
            'vq_code': vq_code,
            'avg_r': avg_r,
            'tau_used': tau_used,
            'pre_quant_emb': vq_emb,  # used for ASR probe (gradients detached later)
            'cu_seqlens': cu_seqlens,
            'max_seqlen': max_seqlen,
        }
        return output
    
    @torch.inference_mode()
    def inference(self, wav):
        vq_emb = self.encoder(wav.unsqueeze(1))
        vq_post_emb, vq_code, vq_loss = self.decoder(vq_emb, vq=True)
        y_ = self.decoder(vq_post_emb, vq=False).squeeze(1)  # [B, T]
        return y_

    @torch.compile
    def compute_disc_loss(self, batch, output):
        y, y_ = output['gt_wav'], output['gen_wav']
        y_ = y_.detach()
        p = self.discriminator(y)
        p_ = self.discriminator(y_)

        real_loss_list, fake_loss_list = [], []
        for i in range(len(p)):
            real_loss, fake_loss = self.criteria['gan_loss'].disc_loss(p[i][-1], p_[i][-1])
            real_loss_list.append(real_loss)
            fake_loss_list.append(fake_loss)

        if hasattr(self, 'spec_discriminator'):
            sd_p = self.spec_discriminator(y)
            sd_p_ = self.spec_discriminator(y_)

            for i in range(len(sd_p)):
                real_loss, fake_loss = self.criteria['gan_loss'].disc_loss(sd_p[i][-1], sd_p_[i][-1])
                real_loss_list.append(real_loss)
                fake_loss_list.append(fake_loss)

        real_loss = sum(real_loss_list)
        fake_loss = sum(fake_loss_list)

        disc_loss = real_loss + fake_loss
        disc_loss = self.cfg.train.lambdas.lambda_disc * disc_loss

        output = {
            'real_loss': real_loss,
            'fake_loss': fake_loss,
            'disc_loss': disc_loss,
        }
        return output
    
    @torch.compile
    def compute_gen_loss(self, batch, output):
        y, y_ = output['gt_wav'], output['gen_wav']
        vq_loss, vq_code = output['vq_loss'], output['vq_code']
        # perceptual_se_loss_l2 = output['perceptual_se_loss_l2']
        # x_feat_recon_loss = output['x_feat_recon_loss']
        gen_loss = 0.0
        self.set_discriminator_gradients(False)
        output_dict = {}
        cfg = self.cfg.train

        # Mel spectrogram loss
        if cfg.use_mel_loss:
            mel_loss = self.criteria['mel_loss'](y_.squeeze(1), y.squeeze(1))
            gen_loss += mel_loss * cfg.lambdas.lambda_mel_loss
            output_dict['mel_loss'] = mel_loss

        # GAN loss
        p_ = self.discriminator(y_)
        adv_loss_list = []
        for i in range(len(p_)):
            adv_loss_list.append(self.criteria['gan_loss'].gen_loss(p_[i][-1]))
        if hasattr(self, 'spec_discriminator'):
            sd_p_ = self.spec_discriminator(y_)
            for i in range(len(sd_p_)):
                adv_loss_list.append(self.criteria['gan_loss'].gen_loss(sd_p_[i][-1]))
        adv_loss = sum(adv_loss_list)
        gen_loss += adv_loss * cfg.lambdas.lambda_adv
        output_dict['adv_loss'] = adv_loss

        # Feature Matching loss
        if cfg.use_feat_match_loss:
            fm_loss = 0.0
            with torch.no_grad():
                p = self.discriminator(y)
            for i in range(len(p_)):
                for j in range(len(p_[i]) - 1):
                    fm_loss += self.criteria['fm_loss'](p_[i][j], p[i][j].detach())
            gen_loss += fm_loss * cfg.lambdas.lambda_feat_match_loss
            output_dict['fm_loss'] = fm_loss
            if hasattr(self, 'spec_discriminator'):
                spec_fm_loss = 0.0
                with torch.no_grad():
                    sd_p = self.spec_discriminator(y)
                for i in range(len(sd_p_)):
                    for j in range(len(sd_p_[i]) - 1):
                        spec_fm_loss += self.criteria['fm_loss'](sd_p_[i][j], sd_p[i][j].detach())
                gen_loss += spec_fm_loss * cfg.lambdas.lambda_feat_match_loss
                output_dict['spec_fm_loss'] = spec_fm_loss

        # VQ loss
        if vq_loss is not None:
            vq_loss = sum(vq_loss)
            gen_loss += vq_loss
            output_dict['vq_loss'] = vq_loss
        
        if 'entropy_loss' in output:
            gen_loss += output['entropy_loss']
            output_dict['entropy_loss'] = output['entropy_loss']

        # Perceptual loss
        # output_dict['perceptual_se_loss_l2'] = perceptual_se_loss_l2
        # gen_loss += output_dict['perceptual_se_loss_l2'] * cfg.lambdas.lambda_perceptual_loss
        
        self.set_discriminator_gradients(True)
        output_dict['gen_loss'] = gen_loss
        return output_dict

    def text_to_int(self, text):
        # Converts transcript string to list of ints (CTC targets)
        return [self.asr_token_map[ch] for ch in text if ch in self.asr_token_map]

    def int_to_text(self, indices):
        return ''.join([self.asr_vocab[i] for i in indices if i != self.blank_id])

    def ctc_greedy_decode(self, log_probs, input_lengths):
        # log_probs: (T, B, V)
        preds = log_probs.argmax(dim=-1)  # (T, B)
        preds = preds.transpose(0, 1)     # (B, T)
        decoded = []
        for b, length in enumerate(input_lengths.tolist()):
            prev = self.blank_id
            tokens = []
            for t in range(length):
                idx = preds[b, t].item()
                if idx != self.blank_id and idx != prev:
                    tokens.append(idx)
                prev = idx
            decoded.append(self.int_to_text(tokens))
        return decoded

    def compute_asr_probe(self, batch, output):
        if not self.use_asr_probe or 'transcript' not in batch:
            return None

        transcripts = batch['transcript']
        if len(transcripts) == 0:
            return None

        feats = output['pre_quant_emb'].detach()
        cu_seqlens = output.get('cu_seqlens')
        max_seqlen = output.get('max_seqlen')

        if cu_seqlens is not None and max_seqlen is not None:
            lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)
            seqs = torch.split(feats, lengths.tolist())
            padded = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)  # (B, T, C)
            input_lengths = lengths
        else:
            padded = feats  # (B, T, C)
            input_lengths = torch.full((padded.size(0),), padded.size(1), device=padded.device, dtype=torch.long)

        logits = self.asr_head(padded)
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T, B, V)

        targets = [torch.tensor(self.text_to_int(t), device=log_probs.device, dtype=torch.long) for t in transcripts]
        target_lengths = torch.tensor([len(t) for t in targets], device=log_probs.device, dtype=torch.long)

        # Skip if any target is empty (CTC cannot handle zero-length)
        if torch.any(target_lengths == 0):
            return None

        flat_targets = torch.cat(targets)
        loss = self.ctc_loss(log_probs, flat_targets, input_lengths, target_lengths)

        decoded = self.ctc_greedy_decode(log_probs.detach(), input_lengths)
        return {
            'ctc_loss': loss,
            'predicted_text': decoded,
            'target_text': transcripts,
        }
    
    @torch.compile
    def training_step(self, batch, batch_idx):
        output = self(batch)
        
        gen_opt, disc_opt = self.optimizers()
        gen_sche, disc_sche = self.lr_schedulers()
        
        # discriminator 
        disc_losses = self.compute_disc_loss(batch, output)
        disc_loss = disc_losses['disc_loss']
        disc_opt.zero_grad()
        self.manual_backward(disc_loss)
        self.clip_gradients(disc_opt, gradient_clip_val=self.cfg.train.disc_grad_clip, gradient_clip_algorithm='norm')
        disc_opt.step()
        disc_sche.step()

        # generator
        gen_losses = self.compute_gen_loss(batch, output)
        gen_loss = gen_losses['gen_loss']

        # ASR probe (no encoder gradients; features detached inside compute_asr_probe)
        asr_out = self.compute_asr_probe(batch, output)
        if asr_out is not None:
            lambda_ctc = float(getattr(self.cfg.train.lambdas, "lambda_ctc", 0.0))
            gen_loss = gen_loss + asr_out['ctc_loss'] * lambda_ctc
            gen_losses['ctc_loss'] = asr_out['ctc_loss']

        gen_opt.zero_grad()
        self.manual_backward(gen_loss)
        self.clip_gradients(gen_opt, gradient_clip_val=self.cfg.train.gen_grad_clip, gradient_clip_algorithm='norm')
        gen_opt.step()
        gen_sche.step()

        self.log_dict(disc_losses, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.cfg.dataset.train.batch_size, sync_dist=True)
        self.log_dict(gen_losses, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.cfg.dataset.train.batch_size, sync_dist=True)        
        if self.use_dtp:
            self.log_dict({'train_avg_r': output['avg_r'], 'train_tau_used': output['tau_used']}, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.cfg.dataset.train.batch_size, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            output = self(batch)
        y, y_, vq_code = output['gt_wav'], output['gen_wav'].float(), output['vq_code']
        rs_y_ = torchaudio.functional.resample(y_, self.cfg.dataset.sample_rate, 16000)
        rs_y = torchaudio.functional.resample(y, self.cfg.dataset.sample_rate, 16000)
        si_snr = self.val_metrics['si_snr'].update(y_, y)
        si_sdr = self.val_metrics['si_sdr'].update(y_, y)
        stoi = self.val_metrics['stoi'].update(rs_y_, rs_y)
        if self.use_dtp:
            avg_r = self.val_metrics['avg_r'].update(output['avg_r'])
        try:
            pesq = self.val_metrics['pesq'].update(rs_y_, rs_y)
        except NoUtterancesError:
            pass
        perplexity = self.val_metrics['codebook_perplexity'].update(vq_code)
        utilization = self.val_metrics['codebook_utilization'].update(vq_code)
        if self.use_asr_probe:
            asr_out = self.compute_asr_probe(batch, output)
            if asr_out is not None:
                self.log_dict({'val_ctc_loss': asr_out['ctc_loss']}, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.cfg.dataset.val.batch_size, sync_dist=True)
                self.val_cer.update(asr_out['predicted_text'], asr_out['target_text'])
                self.val_wer.update(asr_out['predicted_text'], asr_out['target_text'])
        if batch_idx in self.cfg.dataset.val.log_idxs and wandb is not None and hasattr(self.logger, "experiment"):
            y_ = y_[0].squeeze().float().cpu().numpy()
            y = y[0].squeeze().float().cpu().numpy()

            sample_rate = self.cfg.dataset.sample_rate
            y_spec = librosa.power_to_db(librosa.feature.melspectrogram(y=y_, sr=sample_rate, n_mels=128))[::-1]
            y_gt_spec = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sample_rate, n_mels=128))[::-1]

            self.logger.experiment.log({
                f"val/batch_{batch_idx}/reconstructed_audio": wandb.Audio(y_, caption=f"Step: {self.global_step}", sample_rate=sample_rate),
                f"val/batch_{batch_idx}/reconstructed_spectrogram": wandb.Image(y_spec, caption=f"Step: {self.global_step}"),
                f"val/batch_{batch_idx}/original_audio": wandb.Audio(y, caption=f"Step: {self.global_step}", sample_rate=sample_rate),
                f"val/batch_{batch_idx}/original_spectrogram": wandb.Image(y_gt_spec, caption=f"Step: {self.global_step}"),
            }, commit=False)
    
    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), logger=True, batch_size=self.cfg.dataset.val.batch_size, sync_dist=True)
        if self.use_asr_probe:
            self.log_dict({'val_cer': self.val_cer.compute(), 'val_wer': self.val_wer.compute()}, logger=True, batch_size=self.cfg.dataset.val.batch_size, sync_dist=True)
            self.val_cer.reset()
            self.val_wer.reset()
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            output = self(batch)
        y, y_, vq_code = output['gt_wav'], output['gen_wav'].float(), output['vq_code']
        rs_y_ = torchaudio.functional.resample(y_, self.cfg.dataset.sample_rate, 16000)
        rs_y = torchaudio.functional.resample(y, self.cfg.dataset.sample_rate, 16000)
        si_snr = self.test_metrics['si_snr'].update(y_, y)
        si_sdr = self.test_metrics['si_sdr'].update(y_, y)
        stoi = self.test_metrics['stoi'].update(rs_y_, rs_y)
        if self.use_dtp:
            avg_r = self.test_metrics['avg_r'].update(output['avg_r'])
        try:
            pesq = self.test_metrics['pesq'].update(rs_y_, rs_y)
        except NoUtterancesError:
            pass
        perplexity = self.test_metrics['codebook_perplexity'].update(vq_code)
        utilization = self.test_metrics['codebook_utilization'].update(vq_code)
        if self.use_asr_probe:
            asr_out = self.compute_asr_probe(batch, output)
            if asr_out is not None:
                self.log_dict({'test_ctc_loss': asr_out['ctc_loss']}, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.cfg.dataset.test.batch_size, sync_dist=True)
                self.test_cer.update(asr_out['predicted_text'], asr_out['target_text'])
                self.test_wer.update(asr_out['predicted_text'], asr_out['target_text'])
        if batch_idx in self.cfg.dataset.test.log_idxs and wandb is not None and hasattr(self.logger, "experiment"):
            y_ = y_[0].squeeze().float().cpu().numpy()
            y = y[0].squeeze().float().cpu().numpy()

            sample_rate = self.cfg.dataset.sample_rate
            y_spec = librosa.power_to_db(librosa.feature.melspectrogram(y=y_, sr=sample_rate, n_mels=128))[::-1]
            y_gt_spec = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sample_rate, n_mels=128))[::-1]

            self.logger.experiment.log({
                f"test/batch_{batch_idx}/reconstructed_audio": wandb.Audio(y_, caption=f"Step: {self.global_step}", sample_rate=sample_rate),
                f"test/batch_{batch_idx}/reconstructed_spectrogram": wandb.Image(y_spec, caption=f"Step: {self.global_step}"),
                f"test/batch_{batch_idx}/original_audio": wandb.Audio(y, caption=f"Step: {self.global_step}", sample_rate=sample_rate),
                f"test/batch_{batch_idx}/original_spectrogram": wandb.Image(y_gt_spec, caption=f"Step: {self.global_step}"),
            }, commit=False)

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), logger=True, batch_size=self.cfg.dataset.test.batch_size, sync_dist=True)
        if self.use_asr_probe:
            self.log_dict({'test_cer': self.test_cer.compute(), 'test_wer': self.test_wer.compute()}, logger=True, batch_size=self.cfg.dataset.test.batch_size, sync_dist=True)
            self.test_cer.reset()
            self.test_wer.reset()
        self.test_metrics.reset()

    def configure_optimizers(self):
        disc_params = [p for p in self.discriminator.parameters() if p.requires_grad]
        disc_params.extend([p for p in self.spec_discriminator.parameters() if p.requires_grad])

        gen_params = [p for p in self.encoder.parameters() if p.requires_grad]
        gen_params.extend([p for p in self.decoder.parameters() if p.requires_grad])
        if self.use_asr_probe and self.asr_head is not None:
            gen_params.extend([p for p in self.asr_head.parameters() if p.requires_grad])

        if len(gen_params) == 0:
            raise ValueError("No trainable generator parameters found. Check trainable flags in config.")
        if len(disc_params) == 0:
            raise ValueError("No trainable discriminator parameters found.")

        gen_opt = optim.AdamW(gen_params, **self.cfg.train.gen_optim_params)
        disc_opt = optim.AdamW(disc_params, **self.cfg.train.disc_optim_params)

        gen_sche = WarmupLR(gen_opt, **self.cfg.train.gen_schedule_params)
        disc_sche = WarmupLR(disc_opt, **self.cfg.train.disc_schedule_params)

        print(f'Generator optim: {gen_opt}')
        print(f'Discriminator optim: {disc_opt}')

        return [gen_opt, disc_opt], [gen_sche, disc_sche]

    def set_discriminator_gradients(self, flag=True):
        for p in self.discriminator.parameters():
            p.requires_grad = flag

        if hasattr(self, 'spec_discriminator'):
            for p in self.spec_discriminator.parameters():
                p.requires_grad = flag

class CodecLLM(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.construct_model()
        self.construct_criteria()
        self.save_hyperparameters()
        self.automatic_optimization = False

    def construct_model(self):
        enccfg = self.cfg.model.codec_encoder
        codec = CodecLightningModule.load_from_checkpoint(self.cfg.ckpt, cfg=self.cfg)
        self.codec_encoder = codec.encoder.eval()
        self.quantizer = codec.decoder.quantizer.eval()
        del codec

        vocab_size = self.cfg.model.codec_decoder.codebook_size
        self.llm_config = LlamaConfig(
            vocab_size=vocab_size+2,
            hidden_size=256,
            intermediate_size=1024,
            num_hidden_layers=4,
            num_attention_heads=4,
            max_position_embeddings=1024,
            bos_token_id=vocab_size,
            eos_token_id=vocab_size+1,
        )
        self.llm = LlamaForCausalLM(self.llm_config)
        print(self.model)
        
    def construct_criteria(self):
        criteria = nn.ModuleDict()
        criteria['cce_loss'] = nn.CrossEntropyLoss()
        self.criteria = criteria
        print(criteria)

    def forward(self, batch):
        wav = batch['wav']
        with torch.no_grad():
            vq_emb = self.codec_encoder(wav.unsqueeze(1))
            vq_post_emb, indices, vq_loss = self.quantizer(vq_emb)
        indices = indices.squeeze(0)
        inputs = torch.cat([torch.tensor([[self.llm_config.bos_token_id]], device=self.device).repeat(indices.shape[0], 1), indices], dim=1)
        target = torch.cat([indices, torch.tensor([[self.llm_config.eos_token_id]], device=self.device).repeat(indices.shape[0], 1)], dim=1)
        prediction = self.llm(inputs).logits
        output = {
            'y_true': target,
            'y_pred': prediction,
        }      
        return output

    def training_step(self, batch, batch_idx):
        output = self(batch)
        
        opt = self.optimizers()
        sche = self.lr_schedulers()
        
        # discriminator 
        loss = self.criteria['cce_loss'](output['y_pred'].transpose(1, 2), output['y_true'])
        ppl = torch.exp(loss)
        opt.zero_grad()
        self.manual_backward(loss)
        self.clip_gradients(opt, gradient_clip_val=self.cfg.train.gen_grad_clip, gradient_clip_algorithm='norm')
        opt.step()
        sche.step()

        self.log_dict({'loss':loss}, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.cfg.dataset.train.batch_size, sync_dist=True)
        self.log_dict({'ppl':ppl}, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.cfg.dataset.train.batch_size, sync_dist=True)    
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            output = self(batch)
        
        loss = self.criteria['cce_loss'](output['y_pred'].transpose(1, 2), output['y_true'])
        ppl = torch.exp(loss)

        self.log_dict({'val_loss':loss}, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.cfg.dataset.train.batch_size, sync_dist=True)
        self.log_dict({'val_ppl':ppl}, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.cfg.dataset.train.batch_size, sync_dist=True)    
    
    def configure_optimizers(self):
        opt = optim.AdamW(self.llm.parameters(), **self.cfg.train.gen_optim_params)
        sche = WarmupLR(opt, **self.cfg.train.gen_schedule_params)
        print(f'Generator optim: {opt}')
        return [opt], [sche]
