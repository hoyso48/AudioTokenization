import os
import random
import hydra
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import torchmetrics
import torchaudio

import wandb
from vq import BigCodecEncoder, BigCodecDecoder, ConformerEncoderSTFT, ConformerDecoderISTFT
from module import HiFiGANMultiPeriodDiscriminator, SpecDiscriminator
from criterions import GANLoss, MultiResolutionMelSpectrogramLoss
from common.schedulers import WarmupLR
from transformers import LlamaForCausalLM, LlamaConfig
from transformers import AutoModel
from vq.module import SemanticDecoder,SemanticEncoder
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
from torchmetrics.audio import ShortTimeObjectiveIntelligibility, PerceptualEvaluationSpeechQuality, ScaleInvariantSignalNoiseRatio, ScaleInvariantSignalDistortionRatio
from pesq import NoUtterancesError

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
        
        # 0이 아닌 확률에 대해서만 엔트로피 계산
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
        self.save_hyperparameters()
        self.automatic_optimization = False

    def construct_model(self):
        enccfg = self.cfg.model.codec_encoder
        if enccfg.type == 'bigcodec':
            self.encoder = BigCodecEncoder(
                        ngf=enccfg.ngf,
                        use_rnn=enccfg.use_rnn,
                        rnn_bidirectional=enccfg.rnn_bidirectional,
                        rnn_num_layers=enccfg.rnn_num_layers,
                        up_ratios=enccfg.up_ratios,
                        dilations=enccfg.dilations,
                        out_channels=enccfg.out_channels,
                        causal=enccfg.causal,
                        antialias=enccfg.antialias
                    )
        elif enccfg.type == 'conformer_stft':
            self.encoder = ConformerEncoderSTFT(
                hop_length=enccfg.hop_length,
                n_fft=enccfg.n_fft,
                window_size=enccfg.window_size,
                dim=enccfg.dim,
                n_layers=enccfg.n_layers,
                n_head=enccfg.n_head,
                ffn_mult=enccfg.ffn_mult,
                conv_kernel_size=enccfg.conv_kernel_size,
                dropout=enccfg.dropout,
                max_seq_len=enccfg.max_seq_len,
                rope_theta=enccfg.rope_theta,
                causal=enccfg.causal,
                out_channels=enccfg.out_channels,
            )
        else:
            raise ValueError(f'Invalid codec encoder type: {enccfg.type}')
        deccfg = self.cfg.model.codec_decoder
        if deccfg.type == 'bigcodec':
            self.decoder = BigCodecDecoder(
                    in_channels=deccfg.in_channels,
                    upsample_initial_channel=deccfg.upsample_initial_channel,
                    ngf=deccfg.ngf,
                    use_rnn=deccfg.use_rnn,
                    rnn_bidirectional=deccfg.rnn_bidirectional,
                    rnn_num_layers=deccfg.rnn_num_layers,
                    up_ratios=deccfg.up_ratios,
                    dilations=deccfg.dilations,
                    causal=deccfg.causal,
                    antialias=deccfg.antialias,
                    fsq=deccfg.fsq,
                    fsq_levels=deccfg.fsq_levels,
                    vq_num_quantizers=deccfg.vq_num_quantizers,
                    vq_commit_weight=deccfg.vq_commit_weight,
                    vq_full_commit_loss=deccfg.vq_full_commit_loss,
                    codebook_size=deccfg.codebook_size,
                    codebook_dim=deccfg.codebook_dim,
                )
        elif deccfg.type == 'conformer_istft':
            self.decoder = ConformerDecoderISTFT(
                 in_channels=deccfg.in_channels,
                 hop_length=deccfg.hop_length,
                 n_fft=deccfg.n_fft,
                 window_size=deccfg.window_size,
                 dim=deccfg.dim,
                 n_layers=deccfg.n_layers,
                 n_head=deccfg.n_head,
                 ffn_mult=deccfg.ffn_mult,
                 conv_kernel_size=deccfg.conv_kernel_size,
                 dropout=deccfg.dropout,
                 max_seq_len=deccfg.max_seq_len,
                 rope_theta=deccfg.rope_theta,
                 causal=deccfg.causal,
                 fsq=deccfg.fsq,
                 fsq_levels=deccfg.fsq_levels,
                 vq_num_quantizers=deccfg.vq_num_quantizers,
                 vq_commit_weight=deccfg.vq_commit_weight,
                 vq_weight_init=deccfg.vq_weight_init,
                 vq_full_commit_loss=deccfg.vq_full_commit_loss,
                 codebook_size=deccfg.codebook_size,
                 codebook_dim=deccfg.codebook_dim,
            )
        else:
            raise ValueError(f'Invalid codec decoder type: {deccfg.type}')
        mpdcfg = self.cfg.model.mpd
        self.discriminator = HiFiGANMultiPeriodDiscriminator(
                    periods=mpdcfg.periods,
                    max_downsample_channels=mpdcfg.max_downsample_channels,
                    channels=mpdcfg.channels,
                    channel_increasing_factor=mpdcfg.channel_increasing_factor,
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

        if self.cfg.train.use_semantic:
            self.fc_prior = nn.Linear(1024 + enccfg.out_channels, deccfg.in_channels) if self.cfg.train.concat_semantic else nn.Linear(enccfg.out_channels, deccfg.in_channels)
            self.fc_post_a = nn.Linear(deccfg.in_channels,  deccfg.in_channels)
            self.fc_post_s = nn.Linear(deccfg.in_channels,  1024)

            self.SemanticDecoder_module = SemanticDecoder(1024, 1024, 1024)
            self.SemanticEncoder_module = SemanticEncoder(1024, 1024, 1024)
            self.semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0", output_hidden_states=True)
            self.semantic_model.eval()
            self.semantic_model.requires_grad_(False)

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
        metrics['codebook_perplexity'] = CodebookPerplexity(codebook_size=self.cfg.model.codec_decoder.codebook_size)
        metrics['codebook_utilization'] = CodebookUtilization(codebook_size=self.cfg.model.codec_decoder.codebook_size)
        return torchmetrics.MetricCollection(prefix=prefix, metrics=metrics)
    
    # @torch.compile
    def forward(self, batch):
        if self.cfg.train.use_semantic:
            wav = batch['wav']
            feats = batch['feats']
        
            vq_emb = self.encoder(wav.unsqueeze(1))
            # vq_emb = vq_emb.transpose(1, 2)
            with torch.no_grad():
                semantic_target = self.semantic_model(feats[:,0,:,:])

                semantic_target = semantic_target.hidden_states[16]
                semantic_target = semantic_target.detach()

            semantic_target = semantic_target.transpose(1, 2)
            semantic_target_processed = self.SemanticEncoder_module(semantic_target)
            # print(vq_emb.shape, semantic_target_processed.shape)

            if self.cfg.train.concat_semantic:
                vq_emb = torch.cat([semantic_target_processed, vq_emb], dim=1)

            vq_emb = self.fc_prior(vq_emb.transpose(1, 2)).transpose(1, 2)

            vq_post_emb, vq_code, vq_loss = self.decoder(vq_emb, vq=True)
            semantic_recon = self.fc_post_s(vq_post_emb.transpose(1, 2)).transpose(1, 2)
            semantic_recon = self.SemanticDecoder_module(semantic_recon)

            y_ = self.decoder(
                self.fc_post_a(vq_post_emb.transpose(1, 2)).transpose(1, 2),
                vq=False
            )
            y = wav.unsqueeze(1)

            output = {
                'gt_wav': y,
                'gen_wav': y_,
                'vq_loss': vq_loss,
                'vq_code': vq_code,
                'semantic_recon_loss': F.mse_loss(semantic_recon, semantic_target),
                # 'perceptual_se_loss_l2': perceptual_se_loss_l2,
    
            }
        else:
            wav = batch['wav']
            vq_emb = self.encoder(wav.unsqueeze(1))
            vq_post_emb, vq_code, vq_loss = self.decoder(vq_emb, vq=True)
            y_ = self.decoder(vq_post_emb, vq=False) # [B, 1, T]
            y = wav.unsqueeze(1)
            output = {
                'gt_wav': y,
                'gen_wav': y_,
                'vq_loss': vq_loss,
                'vq_code': vq_code
            }
        return output
    
    @torch.inference_mode()
    def inference(self, wav):
        vq_emb = self.encoder(wav.unsqueeze(1))
        vq_post_emb, vq_code, vq_loss = self.decoder(vq_emb, vq=True)
        y_ = self.decoder(vq_post_emb, vq=False).squeeze(1)  # [B, T]
        return y_

    # @torch.compile
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

        # Semantic reconstruction loss
        if self.cfg.train.use_semantic:
            output_dict['semantic_recon_loss'] = output['semantic_recon_loss']
            gen_loss += output_dict['semantic_recon_loss'] * cfg.lambdas.lambda_semantic_loss

        # Perceptual loss
        # output_dict['perceptual_se_loss_l2'] = perceptual_se_loss_l2
        # gen_loss += output_dict['perceptual_se_loss_l2'] * cfg.lambdas.lambda_perceptual_loss
        
        self.set_discriminator_gradients(True)
        output_dict['gen_loss'] = gen_loss
        return output_dict
    
    # @torch.compile
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
        gen_opt.zero_grad()
        self.manual_backward(gen_loss)
        self.clip_gradients(gen_opt, gradient_clip_val=self.cfg.train.gen_grad_clip, gradient_clip_algorithm='norm')
        gen_opt.step()
        gen_sche.step()

        self.log_dict(disc_losses, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.cfg.dataset.train.batch_size, sync_dist=True)
        self.log_dict(gen_losses, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.cfg.dataset.train.batch_size, sync_dist=True)    
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            output = self(batch)
        y, y_, vq_code = output['gt_wav'], output['gen_wav'], output['vq_code']
        rs_y_ = torchaudio.functional.resample(y_, self.cfg.dataset.sample_rate, 16000)
        rs_y = torchaudio.functional.resample(y, self.cfg.dataset.sample_rate, 16000)
        si_snr = self.val_metrics['si_snr'].update(y_, y)
        si_sdr = self.val_metrics['si_sdr'].update(y_, y)
        stoi = self.val_metrics['stoi'].update(rs_y_, rs_y)
        try:
            pesq = self.val_metrics['pesq'].update(rs_y_, rs_y)
        except NoUtterancesError:
            pass
        perplexity = self.val_metrics['codebook_perplexity'].update(vq_code)
        utilization = self.val_metrics['codebook_utilization'].update(vq_code)
        if batch_idx in self.cfg.dataset.val.log_idxs:
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
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            output = self(batch)
        y, y_, vq_code = output['gt_wav'], output['gen_wav'], output['vq_code']
        rs_y_ = torchaudio.functional.resample(y_, self.cfg.dataset.sample_rate, 16000)
        rs_y = torchaudio.functional.resample(y, self.cfg.dataset.sample_rate, 16000)
        si_snr = self.test_metrics['si_snr'].update(y_, y)
        si_sdr = self.test_metrics['si_sdr'].update(y_, y)
        stoi = self.test_metrics['stoi'].update(rs_y_, rs_y)
        try:
            pesq = self.test_metrics['pesq'].update(rs_y_, rs_y)
        except NoUtterancesError:
            pass
        perplexity = self.test_metrics['codebook_perplexity'].update(vq_code)
        utilization = self.test_metrics['codebook_utilization'].update(vq_code)
        if batch_idx in self.cfg.dataset.test.log_idxs:
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
        self.test_metrics.reset()

    def configure_optimizers(self):
        from itertools import chain

        disc_params = self.discriminator.parameters()
        # if hasattr(self, 'spec_discriminator'):
        disc_params = chain(disc_params, self.spec_discriminator.parameters())

        gen_params = chain(
            self.encoder.parameters(),
            self.decoder.parameters(),
        )
        if self.cfg.train.use_semantic:
            gen_params = chain(
                gen_params,
                self.fc_prior.parameters(),
                self.fc_post_a.parameters(),
                self.fc_post_s.parameters(),
                self.SemanticDecoder_module.parameters(),
                self.SemanticEncoder_module.parameters()
            )

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
        # print(indices.shape)
        inputs = torch.cat([torch.tensor([[self.llm_config.bos_token_id]], device=self.device).repeat(indices.shape[0], 1), indices], dim=1)
        target = torch.cat([indices, torch.tensor([[self.llm_config.eos_token_id]], device=self.device).repeat(indices.shape[0], 1)], dim=1)
        prediction = self.llm(inputs).logits
        # print(inputs.shape, target.shape, prediction.shape)
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