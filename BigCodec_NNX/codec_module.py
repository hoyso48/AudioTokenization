from flax import nnx
import jax
import jax.numpy as jnp
from typing import Sequence
from criterions.mel_loss_jax import MultiResolutionMelSpectrogramLoss
from criterions.gan_loss_jax import GANLoss
from criterions.mel_loss_jax import L1Loss
# from vq.codec_encoder_jax import BigCodecEncoder
# from vq.codec_decoder_jax import CodecDecoder
from module.mpd_jax import HiFiGANMultiPeriodDiscriminator
from module.mstft_jax import SpecDiscriminator
from vq.codec_encoder_jax import ConformerEncoderSTFT as ConformerEncoderSTFT
from vq.codec_decoder_jax import ConformerDecoderISTFT as ConformerDecoderISTFT
from dtp.tome_ops_jax import ToMeK2New
from vq.module_jax import Downsample, Upsample
import dtp

class CodecModule(nnx.Module):
    """
    A codec module composed of ConformerEncoderSTFT and ConformerDecoderISTFT.

    This class mirrors the structure of the PyTorch reference (CP Conformer encoder/decoder)
    and is intended for forward/grad compatibility checks. The forward method returns only
    the generated audio (B, 1, T) to align with the compatibility harness expectations.
    """
    def __init__(self, cfg, rngs: nnx.Rngs):
        # Save cfg
        self.cfg = cfg

        # Training flags and lambdas (overlap with lightning_module.py)
        train_cfg = cfg.train
        self.use_mel_loss = getattr(train_cfg, 'use_mel_loss', True)
        self.use_feat_match_loss = getattr(train_cfg, 'use_feat_match_loss', True)
        lambdas = getattr(train_cfg, 'lambdas', None)
        if lambdas is not None:
            self.lambda_disc = getattr(lambdas, 'lambda_disc', 1.0)
            self.lambda_mel_loss = getattr(lambdas, 'lambda_mel_loss', 15.0)
            self.lambda_adv = getattr(lambdas, 'lambda_adv', 1.0)
            self.lambda_feat_match_loss = getattr(lambdas, 'lambda_feat_match_loss', 1.0)
        else:
            self.lambda_disc = 1.0
            self.lambda_mel_loss = 15.0
            self.lambda_adv = 1.0
            self.lambda_feat_match_loss = 1.0

        # Models from cfg (only using overlapping args with lightning_module.py)
        enccfg = cfg.model.codec_encoder
        self.encoder = ConformerEncoderSTFT(
            hop_length=enccfg.hop_length,
            n_fft=enccfg.n_fft,
            window_size=enccfg.window_size,
            dim=enccfg.dim,
            n_layers_stage0=enccfg.n_layers_stage0,
            n_layers_stage1=enccfg.n_layers_stage1,
            r=enccfg.r,
            n_head=enccfg.n_head,
            ffn_mult=enccfg.ffn_mult,
            conv_kernel_size=enccfg.conv_kernel_size,
            dropout=enccfg.dropout,
            max_position_embeddings=enccfg.max_position_embeddings,
            original_max_position_embeddings=enccfg.original_max_position_embeddings,
            base=enccfg.base,
            causal=enccfg.causal,
            out_channels=enccfg.out_channels,
            rngs=rngs,
        )

        deccfg = cfg.model.codec_decoder
        self.decoder = ConformerDecoderISTFT(
            in_channels=deccfg.in_channels,
            hop_length=deccfg.hop_length,
            n_fft=deccfg.n_fft,
            window_size=deccfg.window_size,
            dim=deccfg.dim,
            n_layers_stage0=deccfg.n_layers_stage0,
            n_layers_stage1=deccfg.n_layers_stage1,
            r=deccfg.r,
            n_head=deccfg.n_head,
            ffn_mult=deccfg.ffn_mult,
            conv_kernel_size=deccfg.conv_kernel_size,
            dropout=deccfg.dropout,
            max_position_embeddings=deccfg.max_position_embeddings,
            original_max_position_embeddings=deccfg.original_max_position_embeddings,
            base=deccfg.base,
            causal=deccfg.causal,
            fsq=deccfg.fsq,
            fsq_levels=deccfg.fsq_levels,
            vq_num_quantizers=deccfg.vq_num_quantizers,
            vq_commit_weight=deccfg.vq_commit_weight,
            vq_weight_init=deccfg.vq_weight_init,
            vq_full_commit_loss=deccfg.vq_full_commit_loss,
            codebook_size=deccfg.codebook_size,
            codebook_dim=deccfg.codebook_dim,
            rngs=rngs,
        )

        mpdcfg = cfg.model.mpd
        self.discriminator = HiFiGANMultiPeriodDiscriminator(
            periods=mpdcfg.periods,
            in_channels=1,
            out_channels=1,
            kernel_sizes=(5, 3),
            channels=mpdcfg.channels,
            downsample_scales=(3, 3, 3, 3, 1),
            channel_increasing_factor=mpdcfg.channel_increasing_factor,
            max_downsample_channels=mpdcfg.max_downsample_channels,
            rngs=rngs,
        )

        mstftcfg = cfg.model.mstft
        self.spec_discriminator = SpecDiscriminator(
            stft_params=mstftcfg.stft_params,
            in_channels=mstftcfg.in_channels,
            out_channels=mstftcfg.out_channels,
            kernel_sizes=mstftcfg.kernel_sizes,
            channels=mstftcfg.channels,
            max_downsample_channels=mstftcfg.max_downsample_channels,
            downsample_scales=mstftcfg.downsample_scales,
            rngs=rngs,
        )

        # Criteria
        self.mel_loss_criterion = MultiResolutionMelSpectrogramLoss()
        self.gan_loss_criterion = GANLoss()
        self.fm_loss_criterion = L1Loss()

        # Token Merging + projection mirroring lightning_module.py
        tome_cfg = cfg.model.tome
        if tome_cfg.use_tome:
            self.tome = getattr(dtp.tome_ops_jax, tome_cfg.class_name)(**tome_cfg.tome_params)
            if tome_cfg.proj:
                self.tome_proj = nnx.Linear(enccfg.out_channels, tome_cfg.proj_dim, rngs=rngs)
        else:
            self.downsample = Downsample(in_channels=enccfg.out_channels, out_channels=enccfg.out_channels, stride=2, rngs=rngs)
            self.upsample = Upsample(in_channels=enccfg.out_channels, out_channels=enccfg.out_channels, stride=2, rngs=rngs)

    def __call__(self, batch):
        wav = batch['wav']
        # x: (B, 1, T) or (B, T) — encoder accepts both
        if wav.ndim == 2:
            wav_in = jnp.expand_dims(wav, 1)
        else:
            wav_in = wav
        vq_emb = self.encoder(wav_in, stage=0)
        if getattr(self, 'tome', None) is not None:
            merged, merge_btree, avg_sim = self.tome.compute_merge(self.tome_proj(vq_emb))
            direct_to_root_map = self.tome.btree_to_root_map(merge_btree)
            vq_emb = self.tome.merge(vq_emb, direct_to_root_map)
        elif getattr(self, 'downsample', None) is not None:
            vq_emb = self.downsample(vq_emb)
        vq_emb = self.encoder(vq_emb, stage=1)
        # Quantize -> decode features in two stages, mirroring Torch reference
        vq_post_emb, vq_code, vq_loss = self.decoder(vq_emb, vq=True)
        # Stage 0 of decoder (feature processing)
        vq_post_emb = self.decoder(vq_post_emb, vq=False, stage=0)
        # Unmerge/Upsample back to original temporal resolution
        if getattr(self, 'tome', None) is not None:
            vq_post_emb = self.tome.unmerge(vq_post_emb, direct_to_root_map)
        elif getattr(self, 'upsample', None) is not None:
            vq_post_emb = self.upsample(vq_post_emb)
        # Stage 1 of decoder to ISTFT audio
        y_ = self.decoder(vq_post_emb, vq=False, stage=1)
        y = wav_in
        output = {
            'gt_wav': y,
            'gen_wav': y_,
            'vq_loss': vq_loss,
            'vq_code': vq_code,
        }
        if hasattr(self, 'tome'):
            output['avg_sim'] = avg_sim
        return output

    def compute_disc_loss(self, output):
        y, y_ = output['gt_wav'], output['gen_wav']
        y_ = jax.lax.stop_gradient(y_)

        # Convert to channels-last (B, T, C) for JAX discriminators
        y_cl = jnp.swapaxes(y, 1, 2)
        y_cl_ = jnp.swapaxes(y_, 1, 2)

        p = self.discriminator(y_cl)
        p_ = self.discriminator(y_cl_)

        real_loss_list, fake_loss_list = [], []
        for i in range(len(p)):
            real_loss, fake_loss = self.gan_loss_criterion.disc_loss(p[i][-1].astype(jnp.float32), p_[i][-1].astype(jnp.float32))
            real_loss_list.append(real_loss)
            fake_loss_list.append(fake_loss)

        if hasattr(self, 'spec_discriminator') and self.spec_discriminator is not None:
            sd_p = self.spec_discriminator(y_cl)
            sd_p_ = self.spec_discriminator(y_cl_)
            for i in range(len(sd_p)):
                real_loss, fake_loss = self.gan_loss_criterion.disc_loss(sd_p[i][-1].astype(jnp.float32), sd_p_[i][-1].astype(jnp.float32))
                real_loss_list.append(real_loss)
                fake_loss_list.append(fake_loss)

        real_loss = sum(real_loss_list)
        fake_loss = sum(fake_loss_list)
        disc_loss = self.lambda_disc * (real_loss + fake_loss)
        return {
            'real_loss': real_loss,
            'fake_loss': fake_loss,
            'disc_loss': disc_loss,
        }

    def compute_gen_loss(self, output):
        y, y_ = output['gt_wav'], output['gen_wav']
        vq_loss, vq_code = output.get('vq_loss', 0.0), output.get('vq_code', None)
        gen_loss = 0.0
        gen_output_metrics = {}

        # For mel loss, reduce channel axis (B, 1, T) -> (B, T)
        if self.use_mel_loss:
            mel_loss = self.mel_loss_criterion(y_.squeeze(1).astype(jnp.float32), y.squeeze(1).astype(jnp.float32))
            gen_loss += mel_loss * self.lambda_mel_loss
            gen_output_metrics['mel_loss'] = mel_loss

        # Convert to channels-last (B, T, C) for discriminators
        y_cl = jnp.swapaxes(y, 1, 2)
        y_cl_ = jnp.swapaxes(y_, 1, 2)

        p_ = self.discriminator(y_cl_)
        adv_loss_list = []
        for i in range(len(p_)):
            adv_loss_list.append(self.gan_loss_criterion.gen_loss(p_[i][-1].astype(jnp.float32)))
        if hasattr(self, 'spec_discriminator') and self.spec_discriminator is not None:
            sd_p_ = self.spec_discriminator(y_cl_)
            for i in range(len(sd_p_)):
                adv_loss_list.append(self.gan_loss_criterion.gen_loss(sd_p_[i][-1].astype(jnp.float32)))
        adv_loss = sum(adv_loss_list)
        gen_loss += adv_loss * self.lambda_adv
        gen_output_metrics['adv_loss'] = adv_loss

        if self.use_feat_match_loss:
            fm_loss = 0.0
            p = self.discriminator(y_cl)
            for i in range(len(p_)):
                for j in range(len(p_[i]) - 1):
                    fm_loss += self.fm_loss_criterion(p_[i][j].astype(jnp.float32), jax.lax.stop_gradient(p[i][j].astype(jnp.float32)))
            gen_loss += fm_loss * self.lambda_feat_match_loss
            gen_output_metrics['fm_loss'] = fm_loss

            if hasattr(self, 'spec_discriminator') and self.spec_discriminator is not None:
                spec_fm_loss = 0.0
                sd_p = self.spec_discriminator(y_cl)
                for i in range(len(sd_p_)):
                    for j in range(len(sd_p_[i]) - 1):
                        spec_fm_loss += self.fm_loss_criterion(sd_p_[i][j].astype(jnp.float32), jax.lax.stop_gradient(sd_p[i][j].astype(jnp.float32)))
                gen_loss += spec_fm_loss * self.lambda_feat_match_loss
                gen_output_metrics['spec_fm_loss'] = spec_fm_loss

        if vq_loss is not None:
            vq_loss_sum = jnp.sum(vq_loss)
            gen_loss += vq_loss_sum
            gen_output_metrics['vq_loss'] = vq_loss_sum

        gen_output_metrics['gen_loss'] = gen_loss
        return gen_output_metrics