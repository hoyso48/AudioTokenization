# DTMAE (AudioTokenization/DTMAE) code map

Entry points
- AudioTokenization/DTMAE/train.py:19 `train` (Hydra) builds `DataModule` and `CodecLightningModule`, launches Lightning training/validation/test.

Model architecture (DTMAE)
- AudioTokenization/DTMAE/lightning_module.py:101 `CodecLightningModule` constructs encoder/decoder, discriminators, DTP, and resamplers.
- AudioTokenization/DTMAE/lightning_module.py:325 `forward` defines the DTM pipeline: level-1 encoder -> DTP mask -> downsample -> level-2 encoder -> quantizer -> level-2 decoder -> upsample -> level-1 decoder (ISTFT).

Encoder / decoder definitions
- AudioTokenization/DTMAE/vq/codec_encoder.py:17 `STFT` and :49 `TransformerEncoderSTFT` (STFT + conv + transformer levels).
- AudioTokenization/DTMAE/vq/codec_decoder.py:10 `ISTFT` + :94 `ISTFTHead` (spectrogram to waveform) and :156 `TransformerDecoderISTFT` (quantizer + transformer decoding).
- AudioTokenization/DTMAE/vq/module.py:252 `SelfAttention` (FlashAttention dense/varlen) + :412 `TransformerLayer` + :430 `Transformer` backbone.
- AudioTokenization/DTMAE/vq/module.py:552 `ConvDownsample` + :574 `ConvUpsample` used in encoder/decoder conv paths.

Dynamic token masking (DTP) + resampling
- AudioTokenization/DTMAE/dtp/ops.py:384 `SigmoidSTE`, :705 `PLEBatchTopK`, :817 `PLEBatchTopKJitter`, :954 `BatchTopK`, :1028 `BatchGreedy`.
- AudioTokenization/DTMAE/dtp/ops.py:1117 `PLEBatchTopKTrainPerSeq`, :1290 `BatchTopKTrainPerSeq`, :1424 `BatchGreedyTrainPerSeq` (train-time greedy TODO).
- AudioTokenization/DTMAE/dtp/resampler.py:58 `FixedPatternMasking` (returns fixed interval mask), :187 `FrontierDownsampler`, :210 `AverageDownsampler`, :286 `FixedPatternMaskingDownsampler`, :304 `FixedPatternMaskingUpsampler`, :129 `MaskUpsampler`, :77 `RepeatUpsampler`.

Quantizers
- AudioTokenization/DTMAE/vq/quantizers.py:26 registry for `ResidualVQ`, `DitheredFSQ`, `FSQ`, `SimVQ` (used by `TransformerDecoderISTFT`).
- AudioTokenization/DTMAE/vq/dithered_fsq.py:44 `DitheredFSQ` (TAAE-style FSQ with train/eval levels + residual decomposition).
- AudioTokenization/DTMAE/vq/residual_vq.py:6 `ResidualVQ` (stacked FactorizedVQ).

Discriminators and losses
- AudioTokenization/DTMAE/module/mpd.py:109 `HiFiGANMultiPeriodDiscriminator` (time-domain GAN discriminator).
- AudioTokenization/DTMAE/module/mstft.py:11 `SpecDiscriminator` (multi-resolution STFT discriminator).
- AudioTokenization/DTMAE/criterions/mel_loss.py:9 `MultiResolutionMelSpectrogramLoss`.
- AudioTokenization/DTMAE/criterions/gan_loss.py:6 `GANLoss` (LSGAN losses).

Training loop and metrics
- AudioTokenization/DTMAE/lightning_module.py:410 `compute_disc_loss`, :445 `compute_gen_loss`, :585 `training_step` (manual opt steps, optional ASR probe).
- AudioTokenization/DTMAE/lightning_module.py:292 `construct_metrics` adds STOI/PESQ/SI-SNR/SI-SDR + codebook usage metrics.

Data pipeline
- AudioTokenization/DTMAE/data_module.py:27 `DataModule` selects backend (filelist vs HF streaming) and builds loaders.
- AudioTokenization/DTMAE/data_module.py:169 `FSDataset` for filelist-based training.
- AudioTokenization/DTMAE/hf_streaming_dataset.py:30 `LibriLightStreamingSpec` / :209 `LibriLightStreamingDataset` and :309 `MlsEngStreamingDataset` for streaming datasets.

Schedules and audio utilities
- AudioTokenization/DTMAE/common/schedulers.py:4 `WarmupLR` learning-rate schedule used in `configure_optimizers`.
- AudioTokenization/DTMAE/common/audio.py:6 `stft` helper used by `SpecDiscriminator`.

Configs (Hydra)
- AudioTokenization/DTMAE/config_base*/default.yaml and config_default_mls/default.yaml define model/dataset/train settings used by Hydra (train.py uses config_name=default).
