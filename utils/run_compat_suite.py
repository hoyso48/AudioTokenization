from typing import Dict, Any, Tuple
import sys
sys.path.insert(0, '/home/hoyeol')

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

try:
    import jax
    import jax.numpy as jnp
    from flax import nnx
except Exception:
    jax = None
    jnp = None
    nnx = None

from AudioTokenization.CP.vq import module as pt_mod
from AudioTokenization.CP.vq.codec_encoder import ConformerEncoderSTFT as PT_ConformerEncoderSTFT
from AudioTokenization.BigCodec_NNX.vq import module_jax as jx_mod
from AudioTokenization.BigCodec_NNX.vq.codec_encoder_jax import ConformerEncoderSTFT as JX_ConformerEncoderSTFT
from AudioTokenization.CP.vq.codec_decoder import ConformerDecoderISTFT as PT_ConformerDecoderISTFT
from AudioTokenization.BigCodec_NNX.vq.codec_decoder_jax import ConformerDecoderISTFT as JX_ConformerDecoderISTFT
from AudioTokenization.BigCodec_NNX.common.conv_weightnorm import WNConv1d as JX_WNConv1d
from AudioTokenization.utils.verify_compat import build_and_verify_pair
from AudioTokenization.BigCodec_NNX.codec_module import CodecModule as JX_CodecModule
from AudioTokenization.CP.module.mpd import HiFiGANMultiPeriodDiscriminator as PT_MPD
from AudioTokenization.BigCodec_NNX.module.mpd_jax import HiFiGANMultiPeriodDiscriminator as JX_MPD
from AudioTokenization.CP.module.mstft import SpecDiscriminator as PT_MSTFT
from AudioTokenization.BigCodec_NNX.module.mstft_jax import SpecDiscriminator as JX_MSTFT
from AudioTokenization.BigCodec_NNX.common.spectral import stft as JX_STFT, _get_window as JX_GET_WINDOW, spectrogram as JX_SPECTROGRAM, melscale_fbanks as JX_MELSCALE
from omegaconf import OmegaConf


def _make_seq_input(precision: str, batch: int = 2, channels: int = 8, length: int = 16) -> Tuple[torch.Tensor]:
    assert torch is not None
    x = torch.randn(batch, channels, length)
    return (x,)


def _make_embed_input(precision: str, batch: int = 2, time: int = 16, dim: int = 16) -> Tuple[torch.Tensor]:
    assert torch is not None
    x = torch.randn(batch, time, dim)
    return (x,)

def _make_embed_input_nct(precision: str, batch: int = 2, time: int = 16, dim: int = 16) -> Tuple[torch.Tensor]:
    assert torch is not None
    # Channels-first layout for conv-based PyTorch modules expecting N,C,T
    x = torch.randn(batch, dim, time)
    return (x,)

def _make_audio_input(precision: str, batch: int = 2, length: int = 4096) -> Tuple[torch.Tensor]:
    assert torch is not None
    x = torch.randn(batch, 1, length)
    return (x,)

def _make_decoder_feat_input(precision: str) -> Tuple[torch.Tensor]:
    # Fixed (B, T, C) for decoder features (channels-last)
    assert torch is not None
    x = torch.randn(2, 32, 64)
    return (x,)


def run_suite():
    assert torch is not None and jax is not None and nnx is not None

    pairs = [
        # (name, torch_builder, nnx_builder, input_maker, torch_layout, nnx_layout, jitted)
        # (
        #     'CausalConv1d',
        #     lambda: pt_mod.CausalConv1d(8, 8, 3, stride=1, dilation=1, groups=1, bias=True),
        #     lambda: jx_mod.CausalConv1d(8, 8, 3, stride=1, dilation=1, groups=1, bias=True, rngs=nnx.Rngs(0)),
        #     _make_seq_input,
        #     'NCL', 'NLC', True,
        # ),
        # (
        #     'CausalConvTranspose1d',
        #     lambda: pt_mod.CausalConvTranspose1d(8, 8, 4, stride=2, bias=True),
        #     lambda: jx_mod.CausalConvTranspose1d(8, 8, 4, stride=2, bias=True, rngs=nnx.Rngs(0)),
        #     _make_seq_input,
        #     'NCL', 'NLC', True,
        # ),
        # (
        #     'RMSNorm',
        #     lambda: pt_mod.RMSNorm(16),
        #     lambda: jx_mod.RMSNorm(16, rngs=nnx.Rngs(0)),
        #     _make_embed_input,
        #     'NTC', 'NTC', True,
        # ),
        # (
        #     'SelfAttention',
        #     lambda: pt_mod.SelfAttention(16, n_head=4, dropout=0.0, causal=True),
        #     lambda: jx_mod.SelfAttention(16, n_head=4, dropout=0.0, causal=True, rngs=nnx.Rngs(0)),
        #     _make_embed_input,
        #     'NTC', 'NTC', False,  # avoid jit with stateful paths
        # ),
        # (
        #     'FeedForward',
        #     lambda: pt_mod.FeedForward(16, mult=4, dropout=0.0),
        #     lambda: jx_mod.FeedForward(16, mult=4.0, dropout=0.0, rngs=nnx.Rngs(0)),
        #     _make_embed_input,
        #     'NTC', 'NTC', False,
        # ),
        # (
        #     'ConformerConvModule',
        #     lambda: pt_mod.ConformerConvModule(16, kernel_size=7, dropout=0.0, causal=False),
        #     lambda: jx_mod.ConformerConvModule(16, kernel_size=7, dropout=0.0, causal=False, rngs=nnx.Rngs(0)),
        #     _make_embed_input,
        #     'NCT', 'NTC', False,
        # ),
        # (
        #     'ConformerLayer',
        #     lambda: pt_mod.ConformerLayer(16, n_head=4, ffn_mult=2, conv_kernel_size=7, dropout=0.0, conv_first=False, causal=False),
        #     lambda: jx_mod.ConformerLayer(16, n_head=4, ffn_mult=2.0, conv_kernel_size=7, dropout=0.0, conv_first=False, causal=False, rngs=nnx.Rngs(0)),
        #     _make_embed_input_nct,
        #     'NCT', 'NTC', False,
        # ),
        # (
        #     'ConformerBackbone',
        #     lambda: pt_mod.ConformerBackbone(16, n_layers=2, n_head=4, ffn_mult=2, conv_kernel_size=7, dropout=0.0, conv_first=False, causal=False),
        #     lambda: jx_mod.ConformerBackbone(16, n_layers=2, n_head=4, ffn_mult=2.0, conv_kernel_size=7, dropout=0.0, conv_first=False, causal=False, rngs=nnx.Rngs(0)),
        #     _make_embed_input_nct,
        #     'NCT', 'NTC', False,
        # ),
        # (
        #     'Downsample',
        #     lambda: pt_mod.Downsample(8, 8, stride=2),
        #     lambda: jx_mod.Downsample(8, 8, stride=2, rngs=nnx.Rngs(0)),
        #     _make_seq_input,
        #     'NCL', 'NLC', True,
        # ),
        # (
        #     'Upsample',
        #     lambda: pt_mod.Upsample(8, 8, stride=2),
        #     lambda: jx_mod.Upsample(8, 8, stride=2, rngs=nnx.Rngs(0)),
        #     _make_seq_input,
        #     'NCL', 'NLC', True,
        # ),
        # (
        #     'Upsample_Interpolate',
        #     lambda: pt_mod.Upsample_Interpolate(2.0, 'nearest'),
        #     lambda: jx_mod.Upsample_Interpolate(2.0, 'nearest'),
        #     _make_seq_input,
        #     'NCL', 'NLC', True,
        # ),
        # WeightNorm sanity tests (Conv1d without/with weight_norm)
        # (
        #     'WNConv1d_plain',
        #     lambda: nn.utils.weight_norm(nn.Conv1d(8, 8, 3), name='weight'),
        #     lambda: JX_WNConv1d(8, 8, 3, padding='VALID', rngs=nnx.Rngs(0)),
        #     _make_seq_input,
        #     'NCL', 'NLC', True,
        # ),
        # (
        #     'MPD_firstdisc_flat',
        #     lambda: _build_torch_mpd_wrapper(),
        #     lambda: _build_nnx_mpd_wrapper(),
        #     _make_audio_input,
        #     'NCL', 'NLC', True,
        # ),
        # (
        #     'MSTFT_disc0_out',
        #     lambda: _build_torch_mstft_wrapper(),
        #     lambda: _build_nnx_mstft_wrapper(),
        #     _make_audio_input,
        #     'NCL', 'NLC', True,
        # ),
        # (
        #     'CodecModule_full_audio',
        #     lambda: _build_torch_codec_wrapper_from_cfg(),
        #     lambda: _build_nnx_codec_wrapper_from_cfg(),
        #     _make_audio_input,
        #     'NCT', 'NTC', False,
        # ),
        # (
        #     'ConformerEncoderSTFT_stage0',
        #     lambda: PT_ConformerEncoderSTFT(
        #         hop_length=256,
        #         n_fft=1024,
        #         window_size=1024,
        #         dim=64,
        #         n_layers_stage0=5,
        #         n_layers_stage1=0,
        #         n_head=4,
        #         ffn_mult=2,
        #         conv_kernel_size=31,
        #         dropout=0.0,
        #         max_position_embeddings=256,
        #         original_max_position_embeddings=512,
        #         base=10000.0,
        #         causal=False,
        #         out_channels=64,
        #     ),
        #     lambda: JX_ConformerEncoderSTFT(
        #         hop_length=256,
        #         n_fft=1024,
        #         window_size=1024,
        #         dim=64,
        #         n_layers_stage0=5,
        #         n_layers_stage1=0,
        #         n_head=4,
        #         ffn_mult=2.0,
        #         conv_kernel_size=31,
        #         dropout=0.0,
        #         max_position_embeddings=256,
        #         original_max_position_embeddings=512,
        #         base=10000.0,
        #         causal=False,
        #         out_channels=64,
        #         rngs=nnx.Rngs(0)
        #     ),
        #     _make_audio_input,
        #     'NCT', 'NTC', False,
        # ),
        # (
        #     'ConformerDecoderISTFT_audio',
        #     lambda: PT_ConformerDecoderISTFT(
        #         in_channels=64,
        #         hop_length=256,
        #         n_fft=1024,
        #         window_size=1024,
        #         dim=64,
        #         n_layers_stage0=2,
        #         n_layers_stage1=2,
        #         n_head=4,
        #         ffn_mult=2,
        #         conv_kernel_size=7,
        #         dropout=0.0,
        #         max_position_embeddings=256,
        #         original_max_position_embeddings=512,
        #         base=10000.0,
        #         causal=False,
        #     ),
        #     lambda: JX_ConformerDecoderISTFT(
        #         in_channels=64,
        #         hop_length=256,
        #         n_fft=1024,
        #         window_size=1024,
        #         dim=64,
        #         n_layers_stage0=2,
        #         n_layers_stage1=2,
        #         n_head=4,
        #         ffn_mult=2.0,
        #         conv_kernel_size=7,
        #         dropout=0.0,
        #         max_position_embeddings=256,
        #         original_max_position_embeddings=512,
        #         base=10000.0,
        #         causal=False,
        #         rngs=nnx.Rngs(0),
        #     ),
        #     _make_decoder_feat_input,
        #     'NTC', 'NTC', False,
        # ),
        # # same module but test vq=True path (only encoder/quantizer alignment)
        # (
        #     'ConformerDecoderISTFT_vq',
        #     lambda: PT_ConformerDecoderISTFT(
        #         in_channels=64,
        #         hop_length=256,
        #         n_fft=1024,
        #         window_size=1024,
        #         dim=64,
        #         n_layers_stage0=2,
        #         n_layers_stage1=2,
        #         n_head=4,
        #         ffn_mult=2,
        #         conv_kernel_size=7,
        #         dropout=0.0,
        #         max_position_embeddings=256,
        #         original_max_position_embeddings=512,
        #         base=10000.0,
        #         causal=False,
        #     ),
        #     lambda: JX_ConformerDecoderISTFT(
        #         in_channels=64,
        #         hop_length=256,
        #         n_fft=1024,
        #         window_size=1024,
        #         dim=64,
        #         n_layers_stage0=2,
        #         n_layers_stage1=2,
        #         n_head=4,
        #         ffn_mult=2.0,
        #         conv_kernel_size=7,
        #         dropout=0.0,
        #         max_position_embeddings=256,
        #         original_max_position_embeddings=512,
        #         base=10000.0,
        #         causal=False,
        #         rngs=nnx.Rngs(0),
        #     ),
        #     _make_decoder_feat_input,
        #     'NTC', 'NTC', False,
        # ),
        # Composite codec module pair removed until a stable PT counterpart is finalized
    ]

    precisions = ['bf16', 'fp16']

    results: Dict[str, Dict[str, Any]] = {}
    for name, build_t, build_n, make_input, torch_layout, nnx_layout, jitted in pairs:
        results[name] = {}
        for prec in precisions:
            try:
                extra = {}
                if 'DecoderISTFT_audio' in name:
                    extra['call_kwargs'] = {'vq': False, 'stage': 1}
                    extra['compare_vq_indices'] = False
                elif 'ConformerDecoderISTFT_vq' in name:
                    extra['call_kwargs'] = {'vq': True}
                    # Enable strict equality check for VQ codes (indices)
                    extra['compare_vq_indices'] = True
                res = build_and_verify_pair(
                    build_t, build_n, make_input,
                    precision=prec, jitted=jitted,
                    torch_layout=torch_layout, nnx_layout=nnx_layout,
                    **extra,
                )
                results[name][prec] = res
                # Summary line
                print(f"[OK] {name} @ {prec}: fwd mean_abs_diff={res['forward_stats']['mean_abs_diff']:.3e}, grad mean_abs_diff={res['grad_overall_stats']['mean_abs_diff']:.3e}")
                # If VQ stats present, print detailed info
                if 'vq_code_stats' in res:
                    vqs = res['vq_code_stats']
                    if 'error' in vqs:
                        print(f"        [VQ] error: {vqs['error']}")
                    else:
                        if vqs.get('shape_equal', False):
                            print(f"        [VQ] indices: all_equal={vqs.get('all_equal', False)}, fraction_equal={vqs.get('fraction_equal', 0.0):.6f}, shape={vqs.get('shape')}")
                        else:
                            print(f"        [VQ] indices: shape_mismatch torch={vqs.get('torch_shape')} jax={vqs.get('jax_shape')}")
            except Exception as e:
                results[name][prec] = {'error': str(e)}
                print(f"[FAIL] {name} @ {prec}: {e}")

    # Run spectral ops compatibility checks
    spec_results = _run_spectral_tests()
    results.update(spec_results)
    return results


def _build_torch_mpd_wrapper():
    assert torch is not None and nn is not None

    class PT_MPD_Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.mpd = PT_MPD(
                periods=[2, 3],
                in_channels=1,
                out_channels=1,
                kernel_sizes=[5, 3],
                channels=8,
                downsample_scales=[3, 3, 1],
                channel_increasing_factor=2,
                max_downsample_channels=64,
                nonlinear_activation_params={"negative_slope": 0.1},
                use_weight_norm=True,
            )

        def forward(self, x):  # x: (B, 1, T)
            outs = self.mpd(x)
            first_disc = outs[0]
            return first_disc[-1]

    return PT_MPD_Wrap()


def _build_nnx_mpd_wrapper():
    assert jax is not None and nnx is not None

    class JX_MPD_Wrap(nnx.Module):
        def __init__(self, *, rngs: nnx.Rngs):
            self.mpd = JX_MPD(
                periods=(2, 3),
                in_channels=1,
                out_channels=1,
                kernel_sizes=(5, 3),
                channels=8,
                downsample_scales=(3, 3, 1),
                channel_increasing_factor=2,
                max_downsample_channels=64,
                rngs=rngs,
            )

        def __call__(self, x):  # x: (B, T, C)
            outs = self.mpd(x)
            first_disc = outs[0]
            return first_disc[-1]

    return JX_MPD_Wrap(rngs=nnx.Rngs(0))


def _build_torch_mstft_wrapper():
    assert torch is not None and nn is not None

    class PT_MSTFT_Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.disc = PT_MSTFT(
                in_channels=1,
                out_channels=1,
                kernel_sizes=(7, 3),
                channels=8,
                max_downsample_channels=64,
                downsample_scales=(2, 2, 2),
                use_weight_norm=True,
            )

        def forward(self, x):  # x: (B, 1, T)
            outs = self.disc(x)
            y = outs[0][-1]  # (B, 1, F, T)
            b, c, f, t = y.shape
            return y.view(b, c, f * t)

    return PT_MSTFT_Wrap()


def _build_nnx_mstft_wrapper():
    assert jax is not None and nnx is not None

    class JX_MSTFT_Wrap(nnx.Module):
        def __init__(self, *, rngs: nnx.Rngs):
            self.disc = JX_MSTFT(
                in_channels=1,
                out_channels=1,
                kernel_sizes=(7, 3),
                channels=8,
                max_downsample_channels=64,
                downsample_scales=(2, 2, 2),
                rngs=rngs,
            )

        def __call__(self, x):  # x: (B, T, C)
            outs = self.disc(x)
            y = outs[0][-1]  # (B, F, T, 1)
            b, f, t, c = y.shape
            return y.reshape(b, f * t, c)

    return JX_MSTFT_Wrap(rngs=nnx.Rngs(0))


def _load_cfg_from_test_notebook_path() -> Any:
    # Follow test.ipynb: cfg = OmegaConf.load('/home/hoyeol/AudioTokenization/ckpts/config.yaml')
    return OmegaConf.load('/home/hoyeol/AudioTokenization/ckpts/config.yaml')


def _build_torch_codec_wrapper_from_cfg():
    assert torch is not None and nn is not None
    from AudioTokenization.CP.lightning_module import CodecLightningModule as PT_CodecModule

    class PT_Codec_Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            cfg = _load_cfg_from_test_notebook_path()
            self.module = PT_CodecModule(cfg=cfg)
        def forward(self, x):  # x: (B, 1, T)
            batch = {'wav': x.squeeze(1)}  # lightning forward expects (B, T)
            out = self.module(batch)
            y = out['gen_wav']  # (B, 1, T)
            return y

    return PT_Codec_Wrap()


def _build_nnx_codec_wrapper_from_cfg():
    assert jax is not None and nnx is not None
    from AudioTokenization.BigCodec_NNX.codec_module import CodecModule as JX_CodecModule

    class JX_Codec_Wrap(nnx.Module):
        def __init__(self, *, rngs: nnx.Rngs):
            cfg = _load_cfg_from_test_notebook_path()
            self.module = JX_CodecModule(cfg=cfg, rngs=rngs)
        def __call__(self, x):  # x: (B, T, C)
            batch = {'wav': x.squeeze(-1)}  # (B, T)
            out = self.module(batch)
            return out['gen_wav']

    return JX_Codec_Wrap(rngs=nnx.Rngs(0))


def _run_spectral_tests():
    assert torch is not None and jax is not None and nnx is not None
    out: Dict[str, Dict[str, Any]] = {}

    # STFT: Torch reference via librosa/torchaudio is complex; use PyTorch FFT on framed windows for a small case
    try:
        import numpy as _np
        # Small deterministic signal
        rng = _np.random.default_rng(0)
        x_np = rng.standard_normal((2, 4096)).astype(_np.float32)

        # Torch compute (frames using unfold-like trick)
        x_t = torch.from_numpy(x_np)
        n_fft = 1024
        hop = 256
        win = torch.hann_window(n_fft)
        # Centered pad like JAX path
        x_t_pad = torch.nn.functional.pad(x_t, (n_fft // 2, n_fft // 2), mode='reflect')
        # frame extraction
        num_frames = (x_t_pad.shape[1] - n_fft) // hop + 1
        frames = torch.stack([x_t_pad[:, i * hop:i * hop + n_fft] for i in range(num_frames)], dim=1)  # (B, F, n_fft)
        frames = frames * win.to(frames)
        stft_t = torch.fft.rfft(frames, n=n_fft, dim=-1)  # (B, F, n_fft//2+1)
        stft_t = stft_t.movedim(-1, -2).detach().cpu().numpy().astype(_np.complex64)  # (B, freq, frames)

        # JAX compute
        x_j = jnp.asarray(x_np)
        win_j = JX_GET_WINDOW(n_fft, 'hann')
        stft_j = JX_STFT(x_j, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win_j, center=True, pad_mode='reflect', normalized=False)
        stft_j = _np.asarray(stft_j).astype(_np.complex64)

        fwd = _np.abs(stft_t - stft_j)
        out['spectral_stft'] = {
            'bf16': {
                'forward_stats': {
                    'min_abs_diff': float(fwd.min()),
                    'mean_abs_diff': float(fwd.mean()),
                    'max_abs_diff': float(fwd.max()),
                }
            }
        }
    except Exception as e:
        out['spectral_stft'] = {'error': str(e)}

    # Spectrogram power=1
    try:
        import numpy as _np
        rng = _np.random.default_rng(1)
        x_np = rng.standard_normal((2, 4096)).astype(_np.float32)
        x_t = torch.from_numpy(x_np)
        n_fft = 1024
        hop = 256
        win = torch.hann_window(n_fft)
        x_t_pad = torch.nn.functional.pad(x_t, (n_fft // 2, n_fft // 2), mode='reflect')
        num_frames = (x_t_pad.shape[1] - n_fft) // hop + 1
        frames = torch.stack([x_t_pad[:, i * hop:i * hop + n_fft] for i in range(num_frames)], dim=1)
        frames = frames * win.to(frames)
        stft_t = torch.fft.rfft(frames, n=n_fft, dim=-1)
        spec_t = torch.abs(stft_t).movedim(-1, -2).detach().cpu().numpy()

        x_j = jnp.asarray(x_np)
        win_j = JX_GET_WINDOW(n_fft, 'hann')
        spec_j = JX_SPECTROGRAM(x_j, pad=0, window=win_j, n_fft=n_fft, hop_length=hop, win_length=n_fft, power=1.0, normalized=False, center=True, pad_mode='reflect', onesided=True, return_complex=None)
        spec_j = _np.asarray(spec_j)

        fwd = _np.abs(spec_t - spec_j)
        out['spectral_spectrogram_pow1'] = {
            'bf16': {
                'forward_stats': {
                    'min_abs_diff': float(fwd.min()),
                    'mean_abs_diff': float(fwd.mean()),
                    'max_abs_diff': float(fwd.max()),
                }
            }
        }
    except Exception as e:
        out['spectral_spectrogram_pow1'] = {'error': str(e)}

    return out


if __name__ == '__main__':
    run_suite()


