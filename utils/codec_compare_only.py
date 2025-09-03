import sys, os
sys.path.insert(0, '/home/hoyeol')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

from omegaconf import OmegaConf
import numpy as np

import torch
from flax import nnx

from AudioTokenization.CP.lightning_module import CodecLightningModule as PT_Codec
from AudioTokenization.BigCodec_NNX.codec_module import CodecModule as JX_Codec
from AudioTokenization.utils.torch_nnx_port import Torch2NNX


def _np(x):
    if hasattr(x, 'detach'):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _stats(a, b):
    a = _np(a).astype(np.float32)
    b = _np(b).astype(np.float32)
    d = np.abs(a - b)
    return {
        'min_abs_diff': float(d.min()),
        'mean_abs_diff': float(d.mean()),
        'max_abs_diff': float(d.max()),
        'shape_a': tuple(a.shape),
        'shape_b': tuple(b.shape),
    }


def main():
    cfg_path = '/home/hoyeol/AudioTokenization/ckpts/config.yaml'
    cfg = OmegaConf.load(cfg_path)

    pt = PT_Codec(cfg=cfg)
    jx = JX_Codec(cfg=cfg, rngs=nnx.Rngs(0))
    Torch2NNX().copy(pt, jx)
    pt.eval(); jx.eval()

    # Generate deterministic random input with valid length (>=16000 and divisible by 320)
    rng = np.random.default_rng(0)
    target_len = 16000
    if target_len % 320 != 0:
        target_len = ((target_len + 319) // 320) * 320
    wav = rng.standard_normal((2, target_len)).astype('float32')

    pt_batch = {'wav': torch.from_numpy(wav)}
    jx_batch = {'wav': wav}

    with torch.no_grad():
        pt_out = pt(pt_batch)
    jx_out = jx(jx_batch)

    # Compare forward keys shared by both
    common_keys = sorted(set(pt_out.keys()).intersection(jx_out.keys()))
    print('[forward diffs]')
    for k in common_keys:
        if k == 'vq_code':
            a = _np(pt_out[k]).astype(np.int64)
            b = _np(jx_out[k]).astype(np.int64)
            same_shape = tuple(a.shape) == tuple(b.shape)
            if same_shape:
                eq = (a == b)
                frac = float(eq.sum() / eq.size)
                print(f" {k}: shape_equal={same_shape}, fraction_equal={frac:.6f}, shape={a.shape}")
            else:
                print(f" {k}: shape_mismatch torch={a.shape} jax={b.shape}")
        else:
            s = _stats(pt_out[k], jx_out[k])
            print(f" {k}: mean_abs_diff={s['mean_abs_diff']:.3e}, max_abs_diff={s['max_abs_diff']:.3e}, shapes=({s['shape_a']},{s['shape_b']})")

    # Compute and compare discriminator/gen losses
    with torch.no_grad():
        pt_disc = pt.compute_disc_loss(pt_batch, pt_out)
    jx_disc = jx.compute_disc_loss(jx_out)

    disc_keys = sorted(set(pt_disc.keys()).intersection(jx_disc.keys()))
    print('[disc_loss diffs]')
    for k in disc_keys:
        s = _stats(pt_disc[k], jx_disc[k])
        print(f" {k}: mean_abs_diff={s['mean_abs_diff']:.3e}, max_abs_diff={s['max_abs_diff']:.3e}, shapes=({s['shape_a']},{s['shape_b']})")

    with torch.no_grad():
        pt_gen = pt.compute_gen_loss(pt_batch, pt_out)
    jx_gen = jx.compute_gen_loss(jx_out)

    gen_keys = sorted(set(pt_gen.keys()).intersection(jx_gen.keys()))
    print('[gen_loss diffs]')
    for k in gen_keys:
        s = _stats(pt_gen[k], jx_gen[k])
        print(f" {k}: mean_abs_diff={s['mean_abs_diff']:.3e}, max_abs_diff={s['max_abs_diff']:.3e}, shapes=({s['shape_a']},{s['shape_b']})")


if __name__ == '__main__':
    main()


