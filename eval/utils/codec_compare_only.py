import sys, os
sys.path.insert(0, '/home/hoyeol')
# We'll manage precedence for these two explicitly right before dynamic imports
SSL_PATH = '/home/hoyeol/AudioTokenization/BigCodec_SSL'
NNX_PATH = '/home/hoyeol/AudioTokenization/BigCodec_NNX'
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

from omegaconf import OmegaConf
import numpy as np

import torch
from flax import nnx
import importlib.util

from AudioTokenization.utils.torch_nnx_port import Torch2NNX


def load_module_from_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prefer_paths_prepend(preferred_paths):
    """Move preferred_paths (in order) to the front of sys.path (after '/home/hoyeol')."""
    # Keep top-level '/home/hoyeol' at index 0 if present
    top = ['/home/hoyeol'] if sys.path and sys.path[0] == '/home/hoyeol' else []
    rest = [p for p in sys.path if p not in preferred_paths and p not in top]
    sys.path[:] = top + list(preferred_paths) + rest


def _clear_conflicting_packages():
    """Ensure top-level ambiguous package names will be re-imported from the right path."""
    targets = ('vq', 'module', 'criterions', 'dtp')
    for modname in list(sys.modules.keys()):
        for tgt in targets:
            if modname == tgt or modname.startswith(tgt + '.'):
                del sys.modules[modname]
                break


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
    cfg_path = '/home/hoyeol/outputs/2025-09-03/07-10-33/hydra/config.yaml'
    cfg = OmegaConf.load(cfg_path)

    # Dynamically load PT and JAX modules to avoid package path issues
    _prefer_paths_prepend([SSL_PATH])
    _clear_conflicting_packages()
    pt_mod = load_module_from_path(
        'pt_lightning', '/home/hoyeol/AudioTokenization/BigCodec_SSL/lightning_module.py'
    )
    # Now prepare env for JAX side and import
    _prefer_paths_prepend([NNX_PATH])
    _clear_conflicting_packages()
    jx_mod = load_module_from_path(
        'jx_codec_module', '/home/hoyeol/AudioTokenization/BigCodec_NNX/codec_module.py'
    )

    PT_Codec = getattr(pt_mod, 'CodecLightningModule')
    JX_Codec = getattr(jx_mod, 'CodecModule')

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


