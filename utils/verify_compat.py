from typing import Any, Callable, Dict, Optional, Tuple

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

import numpy as np

from AudioTokenization.utils.torch_nnx_port import Torch2NNX


def _set_precision_torch(precision: str):
    if torch is None:
        return None
    if not torch.cuda.is_available():
        return torch.float32
    if precision == 'bf16':
        return torch.bfloat16
    elif precision == 'fp16':
        return torch.float16
    else:
        return torch.float32


def _set_precision_jax(precision: str):
    if jnp is None:
        return np.float32
    if jax.default_backend() == 'cpu':
        return jnp.float32
    if precision == 'bf16':
        return jnp.bfloat16
    elif precision == 'fp16':
        return jnp.float16
    else:
        return jnp.float32


def _jax_from_pt(x: torch.Tensor, dtype) -> jnp.ndarray:
    arr = x.detach().cpu().numpy()
    return jnp.asarray(arr).astype(dtype)


def _abs_diff_stats(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    diff = np.abs(a - b)
    return {
        'min_abs_diff': float(diff.min()),
        'mean_abs_diff': float(diff.mean()),
        'max_abs_diff': float(diff.max()),
    }


def _axes_order(src_layout: str, dst_layout: str):
    if len(src_layout) != 3 or len(dst_layout) != 3:
        return None
    idx = {ch: i for i, ch in enumerate(src_layout)}
    try:
        return tuple(idx[ch] for ch in dst_layout)
    except KeyError:
        return None


def _permute_torch(x: torch.Tensor, src_layout: str, dst_layout: str) -> torch.Tensor:
    if x.ndim != 3 or src_layout == dst_layout:
        return x
    order = _axes_order(src_layout, dst_layout)
    if order is None:
        return x
    return x.permute(*order)


def _permute_np(x: np.ndarray, src_layout: str, dst_layout: str) -> np.ndarray:
    if x.ndim != 3 or src_layout == dst_layout:
        return x
    order = _axes_order(src_layout, dst_layout)
    if order is None:
        return x
    return np.transpose(x, order)


def _flatten_nnx_state_to_arrays(state: Any) -> Dict[str, np.ndarray]:
    # Convert nnx state tree into flat dict of numpy arrays
    flat: Dict[str, np.ndarray] = {}
    try:
        pure = nnx.tree_util.state_to_pure_dict(state)
    except Exception:
        pure = {}
    def _recurse(d: Dict[str, Any], prefix: str = ""):
        for k, v in d.items():
            name = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                _recurse(v, name)
            else:
                try:
                    arr = np.asarray(v)
                    flat[name] = arr
                except Exception:
                    pass
    _recurse(pure)
    return flat


def verify_forward_and_grads(
    torch_module: Any,
    nnx_module: Any,
    make_input: Callable[[str], Tuple[torch.Tensor, ...]],
    *,
    precision: str = 'fp32',
    jitted: bool = True,
    loss_fn: Optional[Callable[[Any, Any], Any]] = None,
    torch_layout: str = 'NCL',
    nnx_layout: str = 'NLC',
    roundtrip_check: bool = True,
    build_fresh_torch: Optional[Callable[[], Any]] = None,
    user_key_mapping: Optional[Dict[str, str]] = None,
    call_kwargs: Optional[Dict[str, Any]] = None,
    compare_vq_indices: bool = False,
) -> Dict[str, Any]:
    assert torch is not None and jax is not None and nnx is not None

    torch_dtype = _set_precision_torch(precision)
    jax_dtype = _set_precision_jax(precision)

    pt_inputs = make_input(precision)
    pt_inputs = tuple(x.to(dtype=torch_dtype).requires_grad_(True) for x in pt_inputs)

    torch_module.eval()
    call_kwargs = call_kwargs or {}
    torch_out = torch_module(*pt_inputs, **call_kwargs)

    jax_inputs = []
    for x in pt_inputs:
        x_j = _permute_torch(x, torch_layout, nnx_layout)
        jax_inputs.append(_jax_from_pt(x_j, jax_dtype))
    jax_inputs = tuple(jax_inputs)

    def jax_forward(*xs):
        nnx_module.eval()
        return nnx_module(*xs, **call_kwargs)

    jax_forward_comp = jax.jit(jax_forward) if jitted else jax_forward
    jax_out = jax_forward_comp(*jax_inputs)

    # Select primary tensor if module returns a tuple/list (e.g., (x, q, loss))
    torch_primary = torch_out[0] if isinstance(torch_out, (tuple, list)) else torch_out
    jax_primary = jax_out[0] if isinstance(jax_out, (tuple, list)) else jax_out

    jax_out_np = np.asarray(jax_primary)
    # Only permute output if shapes differ; avoids mis-permuting audio tensors already aligned
    try:
        torch_shape = tuple(torch_primary.shape) if hasattr(torch_primary, 'shape') else None
        jax_shape = tuple(jax_primary.shape) if hasattr(jax_primary, 'shape') else None
        if torch_shape is not None and jax_shape is not None and torch_shape != jax_shape and jax_out_np.ndim == 3:
            jax_out_np = _permute_np(jax_out_np, nnx_layout, torch_layout)
    except Exception:
        # Fallback: leave as-is
        pass

    out_stats = _abs_diff_stats(np.asarray(torch_primary.detach().cpu().float()), jax_out_np.astype(np.float32))

    # Optional: compare exact equality of VQ indices (q) when available
    vq_stats: Optional[Dict[str, Any]] = None
    if compare_vq_indices:
        try:
            # Expect tuple/list outputs: (quantized, q, commit_loss)
            torch_q = torch_out[1]
            jax_q = jax_out[1]
            # Convert to numpy
            tq = np.asarray(torch_q.detach().cpu().numpy())
            jq = np.asarray(jax_q)
            # Try to reconcile potential axis permutations
            if tq.shape != jq.shape and tq.ndim == jq.ndim == 3:
                import itertools as _it
                matched = False
                for perm in _it.permutations(range(3)):
                    if jq.transpose(perm).shape == tq.shape:
                        jq = jq.transpose(perm)
                        matched = True
                        break
                if not matched:
                    # leave as-is; shape mismatch will be reported
                    pass
            all_equal = False
            frac_equal = 0.0
            shape_equal = tuple(tq.shape) == tuple(jq.shape)
            if shape_equal:
                eq = (tq == jq)
                num_equal = int(eq.sum())
                total = int(eq.size)
                frac_equal = float(num_equal) / float(max(total, 1))
                all_equal = (num_equal == total)
                vq_stats = {
                    'shape_equal': shape_equal,
                    'all_equal': all_equal,
                    'fraction_equal': frac_equal,
                    'shape': tuple(tq.shape),
                }
            else:
                vq_stats = {
                    'shape_equal': shape_equal,
                    'all_equal': False,
                    'fraction_equal': 0.0,
                    'torch_shape': tuple(tq.shape),
                    'jax_shape': tuple(jq.shape),
                }
        except Exception as e:
            vq_stats = {'error': str(e)}

    if loss_fn is None:
        def _default_loss(y):
            y0 = y[0] if isinstance(y, (tuple, list)) else y
            return (y0.float() ** 2).mean()
        loss_fn = _default_loss

    torch_loss = loss_fn(torch_out)
    torch_loss.backward()

    def loss_on_nnx(model, *xs):
        y = model(*xs, **call_kwargs)
        y0 = y[0] if isinstance(y, (tuple, list)) else y
        y32 = y0.astype(jnp.float32)
        return jnp.mean(y32 * y32)

    # Do not jit over model reference
    _, grads = nnx.value_and_grad(lambda m: loss_on_nnx(m, *jax_inputs))(nnx_module)

    pt_grads: Dict[str, np.ndarray] = {}
    for name, p in torch_module.named_parameters():
        if p.grad is not None:
            pt_grads[name] = p.grad.detach().cpu().float().numpy()

    flat_nnx_grads = _flatten_nnx_state_to_arrays(grads)

    pt_all = np.concatenate([g.ravel() for g in pt_grads.values()]) if pt_grads else np.array([0.0], dtype=np.float32)
    nnx_all = np.concatenate([g.ravel() for g in flat_nnx_grads.values()]) if flat_nnx_grads else np.array([0.0], dtype=np.float32)
    grad_overall = _abs_diff_stats(pt_all.astype(np.float32), nnx_all.astype(np.float32))

    result: Dict[str, Any] = {
        'forward_stats': out_stats,
        'grad_overall_stats': grad_overall,
        'torch_out_shape': tuple(torch_primary.shape) if hasattr(torch_primary, 'shape') else None,
        'jax_out_shape': tuple(jax_primary.shape) if hasattr(jax_primary, 'shape') else None,
    }

    if vq_stats is not None:
        result['vq_code_stats'] = vq_stats

    return result


def build_and_verify_pair(
    build_torch: Callable[[], Any],
    build_nnx: Callable[[], Any],
    make_input: Callable[[str], Tuple[torch.Tensor, ...]],
    *,
    precision: str = 'fp32',
    jitted: bool = True,
    torch_layout: str = 'NCL',
    nnx_layout: str = 'NLC',
    call_kwargs: Optional[Dict[str, Any]] = None,
    compare_vq_indices: bool = False,
) -> Dict[str, Any]:
    assert torch is not None and jax is not None and nnx is not None

    t_mod = build_torch()
    n_mod = build_nnx()

    # Copy weights PT -> NNX
    t2f = Torch2NNX()
    mapping = t2f.copy(t_mod, n_mod)

    # Verify
    return verify_forward_and_grads(
        t_mod,
        n_mod,
        make_input,
        precision=precision,
        jitted=jitted,
        torch_layout=torch_layout,
        nnx_layout=nnx_layout,
        roundtrip_check=False,
        build_fresh_torch=None,
        user_key_mapping=mapping,
        call_kwargs=call_kwargs,
        compare_vq_indices=compare_vq_indices,
    )


