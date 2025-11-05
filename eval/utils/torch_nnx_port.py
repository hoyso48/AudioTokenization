import math
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

try:
    import jax.numpy as jnp
    from flax import nnx
except Exception:
    jnp = None
    nnx = None

# Optional repo-local modules for custom mappings
try:
    from AudioTokenization.CP.vq import module as pt_mod
except Exception:
    pt_mod = None

try:
    from AudioTokenization.BigCodec_NNX.vq import module_jax as jax_mod
except Exception:
    jax_mod = None


def _to_numpy(x: Any):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def _permute_pt_conv1d_weight_to_nnx(weight):
    # PT: (outC, inC/groups, k) -> NNX: (k, inC/groups, outC)
    return weight.permute(2, 1, 0) if getattr(weight, 'ndim', 0) == 3 else weight


def _permute_pt_convtranspose1d_weight_to_nnx(weight):
    # PT ConvTranspose1d: (inC, outC/groups, k) -> NNX ConvTranspose(transpose_kernel=True): (k, outC/groups, inC)
    return weight.permute(2, 1, 0) if getattr(weight, 'ndim', 0) == 3 else weight


def _transpose_linear_kernel_pt_to_nnx(weight):
    # PT: (out, in) -> NNX: (in, out)
    return weight.t() if getattr(weight, 'ndim', 0) == 2 else weight


def _squeeze_weight_g_to_vector(weight_g):
    if getattr(weight_g, 'ndim', 0) >= 1:
        try:
            return weight_g.reshape((weight_g.shape[0],))
        except Exception:
            return weight_g
    return weight_g


class Torch2NNX:
    """
    General torch -> NNX state copier (parameters and buffers), with warnings for unmapped items.

    - Recursively traverses module graphs, including lists/Sequential/ModuleList.
    - Handles Conv1d/ConvTranspose1d kernel permutations and Linear kernel transposes.
    - Maps LayerNorm weight/bias -> scale/bias.
    - Supports custom RMSNorm types from this repository.
    - Records mapping from torch param full names to NNX state keys for later comparison.
    """

    def __init__(self):
        self.modules_mapping_info: Dict[Type[Any], Dict[str, Any]] = {}

        if nn is not None:
            self.modules_mapping_info.update({
                getattr(nn, 'Conv2d', type('X', (), {})): {
                    "type": getattr(jax_mod, "WNConv2d", object),
                    "params_mapping": {
                        # Torch Conv2d: weight [out, in/groups, kH, kW]
                        # NNX WNConv2d: weight_v [kH, kW, in/groups, out]
                        "weight": "weight_v",
                        "bias": "bias",
                        "weight_v": "weight_v",
                        "weight_g": "weight_g",
                    },
                    "params_transform": lambda name, p: (
                        p.permute(2, 3, 1, 0) if name in ("weight", "weight_v") and getattr(p, 'ndim', 0) == 4 else (
                            _squeeze_weight_g_to_vector(p) if name == "weight_g" else p
                        )
                    ),
                },
                nn.Conv1d: {
                    "type": (getattr(nnx, "Conv", object), getattr(jax_mod, "DepthwiseConv1d", object), getattr(jax_mod, "WNConv1d", object), getattr(jax_mod, "WeightNorm", object)),
                    "params_mapping": {
                        # Plain conv params
                        "weight": "kernel",
                        "bias": "bias",
                        # WeightNorm params if present
                        "weight_v": "weight_v",
                        "weight_g": "weight_g",
                    },
                    "params_transform": lambda name, p: (
                        _permute_pt_conv1d_weight_to_nnx(p) if name in ("weight", "weight_v") else (
                            _squeeze_weight_g_to_vector(p) if name == "weight_g" else p
                        )
                    ),
                },
                nn.ConvTranspose1d: {
                    "type": (getattr(nnx, "ConvTranspose", object), getattr(jax_mod, "WNConvTranspose1d", object)),
                    "params_mapping": {
                        "weight": "kernel",
                        "bias": "bias",
                        "weight_v": "weight_v",
                        "weight_g": "weight_g",
                    },
                    "params_transform": lambda name, p: (
                        _permute_pt_convtranspose1d_weight_to_nnx(p) if name in ("weight", "weight_v") else (
                            _squeeze_weight_g_to_vector(p) if name == "weight_g" else p
                        )
                    ),
                },
                nn.Linear: {
                    "type": (getattr(nnx, "Linear", object), getattr(jax_mod if jax_mod else object, "WNConv1d", object)),
                    "params_mapping": {
                        # For nnx.Linear target
                        "weight": "kernel",
                        "bias": "bias",
                        # For WNConv1d target (weight-norm pointwise conv)
                        "weight_v": "weight_v",
                        "weight_g": "weight_g",
                    },
                    "params_transform": lambda name, p: (
                        # Linear kernel to (in, out)
                        _transpose_linear_kernel_pt_to_nnx(p) if name == "weight" else (
                            # weight_v to conv1x1 (1, in, out)
                            _transpose_linear_kernel_pt_to_nnx(p).unsqueeze(0) if name == "weight_v" and getattr(p, 'ndim', 0) == 2 else (
                                _squeeze_weight_g_to_vector(p) if name == "weight_g" else p
                            )
                        )
                    ),
                },
                getattr(nn, 'Embedding', type('X', (), {})): {
                    "type": getattr(nnx, "Embed", object),
                    "params_mapping": {
                        "weight": "embedding",
                    },
                },
                nn.LayerNorm: {
                    "type": getattr(nnx, "LayerNorm", object),
                    "params_mapping": {
                        "weight": "scale",
                        "bias": "bias",
                    },
                },
                nn.Dropout: {
                    "type": getattr(nnx, "Dropout", object),
                    "params_mapping": {},
                },
            })

        # Custom RMSNorm mapping (if available)
        if pt_mod is not None and jax_mod is not None and hasattr(pt_mod, "RMSNorm") and hasattr(jax_mod, "RMSNorm"):
            self.modules_mapping_info[pt_mod.RMSNorm] = {
                "type": jax_mod.RMSNorm,
                "params_mapping": {
                    "weight": "weight",
                },
            }

        # Torch types that are ignorable leaves (no state to copy)
        self.IGNORABLE_TORCH_LEAVES: Tuple[Type[Any], ...] = tuple(t for t in [
            getattr(nn, "Dropout", None),
            getattr(nn, "ReLU", None),
            getattr(nn, "GELU", None),
            getattr(nn, "SiLU", None),
            getattr(nn, "Tanh", None),
            getattr(nn, "Sigmoid", None),
            getattr(nn, "GLU", None),
            getattr(nn, "Identity", None),
        ] if t is not None)

        self.param_mapping_torch_to_nnx: Dict[str, str] = {}
        self.buffer_mapping_torch_to_nnx: Dict[str, str] = {}

    def _warn(self, msg: str):
        print(f"[Torch2NNX][WARN] {msg}")

    def _assert_compatible_type(self, nnx_module: Any, expected_type: Any) -> None:
        ok = False
        if isinstance(expected_type, tuple):
            ok = any(isinstance(nnx_module, t) for t in expected_type if isinstance(t, type))
        else:
            ok = isinstance(nnx_module, expected_type)
        if not ok:
            self._warn(f"Type mismatch: found NNX type={type(nnx_module).__name__}, expected one of={getattr(expected_type, '__name__', expected_type)}")
            raise AssertionError((nnx_module, type(nnx_module), expected_type))

    def _copy_params_buffers(self, torch_nn_module: Any, nnx_module: Any, *, torch_path: str, nnx_path: str) -> None:
        torch_module_type = type(torch_nn_module)
        if torch_module_type not in self.modules_mapping_info:
            self._warn(f"No mapping for module type {torch_module_type} at {torch_path}")
            return
        module_mapping_info = self.modules_mapping_info[torch_module_type]
        self._assert_compatible_type(nnx_module, module_mapping_info["type"])  # type: ignore[index]

        base_param_map: Mapping[str, str] = module_mapping_info.get("params_mapping", {})
        params_transform = module_mapping_info.get("params_transform", lambda name, p: p)

        # Detect torch weight_norm presence
        has_wn = hasattr(torch_nn_module, "weight_v") and hasattr(torch_nn_module, "weight_g")

        # Build dynamic mapping based on target attributes
        param_map: Dict[str, str] = dict(base_param_map)
        if hasattr(nnx_module, "weight_v") and "weight" in param_map:
            # Route plain 'weight' to 'weight_v' when target expects weight-norm style params
            param_map["weight"] = "weight_v"

        # Copy order: copy weight_v/weight_g first; potentially skip computed 'weight'
        keys_order: List[str] = list(param_map.keys())
        if has_wn and "weight" in keys_order:
            keys_order.remove("weight")
            keys_order.append("weight")

        copied_count = 0
        for torch_key in keys_order:
            nnx_key = param_map[torch_key]
            try:
                torch_value = getattr(torch_nn_module, torch_key, None)
            except Exception as e:
                self._warn(f"Error accessing Torch param '{torch_key}' at {torch_path}: {e}")
                continue
            # If NNX missing attribute entirely, warn and skip
            if not hasattr(nnx_module, nnx_key):
                if torch_value is not None:
                    self._warn(f"NNX param '{nnx_key}' missing at {nnx_path} while Torch has '{torch_key}' at {torch_path}")
                continue
            nnx_param = getattr(nnx_module, nnx_key)
            if nnx_param is None:
                if torch_value is not None:
                    self._warn(f"NNX param '{nnx_key}' is None at {nnx_path}; Torch has '{torch_key}'")
                continue

            # Skip copying computed 'weight' when torch module uses weight_norm; we'll copy weight_v/g instead
            if torch_key == "weight" and has_wn:
                continue

            if torch_value is None:
                tgt = getattr(nnx_param, "value", None)
                if tgt is not None:
                    self._warn(f"Torch param '{torch_key}' is None at {torch_path} but NNX expects '{nnx_key}' at {nnx_path}")
                continue

            torch_value = params_transform(torch_key, torch_value)

            # Special case: PT Linear.weight -> NNX WN/Conv1d.weight_v (k=1 conv)
            if torch_key == "weight" and nnx_key == "weight_v" and getattr(torch_value, 'ndim', 0) == 2:
                torch_value = _transpose_linear_kernel_pt_to_nnx(torch_value).unsqueeze(0)

            np_value = _to_numpy(torch_value)
            if jnp is not None:
                np_value = jnp.asarray(np_value)
            tgt = getattr(nnx_param, "value", None)

            # Bridge shapes for 1x1 conv <-> linear
            if tgt is not None and hasattr(np_value, 'shape') and hasattr(tgt, 'shape'):
                src_shape = tuple(np_value.shape)
                tgt_shape = tuple(tgt.shape)
                # Expand leading singleton (in,out)->(1,in,out)
                if len(src_shape) + 1 == len(tgt_shape) and tgt_shape[0] == 1 and src_shape == tgt_shape[1:]:
                    np_value = np_value.reshape(tgt_shape)
                    src_shape = tgt_shape
                # Squeeze leading singleton (1,in,out)->(in,out)
                if len(src_shape) == len(tgt_shape) + 1 and src_shape[0] == 1 and src_shape[1:] == tgt_shape:
                    np_value = np_value.reshape(tgt_shape)

            if tgt is not None and tuple(getattr(np_value, 'shape', ())) != tuple(getattr(tgt, 'shape', ())) :
                self._warn(f"Shape mismatch for {torch_path}.{torch_key} -> {nnx_path}.{nnx_key}: src={getattr(np_value, 'shape', None)} dst={getattr(tgt, 'shape', None)}")
            if hasattr(nnx_param, "value"):
                try:
                    nnx_param.value = np_value
                    copied_count += 1
                except Exception as e:
                    self._warn(f"Failed to assign param {nnx_path}.{nnx_key} from {torch_path}.{torch_key}: {e}")

            torch_full = f"{torch_path}.{torch_key}" if torch_path else torch_key
            nnx_full = f"{nnx_path}.{nnx_key}" if nnx_path else nnx_key
            self.param_mapping_torch_to_nnx[torch_full] = nnx_full

        if copied_count == 0 and base_param_map:
            self._warn(f"No params copied for mapped module at {torch_path} -> {nnx_path}. Check key names and shapes.")

    def _copy_sequential_like(self, torch_nn_seq: Any, nnx_seq: Any, *, torch_path: str, nnx_path: str, skip_modules: Optional[List[str]]):
        if isinstance(torch_nn_seq, (nn.Sequential, nn.ModuleList)):
            length = len(torch_nn_seq)
            nnx_layers = getattr(nnx_seq, "layers", None)
            if nnx_layers is None:
                nnx_layers = nnx_seq
            if len(nnx_layers) != length:
                self._warn(f"Sequential length mismatch at {torch_path} vs {nnx_path}: torch={length} nnx={len(nnx_layers)}; copying up to min length.")
            for i in range(min(length, len(nnx_layers))):
                torch_child = torch_nn_seq[i]
                nnx_child = nnx_layers[i]
                self.copy_module(torch_child, nnx_child,
                                 torch_path=f"{torch_path}.{i}" if torch_path else str(i),
                                 nnx_path=f"{nnx_path}.{i}" if nnx_path else str(i),
                                 skip_modules=skip_modules)
        else:
            raise AssertionError(type(torch_nn_seq))

    def _find_child_by_expected_type(self, nnx_parent: Any, torch_child_type: Type[Any]) -> Optional[Any]:
        mapping = self.modules_mapping_info.get(torch_child_type)
        if not mapping:
            return None
        expected = mapping.get("type")
        if expected is None:
            return None
        # NNX Modules might not have a regular __dict__, so avoid vars() here.
        # Prefer nnx.iter_graph if available to safely traverse sub-objects.
        try:
            from flax import nnx as _nnx  # local import to avoid hard dep when unavailable
            try:
                for _, value in _nnx.iter_graph(nnx_parent):
                    try:
                        if isinstance(expected, tuple):
                            if any(isinstance(value, t) for t in expected if isinstance(t, type)):
                                return value
                        else:
                            if isinstance(value, expected):
                                return value
                    except Exception:
                        continue
            except Exception:
                # Fallback to a conservative dir() traversal
                for attr_name in dir(nnx_parent):
                    # Skip dunder and private attributes
                    if attr_name.startswith('__'):
                        continue
                    try:
                        value = getattr(nnx_parent, attr_name)
                    except Exception:
                        continue
                    try:
                        if isinstance(expected, tuple):
                            if any(isinstance(value, t) for t in expected if isinstance(t, type)):
                                return value
                        else:
                            if isinstance(value, expected):
                                return value
                    except Exception:
                        continue
        except Exception:
            # Absolute fallback: nothing found
            return None
        return None

    def copy_module(self, torch_module: Any, nnx_module: Any, *, torch_path: str = "", nnx_path: str = "", skip_modules: Optional[List[str]] = None):
        if skip_modules is None:
            skip_modules = []

        if type(torch_module) in self.modules_mapping_info:
            self._copy_params_buffers(torch_module, nnx_module, torch_path=torch_path, nnx_path=nnx_path)
            return

        if torch_module.__class__.__name__ in skip_modules:
            return

        # Special case: map Sequential(Conv + Act) directly into a single NNX conv module
        if nn is not None and isinstance(torch_module, nn.Sequential) and hasattr(nnx_module, 'weight_v'):
            for child_name, torch_child in torch_module.named_children():
                if isinstance(torch_child, (getattr(nn, 'Conv2d', tuple()), getattr(nn, 'Conv1d', tuple()))):
                    self._copy_params_buffers(
                        torch_child,
                        nnx_module,
                        torch_path=f"{torch_path}.{child_name}" if torch_path else child_name,
                        nnx_path=nnx_path,
                    )
                    return
            # Fall through if no conv child found

        if nn is not None and isinstance(torch_module, (nn.Sequential, nn.ModuleList)):
            self._copy_sequential_like(torch_module, nnx_module, torch_path=torch_path, nnx_path=nnx_path, skip_modules=skip_modules)
            return

        # Handle ModuleDict by iterating its children and mapping by attribute name when possible
        if nn is not None and hasattr(nn, 'ModuleDict') and isinstance(torch_module, nn.ModuleDict):
            for child_name, torch_child in torch_module.items():
                # Prefer attribute name match on the NNX side
                nnx_child = getattr(nnx_module, child_name, None)
                child_nnx_path = f"{nnx_path}.{child_name}" if nnx_path else child_name
                if nnx_child is None:
                    # Fallback: try to find by expected type
                    nnx_child = self._find_child_by_expected_type(nnx_module, type(torch_child))
                    if nnx_child is None:
                        if isinstance(torch_child, self.IGNORABLE_TORCH_LEAVES):
                            continue
                        self._warn(f"No matching NNX child for {torch_path}.{child_name} at {nnx_path}")
                        continue
                self.copy_module(
                    torch_child,
                    nnx_child,
                    torch_path=f"{torch_path}.{child_name}" if torch_path else child_name,
                    nnx_path=child_nnx_path,
                    skip_modules=skip_modules,
                )
            return

        named_children = list(torch_module.named_children())
        if len(named_children) == 0:
            # Short-circuit ignorable leaves without params/buffers
            if any(isinstance(torch_module, t) for t in self.IGNORABLE_TORCH_LEAVES):
                return
            # Leaf: direct name matching
            any_mapped = False
            for name, torch_param in torch_module.named_parameters(recurse=False):
                if not hasattr(nnx_module, name):
                    self._warn(f"NNX leaf missing param '{name}' at {nnx_path} while Torch has it at {torch_path}")
                    continue
                nnx_param = getattr(nnx_module, name)
                if nnx_param is None:
                    self._warn(f"NNX leaf param '{name}' is None at {nnx_path}")
                    continue
                np_value = _to_numpy(torch_param)
                if jnp is not None:
                    np_value = jnp.asarray(np_value)
                if hasattr(nnx_param, "value"):
                    tgt = getattr(nnx_param, "value")
                    if tuple(getattr(np_value, 'shape', ())) != tuple(getattr(tgt, 'shape', ())) :
                        self._warn(f"Shape mismatch at leaf {torch_path}.{name} -> {nnx_path}.{name}")
                    nnx_param.value = np_value
                    self.param_mapping_torch_to_nnx[f"{torch_path}.{name}" if torch_path else name] = f"{nnx_path}.{name}" if nnx_path else name
                    any_mapped = True
            for name, torch_buffer in torch_module.named_buffers(recurse=False):
                if not hasattr(nnx_module, name):
                    self._warn(f"NNX leaf missing buffer '{name}' at {nnx_path} while Torch has it at {torch_path}")
                    continue
                nnx_buffer = getattr(nnx_module, name)
                if nnx_buffer is None:
                    self._warn(f"NNX leaf buffer '{name}' is None at {nnx_path}")
                    continue
                np_value = _to_numpy(torch_buffer)
                if jnp is not None:
                    np_value = jnp.asarray(np_value)
                if hasattr(nnx_buffer, "value"):
                    tgt = getattr(nnx_buffer, "value")
                    if tuple(getattr(np_value, 'shape', ())) != tuple(getattr(tgt, 'shape', ())) :
                        self._warn(f"Shape mismatch at leaf buffer {torch_path}.{name} -> {nnx_path}.{name}")
                    nnx_buffer.value = np_value
                    self.buffer_mapping_torch_to_nnx[f"{torch_path}.{name}" if torch_path else name] = f"{nnx_path}.{name}" if nnx_path else name
                    any_mapped = True
            if not any_mapped:
                self._warn(f"No params/buffers mapped for leaf at {torch_path} -> {nnx_path}")
            return

        for child_name, torch_child in named_children:
            nnx_child = getattr(nnx_module, child_name, None)
            child_nnx_path = f"{nnx_path}.{child_name}" if nnx_path else child_name
            if nnx_child is None:
                # If the Torch child is a ModuleDict, try to map its contents directly into current nnx_module
                if nn is not None and hasattr(nn, 'ModuleDict') and isinstance(torch_child, nn.ModuleDict):
                    self.copy_module(
                        torch_child,
                        nnx_module,
                        torch_path=f"{torch_path}.{child_name}" if torch_path else child_name,
                        nnx_path=nnx_path,
                        skip_modules=skip_modules,
                    )
                    continue
                nnx_child = self._find_child_by_expected_type(nnx_module, type(torch_child))
                if nnx_child is None:
                    if isinstance(torch_child, self.IGNORABLE_TORCH_LEAVES):
                        continue
                    self._warn(f"No matching NNX child for {torch_path}.{child_name} at {nnx_path}")
                    continue

            self.copy_module(
                torch_child,
                nnx_child,
                torch_path=f"{torch_path}.{child_name}" if torch_path else child_name,
                nnx_path=child_nnx_path,
                skip_modules=skip_modules,
            )

        for name, torch_buffer in torch_module.named_buffers(recurse=False):
            if name in [n for n, _ in named_children]:
                continue
            if hasattr(nnx_module, name):
                nnx_buffer = getattr(nnx_module, name)
                if nnx_buffer is None:
                    self._warn(f"NNX buffer '{name}' is None at {nnx_path}")
                    continue
                np_value = _to_numpy(torch_buffer)
                if jnp is not None:
                    np_value = jnp.asarray(np_value)
                if hasattr(nnx_buffer, "value"):
                    tgt = getattr(nnx_buffer, "value")
                    if tuple(getattr(np_value, 'shape', ())) != tuple(getattr(tgt, 'shape', ())) :
                        self._warn(f"Shape mismatch buffer {torch_path}.{name} -> {nnx_path}.{name}")
                    nnx_buffer.value = np_value
                    self.buffer_mapping_torch_to_nnx[f"{torch_path}.{name}" if torch_path else name] = f"{nnx_path}.{name}" if nnx_path else name

        for name, torch_param in torch_module.named_parameters(recurse=False):
            if name in [n for n, _ in named_children]:
                continue
            if hasattr(nnx_module, name):
                nnx_param = getattr(nnx_module, name)
                if nnx_param is None:
                    self._warn(f"NNX param '{name}' is None at {nnx_path}")
                    continue
                np_value = _to_numpy(torch_param)
                if jnp is not None:
                    np_value = jnp.asarray(np_value)
                if hasattr(nnx_param, "value"):
                    tgt = getattr(nnx_param, "value")
                    if tuple(getattr(np_value, 'shape', ())) != tuple(getattr(tgt, 'shape', ())) :
                        self._warn(f"Shape mismatch param {torch_path}.{name} -> {nnx_path}.{name}")
                    nnx_param.value = np_value
                    self.param_mapping_torch_to_nnx[f"{torch_path}.{name}" if torch_path else name] = f"{nnx_path}.{name}" if nnx_path else name

    def copy(self, torch_module: Any, nnx_module: Any, *, skip_modules: Optional[List[str]] = None) -> Dict[str, str]:
        self.param_mapping_torch_to_nnx.clear()
        self.buffer_mapping_torch_to_nnx.clear()
        self.copy_module(torch_module, nnx_module, torch_path="", nnx_path="", skip_modules=skip_modules)
        if not self.param_mapping_torch_to_nnx:
            self._warn("No parameters mapped at top-level")
        return dict(self.param_mapping_torch_to_nnx)


