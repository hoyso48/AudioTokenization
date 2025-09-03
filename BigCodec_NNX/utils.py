import jax
import jax.numpy as jnp
from flax import nnx
import logging
import numpy as np
from jax.experimental import mesh_utils

class Constant(nnx.Variable):
    pass

def create_device_mesh(dcn_data_parallelism=-1, 
                        dcn_fsdp_parallelism=1, 
                        dcn_tensor_parallelism=1, 
                        ici_data_parallelism=1, 
                        ici_fsdp_parallelism=-1, 
                        ici_tensor_parallelism=1):
    """Creates a device mesh with each slice in its own data parallel group. If there is only one slice, uses two replicas."""
    devices = jax.devices()
    num_devices = len(devices)
    try:
        num_slices = 1 + max([d.slice_index for d in devices])
    except:
        num_slices = 1
    num_devices_per_slice = num_devices // num_slices
    logging.info(f'Devices: {devices}')
    logging.info(f'Number of devices: {num_devices}')

    multi_slice_env = hasattr(devices[0], 'slice_index')

    dcn_parallelism = [
        dcn_data_parallelism,
        dcn_fsdp_parallelism,
        dcn_tensor_parallelism,
    ]
    ici_parallelism = [
        ici_data_parallelism,
        ici_fsdp_parallelism,
        ici_tensor_parallelism,
    ]

    # Find possible unspecified parallelisms
    dcn_parallelism = fill_unspecified_mesh_axes(
        dcn_parallelism, num_slices, 'DCN'
    )
    ici_parallelism = fill_unspecified_mesh_axes(
        ici_parallelism, num_devices_per_slice, 'ICI'
    )

    if multi_slice_env:
        mesh = mesh_utils.create_hybrid_device_mesh(
        ici_parallelism, dcn_parallelism
        )
    else:
        mesh = mesh_utils.create_device_mesh(ici_parallelism)

    print(f'Decided on mesh: {mesh}')
    print(f'Mesh shape: {mesh.shape}')
#   logging.info(f'Decided on mesh: {mesh}')
#   logging.info(f'Mesh shape: {mesh.shape}')

    return mesh


def fill_unspecified_mesh_axes(
    parallelism_vals, target_product, parallelism_type
    ):
    """Evaluates unspecified DCN/ICI parallelism values"""
    if -1 in parallelism_vals:
        assert parallelism_vals.count(-1) == 1, (
        f'Found unspecified values (-1) for more than one {parallelism_type}   '
        '   parallelism axis. At most one axis can be unspecified.'
        )

        determined_val = target_product / np.prod(parallelism_vals) * -1

        assert determined_val >= 1 and determined_val.is_integer, (
        'Unspecified value unable to be determined with the given     '
        f' {parallelism_type} parallelism values'
        )

        parallelism_vals[parallelism_vals.index(-1)] = int(determined_val)

    target_type = 'slices' if parallelism_type == 'DCN' else 'devices per slice'

    assert np.prod(parallelism_vals) == target_product, (
        f'Number of {target_type} {target_product} does not match    the product'
        f' of the {parallelism_type} parallelism {np.prod(parallelism_vals)}'
    )

    return parallelism_vals