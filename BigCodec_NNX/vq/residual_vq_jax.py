import jax
import jax.numpy as jnp
from flax import nnx
from typing import List, Tuple, Union

from .factorized_vector_quantize_jax import FactorizedVectorQuantize

class ResidualVQ(nnx.Module):
    """Flax NNX implementation of Residual Vector Quantization."""

    def __init__(
        self,
        *,
        num_quantizers: int,
        codebook_size: Union[int, List[int]],
        rngs: nnx.Rngs,
        # Pass FactorizedVectorQuantize args via kwargs (dim, codebook_dim, commitment)
        **kwargs
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        VQ = FactorizedVectorQuantize

        if isinstance(codebook_size, int):
            codebook_sizes = [codebook_size] * num_quantizers
        else:
            assert len(codebook_size) == num_quantizers
            codebook_sizes = codebook_size

        # Explicitly define the type hint for the list of modules
        self.layers: List[FactorizedVectorQuantize] = [
            VQ(codebook_size=size, rngs=rngs, **kwargs)
            for i, size in enumerate(codebook_sizes)
        ]

        # Determine output dimension D from the first layer's kwargs or projections
        if 'dim' not in kwargs:
             raise ValueError("ResidualVQ requires 'dim' in kwargs.")
        self._output_dim = kwargs['dim'] # Store D for vq2emb initialization

    def set_training_mode(self, training: bool):
        """Sets the training mode for all VQ layers."""
        for layer in self.layers:
            if hasattr(layer, 'set_training_mode'):
                layer.set_training_mode(training)

    def __call__(self, x: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Applies residual vector quantization."""
        quantized_out = jnp.zeros_like(x)
        residual = x

        all_losses = []
        all_indices = []

        for layer in self.layers:
            quantized, indices, loss = layer(residual) # layer.__call__
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            all_indices.append(indices)
            all_losses.append(jnp.mean(loss)) # loss shape is (B,)

        stacked_losses = jnp.stack(all_losses, axis=0)
        stacked_indices = jnp.stack(all_indices, axis=0)

        return quantized_out, stacked_indices, stacked_losses

    def vq2emb(self, vq_indices: jax.Array, proj: bool = True) -> jax.Array:
        """
        Converts stacked VQ indices (B, T, num_quantizers) to summed embeddings.
        Replicates the assumed logic from the original PyTorch implementation.
        """
        # Input shape assumption: B, T, num_quantizers
        assert vq_indices.ndim == 3, f"Expected vq_indices shape (B, T, num_quantizers), got {vq_indices.shape}"
        assert vq_indices.shape[-1] == self.num_quantizers, \
            f"Last dim of vq_indices ({vq_indices.shape[-1]}) != num_quantizers ({self.num_quantizers})"

        b, t, _ = vq_indices.shape
        # Initialize output tensor with the final dimension D
        quantized_out = jnp.zeros((b, t, self._output_dim), dtype=jnp.float32) # Use appropriate dtype

        for i in range(self.num_quantizers):
            layer = self.layers[i]
            # Indices for the current layer: shape (B, T)
            indices_for_layer = vq_indices[:, :, i]

            # Embed the indices: shape (B, T, codebook_D)
            embedded = layer.embed_code(indices_for_layer)

            # Determine the contribution for this layer based on 'proj' flag
            quantized_contribution: jax.Array
            if proj:
                # Apply output projection if it exists (handles Identity case)
                quantized_contribution = layer.out_proj(embedded) # Output shape (B, T, D)
            else:
                # If not projecting, use the direct embedding.
                # This requires D == codebook_D for summation to be valid.
                if self._output_dim != layer.codebook_dim:
                    raise ValueError(f"Cannot sum embeddings with proj=False because "
                                     f"output dimension ({self._output_dim}) != "
                                     f"codebook dimension ({layer.codebook_dim}) in layer {i}. "
                                     f"Consider using proj=True or ensure dimensions match.")
                quantized_contribution = embedded # Output shape (B, T, codebook_D)

            quantized_out = quantized_out + quantized_contribution

        return quantized_out

    def get_embedding_weights(self) -> List[jax.Array]:
        """Gets the codebook weights from each layer."""
        # Check for shared codebook (simple object identity check)
        if len(self.layers) > 1 and all(layer is self.layers[0] for layer in self.layers):
             return [self.layers[0].codebook_weights] # Shared
        else:
             return [layer.codebook_weights for layer in self.layers] # Independent