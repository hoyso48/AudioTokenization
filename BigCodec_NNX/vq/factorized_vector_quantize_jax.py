# --- START OF FILE factorized_vector_quantize_nnx_v3.py ---

from typing import Tuple, Optional

import jax
import jax.numpy as jnp
from flax import nnx  # Use the stable flax.nnx
from einops import rearrange # Make sure einops is installed for JAX
from AudioTokenization.BigCodec_NNX.common.conv_weightnorm import WNConv1d

# Helper NNX Identity module
class Identity(nnx.Module):
    def __init__(self):
        pass
    def __call__(self, x):
        return x

def _l2_normalize(x, axis=None, epsilon=1e-12):
  """Normalizes along dimension `axis` using an L2 norm.

  This specialized function exists for numerical stability reasons.

  Args:
    x: An input ndarray.
    axis: Dimension along which to normalize, e.g. `1` to separately normalize
      vectors in a batch. Passing `None` views `t` as a flattened vector when
      calculating the norm (equivalent to Frobenius norm).
    eps: Epsilon to avoid dividing by zero.

  Returns:
    An array of the same shape as 'x' L2-normalized along 'axis'.
  """
  return x * jax.lax.rsqrt((x * x).sum(axis=axis, keepdims=True) + epsilon)

class FactorizedVectorQuantize(nnx.Module):
    """
    Flax NNX implementation of Factorized Vector Quantization.
    All inputs and outputs assume Batch x Time x Dimension (B, T, D) format.
    Uses nnx.Conv1d with kernel_size=1 for projections instead of nnx.Linear.

    Note: weight_norm from the original PyTorch implementation is omitted.
    """
    def __init__(
        self,
        dim: int,               # Input/Output dimension (D in B, T, D)
        codebook_size: int,     # Number of codes in the codebook
        codebook_dim: int,      # Dimension of codes and projected space (D' in B, T, D')
        commitment: float,      # Commitment loss weight
        *,                      # Ensure rngs is keyword-only
        rngs: nnx.Rngs          # RNGs for initializing layers
    ):
        """
        Initializes the FactorizedVectorQuantize module.

        Args:
            dim: Dimension of the input and output vectors (D in B, T, D format).
            codebook_size: The number of vectors in the codebook.
            codebook_dim: The dimension of the vectors in the codebook (D').
                           Also the dimension after the input projection.
            commitment: Weight for the commitment loss term.
            rngs: The nnx.Rngs object for initializing parameters.
        """
        super().__init__()
        self.codebook_size = codebook_size # Static config
        self.codebook_dim = codebook_dim   # Static config
        self.commitment = commitment       # Static config
        self.dim = dim                     # Static config

        # Projections using Conv1d with kernel_size=1
        # Input: (B, T, D), Output: (B, T, codebook_D) or (B, T, D)
        if dim != self.codebook_dim:
            # Conv1d expects input (B, T, D) - features are channels
            self.in_proj = WNConv1d(
                in_features=dim,
                out_features=self.codebook_dim,
                kernel_size=1, # Kernel size of 1 for point-wise transformation
                rngs=rngs
            )
            self.out_proj = WNConv1d(
                in_features=self.codebook_dim,
                out_features=dim,
                kernel_size=1,
                rngs=rngs
            )
        else:
            self.in_proj = Identity()
            self.out_proj = Identity()

        # Codebook Embedding Layer
        # Stores weights of shape (codebook_size, codebook_dim)
        self._codebook = nnx.Embed(codebook_size, self.codebook_dim, rngs=rngs)
        self.deterministic = False

    @property
    def codebook(self) -> nnx.Embed:
        """Returns the nnx.Embed layer instance."""
        return self._codebook

    @property
    def codebook_weights(self) -> jax.Array:
        """Access the codebook embedding weights as a JAX array."""
        return self._codebook.embedding.value

    def set_training_mode(self, training: bool):
        """Explicitly set the training mode."""
        self.training = training

    def __call__(self, z: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Quantizes the input tensor using a fixed codebook and returns
        the corresponding codebook vectors.

        Args:
            z: Input tensor Batch x Time x Dimension (B, T, D).

        Returns:
            A tuple containing:
            - z_q: Quantized continuous representation (B, T, D).
            - indices: Codebook indices (B, T).
            - commit_loss: Commitment loss scalar per batch element (B,).
        """
        # Input shape is assumed B T D
        assert z.ndim == 3, f"Input tensor z must have 3 dimensions (B, T, D), got {z.ndim}"
        assert z.shape[-1] == self.dim, f"Input last dimension {z.shape[-1]} != self.dim {self.dim}"
        b, t, d = z.shape

        # Project input into low-dimensional space using Conv1d
        # Input: (B, T, D), Output: (B, T, codebook_D)
        z_e = self.in_proj(z)

        # --- Quantization ---
        # decode_latents needs (B*T, codebook_D) format internally
        # It returns indices (B, T) and z_q (B, T, codebook_D)
        z_q_quantized, indices = self.decode_latents(z_e) # z_q: (B, T, codebook_D), indices: (B, T)
        # --- End Quantization ---

        # Calculate commitment loss
        # In eval mode (deterministic=True), mirror Torch behavior and return zero commit loss
        if getattr(self, 'deterministic', False):
            commit_loss = jnp.zeros(b, dtype=z.dtype)
        else:
            # Use stop_gradient which is equivalent to detach
            # Both z_e and z_q_quantized are (B, T, codebook_D)
            detached_z_q = jax.lax.stop_gradient(z_q_quantized)
            detached_z_e = jax.lax.stop_gradient(z_e)

            # MSE Loss per batch element (mean over T, D' dimensions)
            commitment_loss_per_element = jnp.mean((z_e - detached_z_q) ** 2, axis=(1, 2)) * self.commitment
            codebook_loss_per_element = jnp.mean((z_q_quantized - detached_z_e) ** 2, axis=(1, 2))

            commit_loss = commitment_loss_per_element + codebook_loss_per_element # Shape: (B,)

        # Straight-Through Estimator (STE)
        # Both z_e and z_q_quantized are (B, T, codebook_D)
        z_q_ste = z_e + jax.lax.stop_gradient(z_q_quantized - z_e) # Shape: (B, T, codebook_D)

        # Project back to original dimension using Conv1d
        # Input: (B, T, codebook_D), Output: (B, T, D)
        z_q = self.out_proj(z_q_ste)
        return z_q, indices, commit_loss

    def decode_latents(self, z_e: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Finds the nearest codebook vectors for the projected latent vectors z_e.

        Args:
            z_e: Projected latent tensor of shape (B, T, codebook_D).

        Returns:
            A tuple containing:
            - z_q: Quantized vectors corresponding to indices (B, T, codebook_D).
            - indices: Codebook indices for the latents (B, T).
        """
        b, t, d = z_e.shape
        assert d == self.codebook_dim, f"Latent dimension {d} != codebook dimension {self.codebook_dim}"

        # Reshape latents to (B*T, codebook_D) for distance calculation
        encodings = rearrange(z_e, "b t d -> (b t) d")

        # Get codebook weights
        codebook_w = self.codebook_weights # Shape: (codebook_N, codebook_D)

        # L2 normalize encodings and codebook
        epsilon = 1e-6
        encodings_norm = _l2_normalize(encodings, axis=-1, epsilon=epsilon)
        codebook_norm = _l2_normalize(codebook_w, axis=-1, epsilon=epsilon)

        # Compute cosine similarity (equivalent to minimizing L2 for normalized vectors)
        cosine_sim = encodings_norm @ codebook_norm.T # Shape: (B*T, codebook_N)

        # Find the index of the closest codebook vector
        indices_flat = jnp.argmax(cosine_sim, axis=-1) # Shape: (B*T,)

        # Reshape indices back to (B, T)
        indices = rearrange(indices_flat, "(b t) -> b t", b=b, t=t) # Shape: (B, T)

        # Get the quantized vectors from the codebook using the indices
        # embed_code returns (B, T, codebook_D) directly
        z_q = self.embed_code(indices) # Shape: (B, T, codebook_D)

        return z_q, indices

    # --- Helper methods consistent with (B, T, D) ---

    def embed_code(self, embed_id: jax.Array) -> jax.Array:
        """Embeds codebook indices into vectors."""
        # Input embed_id shape: e.g., (B, T)
        # Output shape: (B, T, codebook_D)
        return self._codebook(embed_id)

    def decode_code(self, embed_id: jax.Array) -> jax.Array:
        """
        Gets embedding vectors.
        In this (B, T, D) setup, no transpose is needed after embedding.
        This method might be less necessary or just call embed_code.
        Keeping it for potential compatibility/clarity.
        """
        # Input embed_id shape: (B, T)
        # Output shape: (B, T, codebook_D)
        return self.embed_code(embed_id)
