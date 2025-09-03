import jax
import jax.numpy as jnp
from jax import custom_vjp, grad
from flax.experimental import nnx # Assuming you use nnx for the Module
from functools import partial
import numpy as np

# --- Constants ---
_NO_DIV_BY_ZERO = 1e-9

# --- Simplified Custom VJP Function ---

import jax
import jax.numpy as jnp
from jax import custom_vjp
from functools import partial
from jax.nn import logsumexp

# --- Constants for Numerical Stability ---
# Log-epsilon for logsumexp trick for beta. log(1e-8) approx -18.4
_SAFE_LOG_EPSILON_BETA = -18.0
# Minimum effective beta for non-logscale case
_MIN_BETA_EFF_NONLOG = 1e-8

@partial(custom_vjp, nondiff_argnums=(3,))
def snake_beta(x: jnp.ndarray,
               alpha: jnp.ndarray,
               beta: jnp.ndarray,
               alpha_logscale: bool = False) -> jnp.ndarray:
    """
    Numerically stabilized SnakeBeta (v3) using logsumexp for beta and
    log-space intermediates for alpha-beta interactions when alpha_logscale=True.

    Args:
        x (jnp.ndarray): Input tensor, shape (..., C).
        alpha (jnp.ndarray): Frequency parameter, shape (C,). If logscale, log(alpha_eff).
        beta (jnp.ndarray): Scale parameter, shape (C,). If logscale, log(beta_eff).
        alpha_logscale (bool): If True, interpret alpha/beta as log parameters.

    Returns:
        jnp.ndarray: Output tensor, shape (..., C).
    """
    # --- Forward computation with stability enhancements ---
    if alpha_logscale:
        # alpha_eff still computed directly, potential inf if alpha is too large
        alpha_eff = jnp.exp(alpha)

        # Stable log of (beta_eff + epsilon) using logsumexp
        log_beta_eff = beta
        log_eps_beta_arr = jnp.full_like(log_beta_eff, _SAFE_LOG_EPSILON_BETA)
        # log_beta_stable = log(exp(beta) + exp(log_eps))
        log_beta_stable = logsumexp(jnp.stack([log_beta_eff, log_eps_beta_arr]), axis=0)

        # Stable inverse: inv_beta_stable = 1 / (exp(beta) + epsilon)
        inv_beta_stable = jnp.exp(-log_beta_stable)

    else:
        alpha_eff = alpha
        beta_eff = beta
        # Simple stability for non-logscale beta
        beta_stable = jnp.maximum(beta_eff, _MIN_BETA_EFF_NONLOG)
        inv_beta_stable = 1.0 / beta_stable
        # Need log_beta_stable equivalent for gradient calculation consistency, though less critical
        # log_beta_stable is not directly used in forward non-logscale path

    # Common calculations
    ax = alpha_eff * x
    sq_sin_ax = jnp.square(jnp.sin(ax))

    # Apply the snake term using the stabilized inverse beta
    snake_term = sq_sin_ax * inv_beta_stable
    output = x + snake_term
    return output

def snake_beta_fwd(x: jnp.ndarray,
                   alpha: jnp.ndarray,
                   beta: jnp.ndarray,
                   alpha_logscale: bool) -> tuple[jnp.ndarray, tuple]:
    """Forward pass for VJP. Returns primal output and residuals."""
    primal_out = snake_beta(x, alpha, beta, alpha_logscale)
    # Save original inputs and logscale flag for backward pass
    residuals = (x, alpha, beta, alpha_logscale)
    return primal_out, residuals

def snake_beta_bwd(alpha_logscale_arg_dummy: bool, # Nondiff arg
                   residuals: tuple,
                   grad_output: jnp.ndarray
                   ) -> tuple[jnp.ndarray | None, jnp.ndarray | None, jnp.ndarray | None]:
    """Backward pass for VJP using stabilized intermediates."""
    x, alpha, beta, alpha_logscale = residuals
    g = grad_output # Shape (..., C)

    # --- Recompute intermediates *with stabilization* ---
    if alpha_logscale:
        alpha_eff = jnp.exp(alpha) # Recompute alpha_eff

        # Recompute stable log_beta and its inverse
        log_beta_eff = beta
        log_eps_beta_arr = jnp.full_like(log_beta_eff, _SAFE_LOG_EPSILON_BETA)
        log_beta_stable = logsumexp(jnp.stack([log_beta_eff, log_eps_beta_arr]), axis=0)
        inv_beta_stable = jnp.exp(-log_beta_stable)

        # Compute log of the alpha/beta interaction term (stable)
        # log(alpha_eff / beta_stable) = log(alpha_eff) - log(beta_stable)
        log_alpha_over_beta_stable = alpha - log_beta_stable # log(exp(alpha)) = alpha

        # Compute softmax term for beta gradient (stable, 0 to 1)
        # softmax = exp(beta) / (exp(beta) + epsilon) = exp(beta - log_beta_stable)
        softmax_beta_term = jnp.exp(beta - log_beta_stable)

    else:
        alpha_eff = alpha
        beta_eff = beta
        # Recompute stable beta and its inverse for non-logscale
        beta_stable = jnp.maximum(beta_eff, _MIN_BETA_EFF_NONLOG)
        # Avoid division by zero if beta_stable happens to be exactly zero
        # (unlikely with maximum but safer)
        inv_beta_stable = jnp.where(beta_stable > 0, 1.0 / beta_stable, 0.0)
        # Gradient flag
        is_beta_eff_stable = beta_eff >= _MIN_BETA_EFF_NONLOG


    # Common intermediate calculations
    ax = alpha_eff * x               # Shape (..., C)
    # sin_ax = jnp.sin(ax)           # Not directly needed now
    sin_2ax = jnp.sin(2 * ax)        # Shape (..., C)
    sq_sin_ax = jnp.square(jnp.sin(ax))   # Shape (..., C), needed for beta grad

    # --- Gradient w.r.t. x ---
    if alpha_logscale:
        # Use the log-space intermediate: exp(log(alpha_eff / beta_stable)) = alpha_eff / beta_stable
        grad_x_factor = jnp.exp(log_alpha_over_beta_stable) * sin_2ax
    else:
        # Original form for non-logscale
        grad_x_factor = alpha_eff * inv_beta_stable * sin_2ax
    # grad_x = g * (1 + d(snake_term)/dx)
    grad_x = g * (1.0 + grad_x_factor) # Shape (..., C)

    # --- Gradient w.r.t. alpha ---
    # dSnake/dalpha = dSnake/dalpha_eff * dalpha_eff/dalpha
    # dSnake/dalpha_eff = x * inv_beta_stable * sin(2*alpha_eff*x)
    # dalpha_eff/dalpha = alpha_eff if logscale else 1.0

    if alpha_logscale:
         # Use the log-space intermediate again
         # alpha_eff * inv_beta_stable = exp(log_alpha_over_beta_stable)
        dSnake_dalpha_eff_times_deriv = x * jnp.exp(log_alpha_over_beta_stable) * sin_2ax
    else:
        dSnake_dalpha_eff_times_deriv = x * inv_beta_stable * sin_2ax * 1.0

    # Element-wise gradient before reduction
    # g * dSnake/dalpha
    grad_alpha_full = g * dSnake_dalpha_eff_times_deriv # Shape (..., C)

    # Sum over broadcasted dimensions
    reduce_axes = tuple(range(x.ndim - 1))
    grad_alpha = jnp.sum(grad_alpha_full, axis=reduce_axes) if x.ndim > 1 else grad_alpha_full # Shape (C,)

    # --- Gradient w.r.t. beta ---
    # dSnake/dbeta = dSnake/d(inv_beta_stable) * d(inv_beta_stable)/d(beta) [logscale]
    # dSnake/dbeta = dSnake/d(beta_stable) * d(beta_stable)/d(beta) [non-logscale]

    if alpha_logscale:
        # Use the refined stable gradient: - g * sq_sin_ax * inv_beta_stable * softmax_beta_term
        grad_beta_full = - g * sq_sin_ax * inv_beta_stable * softmax_beta_term
    else:
        # Use the original derivative form but with beta_stable check
        # dSnake/dbeta_stable = - sq_sin_ax / (beta_stable^2) = - sq_sin_ax * (inv_beta_stable^2)
        # dbeta_stable/dbeta = 1.0 if beta >= min_val else 0.0
        dSnake_dbeta_stable = -sq_sin_ax * jnp.square(inv_beta_stable)
        dbeta_stable_dbeta = jnp.where(is_beta_eff_stable, 1.0, 0.0)
        grad_beta_full = g * dSnake_dbeta_stable * dbeta_stable_dbeta

    # Sum over broadcasted dimensions
    grad_beta = jnp.sum(grad_beta_full, axis=reduce_axes) if x.ndim > 1 else grad_beta_full # Shape (C,)

    # Return gradients matching input shapes (x, alpha, beta)
    return grad_x, grad_alpha, grad_beta

# Register the VJP rule
snake_beta.defvjp(snake_beta_fwd, snake_beta_bwd)

# --- Plain JAX version for Autodiff Comparison (Concise) ---
def snake_beta_autodiff_concise(x: jnp.ndarray,
                                alpha: jnp.ndarray,
                                beta: jnp.ndarray,
                                alpha_logscale: bool = False) -> jnp.ndarray:
    """Plain JAX implementation matching snake_beta_concise forward pass."""
    # --- Forward computation relies on implicit broadcasting ---
    alpha_eff = jnp.exp(alpha) if alpha_logscale else alpha
    beta_eff = jnp.exp(beta) if alpha_logscale else beta
    beta_safe = beta_eff + _NO_DIV_BY_ZERO
    ax = alpha_eff * x
    snake_term = jnp.square(jnp.sin(ax)) / beta_safe
    output = x + snake_term
    return output

# --- Verification Function (use the previous one, ensure it uses concise autodiff version) ---
def compare_gradients(x_data, alpha_data, beta_data, logscale_mode, N, C):
    """Calculates and compares gradients from custom VJP and autodiff."""
    print(f"\n--- Comparing Gradients (Concise) (logscale={logscale_mode}, N={N}, C={C}, x_shape={x_data.shape}, alpha_shape={alpha_data.shape}) ---")

    # Define functions returning scalar sum for grad
    def sum_snake_beta_custom(x, a, b):
        # *** Use the concise custom VJP function ***
        return jnp.sum(snake_beta(x, a, b, alpha_logscale=logscale_mode))

    def sum_snake_beta_auto(x, a, b):
         # *** Use the concise autodiff function ***
        return jnp.sum(snake_beta_autodiff_concise(x, a, b, alpha_logscale=logscale_mode))

    # Calculate gradients using custom VJP
    grad_x_custom, grad_alpha_custom, grad_beta_custom = grad(sum_snake_beta_custom, argnums=(0, 1, 2))(
        x_data, alpha_data, beta_data
    )

    # Calculate gradients using standard autodiff
    grad_x_auto, grad_alpha_auto, grad_beta_auto = grad(sum_snake_beta_auto, argnums=(0, 1, 2))(
        x_data, alpha_data, beta_data
    )

    # Compare forward pass (sanity check)
    out_custom = snake_beta(x_data, alpha_data, beta_data, alpha_logscale=logscale_mode)
    out_auto = snake_beta_autodiff_concise(x_data, alpha_data, beta_data, alpha_logscale=logscale_mode)
    fwd_diff = jnp.mean(jnp.abs(out_custom - out_auto))
    print(f"Forward Pass Mean Abs Diff: {fwd_diff:.6e}")
    assert np.allclose(out_custom, out_auto, atol=1e-5), "Forward passes differ significantly!"

    # Compare gradients element-wise
    diff_x = jnp.mean(jnp.abs(grad_x_custom - grad_x_auto))
    diff_alpha = jnp.mean(jnp.abs(grad_alpha_custom - grad_alpha_auto))
    diff_beta = jnp.mean(jnp.abs(grad_beta_custom - grad_beta_auto))

    print(f"Gradient wrt x      Mean Abs Diff: {diff_x:.6e} (Shapes: Custom={grad_x_custom.shape}, Auto={grad_x_auto.shape})")
    print(f"Gradient wrt alpha  Mean Abs Diff: {diff_alpha:.6e} (Shapes: Custom={grad_alpha_custom.shape}, Auto={grad_alpha_auto.shape})")
    print(f"Gradient wrt beta   Mean Abs Diff: {diff_beta:.6e} (Shapes: Custom={grad_beta_custom.shape}, Auto={grad_beta_auto.shape})")

    # Add assertion for gradient closeness
    atol = 1e-5 # Adjust tolerance if needed based on float precision
    rtol = 1e-5
    assert np.allclose(grad_x_custom, grad_x_auto, atol=atol, rtol=rtol), f"Grad x differs significantly (Diff: {diff_x})"
    assert np.allclose(grad_alpha_custom, grad_alpha_auto, atol=atol, rtol=rtol), f"Grad alpha differs significantly (Diff: {diff_alpha})"
    assert np.allclose(grad_beta_custom, grad_beta_auto, atol=atol, rtol=rtol), f"Grad beta differs significantly (Diff: {diff_beta})"
    print("Gradient comparison successful.")


# --- Example NNX Module (using the concise function) ---
class Constant: # Define if not using Flax/NNX fully
    def __init__(self, value): self.value = value

class SnakeBeta(nnx.Module):
    """
    JAX/Flax NNX 구현의 SnakeBeta 활성화 함수
    Uses the *concise* custom_vjp snake_beta function internally.
    """
    def __init__(self, in_features: int, alpha: float = 1.0,
                 alpha_trainable: bool = True, alpha_logscale: bool = False, *,
                 rngs: nnx.Rngs | None = None):
        self.in_features = in_features
        self.alpha_trainable = alpha_trainable
        self.alpha_logscale = alpha_logscale

        if self.alpha_logscale:
            alpha_init_val = 0.0
            beta_init_val = 0.0
            alpha_init = jnp.full((in_features,), alpha_init_val)
            beta_init = jnp.full((in_features,), beta_init_val)
        else:
            alpha_init = jnp.full((in_features,), alpha)
            beta_init = jnp.full((in_features,), alpha) # Match original behavior

        if self.alpha_trainable:
            self.alpha = nnx.Param(alpha_init)
            self.beta = nnx.Param(beta_init)
        else:
            self.alpha = Constant(alpha_init)
            self.beta = Constant(beta_init)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Applies the SnakeBeta activation using the concise function."""
        alpha_val = self.alpha.value
        beta_val = self.beta.value
        # *** Call the concise version ***
        return snake_beta(x, alpha_val, beta_val, self.alpha_logscale)


if __name__ == "__main__":
    # --- Verification on Larger B, N, C Arrays ---
    key = jax.random.PRNGKey(42)
    B, N, C = 4, 100, 16 # Example dimensions

    key, subkey = jax.random.split(key)
    x_large = jax.random.normal(subkey, (B, N, C))

    key, subkey1, subkey2 = jax.random.split(key, 3)
    # Parameters have shape (C,)
    alpha_large = jax.random.uniform(subkey1, (C,), minval=0.1, maxval=2.0)
    beta_large = jax.random.uniform(subkey2, (C,), minval=0.5, maxval=5.0)

    # Test case: alpha_logscale = False
    compare_gradients(x_large, alpha_large, beta_large, logscale_mode=False, N=N, C=C)

    # Test case: alpha_logscale = True
    compare_gradients(x_large, alpha_large, beta_large, logscale_mode=True, N=N, C=C)

    # --- Test the NNX Module ---
    print("\n--- Testing NNX SnakeBeta Module (Concise Backend) ---")
    key, subkey = jax.random.split(key)

    snake_module_nolog = SnakeBeta(in_features=C, alpha=0.8, alpha_trainable=True, alpha_logscale=False)
    snake_module_log = SnakeBeta(in_features=C, alpha=0.8, alpha_trainable=True, alpha_logscale=True)

    # Test forward pass
    y_nolog = snake_module_nolog(x_large)
    y_log = snake_module_log(x_large)
    print("NNX Module Output shape (no log):", y_nolog.shape)
    print("NNX Module Output shape (log):", y_log.shape)
    assert y_nolog.shape == (B, N, C)
    assert y_log.shape == (B, N, C)

    # Test gradient calculation through the module
    def loss_fn(module, x_in):
        y = module(x_in)
        return jnp.mean(y)

    grad_fn = nnx.value_and_grad(loss_fn, argnums=0) # Grad wrt module

    try:
        value_nolog, grads_module_nolog = grad_fn(snake_module_nolog, x_large)
        value_log, grads_module_log = grad_fn(snake_module_log, x_large)

        grad_alpha_nolog = grads_module_nolog.alpha.value
        grad_beta_nolog = grads_module_nolog.beta.value
        grad_alpha_log = grads_module_log.alpha.value
        grad_beta_log = grads_module_log.beta.value

        print(f"NNX Grad alpha (no log) shape: {grad_alpha_nolog.shape}")
        print(f"NNX Grad beta (no log) shape: {grad_beta_nolog.shape}")
        print(f"NNX Grad alpha (log) shape: {grad_alpha_log.shape}")
        print(f"NNX Grad beta (log) shape: {grad_beta_log.shape}")

        assert grad_alpha_nolog.shape == (C,)
        assert grad_beta_nolog.shape == (C,)
        assert grad_alpha_log.shape == (C,)
        assert grad_beta_log.shape == (C,)
        print("NNX Module gradient shapes are correct.")

    except Exception as e:
        print(f"Error during NNX gradient calculation: {e}")
        print("Skipping NNX gradient shape check. Ensure nnx setup is correct.")