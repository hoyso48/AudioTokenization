import jax
import jax.numpy as jnp
from flax import nnx
import jax
from AudioTokenization.BigCodec_NNX.utils import Constant

from jax import custom_vjp
from functools import partial
from jax.nn import logsumexp

# # --- Constants for Numerical Stability ---
# # Log-epsilon for logsumexp trick for beta. log(1e-8) approx -18.4
# _SAFE_LOG_EPSILON_BETA = -18.0
# # Minimum effective beta for non-logscale case
# _MIN_BETA_EFF_NONLOG = 1e-5

# @partial(custom_vjp, nondiff_argnums=(3,))
# def snake_beta(x: jnp.ndarray,
#                alpha: jnp.ndarray,
#                beta: jnp.ndarray,
#                alpha_logscale: bool = False) -> jnp.ndarray:
#     """
#     Numerically stabilized SnakeBeta (v3) using logsumexp for beta and
#     log-space intermediates for alpha-beta interactions when alpha_logscale=True.

#     Args:
#         x (jnp.ndarray): Input tensor, shape (..., C).
#         alpha (jnp.ndarray): Frequency parameter, shape (C,). If logscale, log(alpha_eff).
#         beta (jnp.ndarray): Scale parameter, shape (C,). If logscale, log(beta_eff).
#         alpha_logscale (bool): If True, interpret alpha/beta as log parameters.

#     Returns:
#         jnp.ndarray: Output tensor, shape (..., C).
#     """
#     # --- Forward computation with stability enhancements ---
#     if alpha_logscale:
#         # alpha_eff still computed directly, potential inf if alpha is too large
#         alpha_eff = jnp.exp(alpha)

#         # Stable log of (beta_eff + epsilon) using logsumexp
#         log_beta_eff = beta
#         log_eps_beta_arr = jnp.full_like(log_beta_eff, _SAFE_LOG_EPSILON_BETA)
#         # log_beta_stable = log(exp(beta) + exp(log_eps))
#         log_beta_stable = logsumexp(jnp.stack([log_beta_eff, log_eps_beta_arr]), axis=0)

#         # Stable inverse: inv_beta_stable = 1 / (exp(beta) + epsilon)
#         inv_beta_stable = jnp.exp(-log_beta_stable)

#     else:
#         alpha_eff = alpha
#         beta_eff = beta
#         # Simple stability for non-logscale beta
#         beta_stable = jnp.maximum(beta_eff, _MIN_BETA_EFF_NONLOG)
#         inv_beta_stable = 1.0 / beta_stable
#         # Need log_beta_stable equivalent for gradient calculation consistency, though less critical
#         # log_beta_stable is not directly used in forward non-logscale path

#     # Common calculations
#     ax = alpha_eff * x
#     sq_sin_ax = jnp.square(jnp.sin(ax))

#     # Apply the snake term using the stabilized inverse beta
#     snake_term = sq_sin_ax * inv_beta_stable
#     output = x + snake_term
#     return output

# def snake_beta_fwd(x: jnp.ndarray,
#                    alpha: jnp.ndarray,
#                    beta: jnp.ndarray,
#                    alpha_logscale: bool) -> tuple[jnp.ndarray, tuple]:
#     """Forward pass for VJP. Returns primal output and residuals."""
#     primal_out = snake_beta(x, alpha, beta, alpha_logscale)
#     # Save original inputs and logscale flag for backward pass
#     residuals = (x, alpha, beta, alpha_logscale)
#     return primal_out, residuals

# def snake_beta_bwd(alpha_logscale_arg_dummy: bool, # Nondiff arg
#                    residuals: tuple,
#                    grad_output: jnp.ndarray
#                    ) -> tuple[jnp.ndarray | None, jnp.ndarray | None, jnp.ndarray | None]:
#     """Backward pass for VJP using stabilized intermediates."""
#     x, alpha, beta, alpha_logscale = residuals
#     g = grad_output # Shape (..., C)

#     # --- Recompute intermediates *with stabilization* ---
#     if alpha_logscale:
#         alpha_eff = jnp.exp(alpha) # Recompute alpha_eff

#         # Recompute stable log_beta and its inverse
#         log_beta_eff = beta
#         log_eps_beta_arr = jnp.full_like(log_beta_eff, _SAFE_LOG_EPSILON_BETA)
#         log_beta_stable = logsumexp(jnp.stack([log_beta_eff, log_eps_beta_arr]), axis=0)
#         inv_beta_stable = jnp.exp(-log_beta_stable)

#         # Compute log of the alpha/beta interaction term (stable)
#         # log(alpha_eff / beta_stable) = log(alpha_eff) - log(beta_stable)
#         log_alpha_over_beta_stable = alpha - log_beta_stable # log(exp(alpha)) = alpha

#         # Compute softmax term for beta gradient (stable, 0 to 1)
#         # softmax = exp(beta) / (exp(beta) + epsilon) = exp(beta - log_beta_stable)
#         softmax_beta_term = jnp.exp(beta - log_beta_stable)

#     else:
#         alpha_eff = alpha
#         beta_eff = beta
#         # Recompute stable beta and its inverse for non-logscale
#         beta_stable = jnp.maximum(beta_eff, _MIN_BETA_EFF_NONLOG)
#         # Avoid division by zero if beta_stable happens to be exactly zero
#         # (unlikely with maximum but safer)
#         inv_beta_stable = jnp.where(beta_stable > 0, 1.0 / beta_stable, 0.0)
#         # Gradient flag
#         is_beta_eff_stable = beta_eff >= _MIN_BETA_EFF_NONLOG


#     # Common intermediate calculations
#     ax = alpha_eff * x               # Shape (..., C)
#     # sin_ax = jnp.sin(ax)           # Not directly needed now
#     sin_2ax = jnp.sin(2 * ax)        # Shape (..., C)
#     sq_sin_ax = jnp.square(jnp.sin(ax))   # Shape (..., C), needed for beta grad

#     # --- Gradient w.r.t. x ---
#     if alpha_logscale:
#         # Use the log-space intermediate: exp(log(alpha_eff / beta_stable)) = alpha_eff / beta_stable
#         grad_x_factor = jnp.exp(log_alpha_over_beta_stable) * sin_2ax
#     else:
#         # Original form for non-logscale
#         grad_x_factor = alpha_eff * inv_beta_stable * sin_2ax
#     # grad_x = g * (1 + d(snake_term)/dx)
#     grad_x = g * (1.0 + grad_x_factor) # Shape (..., C)

#     # --- Gradient w.r.t. alpha ---
#     # dSnake/dalpha = dSnake/dalpha_eff * dalpha_eff/dalpha
#     # dSnake/dalpha_eff = x * inv_beta_stable * sin(2*alpha_eff*x)
#     # dalpha_eff/dalpha = alpha_eff if logscale else 1.0

#     if alpha_logscale:
#          # Use the log-space intermediate again
#          # alpha_eff * inv_beta_stable = exp(log_alpha_over_beta_stable)
#         dSnake_dalpha_eff_times_deriv = x * jnp.exp(log_alpha_over_beta_stable) * sin_2ax
#     else:
#         dSnake_dalpha_eff_times_deriv = x * inv_beta_stable * sin_2ax * 1.0

#     # Element-wise gradient before reduction
#     # g * dSnake/dalpha
#     grad_alpha_full = g * dSnake_dalpha_eff_times_deriv # Shape (..., C)

#     # Sum over broadcasted dimensions
#     reduce_axes = tuple(range(x.ndim - 1))
#     grad_alpha = jnp.sum(grad_alpha_full, axis=reduce_axes) if x.ndim > 1 else grad_alpha_full # Shape (C,)

#     # --- Gradient w.r.t. beta ---
#     # dSnake/dbeta = dSnake/d(inv_beta_stable) * d(inv_beta_stable)/d(beta) [logscale]
#     # dSnake/dbeta = dSnake/d(beta_stable) * d(beta_stable)/d(beta) [non-logscale]

#     if alpha_logscale:
#         # Use the refined stable gradient: - g * sq_sin_ax * inv_beta_stable * softmax_beta_term
#         grad_beta_full = - g * sq_sin_ax * inv_beta_stable * softmax_beta_term
#     else:
#         # Use the original derivative form but with beta_stable check
#         # dSnake/dbeta_stable = - sq_sin_ax / (beta_stable^2) = - sq_sin_ax * (inv_beta_stable^2)
#         # dbeta_stable/dbeta = 1.0 if beta >= min_val else 0.0
#         dSnake_dbeta_stable = -sq_sin_ax * jnp.square(inv_beta_stable)
#         dbeta_stable_dbeta = jnp.where(is_beta_eff_stable, 1.0, 0.0)
#         grad_beta_full = g * dSnake_dbeta_stable * dbeta_stable_dbeta

#     # Sum over broadcasted dimensions
#     grad_beta = jnp.sum(grad_beta_full, axis=reduce_axes) if x.ndim > 1 else grad_beta_full # Shape (C,)

#     # Return gradients matching input shapes (x, alpha, beta)
#     return grad_x, grad_alpha, grad_beta

# # Register the VJP rule
# snake_beta.defvjp(snake_beta_fwd, snake_beta_bwd)

class Snake(nnx.Module):
    """
    JAX/Flax NNX 구현의 Snake 활성화 함수
    
    원본 논문: https://arxiv.org/abs/2006.08195
    """
    def __init__(self, in_features: int, alpha: float = 1.0, 
                 alpha_trainable: bool = True, alpha_logscale: bool = False,
                 rngs: nnx.Rngs = None):
        self.in_features = in_features
        self.alpha_trainable = alpha_trainable
        self.alpha_logscale = alpha_logscale
        
        if self.alpha_logscale:
            alpha_init = jnp.zeros((in_features,)) * alpha
        else:
            alpha_init = jnp.ones((in_features,)) * alpha
        
        self.alpha = nnx.Param(alpha_init)
    
    def __call__(self, x):
        alpha = self.alpha
        if not self.alpha_trainable:
            alpha = jax.lax.stop_gradient(alpha)
        
        alpha = jnp.reshape(alpha, (1, 1, -1))
        alpha = jnp.reshape(alpha, (1, 1, -1))
        
        if self.alpha_logscale:
            alpha = jnp.exp(alpha)
        
        no_div_by_zero = 1e-9
        return x + (1.0 / (alpha + no_div_by_zero)) * jnp.square(jnp.sin(x * alpha))

# @jax.jit
def snake(x, alpha, beta):
    no_div_by_zero = 1e-9
    return x + (1.0 / (beta + no_div_by_zero)) * jnp.square(jnp.sin((x * alpha)))
    # return x + (1-jnp.cos(2 * x * alpha)) / (2 * beta + no_div_by_zero)

class SnakeBeta(nnx.Module):
    """
    JAX/Flax NNX 구현의 SnakeBeta 활성화 함수
    
    원본 논문 수정 버전: https://arxiv.org/abs/2006.08195
    """
    def __init__(self, in_features: int, alpha: float = 1.0, 
                 alpha_trainable: bool = True, alpha_logscale: bool = False,
                 rngs: nnx.Rngs = None):
        self.in_features = in_features
        self.alpha_trainable = alpha_trainable
        self.alpha_logscale = alpha_logscale
        
        # if self.alpha_logscale:
        #     alpha_init = jnp.zeros((in_features,)) * alpha
        #     beta_init = jnp.zeros((in_features,)) * alpha
        # else:
        #     alpha_init = jnp.ones((in_features,)) * alpha
        #     beta_init = jnp.ones((in_features,)) * alpha
        
        # if self.alpha_trainable:
        #     self.alpha = nnx.Param(alpha_init)
        #     self.beta = nnx.Param(beta_init)
        # else:
        #     self.alpha = Constant(alpha_init)
        #     self.beta = Constant(beta_init)

        # self.no_div_by_zero = 1e-9
    
    def __call__(self, x):
        return jax.nn.swish(x)
        # return jax.nn.leaky_relu(x)
        
        # alpha = self.alpha.value
        # beta = self.beta.value
        # if self.alpha_logscale:
        #     alpha = jnp.exp(alpha)
        #     beta = jnp.exp(beta)
        # alpha = jnp.reshape(alpha, (1, 1, -1))
        # beta = jnp.reshape(beta, (1, 1, -1))
        # return x + (1.0 / (beta + self.no_div_by_zero)) * jnp.power(jnp.sin((x * alpha)), 2) #snake(x, alpha, beta) # jax.nn.leaky_relu(x)
        # alpha = self.alpha.value
        # beta = self.beta.value
        # # alpha = jnp.reshape(alpha, (1, 1, -1))
        # # beta = jnp.reshape(beta, (1, 1, -1))
        # return snake_beta(x, alpha, beta, self.alpha_logscale)