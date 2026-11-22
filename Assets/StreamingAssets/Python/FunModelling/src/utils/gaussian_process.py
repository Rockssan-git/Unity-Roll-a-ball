"""
Gaussian-process utilities in JAX (differentiable, so hyperparameters can be fit by
gradient ascent on the log marginal likelihood).

Kernel used everywhere::

    k(a, b) = rbf_var · exp(-||a - b||² / length_scale)  +  dot_var · (sigma_0² + a·b)

with rbf_var = sigma_f² and dot_var = sigma_dot². The dot-product term is off by default
(sigma_dot = 0), which gives back the plain RBF GP.
NB: length_scale here divides ||a - b||² directly; sklearn's RBF(l) corresponds to length_scale = 2 l².

Observation noise sits on the diagonal of K_y = K + diag(noise_var) + jitter·I, where
noise_var is a scalar (homoskedastic) or one value per training point (heteroskedastic).
Internals work with variances rather than standard deviations so gradients stay finite
when a noise term is exactly 0.
"""
from functools import partial

import jax
jax.config.update("jax_enable_x64", True)  # Cholesky of K_y is unstable in float32
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

def kernel_matrix(A, B, length_scale, rbf_var=1., dot_var=0., sigma_0=0.):
    """k(a, b) for every pair (a in A, b in B) -> (len(A), len(B))."""
    sq_dist = jnp.sum((A[:, None, :] - B[None, :, :])**2, axis=-1)
    return rbf_var * jnp.exp(-sq_dist / length_scale) + dot_var * (sigma_0**2 + A @ B.T)


def kernel_diag(A, length_scale, rbf_var=1., dot_var=0., sigma_0=0.):
    """k(a, a) for every a in A -> (len(A),)."""
    return rbf_var + dot_var * (sigma_0**2 + jnp.sum(A**2, axis=-1))


def _kernel_params(length_scale, sigma_f, sigma_dot, sigma_0):
    return dict(length_scale=length_scale, rbf_var=sigma_f**2, dot_var=sigma_dot**2, sigma_0=sigma_0)


def _factor_K_y(X, noise_var, kp, jitter, mask=None):
    """
    Cholesky factor of K_y = K + diag(noise_var) + jitter·I.
    mask (n,) of 0/1: rows with 0 are padding; their block of K_y becomes the identity,
    decoupled from the real points, so they change neither the solves nor log|K_y|.
    """
    n = X.shape[0]
    K = kernel_matrix(X, X, **kp)
    diag = jnp.broadcast_to(noise_var + jitter, (n,))
    if mask is not None:
        K = K * mask[:, None] * mask[None, :]
        diag = jnp.where(mask > 0, diag, 1.)
    return cho_factor(K + jnp.diag(diag), lower=True)


def _mask_rows(y, mask):
    return y if mask is None else y * (mask if y.ndim == 1 else mask[:, None])


# ---------------------------------------------------------------------------
# Core (variance parameterisation, jitted)
# Optional `mask` lets callers pad the training set to a fixed size, so JAX doesn't
# recompile every time a data point is added.
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames="return_variance")
def gp_posterior(X_star, X, y, noise_var, kp, jitter=1e-6, return_variance=False, mask=None):
    """
    Posterior of the latent function at X_star.
    y: (n,) or (n, n_targets) -> mean (m,) or (m, n_targets); variance (m,) (shared by all targets).
    noise_var: scalar or (n,). kp: dict(length_scale, rbf_var, dot_var, sigma_0).
    """
    c_low = _factor_K_y(X, noise_var, kp, jitter, mask)
    k_star = kernel_matrix(X_star, X, **kp)  # (m, n)
    if mask is not None:
        k_star = k_star * mask[None, :]
    mean = k_star @ cho_solve(c_low, _mask_rows(y, mask))

    if not return_variance:
        return mean
    K_inv_k = cho_solve(c_low, k_star.T)  # (n, m)
    variance = kernel_diag(X_star, **kp) - jnp.sum(k_star * K_inv_k.T, axis=-1)
    return mean, variance


@jax.jit
def gp_log_marginal_likelihood(X, y, noise_var, kp, jitter=1e-6, mask=None):
    """log p(y | X), summed over targets if y is (n, n_targets) (targets share the kernel)."""
    n = X.shape[0] if mask is None else jnp.sum(mask)
    n_targets = 1 if y.ndim == 1 else y.shape[1]
    y = _mask_rows(y, mask)
    c_low = _factor_K_y(X, noise_var, kp, jitter, mask)    # K_y = L Lᵀ (lower-triangular)
    alpha = cho_solve(c_low, y)                             # K_y⁻¹ y, via triangular solves

    lml  = -0.5 * jnp.sum(y * alpha)                                  # data-fit:  -½ yᵀ K_y⁻¹ y
    lml -= n_targets * jnp.sum(jnp.log(jnp.diag(c_low[0])))           # -½ log|K_y|  (|K_y| = ∏ diag(L)²)
    lml -= 0.5 * n * n_targets * jnp.log(2 * jnp.pi)                  # normalization
    return lml


# ---------------------------------------------------------------------------
# Plain (homoskedastic or user-given per-point noise) GP
# ---------------------------------------------------------------------------

def predict_gaussian_process(x_star, X, y, length_scale, sigma_noise, sigma_f=1., return_variance=False, jitter=1e-6,
                             sigma_dot=0., sigma_0=0.):
    """Single test point x_star (d,) -> mean (and variance)."""
    x_star = jnp.atleast_2d(jnp.asarray(x_star, dtype=float))
    out = predict_vec_gaussian_process(x_star, X, y, length_scale, sigma_noise, sigma_f, return_variance, jitter,
                                       sigma_dot=sigma_dot, sigma_0=sigma_0)
    if not return_variance:
        return out[0]
    mean, variance = out
    return mean[0], variance[0]


#@measure_execution_time
def predict_vec_gaussian_process(X_star, X, y, length_scale, sigma_noise, sigma_f=1., return_variance=False, jitter=1e-6,
                                 sigma_dot=0., sigma_0=0.):
    """Batch of test points X_star (m, d) -> mean (m,) (and variance (m,)). sigma_noise: scalar or (n,)."""
    X_star = jnp.asarray(X_star, dtype=float)
    X = jnp.asarray(X, dtype=float)
    y = jnp.asarray(y, dtype=float)
    noise_var = jnp.asarray(sigma_noise, dtype=float)**2
    kp = _kernel_params(length_scale, sigma_f, sigma_dot, sigma_0)
    return gp_posterior(X_star, X, y, noise_var, kp, jitter, return_variance=return_variance)


def compute_gp_weights(X_star, X, length_scale, sigma_noise, sigma_f=1., jitter=1e-6, sigma_dot=0., sigma_0=0.):
    """
    Linear weights the GP mean places on each training point: mean(X_star) = weights @ y.
    weights[i, j] is the contribution of X[j] to the prediction at X_star[i]; a larger
    sigma_noise[j] (e.g. a stale/low-recency observation) shrinks column j toward 0
    since sigma_noise sits on the diagonal of K_y before it's inverted.
    Use weights_to_pct_contribution to turn a row into a % breakdown.
    """
    X_star = jnp.asarray(X_star, dtype=float)
    X = jnp.asarray(X, dtype=float)
    kp = _kernel_params(length_scale, sigma_f, sigma_dot, sigma_0)

    c_low = _factor_K_y(X, jnp.asarray(sigma_noise, dtype=float)**2, kp, jitter)
    k_star = kernel_matrix(X_star, X, **kp)
    return cho_solve(c_low, k_star.T).T  # (n_star, n)


def weights_to_pct_contribution(weights):
    """
    % contribution of each training point per row of `weights`, based on ``|weight|``.
    Uses abs() because RBF-GP weights can go slightly negative (e.g. for a point
    sandwiched between two closer, opposite-signed neighbors), so a plain sum
    would understate those points' actual influence.
    """
    abs_w = jnp.abs(weights)
    return 100 * abs_w / abs_w.sum(axis=-1, keepdims=True)


#@measure_execution_time
def log_marginal_likelihood_vec_gaussian_process(X, y, length_scale, sigma_noise, sigma_f=1., jitter=1e-6,
                                                 sigma_dot=0., sigma_0=0.):
    """log p(y | X) of the plain GP. sigma_noise: scalar or (n,)."""
    X = jnp.asarray(X, dtype=float)
    y = jnp.asarray(y, dtype=float)
    kp = _kernel_params(length_scale, sigma_f, sigma_dot, sigma_0)
    return gp_log_marginal_likelihood(X, y, jnp.asarray(sigma_noise, dtype=float)**2, kp, jitter)


# ---------------------------------------------------------------------------
# Heteroskedastic GP: older observations get more noise, so they count less
# ---------------------------------------------------------------------------

def compute_recency(t, t_current=None, t_max=None):
    """
    Normalised age of each observation: (t_current - t) / t_max.
    t_current defaults to the latest time in t; t_max defaults to max(t_current, 1).
    """
    t = jnp.asarray(t, dtype=float)
    if t_current is None:
        t_current = jnp.max(t)
    if t_max is None:
        t_max = jnp.maximum(t_current, 1.)
    return (t_current - t) / t_max


def heteroskedastic_noise_var(t, noise_var, recency_weight, t_max=None, t_current=None):
    """
    Per-observation noise variance: noise_var + exp(recency_weight · recency) - 1.
    The newest point (recency 0) keeps noise_var; the extra noise grows
    exponentially with age, and recency_weight = 0 gives back the plain GP.
    """
    recency = compute_recency(t, t_current=t_current, t_max=t_max)
    return noise_var + jnp.expm1(recency_weight * recency)


def heteroskedastic_sigma_noise(t, sigma_noise, recency_weight, t_max=None, t_current=None):
    """Per-observation noise std: sqrt(sigma_noise² + exp(recency_weight · recency) - 1)."""
    return jnp.sqrt(heteroskedastic_noise_var(t, sigma_noise**2, recency_weight, t_max=t_max, t_current=t_current))


def predict_heteroskedastic_gaussian_process(x_star, X, y, length_scale, sigma_noise, sigma_f=1., t=None,
                                             recency_weight=0., t_max=None, t_current=None,
                                             return_variance=False, jitter=1e-6, sigma_dot=0., sigma_0=0.):
    """Single test point x_star (d,) -> mean (and variance). See predict_vec_heteroskedastic_gaussian_process."""
    x_star = jnp.atleast_2d(jnp.asarray(x_star, dtype=float))
    out = predict_vec_heteroskedastic_gaussian_process(x_star, X, y, length_scale, sigma_noise, sigma_f,
                                                       t=t, recency_weight=recency_weight, t_max=t_max,
                                                       t_current=t_current, return_variance=return_variance,
                                                       jitter=jitter, sigma_dot=sigma_dot, sigma_0=sigma_0)
    if not return_variance:
        return out[0]
    mean, variance = out
    return mean[0], variance[0]


def _hk_noise_var(X, sigma_noise, t, recency_weight, t_max, t_current):
    noise_var = jnp.asarray(sigma_noise, dtype=float)**2
    if t is None:
        return noise_var
    if len(t) != len(X):
        raise ValueError(f"t has {len(t)} entries but X has {len(X)} rows")
    return heteroskedastic_noise_var(t, noise_var, recency_weight, t_max=t_max, t_current=t_current)


def predict_vec_heteroskedastic_gaussian_process(X_star, X, y, length_scale, sigma_noise, sigma_f=1., t=None,
                                                 recency_weight=0., t_max=None, t_current=None,
                                                 return_variance=False, jitter=1e-6, sigma_dot=0., sigma_0=0.):
    """
    GP prediction where each training point X[i], observed at time t[i], has noise variance
    heteroskedastic_noise_var(t, sigma_noise², recency_weight, t_max, t_current).
    t: (n,) time of each row of X. recency_weight, t_max: scalars.
    t_current: time at which we predict (defaults to max(t)).
    If t is None, this is the plain (homoskedastic) GP.
    """
    X_star = jnp.asarray(X_star, dtype=float)
    X = jnp.asarray(X, dtype=float)
    y = jnp.asarray(y, dtype=float)
    noise_var = _hk_noise_var(X, sigma_noise, t, recency_weight, t_max, t_current)
    kp = _kernel_params(length_scale, sigma_f, sigma_dot, sigma_0)
    return gp_posterior(X_star, X, y, noise_var, kp, jitter, return_variance=return_variance)


def log_marginal_likelihood_heteroskedastic_gaussian_process(X, y, length_scale, sigma_noise, sigma_f=1., t=None,
                                                             recency_weight=0., t_max=None, t_current=None,
                                                             jitter=1e-6, sigma_dot=0., sigma_0=0.):
    """log p(y | X) with recency-dependent noise; arguments as in predict_vec_heteroskedastic_gaussian_process."""
    X = jnp.asarray(X, dtype=float)
    y = jnp.asarray(y, dtype=float)
    noise_var = _hk_noise_var(X, sigma_noise, t, recency_weight, t_max, t_current)
    kp = _kernel_params(length_scale, sigma_f, sigma_dot, sigma_0)
    return gp_log_marginal_likelihood(X, y, noise_var, kp, jitter)
