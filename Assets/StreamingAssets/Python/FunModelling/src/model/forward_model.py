# common librairies
from abc import ABC, abstractmethod
import copy
import numpy as np

import warnings
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, DotProduct,WhiteKernel
from sklearn.preprocessing import StandardScaler

# my scripts
from src.utils.gaussian_process import gp_posterior, gp_log_marginal_likelihood, heteroskedastic_noise_var



class ForwardModel(ABC):
    """Learns to predict the outcome of an action."""
    @abstractmethod
    def update(self,action:np.ndarray,obs:np.ndarray) -> dict:
        """
        Add one (action, outcome) pair and refit.

        Parameters
        ----------
        action : np.ndarray
            new action tried

        obs : np.ndarray
            observation following this action 

        Returns
        -------
        info : dict
            Information about the fit of the forward model
        """
        pass


    @abstractmethod
    def __call__(self,actions:np.ndarray)->np.ndarray:
        """Predicted outcomes ``(n, out_dim)`` of ``actions`` ``(n, action_dim)``."""
        pass



class GaussianProcessForwardModel(ForwardModel):
    """
    Forward model based on scikit-learn's ``GaussianProcessRegressor``, with kernel
    ``C * DotProduct + C * RBF + White``. Actions are standardised and outcomes normalised;
    the GP is refit on all data after every update.

    Parameters
    ----------
    out_dim : int
        Dimension of the predicted outcome.
    sigma_0 : float, default 0.1
        Initial ``sigma_0`` of the DotProduct kernel.
    length_scale : float, default 1.0
        Initial RBF length scale.
    length_scale_bounds : tuple[float, float], default (1e-2, 1e2)
    n_restarts_optimizer : int, default 9
        Extra restarts of the hyperparameter optimiser.
    """
    def __init__(self, 
                 out_dim: int, 
                 sigma_0=.1, 
                 length_scale: float = 1.0,
                 length_scale_bounds: tuple[float, float] = (1e-2, 1e2),
                 n_restarts_optimizer: int = 9):
        kernel = ConstantKernel(1.0)*DotProduct(sigma_0=sigma_0) + ConstantKernel(1.0) * RBF(length_scale=length_scale,length_scale_bounds=length_scale_bounds) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
        self.gaussian_process = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=n_restarts_optimizer,
            n_targets=out_dim,
            normalize_y=True,          # standardizes y per output dim, undone at predict
        )
        self.x_scaler = StandardScaler()
        self.X_train = None   # raw actions, (n_samples, in_dim)
        self.y_train = None   # raw observations, (n_samples, out_dim)
        self.out_dim = out_dim

    def update(self, action: np.ndarray, obs: np.ndarray):
        """
        Add one (action, outcome) pair and refit the GP on all data.

        Returns
        -------
        dict
            Attributes of the fitted ``GaussianProcessRegressor``.
        """
        action = np.asarray(action, dtype=float)[np.newaxis, :]
        obs = np.asarray(obs, dtype=float).reshape(1, self.out_dim)
        if self.X_train is None:
            self.X_train, self.y_train = action, obs
        else:
            self.X_train = np.concatenate([self.X_train, action], axis=0)
            self.y_train = np.concatenate([self.y_train, obs], axis=0)

        X_scaled = self.x_scaler.fit_transform(self.X_train)   # refit on all data

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.gaussian_process.fit(X_scaled, self.y_train)

        return self.gaussian_process.__dict__

    def __call__(self, actions: np.ndarray) -> np.ndarray:
        """GP posterior mean ``(n, out_dim)`` at ``actions``; zeros before the first update."""
        actions = np.atleast_2d(np.asarray(actions, dtype=float))
        if self.X_train is None:
            # No data yet: GP prior mean, which is 0 in raw units
            return np.zeros((len(actions), self.out_dim))

        y = self.gaussian_process.predict(self.x_scaler.transform(actions))
        return y.reshape(len(actions), self.out_dim)


####################################################################################################################################
# Heteroskedastic Gaussian process (recency-weighted noise)
####################################################################################################################################

# Hard-coded kernel, same meaning as sklearn's
#   ConstantKernel(c_dot) * DotProduct(sigma_0) + ConstantKernel(c_rbf) * RBF(length_scale) + WhiteKernel(noise_level)
# plus a recency term making old observations noisier:
#   noise_var_i = noise_level + exp(recency_weight * (t_current - t_i) / t_max) - 1
# Every component can be switched off with "active": False; every hyperparameter can be frozen
# with "<name>_bounds": "fixed" (as in sklearn).
DEFAULT_HK_KERNEL_PARAMS = {
    "dot_product": {"active": True,
                    "constant_value": 1.0, "constant_value_bounds": (1e-5, 1e5),
                    "sigma_0": 0.1, "sigma_0_bounds": (1e-5, 1e5)},
    "rbf": {"active": True,
            "constant_value": 1.0, "constant_value_bounds": (1e-5, 1e5),
            "length_scale": 1.0, "length_scale_bounds": (1e-5, 1e5)},
    "white": {"active": True,
              "noise_level": 1e-5, "noise_level_bounds": (1e-10, 1e-1)},
    "recency": {"active": True,
                "recency_weight": 1.0, "recency_weight_bounds": "fixed",
                "t_max": None},  # None -> max(t_current, 1)
}

# (component, hyperparameter) pairs that can be optimised, in log space
_HK_HYPERPARAMETERS = [("dot_product", "constant_value"), ("dot_product", "sigma_0"),
                       ("rbf", "constant_value"), ("rbf", "length_scale"),
                       ("white", "noise_level"),
                       ("recency", "recency_weight")]


def _merge_kernel_params(kernel_params):
    """Defaults updated (per component) with the user's kernel_params."""
    merged = copy.deepcopy(DEFAULT_HK_KERNEL_PARAMS)
    for component, params in (kernel_params or {}).items():
        if component not in merged:
            raise KeyError(f"unknown kernel component '{component}', expected one of {list(merged)}")
        unknown = set(params) - set(merged[component])
        if unknown:
            raise KeyError(f"unknown parameter(s) {unknown} for kernel component '{component}'")
        merged[component].update(params)
    if not (merged["rbf"]["active"] or merged["dot_product"]["active"]):
        raise ValueError("at least one of the 'rbf' and 'dot_product' kernels must be active")
    return merged


def _hk_gp_arguments(theta, fixed_values, free, active, t_max, t, t_current):
    """
    Hyperparameters -> (noise_var, kp) for the utils functions.
    theta: log of the free hyperparameters (in the order of `free`); fixed_values: {"comp.name": value}.
    active: (dot_product, rbf, white, recency) flags.
    """
    values = dict(fixed_values)
    for i, (c, n) in enumerate(free):
        values[f"{c}.{n}"] = jnp.exp(theta[i])
    dot_on, rbf_on, white_on, rec_on = active
    kp = dict(
        length_scale=2 * values["rbf.length_scale"]**2 if rbf_on else 1.,  # sklearn RBF: exp(-d²/(2l²))
        rbf_var=values["rbf.constant_value"] if rbf_on else 0.,
        dot_var=values["dot_product.constant_value"] if dot_on else 0.,
        sigma_0=values["dot_product.sigma_0"] if dot_on else 0.,
    )
    noise_var = values["white.noise_level"] if white_on else 0.
    if rec_on:
        noise_var = heteroskedastic_noise_var(t, noise_var, values["recency.recency_weight"],
                                              t_max=t_max, t_current=t_current)
    return noise_var, kp


def _hk_neg_lml(theta, fixed_values, X, y, t, mask, t_current, alpha, free, active, t_max):
    noise_var, kp = _hk_gp_arguments(theta, fixed_values, free, active, t_max, t, t_current)
    return -gp_log_marginal_likelihood(X, y, noise_var, kp, alpha, mask)


# module level so the compiled function is reused across fits (shapes are padded, see _PAD_MULTIPLE)
_hk_neg_lml_and_grad = jax.jit(jax.value_and_grad(_hk_neg_lml), static_argnames=("free", "active", "t_max"))
_hk_neg_lml_jit = jax.jit(_hk_neg_lml, static_argnames=("free", "active", "t_max"))

# training sets are zero-padded (and masked) to a multiple of this, so adding one
# observation only triggers a JAX recompilation every _PAD_MULTIPLE observations
_PAD_MULTIPLE = 16


def _pad_rows(a, size, fill=0.):
    pad = [(0, size - a.shape[0])] + [(0, 0)] * (a.ndim - 1)
    return np.pad(a, pad, constant_values=fill)


class HeteroskedasticGaussianProcessRegressor:
    """
    sklearn-like GaussianProcessRegressor with a hard-coded DotProduct + RBF + White kernel
    and recency-dependent (heteroskedastic) observation noise. Computations are in JAX
    (src/utils/gaussian_process.py); hyperparameters are fit by L-BFGS-B on the log marginal
    likelihood with exact JAX gradients, restarted n_restarts_optimizer times as in sklearn.

    Parameters
    ----------
    kernel_params : dict, optional
        Partial override of DEFAULT_HK_KERNEL_PARAMS, e.g.
        {"dot_product": {"active": False}, "recency": {"recency_weight": 2., "recency_weight_bounds": (1e-3, 10.)}}
    alpha : float
        Value added to the diagonal of the kernel matrix (as in sklearn).
    n_restarts_optimizer : int
        Extra optimiser runs from log-uniform random starting points within the bounds.
    normalize_y : bool
        Standardise each target before fitting; undone at predict time.
    n_targets : int, optional
        Only checked against y (kept for sklearn compatibility).
    random_state : int or np.random.Generator, optional
    """

    def __init__(self, kernel_params=None, alpha=1e-10, n_restarts_optimizer=0, normalize_y=False,
                 n_targets=None, random_state=None):
        self.kernel_params = _merge_kernel_params(kernel_params)
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.n_targets = n_targets
        self.random_state = random_state

    # ------------------------------------------------------------------ hyperparameters
    def _free_hyperparameters(self):
        """(component, name) of the active, non-fixed hyperparameters."""
        free = []
        for component, name in _HK_HYPERPARAMETERS:
            p = self.kernel_params[component]
            bounds = p[f"{name}_bounds"]
            if p["active"] and not (isinstance(bounds, str) and bounds == "fixed"):
                if bounds[0] <= 0:
                    raise ValueError(f"{component}.{name}_bounds must be > 0 (optimised in log space), got {bounds}")
                free.append((component, name))
        return free

    @staticmethod
    def _static_config(kernel_params):
        """Hashable (active flags, t_max) passed as static arguments to the jitted functions."""
        active = tuple(bool(kernel_params[c]["active"]) for c in ("dot_product", "rbf", "white", "recency"))
        t_max = kernel_params["recency"]["t_max"]
        return active, (None if t_max is None else float(t_max))

    @staticmethod
    def _fixed_values(kernel_params, free=()):
        return {f"{c}.{n}": float(kernel_params[c][n]) for c, n in _HK_HYPERPARAMETERS if (c, n) not in free}

    # ------------------------------------------------------------------ fit
    def fit(self, X, y, t=None, t_current=None):
        """
        X : (n, d), y : (n,) or (n, n_targets)
        t : (n,) time of each observation; defaults to 0..n-1 (row order = time order)
        t_current : time at which recency is measured; defaults to max(t)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = X.shape[0]
        self._y_is_1d = y.ndim == 1
        y2 = y.reshape(n, -1)
        if self.n_targets is not None and y2.shape[1] != self.n_targets:
            raise ValueError(f"y has {y2.shape[1]} targets but n_targets={self.n_targets}")

        if self.normalize_y:
            self._y_train_mean = y2.mean(axis=0)
            std = y2.std(axis=0)
            self._y_train_std = np.where(std == 0, 1., std)
        else:
            self._y_train_mean = np.zeros(y2.shape[1])
            self._y_train_std = np.ones(y2.shape[1])

        t = np.arange(n, dtype=float) if t is None else np.asarray(t, dtype=float)
        if t.shape != (n,):
            raise ValueError(f"t must have shape ({n},), got {t.shape}")
        self.n_train_ = n
        self.t_current_ = float(t.max()) if t_current is None else float(t_current)

        # pad to a multiple of _PAD_MULTIPLE; padded rows are masked out (exact, not an approximation)
        size = -(-n // _PAD_MULTIPLE) * _PAD_MULTIPLE
        self.X_train_ = jnp.asarray(_pad_rows(X, size))
        self.y_train_ = jnp.asarray(_pad_rows((y2 - self._y_train_mean) / self._y_train_std, size))
        self.t_train_ = jnp.asarray(_pad_rows(t, size, fill=self.t_current_))
        self.mask_ = jnp.asarray(_pad_rows(np.ones(n), size))

        kernel_params = self.kernel_params
        free = tuple(self._free_hyperparameters())
        active, t_max = self._static_config(kernel_params)
        fixed_values = self._fixed_values(kernel_params, free)
        data = (fixed_values, self.X_train_, self.y_train_, self.t_train_, self.mask_, self.t_current_, self.alpha)
        static = dict(free=free, active=active, t_max=t_max)

        def lml(theta):
            return -float(_hk_neg_lml_jit(jnp.asarray(theta, dtype=float), *data, **static))

        theta0 = np.array([np.log(kernel_params[c][n_]) for c, n_ in free])
        if len(free):
            def objective(theta):
                value, grad = _hk_neg_lml_and_grad(jnp.asarray(theta), *data, **static)
                value, grad = float(value), np.asarray(grad, dtype=float)
                if not np.isfinite(value) or not np.all(np.isfinite(grad)):
                    return np.inf, np.zeros_like(theta)
                return value, grad

            log_bounds = np.log(np.array([kernel_params[c][f"{n_}_bounds"] for c, n_ in free], dtype=float))
            rng = np.random.default_rng(self.random_state)
            starts = [np.clip(theta0, log_bounds[:, 0], log_bounds[:, 1])]
            starts += [rng.uniform(log_bounds[:, 0], log_bounds[:, 1]) for _ in range(self.n_restarts_optimizer)]

            best_theta, best_value = starts[0], objective(starts[0])[0]
            for start in starts:
                res = minimize(objective, start, jac=True, method="L-BFGS-B", bounds=log_bounds)
                if not res.success:
                    warnings.warn(f"lbfgs failed to converge: {res.message}", ConvergenceWarning)
                if res.fun < best_value:
                    best_theta, best_value = res.x, res.fun
            theta = best_theta
        else:
            theta = theta0

        # fitted kernel, as plain floats
        self.kernel_params_ = copy.deepcopy(kernel_params)
        for (c, n_), v in zip(free, np.exp(theta)):
            self.kernel_params_[c][n_] = float(v)
        self.log_marginal_likelihood_value_ = lml(theta)
        return self

    # ------------------------------------------------------------------ predict
    def predict(self, X, return_std=False, t_current=None):
        """
        X : (m, d) -> mean (m,) or (m, n_targets) [, std of the same shape].
        t_current : time at which recency is measured; defaults to the one used in fit.
        Like sklearn, the predictive std includes the white noise level (not the recency noise).
        """
        X = jnp.asarray(np.atleast_2d(np.asarray(X, dtype=float)))
        t_current = self.t_current_ if t_current is None else float(t_current)
        kp_ = self.kernel_params_
        active, t_max = self._static_config(kp_)
        noise_var, kp = _hk_gp_arguments(jnp.zeros(0), self._fixed_values(kp_), (), active, t_max,
                                         self.t_train_, t_current)

        out = gp_posterior(X, self.X_train_, self.y_train_, noise_var, kp, self.alpha,
                           return_variance=return_std, mask=self.mask_)
        mean, var = (out, None) if not return_std else out
        mean = np.asarray(mean) * self._y_train_std + self._y_train_mean
        if self._y_is_1d:
            mean = mean[:, 0]
        if not return_std:
            return mean

        var = np.asarray(var)
        if kp_["white"]["active"]:
            var = var + kp_["white"]["noise_level"]
        std = np.sqrt(np.maximum(var, 0.))[:, None] * self._y_train_std
        if self._y_is_1d:
            std = std[:, 0]
        return mean, std

    def log_marginal_likelihood(self):
        return self.log_marginal_likelihood_value_


class HKGaussianProcessForwardModel(ForwardModel):
    """
    Same as GaussianProcessForwardModel, but older observations are given more noise
    (HeteroskedasticGaussianProcessRegressor), so the model tracks a changing environment.
    """
    def __init__(self,
                 out_dim: int,
                 sigma_0=.1,
                 length_scale: float = 1.0,
                 length_scale_bounds: tuple[float, float] = (1e-2, 1e2),
                 n_restarts_optimizer: int = 9,
                 recency_weight: float = 1.0,
                 recency_weight_bounds="fixed",
                 t_max: float | None = None,
                 kernel_params: dict | None = None,
                 random_state=None):
        params = {
            "dot_product": {"sigma_0": sigma_0},
            "rbf": {"length_scale": length_scale, "length_scale_bounds": length_scale_bounds},
            "white": {"noise_level": 1e-5, "noise_level_bounds": (1e-10, 1e-1)},
            "recency": {"recency_weight": recency_weight, "recency_weight_bounds": recency_weight_bounds, "t_max": t_max},
        }
        for component, overrides in (kernel_params or {}).items():  # kernel_params has the last word
            params.setdefault(component, {}).update(overrides)

        self.gaussian_process = HeteroskedasticGaussianProcessRegressor(
            kernel_params=params,
            n_restarts_optimizer=n_restarts_optimizer,
            n_targets=out_dim,
            normalize_y=True,          # standardizes y per output dim, undone at predict
            random_state=random_state,
        )
        self.x_scaler = StandardScaler()
        self.X_train = None   # raw actions, (n_samples, in_dim)
        self.y_train = None   # raw observations, (n_samples, out_dim)
        self.t_train = None   # time of each observation, (n_samples,)
        self.out_dim = out_dim

    def update(self, action: np.ndarray, obs: np.ndarray, t: float | None = None):
        """t: time of this observation; defaults to the number of previous observations."""
        action = np.asarray(action, dtype=float)[np.newaxis, :]
        obs = np.asarray(obs, dtype=float).reshape(1, self.out_dim)
        if self.X_train is None:
            t = 0. if t is None else t
            self.X_train, self.y_train, self.t_train = action, obs, np.array([t], dtype=float)
        else:
            t = len(self.t_train) if t is None else t
            self.X_train = np.concatenate([self.X_train, action], axis=0)
            self.y_train = np.concatenate([self.y_train, obs], axis=0)
            self.t_train = np.append(self.t_train, float(t))

        X_scaled = self.x_scaler.fit_transform(self.X_train)   # refit on all data

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.gaussian_process.fit(X_scaled, self.y_train, t=self.t_train)

        return self.gaussian_process.__dict__

    def __call__(self, actions: np.ndarray, t_current: float | None = None) -> np.ndarray:
        """t_current: time at which to predict (recency is measured from it); defaults to the last update."""
        actions = np.atleast_2d(np.asarray(actions, dtype=float))
        if self.X_train is None:
            # No data yet: GP prior mean, which is 0 in raw units
            return np.zeros((len(actions), self.out_dim))

        y = self.gaussian_process.predict(self.x_scaler.transform(actions), t_current=t_current)
        return y.reshape(len(actions), self.out_dim)
