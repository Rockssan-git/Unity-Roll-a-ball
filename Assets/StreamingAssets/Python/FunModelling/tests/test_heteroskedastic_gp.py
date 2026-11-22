"""
Tests for src/utils/gaussian_process.py and the heteroskedastic GP in src/model/forward_model.py.

Run from the project root, either with pytest:
    python -m pytest tests -v
or without it:
    python -m tests.test_heteroskedastic_gp
"""
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, DotProduct, WhiteKernel

from src.utils import gaussian_process as gp
from src.model.forward_model import (HeteroskedasticGaussianProcessRegressor as HKGPR,
                                     HKGaussianProcessForwardModel, GaussianProcessForwardModel)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# fixed hyperparameters, sklearn conventions
C_DOT, SIGMA_0, C_RBF, LENGTH_SCALE, NOISE_LEVEL = 0.7, 0.3, 1.5, 0.8, 1e-3
FIXED_PARAMS = {
    "dot_product": {"constant_value": C_DOT, "constant_value_bounds": "fixed", "sigma_0": SIGMA_0, "sigma_0_bounds": "fixed"},
    "rbf": {"constant_value": C_RBF, "constant_value_bounds": "fixed", "length_scale": LENGTH_SCALE, "length_scale_bounds": "fixed"},
    "white": {"noise_level": NOISE_LEVEL, "noise_level_bounds": "fixed"},
    "recency": {"active": False},
}
# same hyperparameters in the utils-function conventions
UTILS_KW = dict(length_scale=2 * LENGTH_SCALE**2, sigma_noise=np.sqrt(NOISE_LEVEL), sigma_f=np.sqrt(C_RBF),
                sigma_dot=np.sqrt(C_DOT), sigma_0=SIGMA_0)


def make_data(n=25, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = np.c_[np.sin(X).sum(1), X[:, 0] * 0.5 + np.cos(X[:, 1])]
    X_star = rng.normal(size=(6, 2))
    return X, y, X_star


# ---------------------------------------------------------------------------
# utils functions
# ---------------------------------------------------------------------------

def test_utils_rbf_gp_matches_sklearn():
    X, y, X_star = make_data()
    ls, sn, sf = 2.0, 0.1, 1.3
    # sklearn RBF uses exp(-d²/(2 l²)) -> l = sqrt(ls / 2)
    sk = GaussianProcessRegressor(C(sf**2, "fixed") * RBF(np.sqrt(ls / 2), "fixed"),
                                  alpha=sn**2 + 1e-6, optimizer=None).fit(X, y[:, 0])
    m, s = sk.predict(X_star, return_std=True)
    m2, v2 = gp.predict_vec_gaussian_process(X_star, X, y[:, 0], ls, sn, sf, return_variance=True)
    np.testing.assert_allclose(m2, m, atol=1e-10)
    np.testing.assert_allclose(v2, s**2, atol=1e-10)
    np.testing.assert_allclose(gp.log_marginal_likelihood_vec_gaussian_process(X, y[:, 0], ls, sn, sf),
                               sk.log_marginal_likelihood(), rtol=1e-10)


def test_utils_single_point_matches_vec():
    X, y, X_star = make_data()
    m, v = gp.predict_gaussian_process(X_star[0], X, y[:, 0], 2., 0.1, 1.3, return_variance=True)
    m2, v2 = gp.predict_vec_gaussian_process(X_star, X, y[:, 0], 2., 0.1, 1.3, return_variance=True)
    np.testing.assert_allclose([m, v], [m2[0], v2[0]], atol=1e-12)


def test_utils_gp_weights_reproduce_mean():
    X, y, X_star = make_data()
    w = gp.compute_gp_weights(X_star, X, 2., 0.1, 1.3)
    m = gp.predict_vec_gaussian_process(X_star, X, y[:, 0], 2., 0.1, 1.3)
    np.testing.assert_allclose(np.asarray(w) @ y[:, 0], m, atol=1e-10)
    np.testing.assert_allclose(gp.weights_to_pct_contribution(w).sum(1), 100.)


def test_utils_heteroskedastic_matches_manual_noise():
    X, y, X_star = make_data(n=20)
    t, w, t_max, t_current = np.arange(20), 3., 25., 20.
    sigma_noise = np.sqrt(0.1**2 + np.exp(w * (t_current - t) / t_max) - 1)  # the formula, written by hand
    ref = gp.predict_vec_gaussian_process(X_star, X, y[:, 0], 2., sigma_noise, 1.3)
    new = gp.predict_vec_heteroskedastic_gaussian_process(X_star, X, y[:, 0], 2., 0.1, 1.3, t=t, recency_weight=w,
                                                          t_max=t_max, t_current=t_current)
    np.testing.assert_allclose(new, ref, atol=1e-12)


def test_utils_recency_weight_zero_is_plain_gp():
    X, y, X_star = make_data(n=20)
    new = gp.predict_vec_heteroskedastic_gaussian_process(X_star, X, y[:, 0], 2., 0.1, 1.3, t=np.arange(20), recency_weight=0.)
    np.testing.assert_allclose(new, gp.predict_vec_gaussian_process(X_star, X, y[:, 0], 2., 0.1, 1.3), atol=1e-12)


# ---------------------------------------------------------------------------
# HeteroskedasticGaussianProcessRegressor
# ---------------------------------------------------------------------------

def test_regressor_fixed_matches_sklearn():
    X, y, X_star = make_data()
    k = C(C_DOT, "fixed") * DotProduct(SIGMA_0, "fixed") + C(C_RBF, "fixed") * RBF(LENGTH_SCALE, "fixed") \
        + WhiteKernel(NOISE_LEVEL, "fixed")
    sk = GaussianProcessRegressor(k, normalize_y=True).fit(X, y)
    hk = HKGPR(FIXED_PARAMS, normalize_y=True).fit(X, y)
    m, s = sk.predict(X_star, return_std=True)
    m2, s2 = hk.predict(X_star, return_std=True)
    np.testing.assert_allclose(m2, m, atol=1e-10)
    np.testing.assert_allclose(s2, s, atol=1e-10)
    np.testing.assert_allclose(hk.log_marginal_likelihood_value_, sk.log_marginal_likelihood_value_, rtol=1e-8)


def test_regressor_optimised_matches_sklearn():
    X, y, X_star = make_data()
    k = C(1.) * DotProduct(0.1) + C(1.) * RBF(1., (1e-2, 1e2)) + WhiteKernel(1e-5, (1e-10, 1e-1))
    sk = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=9, random_state=0).fit(X, y)
    hk = HKGPR({"rbf": {"length_scale_bounds": (1e-2, 1e2)}, "recency": {"active": False}},
               normalize_y=True, n_restarts_optimizer=9, random_state=0).fit(X, y)
    # both optimisers should reach the same optimum (ours must not be worse)
    assert hk.log_marginal_likelihood_value_ >= sk.log_marginal_likelihood_value_ - 1e-3
    np.testing.assert_allclose(hk.predict(X_star), sk.predict(X_star), atol=1e-4)


def test_regressor_output_shapes():
    X, y, X_star = make_data()
    hk = HKGPR(FIXED_PARAMS, normalize_y=True)
    m, s = hk.fit(X, y[:, 0]).predict(X_star, return_std=True)
    assert m.shape == s.shape == (6,)
    m, s = hk.fit(X, y).predict(X_star, return_std=True)
    assert m.shape == s.shape == (6, 2)


def test_regressor_recency_matches_utils():
    X, y, X_star = make_data()
    t, w, t_max = np.arange(25), 2., 30.
    params = {**FIXED_PARAMS, "recency": {"active": True, "recency_weight": w, "t_max": t_max}}
    hk = HKGPR(params, alpha=1e-6).fit(X, y[:, 0], t=t)
    kw = dict(t=t, recency_weight=w, t_max=t_max, **UTILS_KW)
    np.testing.assert_allclose(hk.predict(X_star),
                               gp.predict_vec_heteroskedastic_gaussian_process(X_star, X, y[:, 0], **kw), atol=1e-10)
    np.testing.assert_allclose(hk.log_marginal_likelihood_value_,
                               gp.log_marginal_likelihood_heteroskedastic_gaussian_process(X, y[:, 0], **kw), rtol=1e-10)


def test_regressor_components_can_be_deactivated():
    X, y, X_star = make_data()
    params = {**FIXED_PARAMS, "dot_product": {"active": False}}
    hk = HKGPR(params, alpha=1e-6).fit(X, y[:, 0])
    ref = gp.predict_vec_gaussian_process(X_star, X, y[:, 0], UTILS_KW["length_scale"], UTILS_KW["sigma_noise"],
                                          UTILS_KW["sigma_f"])
    np.testing.assert_allclose(hk.predict(X_star), ref, atol=1e-10)


def test_regressor_recency_weight_can_be_fitted():
    X, y, _ = make_data()
    hk = HKGPR({"recency": {"recency_weight_bounds": (1e-3, 10.)}}, normalize_y=True,
               n_restarts_optimizer=2, random_state=0).fit(X, y, t=np.arange(25))
    w = hk.kernel_params_["recency"]["recency_weight"]
    assert 1e-3 - 1e-12 <= w <= 10. + 1e-12
    assert np.isfinite(hk.log_marginal_likelihood_value_)


# ---------------------------------------------------------------------------
# HKGaussianProcessForwardModel (slower: one fit per update)
# ---------------------------------------------------------------------------

def run_drifting_world(forward_model, n_updates=40, shift=2., seed=1):
    """First half: obs = [a0 + a1, a0 - a1]; second half the first output is shifted by +shift."""
    rng = np.random.default_rng(seed)
    for i in range(n_updates):
        a = rng.uniform(-1, 1, 2)
        forward_model.update(a, np.array([a.sum() + (shift if i >= n_updates // 2 else 0.), a[0] - a[1]]))
    return forward_model(np.zeros((1, 2)))[0]


def test_forward_model_prior_is_zero():
    fm = HKGaussianProcessForwardModel(out_dim=2)
    np.testing.assert_array_equal(fm(np.zeros((3, 2))), np.zeros((3, 2)))


def test_forward_model_tracks_drift_better_than_sklearn():
    hk_pred = run_drifting_world(HKGaussianProcessForwardModel(out_dim=2, recency_weight=3., n_restarts_optimizer=2,
                                                               random_state=0))
    sk_pred = run_drifting_world(GaussianProcessForwardModel(out_dim=2, n_restarts_optimizer=2))
    # true current value at action 0 is [2, 0]
    assert abs(hk_pred[0] - 2.) < abs(sk_pred[0] - 2.)
    assert abs(hk_pred[1]) < 0.1


# ---------------------------------------------------------------------------
# timing comparison (not a test, printed when run as a script)
# ---------------------------------------------------------------------------

def benchmark(n_updates=40):
    for name, fm in [("HKGaussianProcessForwardModel", HKGaussianProcessForwardModel(out_dim=2, recency_weight=3., random_state=0)),
                     ("GaussianProcessForwardModel (sklearn)", GaussianProcessForwardModel(out_dim=2))]:
        t0 = time.time()
        pred = run_drifting_world(fm, n_updates)
        print(f"  {name}: {n_updates} updates in {time.time() - t0:.1f}s, prediction at 0 = {pred.round(3)} (true [2, 0])")


if __name__ == "__main__":
    tests = [(name, f) for name, f in sorted(globals().items()) if name.startswith("test_") and callable(f)]
    n_failed = 0
    for name, f in tests:
        t0 = time.time()
        try:
            f()
            print(f"PASS  {name} ({time.time() - t0:.1f}s)")
        except Exception as e:
            n_failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - n_failed}/{len(tests)} passed")
    print("\nbenchmark:")
    benchmark()
