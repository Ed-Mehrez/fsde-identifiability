import numpy as np
import pytest

import fsde

DT = 0.05


def _fou_paths(H, size, n=4000, seed=0):
    return fsde.simulate_additive(None, H, DT, n, size=size, linear_kappa=1.0, substeps=4, rng=seed)


def _rel_l2(est, true):
    return float(np.sqrt(np.mean((est - true) ** 2) / np.mean(true ** 2)))


# ---------------------------------------------------------------- Hurst ----

@pytest.mark.parametrize("H", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_hurst_on_fbm(H):
    rng = np.random.default_rng(10)
    est = [fsde.hurst_second_diff(fsde.fbm(2 ** 13, H, 1.0, rng=rng)) for _ in range(10)]
    assert abs(np.mean(est) - H) < 0.015
    assert np.std(est) < 0.03


@pytest.mark.parametrize("H", [0.3, 0.7])
def test_hurst_second_diff_is_drift_robust(H):
    X = _fou_paths(H, 20, seed=11)
    est = [fsde.hurst_second_diff(x) for x in X]
    assert abs(np.mean(est) - H) < 0.015


def test_variogram_baseline_is_biased_by_drift():
    # documents why the legacy first-difference estimator was replaced
    X = _fou_paths(0.7, 20, seed=12)
    assert np.mean([fsde.hurst_variogram(x) for x in X]) < 0.7 - 0.04


def test_hurst_bootstrap_ci():
    x = fsde.fbm(4000, 0.35, 1.0, rng=13)
    ci = fsde.hurst_bootstrap(x, n_boot=100, rng=14)
    assert ci.lower < 0.35 < ci.upper
    assert 0.002 < ci.se < 0.05


# -------------------------------------------------------------- sigma ----

@pytest.mark.parametrize("H", [0.2, 0.8])
def test_second_difference_constant(H):
    d = np.diff(fsde.fbm(2, H, 1.0, size=200000, rng=15), n=2, axis=-1)
    assert np.mean(d ** 2) == pytest.approx(fsde.second_diff_constant(H), rel=0.02)


@pytest.mark.parametrize("H", [0.2, 0.5, 0.7])
def test_sigma_constant_with_true_H(H):
    X = fsde.simulate_fsde(fsde.fou(1.0, 0.7), H, DT, 4000, size=10, rng=16)
    assert np.mean([fsde.sigma_constant(x, DT, H) for x in X]) == pytest.approx(0.7, rel=0.03)


def test_sigma_kernel_debias_removes_localisation_bias():
    # constant sigma, small H: dt^H ~ spread of X, so the plain estimator is U-shaped
    m = fsde.fou(1.0, 0.7)
    x = fsde.simulate_fsde(m, 0.2, 0.02, 30000, burn_in=200, rng=21)
    g = np.linspace(*np.quantile(x, [0.05, 0.95]), 40)
    plain = _rel_l2(fsde.sigma_kernel(x, 0.02, 0.2, g, debias=False), m.sigma(g))
    deb = _rel_l2(fsde.sigma_kernel(x, 0.02, 0.2, g), m.sigma(g))
    assert plain > 0.04 and deb < 0.025


@pytest.mark.parametrize("H", [0.3, 0.7])
def test_sigma_kernel_state_dependent(H):
    m = fsde.hyperbolic(1.0, 0.5)
    x = fsde.simulate_fsde(m, H, 0.01, 20000, burn_in=500, rng=17)
    g = np.linspace(*np.quantile(x, [0.05, 0.95]), 50)
    assert _rel_l2(fsde.sigma_kernel(x, 0.01, H, g), m.sigma(g)) < 0.06


# -------------------------------------------------------------- drift ----

def _slope(fit, y1=-0.5, y2=0.5):
    return float((fit.b(y2) - fit.b(y1))[0] / (y2 - y1))


@pytest.mark.parametrize("H", [0.2, 0.7])
def test_whitened_drift_is_unbiased_on_fou_and_ols_is_not(H):
    X = _fou_paths(H, 30, seed=18)
    gls, cover = [], []
    for x in X:
        f = fsde.estimate_drift_lamperti(x, DT, H, kind="poly", n_basis=2)
        s = _slope(f)
        se = 2 / (f.basis.hi - f.basis.lo) * np.sqrt(f.cov[1, 1])  # Legendre P1 -> slope
        gls.append(s)
        cover.append(abs(s + 1.0) <= 1.96 * se)
    assert abs(np.mean(gls) + 1.0) < 0.08
    assert 0.8 <= np.mean(cover) <= 1.0
    ols = np.mean([_slope(fsde.estimate_drift_lamperti(x, DT, H, kind="poly", n_basis=2,
                                                       whiten=False)) for x in X[:10]])
    assert abs(ols + 1.0) > 0.5  # -6.9 at H=0.2, -0.24 at H=0.7


def test_gls_equals_ols_at_H_half():
    x = _fou_paths(0.5, 1, seed=19)[0]
    a = fsde.estimate_drift_lamperti(x, DT, 0.5, n_basis=6)
    b = fsde.estimate_drift_lamperti(x, DT, 0.5, n_basis=6, whiten=False)
    np.testing.assert_allclose(a.coef, b.coef, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("H", [0.3, 0.7])
def test_end_to_end_state_dependent_sigma(H):
    m = fsde.hyperbolic(1.0, 0.5)
    x = fsde.simulate_fsde(m, H, 0.02, 50000, burn_in=200, rng=20)
    g = np.linspace(*np.quantile(x, [0.05, 0.95]), 60)
    f = fsde.fit_fsde(x, 0.02, n_basis=6)
    assert abs(f.H - H) < 0.02
    assert _rel_l2(f.sigma(g), m.sigma(g)) < 0.08
    assert _rel_l2(f.mu(g), m.mu(g)) < 0.35
