import numpy as np
import pytest

import fsde


@pytest.mark.parametrize("H", [0.2, 0.5, 0.8])
def test_fgn_sampler_matches_autocovariance(H):
    rng = np.random.default_rng(0)
    X = fsde.sample_fgn(64, H, size=20000, rng=rng)
    gamma = fsde.fgn_autocov(4, H)
    emp = [np.mean(X ** 2)] + [np.mean(X[:, :-k] * X[:, k:]) for k in range(1, 4)]
    np.testing.assert_allclose(emp, gamma, atol=0.03)


def test_fgn_autocov_basic():
    assert fsde.fgn_autocov(3, 0.3)[0] == pytest.approx(1.0)
    np.testing.assert_allclose(fsde.fgn_autocov(5, 0.5)[1:], 0.0, atol=1e-15)


@pytest.mark.parametrize("H", [0.01, 0.99])
def test_circulant_embedding_is_psd_at_extreme_H(H):
    # the legacy generator zeroed the middle embedding entry and lost exactness near H=1
    x = fsde.sample_fgn(1000, H, rng=1)
    assert x.shape == (1000,) and np.all(np.isfinite(x))


@pytest.mark.parametrize("H", [0.2, 0.7, 0.9])
def test_toeplitz_pcg_matches_levinson(H):
    rng = np.random.default_rng(2)
    B = rng.standard_normal((3000, 3))
    a = fsde.fgn_solve(B, H, method="pcg")
    b = fsde.fgn_solve(B, H, method="levinson")
    assert np.max(np.abs(a - b)) / np.max(np.abs(b)) < 1e-6
    assert fsde.fgn_solve(B[:, 0], H).shape == (3000,)


@pytest.mark.parametrize("H", [0.3, 0.7])
def test_fbm_variance_scaling(H):
    B = fsde.fbm(200, H, 0.05, size=4000, rng=3)
    assert np.all(B[:, 0] == 0)
    assert B[:, -1].var() == pytest.approx((200 * 0.05) ** (2 * H), rel=0.07)


@pytest.mark.parametrize("H", [0.3, 0.5, 0.7])
def test_fou_stationary_variance_closed_form(H):
    X = fsde.simulate_additive(None, H, 0.005, 2400, size=4000, linear_kappa=1.0, rng=4)
    assert X[:, -1].var() == pytest.approx(fsde.fou_stationary_variance(1.0, 1.0, H), rel=0.07)


def test_linear_filter_matches_euler_loop():
    kw = dict(H=0.4, dt=0.05, n=500, x0=0.3, substeps=2)
    a = fsde.simulate_additive(None, linear_kappa=0.8, rng=5, **kw)
    b = fsde.simulate_additive(lambda y: -0.8 * y, rng=5, **kw)
    np.testing.assert_allclose(a, b, atol=1e-10)


def test_numerical_lamperti_matches_analytic():
    m = fsde.hyperbolic(1.0, 0.5)
    F, Finv = fsde.lamperti(m.sigma, -10, 10, n_grid=40001)
    x = np.linspace(-5, 5, 101)
    np.testing.assert_allclose(F(x), m.F(x), atol=1e-4)
    np.testing.assert_allclose(Finv(m.F(x)), x, atol=1e-4)
