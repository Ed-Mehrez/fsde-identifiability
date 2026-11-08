"""Simulation of fBm and 1-D fractional SDEs

    dX_t = mu(X_t) dt + sigma(X_t) dB^H_t.

State-dependent sigma is handled by the Lamperti transform. With
F(x) = int dx / sigma(x), the pathwise chain rule gives

    dY_t = b(Y_t) dt + dB^H_t,   Y = F(X),   b(y) = mu(F^{-1}(y)) / sigma(F^{-1}(y)),

which has *additive* noise. Euler on Y converges for every H in (0, 1),
whereas left-point Euler on X with state-dependent sigma does not converge to a
meaningful solution for H < 1/2. Mapping back, X = F^{-1}(Y) is the pathwise
solution: Young for H > 1/2, the Doss--Sussmann / rough-path solution for
H < 1/2, and the *Stratonovich* SDE at H = 1/2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import lfilter

from .fgn import sample_fgn

__all__ = [
    "fbm", "fgn", "simulate_additive", "simulate_fsde", "lamperti",
    "FSDEModel", "fou", "double_well", "hyperbolic", "fou_stationary_variance",
]

Array = np.ndarray


def fgn(n: int, H: float, dt: float, size: int | None = None, rng=None) -> Array:
    """fGn increments B_{(i+1)dt} - B_{i dt}, i = 0..n-1."""
    return dt ** H * sample_fgn(n, H, size=size, rng=rng)


def fbm(n: int, H: float, dt: float, size: int | None = None, rng=None) -> Array:
    """fBm on the grid 0, dt, ..., n*dt (n+1 points, starting at 0)."""
    inc = fgn(n, H, dt, size=size, rng=rng)
    pad = np.zeros(inc.shape[:-1] + (1,))
    return np.concatenate([pad, np.cumsum(inc, axis=-1)], axis=-1)


def simulate_additive(b: Callable[[Array], Array], H: float, dt: float, n: int,
                      x0: float = 0.0, scale: float = 1.0, size: int | None = None,
                      substeps: int = 1, rng=None, linear_kappa: float | None = None) -> Array:
    """Euler scheme for dY = b(Y) dt + scale * dB^H on a fine grid of step
    dt/substeps, returned on the coarse grid 0, dt, ..., n*dt.

    If ``linear_kappa`` is given, b(y) = -kappa*y is assumed and a fast linear
    filter is used (``b`` is then ignored).
    """
    h = dt / substeps
    N = n * substeps
    m = 1 if size is None else size
    dB = scale * fgn(N, H, h, size=m, rng=rng)  # (m, N)
    if linear_kappa is not None:
        a = 1.0 - linear_kappa * h
        # Y_{j+1} = a Y_j + dB_j, Y_0 = x0
        zi = np.full((m, 1), a * x0)
        Y = np.concatenate([np.full((m, 1), x0), lfilter([1.0], [1.0, -a], dB, axis=-1, zi=zi)[0]], axis=-1)
    else:
        Y = np.empty((m, N + 1))
        Y[:, 0] = x0
        for j in range(N):
            Y[:, j + 1] = Y[:, j] + b(Y[:, j]) * h + dB[:, j]
    Y = Y[:, ::substeps]
    return Y[0] if size is None else Y


def lamperti(sigma: Callable[[Array], Array], lo: float, hi: float, ref: float = 0.0,
             n_grid: int = 20001):
    """Numerical Lamperti pair (F, F^{-1}) on [lo, hi], F(x) = int_ref^x du / sigma(u).

    Requires sigma > 0 on [lo, hi]. Outside the grid F is extended linearly
    using the end slopes.
    """
    x = np.linspace(lo, hi, n_grid)
    s = np.asarray(sigma(x), dtype=float)
    if np.any(s <= 0):
        raise ValueError("sigma must be strictly positive on the Lamperti grid")
    F = cumulative_trapezoid(1.0 / s, x, initial=0.0)
    F -= np.interp(ref, x, F)
    sl_lo, sl_hi = 1.0 / s[0], 1.0 / s[-1]

    def F_fn(v):
        v = np.asarray(v, dtype=float)
        out = np.interp(v, x, F)
        out = np.where(v < lo, F[0] + (v - lo) * sl_lo, out)
        return np.where(v > hi, F[-1] + (v - hi) * sl_hi, out)

    def Finv_fn(y):
        y = np.asarray(y, dtype=float)
        out = np.interp(y, F, x)
        out = np.where(y < F[0], lo + (y - F[0]) / sl_lo, out)
        return np.where(y > F[-1], hi + (y - F[-1]) / sl_hi, out)

    return F_fn, Finv_fn


@dataclass
class FSDEModel:
    """A 1-D fSDE with optional analytic Lamperti pair."""

    name: str
    mu: Callable[[Array], Array]
    sigma: Callable[[Array], Array]
    F: Callable[[Array], Array] | None = None
    Finv: Callable[[Array], Array] | None = None
    x0: float = 0.0


def simulate_fsde(model: FSDEModel, H: float, dt: float, n: int, size: int | None = None,
                  substeps: int = 4, burn_in: int = 0, rng=None,
                  lamperti_range: tuple[float, float] = (-50.0, 50.0)) -> Array:
    """Simulate X on 0, dt, ..., n*dt via Euler on the Lamperti process.

    ``burn_in`` coarse steps are simulated and discarded (the returned path
    then starts in approximately stationary state for ergodic models).
    """
    if model.F is not None and model.Finv is not None:
        F, Finv = model.F, model.Finv
    else:
        F, Finv = lamperti(model.sigma, *lamperti_range)

    def b(y):
        x = Finv(y)
        return model.mu(x) / model.sigma(x)

    Y = simulate_additive(b, H, dt, n + burn_in, x0=float(F(model.x0)), size=size,
                          substeps=substeps, rng=rng)
    return Finv(Y[..., burn_in:])


# --------------------------------------------------------------------------
# Reference models
# --------------------------------------------------------------------------

def fou(kappa: float = 1.0, sigma: float = 1.0, theta: float = 0.0) -> FSDEModel:
    """Fractional OU: dX = -kappa (X - theta) dt + sigma dB^H."""
    return FSDEModel(
        name=f"fOU(kappa={kappa}, sigma={sigma})",
        mu=lambda x: -kappa * (np.asarray(x) - theta),
        sigma=lambda x: np.full_like(np.asarray(x, dtype=float), sigma),
        F=lambda x: (np.asarray(x) - theta) / sigma,
        Finv=lambda y: theta + sigma * np.asarray(y),
        x0=theta,
    )


def double_well(a: float = 1.0, sigma: float = 0.7) -> FSDEModel:
    """Double well: dX = (a X - X^3) dt + sigma dB^H. Nonlinear drift, additive noise."""
    return FSDEModel(
        name=f"double-well(a={a}, sigma={sigma})",
        mu=lambda x: a * np.asarray(x) - np.asarray(x) ** 3,
        sigma=lambda x: np.full_like(np.asarray(x, dtype=float), sigma),
        F=lambda x: np.asarray(x) / sigma,
        Finv=lambda y: sigma * np.asarray(y),
        x0=np.sqrt(a),
    )


def hyperbolic(kappa: float = 1.0, s: float = 0.5) -> FSDEModel:
    """State-dependent diffusion: dX = -kappa X dt + s sqrt(1 + X^2) dB^H.

    Analytic Lamperti pair F(x) = asinh(x)/s, F^{-1}(y) = sinh(s y); the
    transformed drift b(y) = -(kappa/s) tanh(s y) is bounded and smooth.
    """
    return FSDEModel(
        name=f"hyperbolic(kappa={kappa}, s={s})",
        mu=lambda x: -kappa * np.asarray(x),
        sigma=lambda x: s * np.sqrt(1.0 + np.asarray(x) ** 2),
        F=lambda x: np.arcsinh(np.asarray(x)) / s,
        Finv=lambda y: np.sinh(s * np.asarray(y)),
        x0=0.0,
    )


def fou_stationary_variance(kappa: float, sigma: float, H: float) -> float:
    """Var of the stationary fOU sigma * int_{-inf}^t e^{-kappa(t-s)} dB^H_s,
    equal to sigma^2 Gamma(2H+1) / (2 kappa^{2H}) (Cheridito, Kawaguchi, Maejima 2003)."""
    from scipy.special import gamma as G

    return sigma ** 2 * G(2 * H + 1) / (2 * kappa ** (2 * H))
