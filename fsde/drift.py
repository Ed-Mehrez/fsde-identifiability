"""Drift estimation by fGn-whitened (generalised) least squares on a sieve.

Work on the Lamperti process Y = F(X), which has unit additive noise:

    dY_i = b(Y_{t_i}) dt + dt^H * eps_i,   eps ~ fGn(H) (unit step),

with b(y) = mu(F^{-1}(y)) / sigma(F^{-1}(y)). Expand b in a basis
b(y) = sum_j beta_j phi_j(y) and estimate beta by GLS under the fGn
covariance Gamma_H:

    beta_hat = (Phi' Gamma^{-1} Phi)^{-1} Phi' Gamma^{-1} dY / dt.

GLS is equivalent to whitening with the Cholesky factor of Gamma, a causal
(lower-triangular) filter. After whitening, the regressors phi(Y_{t_i}) are
predetermined with respect to the whitened noise. That is why GLS is unbiased
where plain OLS is not: for H != 1/2 the raw increment noise is correlated
with the past, and so with Y_{t_i}.

Identification of b requires a long horizon T = n dt (ergodicity); on a fixed
horizon the drift is not consistently estimable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.polynomial import legendre
from scipy.interpolate import BSpline

from .fgn import fgn_solve

__all__ = ["Basis", "DriftFit", "estimate_drift_lamperti"]


@dataclass
class Basis:
    """Function basis on [lo, hi]; values outside are clamped to the edge."""

    kind: str  # "bspline" or "poly"
    n_basis: int
    lo: float
    hi: float
    degree: int = 3
    knots: np.ndarray = field(default=None, repr=False)

    @classmethod
    def build(cls, y, kind="bspline", n_basis=8, degree=3, quantiles=(0.005, 0.995)):
        lo, hi = np.quantile(np.asarray(y).ravel(), quantiles)
        b = cls(kind=kind, n_basis=n_basis, lo=float(lo), hi=float(hi), degree=degree)
        if kind == "bspline":
            n_int = n_basis - degree - 1
            if n_int < 0:
                raise ValueError("n_basis must be >= degree + 1 for B-splines")
            interior = np.linspace(lo, hi, n_int + 2)[1:-1]
            b.knots = np.concatenate([[lo] * (degree + 1), interior, [hi] * (degree + 1)])
        elif kind != "poly":
            raise ValueError(f"unknown basis kind {kind!r}")
        return b

    def __call__(self, y) -> np.ndarray:
        y = np.clip(np.asarray(y, dtype=float).ravel(), self.lo, self.hi)
        if self.kind == "bspline":
            return BSpline.design_matrix(y, self.knots, self.degree).toarray()
        z = 2 * (y - self.lo) / (self.hi - self.lo) - 1
        return legendre.legvander(z, self.n_basis - 1)


@dataclass
class DriftFit:
    """Result of the whitened sieve regression for the Lamperti drift b(y)."""

    H: float
    dt: float
    basis: Basis
    coef: np.ndarray
    cov: np.ndarray
    F: Callable | None = None
    sigma: Callable | None = None
    n_obs: int = 0

    def b(self, y) -> np.ndarray:
        return self.basis(y) @ self.coef

    def b_se(self, y) -> np.ndarray:
        P = self.basis(y)
        return np.sqrt(np.einsum("ij,jk,ik->i", P, self.cov, P))

    def mu(self, x) -> np.ndarray:
        """mu(x) = sigma(x) * b(F(x))."""
        x = np.asarray(x, dtype=float)
        return self.sigma(x) * self.b(self.F(x))

    def mu_se(self, x) -> np.ndarray:
        """Pointwise SE of mu_hat, conditional on H, sigma and F."""
        x = np.asarray(x, dtype=float)
        return self.sigma(x) * self.b_se(self.F(x))


def estimate_drift_lamperti(y, dt: float, H: float, basis: Basis | None = None, *,
                            kind: str = "bspline", n_basis: int = 8, degree: int = 3,
                            whiten: bool = True, ridge: float = 0.0,
                            solver: str = "auto") -> DriftFit:
    """Estimate b(y) for dY = b(Y) dt + dB^H from Y sampled at step dt.

    ``y`` is one path (1-D) or several (2-D, rows = paths); normal equations
    are accumulated across paths. ``whiten=False`` gives plain OLS (biased for
    H != 1/2; kept as a baseline). ``ridge`` adds ridge * I to the normal
    matrix (in units of the trace / p).
    """
    y2 = np.atleast_2d(np.asarray(y, dtype=float))
    if basis is None:
        basis = Basis.build(y2[:, :-1], kind=kind, n_basis=n_basis, degree=degree)
    p = basis.n_basis
    A = np.zeros((p, p))
    c = np.zeros(p)
    n_obs = 0
    for path in y2:
        dY = np.diff(path)
        Phi = basis(path[:-1])
        if whiten:
            S = fgn_solve(np.column_stack([Phi, dY]), H, method=solver)
            SPhi, Sy = S[:, :p], S[:, p]
        else:
            SPhi, Sy = Phi, dY
        A += Phi.T @ SPhi
        c += Phi.T @ Sy
        n_obs += dY.size
    if ridge > 0:
        A = A + ridge * np.trace(A) / p * np.eye(p)
    # dY = dt * Phi beta + dt^H eps  =>  beta = A^{-1} c / dt,
    # Cov(beta) = dt^{2H-2} A^{-1} (whitened); OLS covariance is not reported.
    beta = np.linalg.solve(A, c) / dt
    cov = dt ** (2 * H - 2) * np.linalg.inv(A) if whiten else np.full((p, p), np.nan)
    return DriftFit(H=H, dt=dt, basis=basis, coef=beta, cov=cov, n_obs=n_obs)
