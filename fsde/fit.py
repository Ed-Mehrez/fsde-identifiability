"""End-to-end estimation of (H, sigma(x), mu(x)) from a sampled path.

Order matters, and each step plugs the previous estimate into the next:

1. H from second differences (drift-robust, sigma-free).
2. sigma(x) from squared second differences, normalised with H_hat.
3. Lamperti transform Y = F_hat(X) with F_hat' = 1 / sigma_hat.
4. b(y) by fGn(H_hat)-whitened sieve least squares on Y; mu = sigma_hat * b(F_hat).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .diffusion import sigma_constant, sigma_kernel
from .drift import DriftFit, estimate_drift_lamperti
from .hurst import hurst_second_diff
from .simulate import lamperti

__all__ = ["FSDEFit", "fit_fsde"]


@dataclass
class FSDEFit:
    H: float
    sigma_grid: np.ndarray
    sigma_values: np.ndarray
    drift: DriftFit

    def sigma(self, x):
        return np.interp(np.asarray(x, dtype=float), self.sigma_grid, self.sigma_values)

    def mu(self, x):
        return self.drift.mu(x)

    def mu_se(self, x):
        return self.drift.mu_se(x)


def fit_fsde(x, dt: float, *, H: float | None = None, sigma: str | float = "kernel",
             n_grid: int = 200, bandwidth: float | None = None, kind: str = "bspline",
             n_basis: int = 8, degree: int = 3, whiten: bool = True,
             solver: str = "auto") -> FSDEFit:
    """Fit an fSDE to one path (1-D) or several (2-D, rows = paths).

    ``H``: pass a value to fix it, else estimated. ``sigma``: ``"kernel"``
    (state-dependent), ``"constant"``, or a float to fix it.
    """
    x2 = np.atleast_2d(np.asarray(x, dtype=float))
    H_hat = hurst_second_diff(x2) if H is None else float(H)

    lo, hi = np.quantile(x2, [0.001, 0.999])
    pad = 0.05 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, n_grid)
    if isinstance(sigma, (int, float)):
        s_vals = np.full(n_grid, float(sigma))
    elif sigma == "constant":
        s_vals = np.full(n_grid, sigma_constant(x2, dt, H_hat))
    elif sigma == "kernel":
        s_vals, counts = sigma_kernel(x2, dt, H_hat, grid, bandwidth=bandwidth, return_counts=True)
        # where data are scarce, hold the last well-supported value (constant extrapolation)
        ok = np.isfinite(s_vals) & (counts >= 20)
        if not ok.any():
            raise ValueError("too little data for kernel sigma; use sigma='constant'")
        i0, i1 = np.argmax(ok), len(ok) - 1 - np.argmax(ok[::-1])
        s_vals = np.interp(grid, grid[ok], s_vals[ok])
        s_vals[:i0] = s_vals[i0]
        s_vals[i1 + 1:] = s_vals[i1]
    else:
        raise ValueError(f"unknown sigma option {sigma!r}")
    s_vals = np.maximum(s_vals, 1e-8)

    def sig_fn(v):
        return np.interp(np.asarray(v, dtype=float), grid, s_vals)

    F, _ = lamperti(sig_fn, grid[0], grid[-1], ref=float(np.median(x2)), n_grid=4001)
    Y = F(x2)
    dfit = estimate_drift_lamperti(Y, dt, H_hat, kind=kind, n_basis=n_basis, degree=degree,
                                   whiten=whiten, solver=solver)
    dfit.F, dfit.sigma = F, sig_fn
    return FSDEFit(H=H_hat, sigma_grid=grid, sigma_values=s_vals, drift=dfit)
