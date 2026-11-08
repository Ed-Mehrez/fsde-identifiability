"""Fractional Gaussian noise: autocovariance, exact sampling, Toeplitz solves.

All functions here work with *unit-step* fGn, i.e. increments of fBm sampled at
step 1, whose autocovariance is

    gamma(k) = 0.5 * (|k+1|^{2H} - 2|k|^{2H} + |k-1|^{2H}),   gamma(0) = 1.

Increments at step ``dt`` are ``dt**H`` times unit-step fGn (self-similarity).
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import solve_toeplitz

__all__ = ["fgn_autocov", "sample_fgn", "fgn_solve"]


def _check_H(H: float) -> None:
    if not 0.0 < H < 1.0:
        raise ValueError(f"Hurst exponent must lie in (0, 1), got {H}")


def fgn_autocov(n: int, H: float) -> np.ndarray:
    """Autocovariance gamma(0), ..., gamma(n-1) of unit-step fGn."""
    _check_H(H)
    k = np.arange(n, dtype=float)
    return 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * k ** (2 * H) + np.abs(k - 1) ** (2 * H))


def sample_fgn(n: int, H: float, size: int | None = None, rng=None) -> np.ndarray:
    """Exact unit-step fGn by circulant embedding (Davies--Harte / Wood--Chan).

    The circulant first row is ``[g0, g1, ..., g_{n-1}, g_n, g_{n-1}, ..., g1]``
    (length 2n). Its eigenvalues are non-negative for fGn for every H in (0, 1)
    (Craigmile 2003), so the method is exact; we still check and raise if
    round-off produces a materially negative eigenvalue.

    Parameters
    ----------
    n : number of increments per path.
    H : Hurst exponent in (0, 1).
    size : number of independent paths. ``None`` returns a 1-D array.
    rng : ``numpy.random.Generator`` or seed.

    Returns
    -------
    Array of shape ``(n,)`` or ``(size, n)``.
    """
    _check_H(H)
    rng = np.random.default_rng(rng)
    m = 1 if size is None else int(size)
    g = fgn_autocov(n + 1, H)
    row = np.concatenate([g, g[-2:0:-1]])  # length 2n
    M = row.size
    lam = np.fft.fft(row).real
    if lam.min() < -1e-8 * lam.max():
        raise RuntimeError(f"circulant embedding not PSD (min eig {lam.min():.3e})")
    lam = np.clip(lam, 0.0, None)
    # Each complex FFT yields two independent real samples (real and imaginary parts).
    n_draw = (m + 1) // 2
    Z = rng.standard_normal((n_draw, M)) + 1j * rng.standard_normal((n_draw, M))
    W = np.fft.fft(np.sqrt(lam / M) * Z, axis=-1)[:, :n]
    out = np.concatenate([W.real, W.imag], axis=0)[:m]
    return out[0] if size is None else out


def _toeplitz_pcg(gamma: np.ndarray, B: np.ndarray, tol: float, maxiter: int) -> np.ndarray:
    """Solve T X = B for symmetric positive-definite Toeplitz T (first column
    ``gamma``) by preconditioned CG with FFT mat-vecs and T. Chan's optimal
    circulant preconditioner. ``B`` has shape (n, k)."""
    n = gamma.size
    emb = np.concatenate([gamma, [0.0], gamma[:0:-1]])
    f_emb = np.fft.rfft(emb)

    def matvec(X):
        Xp = np.zeros((2 * n, X.shape[1]))
        Xp[:n] = X
        return np.fft.irfft(f_emb[:, None] * np.fft.rfft(Xp, axis=0), n=2 * n, axis=0)[:n]

    k = np.arange(n)
    g_rev = np.concatenate([[0.0], gamma[:0:-1]])  # g_rev[k] = gamma[n-k]
    chan = ((n - k) * gamma + k * g_rev) / n
    lam = np.fft.fft(chan).real
    if lam.min() <= 0:
        lam = np.maximum(lam, 1e-12 * lam.max())

    def precond(R):
        return np.fft.ifft(np.fft.fft(R, axis=0) / lam[:, None], axis=0).real

    X = np.zeros_like(B)
    R = B.copy()
    Z = precond(R)
    P = Z.copy()
    rz = np.sum(R * Z, axis=0)
    bnorm = np.linalg.norm(B, axis=0)
    bnorm[bnorm == 0] = 1.0
    for _ in range(maxiter):
        AP = matvec(P)
        pap = np.sum(P * AP, axis=0)
        alpha = np.divide(rz, pap, out=np.zeros_like(rz), where=pap != 0)
        X += alpha * P
        R -= alpha * AP
        if np.all(np.linalg.norm(R, axis=0) <= tol * bnorm):
            return X
        Z = precond(R)
        rz_new = np.sum(R * Z, axis=0)
        beta = np.divide(rz_new, rz, out=np.zeros_like(rz), where=rz != 0)
        P = Z + beta * P
        rz = rz_new
    warnings.warn("Toeplitz PCG did not reach tolerance; result may be inaccurate", RuntimeWarning)
    return X


def fgn_solve(B: np.ndarray, H: float, method: str = "auto", tol: float = 1e-10,
              maxiter: int = 2000) -> np.ndarray:
    """Solve ``Gamma_H X = B`` where ``Gamma_H`` is the n x n unit-step fGn
    covariance matrix. ``B`` may be 1-D (n,) or 2-D (n, k).

    ``method``: ``"levinson"`` (exact, O(n^2) per column), ``"pcg"`` (FFT-based,
    O(n log n) per iteration) or ``"auto"`` (Levinson for n <= 1000).
    """
    B = np.asarray(B, dtype=float)
    squeeze = B.ndim == 1
    B2 = B[:, None] if squeeze else B
    n = B2.shape[0]
    gamma = fgn_autocov(n, H)
    if method == "auto":
        method = "levinson" if n <= 1000 else "pcg"
    if method == "levinson":
        X = solve_toeplitz(gamma, B2)
        X = X.reshape(B2.shape)
    elif method == "pcg":
        X = _toeplitz_pcg(gamma, B2, tol, maxiter)
    else:
        raise ValueError(f"unknown method {method!r}")
    return X[:, 0] if squeeze else X
