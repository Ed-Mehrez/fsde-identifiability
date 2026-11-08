"""Diffusion coefficient sigma(x) from squared second differences.

For an fSDE sampled at step dt, the second difference
D_i = X_{i+2} - 2 X_{i+1} + X_i satisfies, to leading order,

    E[D_i^2] ~= E[sigma(X)^2] * dt^{2H} * (4 - 2^{2H}),

since Var(B_{t+2h} - 2 B_{t+h} + B_t) = h^{2H} (4 - 2^{2H}) for fBm. We
smooth D_i^2 against X_{i+1} with a Gaussian kernel (Nadaraya--Watson) and
divide by the normalising constant.

Squared, not absolute, differences: E|D| would add the factor
E|Z| = sqrt(2/pi) ~ 0.80, which the legacy code forgot.

Localisation bias (``debias``). The conditioning point X_{i+1} is itself
built from the increments in D. When dt^H is not small relative to the spread
of X (small H, e.g. dt^H ~ 0.46 at H = 0.2 and dt = 0.02), D and X_{i+1} are
correlated, and E[D^2 | X = x] picks up the term E[D | X = x]^2. This gives a
U-shaped bias (0.64 to 0.81 for a constant sigma = 0.7 on fOU at H = 0.2).
By the law of total variance,

    E[D^2] = E[ Var(D | X) ] + E[ E[D | X]^2 ],

and for a stationary Gaussian process Var(D | X) is constant, so

    E[D^2 | x] - E[D | x]^2 + E_X[ E[D | X]^2 ]

recovers Var(D) exactly. ``debias=True`` uses this form. For state-dependent
sigma and non-Gaussian X it is an approximation. Observed sigma rel. L2 error,
plain -> debiased: fOU H=0.2 0.077 -> 0.011; hyperbolic H=0.2 0.045 -> 0.030;
double well H=0.2 0.037 -> 0.044 (bimodal X, where the Gaussian argument is
weakest).

Plug-in sensitivity: an error eps in H scales sigma_hat by dt^{-eps}, i.e. by
exp(eps * log(1/dt)). At dt = 0.01, eps = 0.02 gives a 10% error, so the H
error must shrink faster than 1 / log n.
"""

from __future__ import annotations

import numpy as np

__all__ = ["second_diff_constant", "sigma_constant", "sigma_kernel"]


def second_diff_constant(H: float) -> float:
    """Var(B_{2} - 2 B_{1} + B_0) for standard fBm: 4 - 2^{2H}."""
    return 4.0 - 2.0 ** (2 * H)


def _pairs(x):
    x = np.asarray(x, dtype=float)
    x2 = x[None, :] if x.ndim == 1 else x
    d = x2[:, 2:] - 2 * x2[:, 1:-1] + x2[:, :-2]
    loc = x2[:, 1:-1]
    return loc.ravel(), d.ravel()


def sigma_constant(x, dt: float, H: float) -> float:
    """Estimate a constant sigma (or the RMS of sigma(X) along the path)."""
    _, d = _pairs(x)
    return float(np.sqrt(np.mean(d ** 2) / (dt ** (2 * H) * second_diff_constant(H))))


def _nw(loc, vals, x_eval, h):
    """Nadaraya--Watson smooths of each column of ``vals`` at ``x_eval``.
    Returns (smooths with shape (len(x_eval), k), kernel weight sums)."""
    num = np.zeros((x_eval.size, vals.shape[1]))
    den = np.zeros(x_eval.size)
    step = max(1, 2_000_000 // max(1, x_eval.size))
    for s in range(0, loc.size, step):
        u = (x_eval[:, None] - loc[None, s:s + step]) / h
        w = np.exp(-0.5 * u * u)
        num += w @ vals[s:s + step]
        den += w.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den[:, None], den


def _silverman(loc):
    iqr = np.subtract(*np.quantile(loc, [0.75, 0.25]))
    spread = min(loc.std(), iqr / 1.349) if iqr > 0 else loc.std()
    return 1.06 * spread * loc.size ** (-1 / 5)


def sigma_kernel(x, dt: float, H: float, x_eval, bandwidth: float | None = None,
                 debias: bool = True, return_counts: bool = False):
    """Nadaraya--Watson estimate of sigma(x) at ``x_eval``.

    ``bandwidth`` defaults to Silverman's rule on the sample of X.
    ``debias`` removes the localisation bias described in the module docstring.
    With ``return_counts=True`` also returns the kernel weight sum at each
    evaluation point (an effective sample size, up to the peak weight 1); use
    it to mask regions with too little data.
    """
    loc, d = _pairs(x)
    x_eval = np.atleast_1d(np.asarray(x_eval, dtype=float))
    h = _silverman(loc) if bandwidth is None else bandwidth
    sm, den = _nw(loc, np.column_stack([d ** 2, d]), x_eval, h)
    v = sm[:, 0]
    if debias:
        sub = loc[:: max(1, loc.size // 2000)]
        m1_sub, _ = _nw(loc, d[:, None], sub, h)
        v = v - sm[:, 1] ** 2 + np.nanmean(m1_sub[:, 0] ** 2)
    sig = np.sqrt(np.maximum(v, 0.0) / (dt ** (2 * H) * second_diff_constant(H)))
    return (sig, den) if return_counts else sig
