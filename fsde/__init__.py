"""fsde: audited estimators for 1-D fractional SDEs

    dX_t = mu(X_t) dt + sigma(X_t) dB^H_t.

This package replaces the legacy ``src/`` code (see ``docs/phase0.md`` for why).
"""

from .diffusion import second_diff_constant, sigma_constant, sigma_kernel
from .drift import Basis, DriftFit, estimate_drift_lamperti
from .fgn import fgn_autocov, fgn_solve, sample_fgn
from .fit import FSDEFit, fit_fsde
from .hurst import HurstCI, hurst_bootstrap, hurst_second_diff, hurst_variogram
from .simulate import (FSDEModel, double_well, fbm, fgn, fou, fou_stationary_variance,
                       hyperbolic, lamperti, simulate_additive, simulate_fsde)

__all__ = [
    "fgn_autocov", "sample_fgn", "fgn_solve",
    "fbm", "fgn", "simulate_additive", "simulate_fsde", "lamperti",
    "FSDEModel", "fou", "double_well", "hyperbolic", "fou_stationary_variance",
    "hurst_second_diff", "hurst_variogram", "hurst_bootstrap", "HurstCI",
    "second_diff_constant", "sigma_constant", "sigma_kernel",
    "Basis", "DriftFit", "estimate_drift_lamperti",
    "FSDEFit", "fit_fsde",
]
