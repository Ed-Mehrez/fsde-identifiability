# Fractional SDE Identifiability

Nonparametric estimation of fractional stochastic differential equation parameters.

## The Problem

Given trajectory data from a fractional SDE:

$$dX_t = \mu(X_t) dt + \sigma(X_t) dB^H_t$$

Estimate the **functions** μ(x), σ(x) and **scalar** H without parametric assumptions.

## Key Results

| Parameter | Method | Error |
|-----------|--------|-------|
| **H** (Hurst exponent) | Aggregated variance | 2-3% |
| **σ(x)** (diffusion) | Local quadratic variation | <3% |
| **μ(x)** (drift) | fGN-whitened regression + eigenvalue constraint | 25-40% MRE |

### Identifiability Theorem (Verified)

**(μ, σ, H) are UNIQUELY identifiable** from trajectory data alone:

- **H**: From increment autocorrelation ρ(k) = ½(|k+1|^{2H} - 2|k|^{2H} + |k-1|^{2H})
- **σ(x)**: From local QV: E[(dX)²|X=x] / dt^{2H}
- **μ(x)**: From conditional mean E[dX|X=x] / dt with fGN whitening

No equivalence class - unique solution (follows from Stroock-Varadhan uniqueness).

## Why Whitening is Essential

At typical dt and H<0.5, **noise dominates signal**:
- Drift contribution: O(dt)
- Noise contribution: O(dt^H) >> O(dt) when H < 0.5

**Without whitening, regression learns the NOISE, not the drift!**

The fGN whitening procedure:
1. Estimate H from trajectory
2. Build fGN correlation matrix
3. Cholesky factorize: Σ = L @ L^T
4. Whiten: dx_w = L^{-1} @ dx
5. Regress whitened features on whitened dx

**Key insight**: Cholesky L^{-1} is lower triangular → whitening is CAUSAL (online-capable).

## Quick Start

```bash
# Setup
conda env create -f environment.yml
conda activate fsde

# Run example
python examples/test_generator_estimator.py
```

## Usage

```python
from src.generator_estimator import GeneratorEstimator
from src.rough_paths_generator import FractionalBrownianMotion

# Generate fSDE trajectory
fbm = FractionalBrownianMotion(H=0.3, dt=0.01)
# ... simulate trajectory X

# Estimate parameters
estimator = GeneratorEstimator(backend='koopman', whiten=True)
estimator.fit(X, dt=0.01)

# Get estimates
H_est = estimator.H_
mu_pred = estimator.predict_drift(x_eval)
sigma_pred = estimator.predict_diffusion(x_eval)
```

## Project Structure

```
fsde-identifiability/
├── src/
│   ├── generator_estimator.py    # Core estimator (Profile REML)
│   ├── rough_paths_generator.py  # fBM simulation
│   ├── hurst_estimators.py       # H estimation methods
│   ├── spectral_hurst.py         # Spectral H estimation
│   └── sskf/
│       ├── nystrom_koopman.py    # Nyström eigenfunction extraction
│       └── tensor_features.py    # Signature features
├── examples/
│   ├── test_generator_estimator.py
│   ├── test_nystrom_koopman_signatures.py
│   └── test_signature_consistency_training.py
├── papers/
│   └── jmlr_fsde_identifiability_outline.md
└── docs/
    ├── generator_estimator_walkthrough.md
    └── spectral_roughness_theory.md
```

## Key Innovations

### 1. Profile REML Regularization
GCV degenerates for function estimation when dt appears in design matrix. Profile REML with log-determinant penalty gives ~5x improvement.

### 2. Eigenvalue Amplitude Constraint
Shape is often correct (0.96 corr) but amplitude wrong. Use Koopman eigenvalue λ to constrain:
- For linear drift: λ = exp(-slope·dt)
- Rescale: μ_corrected = μ̂ × (log(λ)/dt) / slope_raw

### 3. Nyström-Koopman for Scalability
O(m³) instead of O(n³) for eigenfunction extraction. Works with both RBF and signature kernels.

## Theory

Based on:
- Hida-Malliavin chaos decomposition
- Signature RKHS isomorphism
- Stroock-Varadhan uniqueness of martingale problems

See [papers/jmlr_fsde_identifiability_outline.md](papers/jmlr_fsde_identifiability_outline.md) for the full paper structure.

## Citation

```bibtex
@article{fsde-identifiability,
  title={Nonparametric Estimation of Fractional SDE Parameters},
  author={Mehrez, Edward},
  journal={In preparation for JMLR},
  year={2026}
}
```

## License

MIT License

---
*Split from [RKHS-KRONIC](https://github.com/Ed-Mehrez/RKHS-KRONIC) repository.*
