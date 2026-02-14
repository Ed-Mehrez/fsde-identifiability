# AGENTS.md

Context for AI coding agents working on this repository.

## Environment

**Use the shared conda environment** - do NOT create a new one:

```bash
conda activate rkhs-kronic-gpu
```

The `environment.yml` in this repo is for CI/collaborators only.

## Project Overview

Nonparametric estimation of fractional SDE parameters (μ, σ, H) from trajectory data.

**Model**: `dX_t = μ(X_t) dt + σ(X_t) dB^H_t`

We estimate **functions** μ(x), σ(x) and **scalar** H without parametric assumptions.

## Key Files

| File | Purpose |
|------|---------|
| `src/generator_estimator.py` | Core estimator with Profile REML regularization |
| `src/sskf/nystrom_koopman.py` | Nyström-accelerated eigenfunction extraction |
| `src/rough_paths_generator.py` | Fractional Brownian motion simulation |
| `papers/jmlr_fsde_identifiability_outline.md` | Paper structure |

## Build & Test

```bash
# Run main example
python examples/test_generator_estimator.py

# Run eigenvalue recovery test
python examples/test_nystrom_koopman_signatures.py
```

## Critical Knowledge

### fGN Whitening is ESSENTIAL
At H<0.5, noise dominates signal. Without whitening, regression learns NOISE, not drift.

```python
# Whitening procedure
# 1. Estimate H from trajectory
# 2. Build fGN correlation matrix
# 3. Cholesky: Σ = L @ L^T
# 4. Whiten: dx_w = L^{-1} @ dx
```

### Profile REML > GCV
GCV degenerates for function estimation. Use Profile REML with log-determinant penalty.

### Eigenvalue Amplitude Constraint
Shape is correct but amplitude wrong? Use Koopman eigenvalue λ to rescale:
- `slope_target = log(λ)/dt`
- `μ_corrected = μ̂ × (slope_target / slope_raw)`

## Conventions

- Python 3.10+
- NumPy/SciPy for numerics
- `iisignature` for signature computation
- Type hints encouraged but not required

## Related Repositories

- Parent: [RKHS-KRONIC](https://github.com/Ed-Mehrez/RKHS-KRONIC)
- Sibling: [pomdp-koopman-control](https://github.com/Ed-Mehrez/pomdp-koopman-control)
- Sibling: [rkhs-koopman-control](https://github.com/Ed-Mehrez/rkhs-koopman-control)
