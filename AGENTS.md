# AGENTS.md

Context for AI coding agents working on this repository.

## Environment

**Use the shared conda environment** - do NOT create a new one:

```bash
conda activate rkhs-kronic-gpu
```

The `environment.yml` in this repo is for CI/collaborators only. The `fsde/`
package needs only NumPy and SciPy (>= 1.11), plus pytest for the tests.

## Project Overview

Nonparametric estimation of fractional SDE parameters (μ, σ, H) from trajectory data.

**Model**: `dX_t = μ(X_t) dt + σ(X_t) dB^H_t`

We estimate **functions** μ(x), σ(x) and **scalar** H without parametric assumptions.

## Status (November 2026)

- **`fsde/` is the canonical code.** It holds the audited Phase 0 rebuild; see `docs/phase0.md`.
- **`src/` is legacy.** An audit found its theory and several implementations wrong. Read it for history only and do not extend it.
- **Docs under `docs/mllm/` and `papers/` predate the audit.** Treat their claims as unverified.

## Key Files

| File | Purpose |
|------|---------|
| `fsde/fgn.py` | fGn autocovariance, exact circulant-embedding sampler, fast Toeplitz solves (PCG) |
| `fsde/simulate.py` | fBm, fSDE simulation via the Lamperti transform, reference models |
| `fsde/hurst.py` | H from second differences (drift-robust), bootstrap CI |
| `fsde/diffusion.py` | σ(x) from squared second differences, with localisation debias |
| `fsde/drift.py` | fGn-whitened GLS sieve regression for the drift |
| `fsde/fit.py` | End-to-end `fit_fsde` |
| `docs/phase0.md` | What changed from the legacy code and why; known limitations |
| `examples/phase0_grid.py` | Accuracy grid → `results/phase0_grid.md` |

## Build & Test

```bash
python -m pytest              # fsde/tests, ~15 s
python examples/phase0_grid.py  # ~10 min
```

## Critical Knowledge

- **Not Markov.** For H ≠ 1/2 the process X is not Markov. Generator, Koopman and
  Stroock–Varadhan arguments do not apply. Do not reintroduce them.
- **Drift needs a long horizon.** μ is identifiable only as T → ∞ under ergodicity. H and σ(x)
  come from fine sampling.
- **Whitening is essential for the drift.** Plain regression of dX on X is badly biased
  for H ≠ 1/2: slope −6.9 at H = 0.2 when the truth is −1. Use GLS under the fGn(H)
  covariance, which `fsde.estimate_drift_lamperti` does.
- **Use second differences, not first, for H and σ.** First-difference variograms are biased
  by drift.
- **Errors in H are amplified by log(1/dt) in σ.** σ̂ ∝ dt^{−(Ĥ−H)}.
- **Simulate state-dependent σ through the Lamperti transform.** Left-point Euler on X does
  not converge for H < 1/2.

## Conventions

- Python 3.10+
- NumPy/SciPy for numerics
- Every new estimator gets a test against a closed form or simulated ground truth
- Type hints encouraged but not required

## Related Repositories

- Parent: [RKHS-KRONIC](https://github.com/Ed-Mehrez/RKHS-KRONIC)
- Sibling: [pomdp-koopman-control](https://github.com/Ed-Mehrez/pomdp-koopman-control)
- Sibling: [rkhs-koopman-control](https://github.com/Ed-Mehrez/rkhs-koopman-control)
