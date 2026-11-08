# Fractional SDE Identifiability

Nonparametric estimation of the Hurst exponent H, diffusion σ(x) and drift μ(x) of

$$dX_t = \mu(X_t)\,dt + \sigma(X_t)\,dB^H_t$$

from discretely sampled paths.

> **Status (November 2026).** The `fsde/` package is an audited rebuild (Phase 0).
>
> The legacy `src/` code and the earlier claims in this README did not hold up:
> - Stroock–Varadhan "unique identifiability" does not apply to this non-Markov process.
> - The Koopman eigenvalue "amplitude constraint" was never implemented and is invalid for H ≠ 1/2.
> - σ error was 22–40%, not under 3%.
> - The benchmarks simulated H = 0.5.
>
> See [`docs/phase0.md`](docs/phase0.md) for the full list. `src/` is kept for reference only.

## What is identifiable, and how

| Quantity | Identified from | Estimator in `fsde/` |
| --- | --- | --- |
| H | fine sampling (dt → 0) | change-of-frequency estimator on second differences (lags 1, 2, 4); drift-robust; bootstrap CI |
| σ(x) | fine sampling | kernel-smoothed squared second differences ÷ dt^{2H}(4 − 2^{2H}), with a localisation debias |
| μ(x) | long horizon (T → ∞), ergodicity | fGn(H)-whitened generalised least squares on a B-spline sieve, after the Lamperti transform |

On a fixed time horizon the drift is not consistently estimable, because laws with different drifts are equivalent.

**Why whitening matters.** For H ≠ 1/2, plain regression of increments on the state is badly biased. On fOU with true slope −1 it gives −6.9 at H = 0.2 and −0.24 at H = 0.7. Whitened GLS gives −0.96 to −1.03 at every H. Accuracy with everything estimated is in [`results/phase0_grid.md`](results/phase0_grid.md).

## Quick start

```bash
pip install -e ".[test]"
python -m pytest                       # ~15 s
python examples/phase0_grid.py         # accuracy grid, ~10 min
```

```python
import numpy as np
import fsde

model = fsde.hyperbolic(kappa=1.0, s=0.5)          # dX = -X dt + 0.5 sqrt(1+X^2) dB^H
x = fsde.simulate_fsde(model, H=0.3, dt=0.02, n=50_000, burn_in=200, rng=0)

fit = fsde.fit_fsde(x, dt=0.02)                     # estimates H, sigma(x), mu(x)
grid = np.linspace(-0.5, 0.5, 5)                    # stay in the bulk of the data;
                                                    # estimates degrade in the tails
print(fit.H)                                        # ~0.30
print(fit.sigma(grid), model.sigma(grid))
print(fit.mu(grid), fit.mu_se(grid), model.mu(grid))

ci = fsde.hurst_bootstrap(x, n_boot=200)            # CI for H
```

## Project structure

```
fsde-identifiability/
├── fsde/                 # audited estimators (use this)
│   ├── fgn.py            # fGn sampler (exact), Toeplitz solves
│   ├── simulate.py       # fBm, Lamperti-based fSDE simulation, models
│   ├── hurst.py          # H estimation + bootstrap CI
│   ├── diffusion.py      # sigma(x)
│   ├── drift.py          # whitened sieve GLS for the drift
│   ├── fit.py            # end-to-end fit_fsde
│   └── tests/
├── examples/phase0_grid.py
├── results/phase0_grid.md
├── docs/phase0.md        # audit findings, method, limitations
├── src/                  # LEGACY: kept for reference, do not extend
├── docs/mllm/, papers/   # pre-audit notes; claims unverified
└── pyproject.toml
```

## Next (Phase 1 onward)

- Rates for (Ĥ, σ̂, μ̂) with plug-in error, including the log(1/dt) amplification of H error in σ̂.
- A data-driven choice of sieve dimension.
- Bands for μ that account for the uncertainty in Ĥ and σ̂.
- Use the estimators as the backbone of a long-memory-fidelity benchmark for generative models.

## License

MIT License

---
*Split from [RKHS-KRONIC](https://github.com/Ed-Mehrez/RKHS-KRONIC) repository.*
