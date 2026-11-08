# Phase 0: rebuilt estimators

The `fsde/` package replaces the legacy `src/` code for estimating

```
dX_t = mu(X_t) dt + sigma(X_t) dB^H_t
```

from one or more paths sampled at step dt. It keeps only the four pieces of the
legacy code that an audit (November 2026) found to be correct, and fixes or
drops everything else. `src/` is kept for reference only; do not build on it.

## What changed and why

| Legacy claim or component | Audit finding | Phase 0 replacement |
| --- | --- | --- |
| (mu, sigma, H) "uniquely identifiable" via Stroock-Varadhan | X is not Markov for H != 1/2, so the argument does not apply. Drift laws are mutually absolutely continuous on a finite horizon | H and sigma(x) from fine sampling; mu(x) only as T -> infinity under ergodicity |
| Koopman eigenvalue fixes the drift amplitude | Not a semigroup for H != 1/2; on fOU (kappa = 1) it gives kappa_hat = 4.22 at H = 0.3 and 0.22 at H = 0.7 | Dropped |
| "Eigenvalue amplitude constraint" | Only in the docs, never implemented | Dropped |
| fBm generator (`rough_paths_generator.py`) | Exact for H <= ~0.8; middle circulant entry set to 0, so not exact near H = 1 | `fsde.sample_fgn`: exact for all H in (0, 1), raises if the embedding is not PSD |
| Left-point Euler with state-dependent sigma | For H < 1/2 it converges to neither the rough nor the Skorohod solution | `fsde.simulate_fsde`: Euler on the Lamperti process (additive noise), mapped back. Gives the pathwise solution (Stratonovich at H = 1/2) |
| H from first-difference variogram, "2-3% error" | Biased by drift: about -0.08 at H = 0.7 on fOU | `fsde.hurst_second_diff`: second differences, lags (1, 2, 4), unbiased to < 0.01 |
| Kernel sigma averaging abs(second differences) | Biased low by E abs(Z) ~ 0.80 | `fsde.sigma_kernel`: squared second differences, constant 4 - 2^{2H}, plus a localisation debias (below) |
| "Binned" sigma | Returned all ones | Dropped |
| Raw E[dX given X]/dt drift | OLS slope -6.9 at H = 0.2, -0.24 at H = 0.7 (truth -1) | Kept only as the `whiten=False` baseline |
| fGn-whitened regression | Correct | `fsde.estimate_drift_lamperti`: GLS under fGn(H), B-spline or Legendre sieve, standard errors |
| "Profile REML" | Used only by the signature backends; RBF and Koopman backends hard-coded the ridge | Dropped for now. Sieve dimension is a user choice until a selection rule is added in Phase 2 |
| Benchmarks exp03-exp06 | Simulated H = 0.5 Brownian motion | `examples/phase0_grid.py`: fractional data at H = 0.2-0.8 |

## The estimator, in order

1. **H.** Change-of-frequency estimator on second differences
   D^(k)_i = X_{i+2k} - 2 X_{i+k} + X_i: regress log mean(D^2) on log k for
   k = 1, 2, 4 and halve the slope. Second differences cancel locally linear
   drift, and the sigma factor cancels in the slope. `hurst_bootstrap` gives a
   parametric-bootstrap confidence interval.
2. **sigma(x).** Nadaraya-Watson smoothing of D_i^2 against X_{i+1}, divided by
   dt^{2H} (4 - 2^{2H}). Found during the rebuild: when dt^H is comparable to
   the spread of X (small H), the conditioning point X_{i+1} correlates with D,
   and the plain estimator becomes U-shaped. It gives 0.64 to 0.81 for a constant
   sigma = 0.7 on fOU at H = 0.2, dt = 0.02. The default `debias=True` uses
   E[D^2 | x] - E[D | x]^2 + E_X[E[D | X]^2], exact for a stationary Gaussian
   process. On that case it cuts the error from 7.7% to 1.1%. It is
   approximate for non-Gaussian X: on the bimodal double well at H = 0.2 the
   error goes from 3.7% to 4.4%.
3. **Lamperti transform.** Y = F(X) with F' = 1 / sigma_hat, computed
   numerically. Y has unit additive fBm noise and drift b(y) = mu / sigma at
   x = F^{-1}(y).
4. **Drift.** Generalised least squares of the increments of Y on a sieve basis
   under the fGn(H_hat) covariance, then mu_hat(x) = sigma_hat(x) b_hat(F(x)).
   The Toeplitz solves use preconditioned CG with FFT mat-vecs (T. Chan
   circulant preconditioner), O(n log n) per iteration. GLS equals Cholesky
   whitening, a causal filter; after whitening the regressors are predetermined,
   which is why GLS is unbiased where OLS is not.

## Results

The test suite (`python -m pytest`, about 12 s) checks every component
against closed forms or ground truth:

- fGn autocovariance;
- the fOU stationary variance sigma^2 Gamma(2H+1) / (2 kappa^{2H});
- the second-difference constant 4 - 2^{2H};
- Toeplitz PCG against Levinson;
- the numerical Lamperti map against the analytic one;
- estimator accuracy on fBm, fOU and a state-dependent model.

Accuracy grid, everything estimated: see [`results/phase0_grid.md`](../results/phase0_grid.md),
produced by `python examples/phase0_grid.py`.

Drift on fOU (kappa = 1, dt = 0.05, T = 200, linear basis, 30 replicates;
truth -1):

| H | GLS, true H | GLS, estimated H | OLS (legacy) |
| --- | --- | --- | --- |
| 0.2 | -0.99 ± 0.11 | -0.96 ± 0.13 | -6.91 ± 0.35 |
| 0.3 | -0.98 ± 0.09 | -0.98 ± 0.12 | -3.69 ± 0.24 |
| 0.5 | -1.02 ± 0.11 | -0.99 ± 0.18 | -1.02 ± 0.11 |
| 0.7 | -1.03 ± 0.10 | -0.98 ± 0.12 | -0.24 ± 0.03 |

## Known limitations

These are inputs to Paper A, not bugs to hide.

- **sigma inherits the H error, amplified by log(1/dt).** An error eps in H
  scales sigma_hat by dt^{-eps}. On fOU at dt = 0.05 and T = 200, the spread of
  sigma_hat is 4-6% with estimated H against 1-1.5% with true H. This is intrinsic:
  sigma is defined at time scale 1, but the data only show scale dt.
- **Drift needs long horizons.** With everything estimated, mu error is about 10-16%
  (state-dependent model) and 25% (double well) at T = 1000. Expect roughly
  1/sqrt(T) improvement.
- **Pointwise bands for mu** condition on H_hat and sigma_hat and ignore sieve
  bias. They cover at about 95% when H and sigma are known; coverage with
  everything estimated is reported in the grid.
- **No sieve-dimension selection yet.** n_basis is fixed by the user.
- **1-D only.** The Lamperti transform has no general multivariate analogue.
- **sigma must be bounded away from 0** on the observed range (fCIR-type
  boundaries are not handled).
- **Sampling regime.** The simulation uses Euler on a grid with 4 substeps per
  observation, so the data contain O(dt/4) discretisation error.
