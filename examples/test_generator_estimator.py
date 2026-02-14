"""
Validation test for GeneratorEstimator.

Tests RBF and Signature backends on fractional CIR benchmark.
Compares functional errors (mu(x), sigma(x)) against UniversalFractionalEstimator baseline.

Key regularization: Profile REML (not GCV) — jointly estimates noise variance
from data and uses log-determinant complexity penalty that doesn't degenerate
for large n. This replaced GCV which suffered from dt^2 scaling in the hat matrix.

Reports scorecard with drift MRE, drift correlation, sigma MRE, H accuracy.
"""

import numpy as np
import sys
import os
import time

sys.path.append(os.path.join(os.getcwd(), "src"))
from generator_estimator import GeneratorEstimator
from rough_paths_generator import FractionalBrownianMotion


def generate_fractional_cir(H, kappa, theta, sigma, n_steps, dt=0.01, seed=42):
    """Generate fractional CIR process: dv = kappa*(theta - v)dt + sigma*sqrt(v)*dB^H."""
    np.random.seed(seed)
    fbm = FractionalBrownianMotion(H=H, dt=dt, seed=seed)
    B = fbm.generate(n_steps + 1, n_paths=1)[0]
    dB = np.diff(B)

    v = np.zeros(n_steps)
    v[0] = theta
    for i in range(1, n_steps):
        drift = kappa * (theta - v[i - 1]) * dt
        noise = sigma * np.sqrt(max(1e-4, v[i - 1])) * dB[i - 1]
        v[i] = v[i - 1] + drift + noise
        v[i] = max(1e-4, v[i])
    return v


def compute_functional_errors(estimator, v, H_true, kappa_true, theta_true, sigma_true, dt):
    """
    Compute functional errors for mu(x) and sigma(x) over the empirical state space.
    Returns dict of errors - NO parametric extraction.
    """
    # Test on empirical state space (where data actually lives)
    x_grid = np.linspace(np.percentile(v, 5), np.percentile(v, 95), 100)

    # True drift: mu(x) = kappa * (theta - x)
    mu_true = kappa_true * (theta_true - x_grid)

    # True diffusion: sigma(x) = sigma * sqrt(x)
    sigma_true_vals = sigma_true * np.sqrt(np.maximum(1e-6, x_grid))

    # Predicted drift
    mu_pred = estimator.predict_drift(x_grid)

    # Predicted diffusion (both methods if available)
    sigma_pred_binned = estimator.predict_diffusion(x_grid, method='binned')

    errors = {
        'H_est': estimator.H,
        'H_error': abs(estimator.H - H_true),
        'H_error_pct': abs(estimator.H - H_true) / H_true * 100,
    }

    # Drift functional error (mean relative error, excluding near-zero drift).
    # Use 10% of max to avoid zero-crossing inflation of MRE.
    nonzero_mask = np.abs(mu_true) > 0.10 * np.max(np.abs(mu_true))
    if np.any(nonzero_mask):
        mu_mre = np.mean(np.abs(mu_pred[nonzero_mask] - mu_true[nonzero_mask])
                         / np.abs(mu_true[nonzero_mask]))
        errors['drift_mre'] = mu_mre * 100
    else:
        errors['drift_mre'] = np.nan

    # Also compute correlation (shape agreement)
    errors['drift_corr'] = np.corrcoef(mu_pred, mu_true)[0, 1]

    # Sigma functional error
    sigma_mre = np.mean(np.abs(sigma_pred_binned - sigma_true_vals) /
                        np.maximum(1e-6, sigma_true_vals))
    errors['sigma_binned_mre'] = sigma_mre * 100

    # Kernel sigma if available
    try:
        sigma_pred_kernel = estimator.predict_diffusion(x_grid, method='kernel')
        sigma_mre_k = np.mean(np.abs(sigma_pred_kernel - sigma_true_vals) /
                              np.maximum(1e-6, sigma_true_vals))
        errors['sigma_kernel_mre'] = sigma_mre_k * 100
    except RuntimeError:
        errors['sigma_kernel_mre'] = np.nan

    return errors, x_grid, mu_true, mu_pred, sigma_true_vals, sigma_pred_binned


def run_baseline_comparison(v, H_true, kappa_true, theta_true, sigma_true, dt):
    """Run UniversalFractionalEstimator for baseline comparison."""
    try:
        from universal_sde import UniversalFractionalEstimator
    except ImportError:
        print("  [SKIP] UniversalFractionalEstimator not available")
        return None

    est = UniversalFractionalEstimator(dt=dt)
    est.fit(v, n_iter=3)

    x_grid = np.linspace(np.percentile(v, 5), np.percentile(v, 95), 100)
    mu_true = kappa_true * (theta_true - x_grid)
    sigma_true_vals = sigma_true * np.sqrt(np.maximum(1e-6, x_grid))

    mu_pred = est.predict_drift(x_grid.reshape(-1, 1)).flatten()
    sigma_pred = est.predict_diffusion(x_grid.reshape(-1, 1)).flatten()

    nonzero_mask = np.abs(mu_true) > 0.01 * np.max(np.abs(mu_true))
    drift_mre = np.mean(np.abs(mu_pred[nonzero_mask] - mu_true[nonzero_mask])
                        / np.abs(mu_true[nonzero_mask])) * 100 if np.any(nonzero_mask) else np.nan
    sigma_mre = np.mean(np.abs(sigma_pred - sigma_true_vals)
                        / np.maximum(1e-6, sigma_true_vals)) * 100

    return {
        'H_est': est.H_,
        'H_error_pct': abs(est.H_ - H_true) / H_true * 100,
        'drift_mre': drift_mre,
        'sigma_mre': sigma_mre,
        'drift_corr': np.corrcoef(mu_pred, mu_true)[0, 1],
    }


def print_scorecard(results):
    """Print a comparison scorecard."""
    print("\n" + "=" * 90)
    print(f"{'Configuration':<30} {'H Error%':>10} {'Drift MRE%':>12} {'Drift Corr':>12} "
          f"{'Sigma Bin%':>12} {'Sigma Ker%':>12}")
    print("-" * 90)

    for name, errs in results.items():
        print(f"{name:<30} "
              f"{errs.get('H_error_pct', np.nan):>9.2f}% "
              f"{errs.get('drift_mre', np.nan):>11.2f}% "
              f"{errs.get('drift_corr', np.nan):>11.4f} "
              f"{errs.get('sigma_binned_mre', errs.get('sigma_mre', np.nan)):>11.2f}% "
              f"{errs.get('sigma_kernel_mre', np.nan):>11.2f}%")

    print("=" * 90)


def test_rbf_backends():
    """Test RBF backend with both H methods (fast test, no sig kernel dependency)."""
    print("\n" + "=" * 70)
    print("TEST: RBF Backend on Fractional CIR")
    print("=" * 70)

    H_true = 0.3
    sigma_true = 0.3
    kappa_true = 1.0
    theta_true = 0.5
    dt = 0.05
    N = 2000

    print(f"Parameters: H={H_true}, kappa={kappa_true}, theta={theta_true}, "
          f"sigma={sigma_true}, N={N}, dt={dt}")

    v = generate_fractional_cir(H_true, kappa_true, theta_true, sigma_true, N, dt)

    results = {}

    # --- RBF + Variation H (Profile REML) ---
    print("\n--- RBF + Variation H (Profile REML) ---")
    t0 = time.time()
    est = GeneratorEstimator(dt=dt, backend='rbf', h_method='variation',
                             sigma_method='both', reg_param='gcv')
    est.fit(v, n_iter=3)
    elapsed = time.time() - t0
    errs, *_ = compute_functional_errors(est, v, H_true, kappa_true, theta_true, sigma_true, dt)
    errs['time'] = f"{elapsed:.1f}s"
    results['RBF + REML'] = errs
    est.summary()

    # Print REML diagnostics
    print(f"\nREML diagnostics:")
    for k, val in est.fit_diagnostics_.items():
        if k.startswith('reml'):
            print(f"  {k}: {val}")

    # --- Baseline: UniversalFractionalEstimator ---
    print("\n--- Baseline: UniversalFractionalEstimator ---")
    baseline = run_baseline_comparison(v, H_true, kappa_true, theta_true, sigma_true, dt)
    if baseline:
        results['Baseline (Universal)'] = baseline

    print_scorecard(results)
    return results


def test_signature_backend():
    """Test signature backend (requires sigkernel + torch)."""
    print("\n" + "=" * 70)
    print("TEST: Signature Backend on Fractional CIR")
    print("=" * 70)

    try:
        import torch
        import sigkernel
    except ImportError:
        print("[SKIP] sigkernel or torch not available. Skipping signature backend test.")
        return None

    H_true = 0.3
    sigma_true = 0.3
    kappa_true = 1.0
    theta_true = 0.5
    dt = 0.05
    N = 2000

    print(f"Parameters: H={H_true}, kappa={kappa_true}, theta={theta_true}, "
          f"sigma={sigma_true}, N={N}, dt={dt}")

    v = generate_fractional_cir(H_true, kappa_true, theta_true, sigma_true, N, dt)

    results = {}

    # --- Signature + Variation H (Profile REML) ---
    print("\n--- Signature + Variation H (Profile REML) ---")
    t0 = time.time()
    est = GeneratorEstimator(dt=dt, backend='signature', h_method='variation',
                             sigma_method='both', rank=30, window_length=15,
                             n_landmarks=80, reg_param='gcv')
    est.fit(v, n_iter=3)
    elapsed = time.time() - t0
    errs, x_grid, mu_true, mu_pred, sigma_true_vals, sigma_pred = \
        compute_functional_errors(est, v, H_true, kappa_true, theta_true, sigma_true, dt)
    errs['time'] = f"{elapsed:.1f}s"
    results['Sig + REML'] = errs
    est.summary()

    print_scorecard(results)
    return results


def test_multi_H():
    """Test estimation accuracy across multiple Hurst values."""
    print("\n" + "=" * 70)
    print("TEST: Multi-H Accuracy (RBF Backend, Profile REML)")
    print("=" * 70)

    dt = 0.05
    N = 2000
    kappa, theta, sigma = 1.0, 0.5, 0.3

    print(f"\n{'H_true':>8} {'H_est':>8} {'H_err%':>8} {'Drift MRE%':>12} {'Corr':>8} {'Sigma MRE%':>12}")
    print("-" * 62)

    for H_true in [0.1, 0.2, 0.3, 0.5, 0.7]:
        v = generate_fractional_cir(H_true, kappa, theta, sigma, N, dt, seed=42)
        est = GeneratorEstimator(dt=dt, backend='rbf', h_method='variation',
                                 sigma_method='binned', reg_param='gcv')
        est.fit(v, n_iter=3)

        errs, *_ = compute_functional_errors(est, v, H_true, kappa, theta, sigma, dt)
        print(f"{H_true:>8.2f} {est.H:>8.4f} {errs['H_error_pct']:>7.2f}% "
              f"{errs['drift_mre']:>11.2f}% {errs['drift_corr']:>7.4f} "
              f"{errs['sigma_binned_mre']:>11.2f}%")


def test_n_scaling():
    """Test how drift accuracy scales with sample size."""
    print("\n" + "=" * 70)
    print("TEST: N-Scaling (RBF Backend, Profile REML)")
    print("=" * 70)

    H_true = 0.3
    kappa, theta, sigma = 1.0, 0.5, 0.3
    dt = 0.05

    print(f"\n{'N':>8} {'REML Lambda':>14} {'Drift MRE%':>12} {'Corr':>8} {'H_err%':>8}")
    print("-" * 54)

    for N in [1000, 2000, 5000]:
        v = generate_fractional_cir(H_true, kappa, theta, sigma, N, dt, seed=42)
        est = GeneratorEstimator(dt=dt, backend='rbf', h_method='variation',
                                 sigma_method='binned', reg_param='gcv')
        est.fit(v, n_iter=3)

        errs, *_ = compute_functional_errors(est, v, H_true, kappa, theta, sigma, dt)
        lam = est.fit_diagnostics_.get('reml_lambda', 'N/A')
        print(f"{N:>8} {lam:>14} {errs['drift_mre']:>11.2f}% "
              f"{errs['drift_corr']:>7.4f} {errs['H_error_pct']:>7.2f}%")


if __name__ == "__main__":
    np.random.seed(42)

    # Test RBF backend (always available)
    rbf_results = test_rbf_backends()

    # Test signature backend (requires sigkernel)
    sig_results = test_signature_backend()

    # Test accuracy across roughness spectrum
    test_multi_H()

    # Test convergence with sample size
    test_n_scaling()

    print("\n\nAll tests complete.")
