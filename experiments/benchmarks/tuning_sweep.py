import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from generator_estimator import GeneratorEstimator

def double_well_drift(x):
    return 4 * (x - x**3)

def simulate_double_well(N, dt, sigma=1.0):
    np.random.seed(42)
    X = np.zeros(N)
    X[0] = 0.0
    for i in range(N-1):
        f = double_well_drift(X[i])
        noise = np.random.randn() * np.sqrt(dt) * sigma
        X[i+1] = X[i] + f * dt + noise
    return X

def run_tuning_sweep():
    print("--- Model Optimization: Tuning Sweep (Double-Well) ---")
    
    # 1. Simulate Data (Fixed)
    dt = 0.01
    N = 10000 
    sigma = 1.0
    print(f"Simulating Data (N={N}, dt={dt})...")
    X = simulate_double_well(N, dt, sigma)
    X = X.reshape(-1, 1)
    
    # Grid
    ranks = [20, 50, 100]
    iters = [3, 10, 20]
    
    results = []
    
    # Evaluation Grid
    x_eval = np.linspace(-2.0, 2.0, 100).reshape(-1, 1)
    drift_true = double_well_drift(x_eval).flatten()
    mask = (x_eval.flatten() > -1.5) & (x_eval.flatten() < 1.5)
    drift_range = np.max(drift_true[mask]) - np.min(drift_true[mask])
    
    print(f"{'Rank':<5} | {'Iter':<5} | {'NRMSE':<8} | {'Time (s)':<8}")
    print("-" * 35)
    
    best_nrmse = float('inf')
    best_params = None
    
    for r in ranks:
        for n_it in iters:
            start_time = time.time()
            
            # Fit
            est = GeneratorEstimator(dt=dt, backend='logsig',
                                     h_method='variation', h_type='scalar',
                                     whiten=True, rank=r, fixed_H=0.5)
            est.fit(X, n_iter=n_it)
            
            # Predict
            drift_pred = est.predict_drift(x_eval).flatten()
            
            # Metric
            rmse = np.sqrt(np.mean((drift_true[mask] - drift_pred[mask])**2))
            nrmse = rmse / drift_range
            
            elapsed = time.time() - start_time
            
            print(f"{r:<5} | {n_it:<5} | {nrmse:.4f}   | {elapsed:.2f}")
            results.append((r, n_it, nrmse, elapsed))
            
            if nrmse < best_nrmse:
                best_nrmse = nrmse
                best_params = (r, n_it)
                
    print("-" * 35)
    print(f"Best NRMSE: {best_nrmse:.4f} with Rank={best_params[0]}, Iter={best_params[1]}")
    
    # Plot Heatmap if possible, or just bar chart
    # Just save the best result plot
    print("Running best config for final plot...")
    r_opt, n_opt = best_params
    est = GeneratorEstimator(dt=dt, backend='logsig',
                             h_method='variation', h_type='scalar',
                             whiten=True, rank=r_opt, fixed_H=0.5)
    est.fit(X, n_iter=n_opt)
    drift_pred = est.predict_drift(x_eval).flatten()
    
    plt.figure(figsize=(8, 5))
    plt.plot(x_eval, drift_true, 'k--', label='True Drift')
    plt.plot(x_eval, drift_pred, 'r-', label=f'Best Fit (R={r_opt}, It={n_opt})')
    plt.title(f"Optimized Double-Well Reconstruction\nNRMSE={best_nrmse:.4f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('experiments/benchmarks/tuning_sweep_results.png', dpi=150)
    print("Saved plot to experiments/benchmarks/tuning_sweep_results.png")

if __name__ == "__main__":
    run_tuning_sweep()
