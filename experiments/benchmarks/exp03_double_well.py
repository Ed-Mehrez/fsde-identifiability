import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from generator_estimator import GeneratorEstimator

def double_well_drift(x):
    # V(x) = (x^2 - 1)^2
    # f(x) = -V'(x) = -2(x^2 - 1)(2x) = -4(x^3 - x) = 4(x - x^3)
    return 4 * (x - x**3)

def double_well_potential(x):
    return (x**2 - 1)**2

def simulate_double_well(N, dt, sigma=1.0):
    np.random.seed(42)
    X = np.zeros(N)
    X[0] = 0.0 # Start at unstable point
    
    for i in range(N-1):
        f = double_well_drift(X[i])
        noise = np.random.randn() * np.sqrt(dt) * sigma
        X[i+1] = X[i] + f * dt + noise
        
    return X

def run_experiment():
    print("--- Experiment 3: Double-Well Potential (Physics/Bistability) ---")
    
    # 1. Simulate
    dt = 0.01
    N = 10000 
    sigma = 1.0 # Sufficient noise for transitions
    
    print(f"Simulating Double-Well (dt={dt}, N={N}, sigma={sigma})...")
    X = simulate_double_well(N, dt, sigma)
    
    # 2. Fit Estimator
    print("Fitting GeneratorEstimator...")
    # Standard SDE methods usually suffice, but let's test our estimator
    est = GeneratorEstimator(dt=dt, backend='logsig',
                             h_method='variation', h_type='scalar',
                             whiten=True, rank=20, fixed_H=0.5) # Fix H=0.5 for standard SDE
    est.fit(X.reshape(-1, 1))
    
    print(f"Estimated H: {est.H_:.4f}")
    
    # 3. Drift Analysis
    print("Analyzing drift reconstruction...")
    x_grid = np.linspace(-2.0, 2.0, 100).reshape(-1, 1)
    
    # True drift
    drift_true = double_well_drift(x_grid).flatten()
    
    # Predicted drift
    drift_pred = est.predict_drift(x_grid).flatten()
    
    # Error metrics
    # Restrict evaluation to populated region approx [-1.5, 1.5]
    mask = (x_grid.flatten() > -1.5) & (x_grid.flatten() < 1.5)
    mae = np.mean(np.abs(drift_true[mask] - drift_pred[mask]))
    rmse = np.sqrt(np.mean((drift_true[mask] - drift_pred[mask])**2))
    nrmse = rmse / (np.max(drift_true[mask]) - np.min(drift_true[mask]))
    r2 = 1 - np.sum((drift_true[mask] - drift_pred[mask])**2) / np.sum((drift_true[mask] - np.mean(drift_true[mask]))**2)
    
    print(f"Drift MAE (in [-1.5, 1.5]): {mae:.4f}")
    print(f"Drift RMSE: {rmse:.4f}")
    print(f"Drift NRMSE: {nrmse:.4f}")
    print(f"Drift R2: {r2:.4f}")
    
    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Trajectory
    ax[0].plot(np.arange(N)*dt, X, alpha=0.7, lw=0.5)
    ax[0].set_title(f"Double-Well Trajectory (H_est={est.H_:.2f})")
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Position x")
    
    # Drift Comparison
    ax[1].plot(x_grid, drift_true, 'k--', label='True Drift $4(x-x^3)$', lw=2)
    ax[1].plot(x_grid, drift_pred, 'r-', label='Estimated Drift', lw=2, alpha=0.8)
    
    # Show density
    ax_hist = ax[1].twinx()
    ax_hist.hist(X, bins=50, density=True, alpha=0.2, color='gray', label='Empirical Density')
    ax_hist.set_yticks([])
    
    ax[1].set_title(f"Drift Reconstruction (MAE={mae:.3f})")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("Drift f(x)")
    ax[1].legend(loc='upper right')
    ax[1].set_ylim([-5, 5])
    ax[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiments/benchmarks/exp03_double_well_results.png', dpi=150)
    print("Saved plot to experiments/benchmarks/exp03_double_well_results.png")

if __name__ == "__main__":
    run_experiment()
