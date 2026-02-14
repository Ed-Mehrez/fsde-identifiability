import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from generator_estimator import GeneratorEstimator

def double_well_drift(x):
    return 4 * (x - x**3)

def simulate_double_well(N, dt, sigma=1.0):
    np.random.seed(42)
    X = np.zeros(N)
    X[0] = 0.1
    for i in range(N-1):
        f = double_well_drift(X[i])
        noise = np.random.randn() * np.sqrt(dt) * sigma
        X[i+1] = X[i] + f * dt + noise
    return X

def run_debug():
    print("--- Debug Double-Well ---")
    
    dt = 0.01
    N = 5000 # Reduced to avoid OOM
    sigma = 1.0 # High noise to explore
    
    print(f"Simulating N={N}...")
    X = simulate_double_well(N, dt, sigma)
    X = X.reshape(-1, 1)
    
    # Grid
    x_eval = np.linspace(-2.5, 2.5, 100).reshape(-1, 1)
    drift_true = double_well_drift(x_eval).flatten()
    
    # 1. Baseline
    print("Fitting Baseline (GCV)...")
    est_gcv = GeneratorEstimator(dt=dt, backend='logsig',
                                 h_method='variation', h_type='scalar',
                                 whiten=True, rank=50, fixed_H=0.5,
                                 reg_param='gcv')
    est_gcv.fit(X, n_iter=5)
    drift_gcv = est_gcv.predict_drift(x_eval).flatten()
    
    # 2. Low Regularization
    print("Fitting Low Reg (1e-5)...")
    est_low = GeneratorEstimator(dt=dt, backend='logsig',
                                 h_method='variation', h_type='scalar',
                                 whiten=True, rank=50, fixed_H=0.5,
                                 reg_param=1e-5)
    est_low.fit(X, n_iter=5)
    drift_low = est_low.predict_drift(x_eval).flatten()
    
    # 3. Polynomial Backend? (Not available, but let's emulate by manual feature map if needed)
    # For now check if low reg helps.
    
    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Density
    ax[0].hist(X.flatten(), bins=50, density=True, alpha=0.5, color='gray', label='Data Density')
    ax[0].set_title("Data Distribution")
    ax[0].set_xlim([-2.5, 2.5])
    
    # Drift
    ax[1].plot(x_eval, drift_true, 'k--', label='True', lw=2)
    ax[1].plot(x_eval, drift_gcv, 'b-', label='GCV', alpha=0.8)
    ax[1].plot(x_eval, drift_low, 'r-', label='Reg=1e-5', alpha=0.8)
    
    # Error
    mask = (x_eval.flatten() > -1.5) & (x_eval.flatten() < 1.5)
    mae_gcv = np.mean(np.abs(drift_true[mask] - drift_gcv[mask]))
    mae_low = np.mean(np.abs(drift_true[mask] - drift_low[mask]))
    
    ax[1].set_title(f"Drift Recovery\nMAE(GCV)={mae_gcv:.3f}, MAE(Low)={mae_low:.3f}")
    ax[1].set_ylim([-25, 25])
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiments/benchmarks/debug_double_well.png', dpi=150)
    print("Saved to experiments/benchmarks/debug_double_well.png")

if __name__ == "__main__":
    run_debug()
