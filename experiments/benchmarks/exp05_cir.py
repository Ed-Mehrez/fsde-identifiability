import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from generator_estimator import GeneratorEstimator

def cir_drift(x, kappa=1.0, theta=1.0):
    return kappa * (theta - x)

def cir_diffusion(x, sigma=0.5):
    return sigma * np.sqrt(np.maximum(x, 0))

def simulate_cir(N, dt, kappa=1.0, theta=1.0, sigma=0.5):
    np.random.seed(42)
    X = np.zeros(N)
    X[0] = theta
    sq_dt = np.sqrt(dt)
    
    # Simple Euler-Maruyama with truncation
    for i in range(N-1):
        x_curr = max(X[i], 1e-6) # Ensure positive for sqrt
        drift = kappa * (theta - x_curr)
        diff = sigma * np.sqrt(x_curr)
        dx = drift * dt + diff * np.random.randn() * sq_dt
        X[i+1] = max(x_curr + dx, 1e-6)
        
    return X

def run_experiment():
    print("--- Experiment 5: CIR Process (Finance/State-Dependent Diffusion) ---")
    
    # 1. Simulate
    dt = 0.01
    N = 5000 
    kappa = 1.0
    theta = 1.0
    sigma = 0.5
    
    print(f"Simulating CIR (dt={dt}, N={N}, k={kappa}, th={theta}, sig={sigma})...")
    X = simulate_cir(N, dt, kappa, theta, sigma)
    
    # 2. Fit Estimator
    print("Fitting GeneratorEstimator...")
    est = GeneratorEstimator(dt=dt, backend='logsig',
                             h_method='variation', h_type='scalar',
                             sigma_method='both', # enable kernel sigma
                             whiten=True, rank=20, fixed_H=0.5) # Fix H=0.5
    # Norm?
    # CIR is positive. Standardizing shifts to mean 0.
    # The estimator handles standardization internally via input scaling usually?
    # But for explicit sigma check, let's feed raw X if possible or track scale.
    # The class standardizes internally? No, the code we viewed earlier takes trajectory as is.
    # Wait, `fit` stores `training_trajectory_`.
    # Whitening operator uses raw increments?
    # Let's standardize manually for stability, then map back.
    
    X_mean = np.mean(X)
    X_std = np.std(X)
    X_norm = (X - X_mean) / X_std
    
    est.fit(X_norm.reshape(-1, 1))
    
    print(f"Estimated H (should be ~0.5): {est.H_:.4f}")
    
    # 3. Analyze Drift and Diffusion
    print("Check Drift and Diffusion...")
    
    # Grid in natural units
    x_grid_raw = np.linspace(0.1, 2.0, 50)
    x_grid_norm = (x_grid_raw - X_mean) / X_std
    
    # Predict
    drift_pred_norm = est.predict_drift(x_grid_norm.reshape(-1, 1)).flatten()
    diffusion_pred_norm = est.predict_diffusion(x_grid_norm.reshape(-1, 1)).flatten()
    
    # Map back
    # dX = f(X) dt + g(X) dW
    # Y = (X - mu)/sig => X = sig Y + mu
    # dY = dX / sig = (f(sig Y + mu)/sig) dt + (g(sig Y + mu)/sig) dW
    # So f_Y(y) = f_X(x)/sig
    #    g_Y(y) = g_X(x)/sig
    
    drift_pred_raw = drift_pred_norm * X_std
    diffusion_pred_raw = diffusion_pred_norm * X_std
    
    # Compare
    drift_true = cir_drift(x_grid_raw, kappa, theta)
    diffusion_true = cir_diffusion(x_grid_raw, sigma)
    
    # Metrics
    metrics_mask = (x_grid_raw > 0.5) & (x_grid_raw < 1.5) # High density region
    mae_drift = np.mean(np.abs(drift_true[metrics_mask] - drift_pred_raw[metrics_mask]))
    mae_diff = np.mean(np.abs(diffusion_true[metrics_mask] - diffusion_pred_raw[metrics_mask]))
    
    # NRMSE / R2
    drift_range = np.max(drift_true[metrics_mask]) - np.min(drift_true[metrics_mask])
    rmse_drift = np.sqrt(np.mean((drift_true[metrics_mask] - drift_pred_raw[metrics_mask])**2))
    nrmse_drift = rmse_drift / drift_range
    r2_drift = 1 - np.sum((drift_true[metrics_mask] - drift_pred_raw[metrics_mask])**2) / np.sum((drift_true[metrics_mask] - np.mean(drift_true[metrics_mask]))**2)

    diff_range = np.max(diffusion_true[metrics_mask]) - np.min(diffusion_true[metrics_mask])
    rmse_diff = np.sqrt(np.mean((diffusion_true[metrics_mask] - diffusion_pred_raw[metrics_mask])**2))
    nrmse_diff = rmse_diff / diff_range
    
    print(f"Drift MAE: {mae_drift:.4f}")
    print(f"Drift NRMSE: {nrmse_drift:.4f}")
    print(f"Drift R2: {r2_drift:.4f}")
    print(f"Diffusion MAE: {mae_diff:.4f}")
    print(f"Diffusion NRMSE: {nrmse_diff:.4f}")
    
    # Plot
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Trajectory
    ax[0].plot(X[:1000])
    ax[0].set_title(f"CIR Path (H_est={est.H_:.2f})")
    
    # Drift
    ax[1].plot(x_grid_raw, drift_true, 'k--', label='True Drift')
    ax[1].plot(x_grid_raw, drift_pred_raw, 'r-', label='Learned Drift')
    ax[1].set_title(f"Drift Recovery (MAE={mae_drift:.3f})")
    ax[1].legend()
    
    # Diffusion
    ax[2].plot(x_grid_raw, diffusion_true, 'k--', label='True Diff $\sigma\sqrt{x}$')
    ax[2].plot(x_grid_raw, diffusion_pred_raw, 'b-', label='Learned Diff')
    ax[2].set_title(f"Diffusion Recovery (MAE={mae_diff:.3f})")
    ax[2].legend()
    
    plt.tight_layout()
    plt.savefig('experiments/benchmarks/exp05_cir_results.png', dpi=150)
    print("Saved plot to experiments/benchmarks/exp05_cir_results.png")

if __name__ == "__main__":
    run_experiment()
