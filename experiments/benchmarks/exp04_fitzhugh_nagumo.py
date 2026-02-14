import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from generator_estimator import GeneratorEstimator

def fhn_drift(state, I=0.5, epsilon=0.08, a=0.7, b=0.8):
    v, w = state
    dv = v - v**3/3 - w + I
    dw = epsilon * (v + a - b*w)
    return np.array([dv, dw])

def simulate_fhn(N, dt, sig_v=0.5, sig_w=0.5):
    np.random.seed(42)
    X = np.zeros((N, 2))
    X[0] = [0.0, 0.0]
    
    for i in range(N-1):
        drift = fhn_drift(X[i])
        noise = np.array([np.random.randn()*sig_v, np.random.randn()*sig_w]) * np.sqrt(dt)
        X[i+1] = X[i] + drift * dt + noise
        
    return X

def run_experiment():
    print("--- Experiment 4: FitzHugh-Nagumo (Neuroscience/Multiscale) ---")
    
    # 1. Simulate
    dt = 0.05 # Needs slightly larger dt for slow variable to move? Or smaller for stability?
    # Fast var v changes O(1), slow w changes O(eps)=0.08.
    # dt=0.05 is standard for FHN.
    N = 5000 
    sig_v, sig_w = 0.5, 0.1 # More noise on voltage usually
    
    print(f"Simulating FHN (dt={dt}, N={N})...")
    X = simulate_fhn(N, dt, sig_v, sig_w)
    
    # 2. Fit Estimator
    print("Fitting GeneratorEstimator (vector H)...")
    est = GeneratorEstimator(dt=dt, backend='logsig',
                             h_method='variation', h_type='vector',
                             whiten=True, rank=20, fixed_H=np.array([0.5, 0.5])) # Fix H=[0.5, 0.5]
    # Norm?
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_norm = (X - X_mean) / X_std
    
    est.fit(X_norm)
    
    print(f"Estimated H (should be ~0.5): {est.H_}")
    
    # 3. Drift Analysis
    print("Analyzing drift field...")
    
    # Grid
    v_grid = np.linspace(-3, 3, 20)
    w_grid = np.linspace(-3, 3, 20)
    V, W = np.meshgrid(v_grid, w_grid)
    points = np.column_stack([V.ravel(), W.ravel()])
    
    # True Drift (unnormalized scale)
    drift_true = np.array([fhn_drift(p) for p in points])
    
    # Predicted Drift (normalized scale)
    # We need to map inputs to normalized space and outputs back to raw space
    points_norm = (points - X_mean) / X_std
    drift_pred_norm = est.predict_drift(points_norm)
    
    # Un-normalize drift prediction:
    # dX = f(X) dt + ...
    # dX_norm = dX / std = f(X)/std dt + ...
    # So f_pred_norm ~ f_true / std
    # f_pred_raw = f_pred_norm * std
    drift_pred_raw = drift_pred_norm * X_std
    
    # Error Metrics
    # Only in visited region
    hull_min = np.min(X, axis=0)
    hull_max = np.max(X, axis=0)
    mask = (points[:,0] > hull_min[0]) & (points[:,0] < hull_max[0]) & \
           (points[:,1] > hull_min[1]) & (points[:,1] < hull_max[1])
           
    mae = np.mean(np.abs(drift_true[mask] - drift_pred_raw[mask]))
    rmse = np.sqrt(np.mean((drift_true[mask] - drift_pred_raw[mask])**2))
    
    # Range for NRMSE
    v_range = np.max(drift_true[mask]) - np.min(drift_true[mask])
    nrmse = rmse / v_range
    r2 = 1 - np.sum((drift_true[mask] - drift_pred_raw[mask])**2) / np.sum((drift_true[mask] - np.mean(drift_true[mask]))**2)
    
    print(f"Drift MAE (in hull): {mae:.4f}")
    print(f"Drift RMSE: {rmse:.4f}")
    print(f"Drift NRMSE: {nrmse:.4f}")
    print(f"Drift R2: {r2:.4f}")
    
    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Trajectory
    ax[0].plot(X[:2000, 0], X[:2000, 1], alpha=0.5, lw=0.5, color='gray')
    ax[0].set_title(f"FHN Phase Space (H_est={est.H_})")
    ax[0].set_xlabel("v (Voltage)")
    ax[0].set_ylabel("w (Recovery)")
    
    # Nullclines
    v_nc = np.linspace(-3, 3, 100)
    w_nc1 = v_nc - v_nc**3/3 + 0.5 # v-nullcline
    w_nc2 = (v_nc + 0.7)/0.8       # w-nullcline
    ax[0].plot(v_nc, w_nc1, 'b--', lw=1, label='dv/dt=0')
    ax[0].plot(v_nc, w_nc2, 'r--', lw=1, label='dw/dt=0')
    ax[0].legend()
    
    # Drift Field
    # Subsample for quiver
    skip = 2
    ax[1].quiver(V[::skip, ::skip], W[::skip, ::skip], 
                 drift_true.reshape(20, 20, 2)[::skip, ::skip, 0], 
                 drift_true.reshape(20, 20, 2)[::skip, ::skip, 1],
                 color='black', alpha=0.3, label='True')
                 
    ax[1].quiver(V[::skip, ::skip], W[::skip, ::skip], 
                 drift_pred_raw.reshape(20, 20, 2)[::skip, ::skip, 0], 
                 drift_pred_raw.reshape(20, 20, 2)[::skip, ::skip, 1],
                 color='red', alpha=0.6, label='Learned')
    
    ax[1].set_title(f"Drift Field (MAE={mae:.3f})")
    ax[1].set_xlabel("v")
    ax[1].legend()
    
    plt.tight_layout()
    plt.savefig('experiments/benchmarks/exp04_fhn_results.png', dpi=150)
    print("Saved plot to experiments/benchmarks/exp04_fhn_results.png")

if __name__ == "__main__":
    run_experiment()
