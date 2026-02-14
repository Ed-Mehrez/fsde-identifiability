import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from generator_estimator import GeneratorEstimator

def generate_fBm(N, dt, H):
    """Simple Cholesky-based fractional Brownian motion generator (1D)."""
    t = np.arange(N)
    # Autocovariance of fGn
    # gamma(k) = 0.5 * (|k+1|^2H + |k-1|^2H - 2|k|^2H) * dt^2H
    # We want standard B_H, so scale is 1.
    k = np.arange(N)
    gamma = 0.5 * (np.abs(k + 1)**(2*H) + np.abs(k - 1)**(2*H) - 2*np.abs(k)**(2*H)) * dt**(2*H)
    
    # Construct covariance matrix
    # Toeplitz
    R = np.zeros((N, N))
    for i in range(N):
        R[i, i:] = gamma[:N-i]
        R[i:, i] = gamma[:N-i]
        
    L = np.linalg.cholesky(R + 1e-9 * np.eye(N))
    noise = L @ np.random.randn(N)
    return np.cumsum(noise)

def run_experiment():
    print("--- Experiment 2: Coupled fOU (Anisotropic H) ---")
    
    # Parameters
    np.random.seed(1337)
    dt = 0.01
    N = 2000 # Reduced from 4000 to speed up Cholesky
    
    H_true = np.array([0.7, 0.4])
    dim = 2
    
    print(f"Generating 2D Coupled fOU (H={H_true}, N={N}, dt={dt})...")
    
    # 1. Generate Noise
    B = np.zeros((N, dim))
    for d in range(dim):
        B[:, d] = generate_fBm(N, dt, H_true[d])
        
    dB = np.diff(B, axis=0) # (N-1, D)
    
    # 2. Simulate fOU
    # dX = A X dt + dB
    # A = [[-1, 0.5], [-0.5, -1]]
    A_true = np.array([[-1.0, 0.5], [-0.5, -1.0]])
    
    X = np.zeros((N, dim))
    X[0] = np.random.randn(dim)
    
    for i in range(N-1):
        drift = A_true @ X[i]
        # Noise is dB[i]
        X[i+1] = X[i] + drift * dt + dB[i]
        
    # Standardize for estimation (helper usually expects unit scale)
    # But let's keep raw to check physical unit recovery if possible?
    # GeneratorEstimator assumes standardization helps with RBF/Sig bandwidths.
    # Let's standardize manually and track scale.
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_norm = (X - X_mean) / X_std
    
    # 3. Fit Estimator
    print("Fitting GeneratorEstimator (vector H)...")
    est = GeneratorEstimator(dt=dt, backend='logsig',
                             h_method='variation', h_type='vector',
                             whiten=True, rank=20)
    est.fit(X_norm)
    
    print(f"True H:      {H_true}")
    print(f"Estimated H: {est.H_}")
    
    # 4. Drift Analysis
    # True drift of X_norm? 
    # dX_norm = dX / X_std = (A X dt + dB) / X_std
    # X = X_std * X_norm + X_mean
    # dX_norm = A (X_std X_norm) / X_std dt + ...
    # Effective A_norm = diag(1/std) @ A @ diag(std)
    
    A_eff = np.diag(1/X_std) @ A_true @ np.diag(X_std)
    
    print("True Effective A (for normalized data):")
    print(A_eff)
    
    # Predict at grid
    grid_lim = 2.0
    g = np.linspace(-grid_lim, grid_lim, 15)
    G1, G2 = np.meshgrid(g, g)
    points = np.column_stack([G1.ravel(), G2.ravel()])
    
    drift_pred = est.predict_drift(points) # (N_grid, 2)
    drift_true = (points @ A_eff.T) # (N_grid, 2)
    
    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Trajectory
    ax[0].plot(X_norm[:1000, 0], X_norm[:1000, 1], alpha=0.6, lw=0.8)
    ax[0].set_title(f"Trajectory (H_est={est.H_})")
    ax[0].set_aspect('equal')
    
    # Vector Field Comparison
    # True (Black), Learned (Red)
    ax[1].quiver(points[:,0], points[:,1], drift_true[:,0], drift_true[:,1], 
                 color='black', alpha=0.3, label='True Linear')
    ax[1].quiver(points[:,0], points[:,1], drift_pred[:,0], drift_pred[:,1], 
                 color='red', alpha=0.6, label='Learned')
    ax[1].legend()
    ax[1].set_title("Drift Field Comparison")
    
    plt.tight_layout()
    plt.savefig('experiments/jmlr/exp02_coupled_fou_results.png', dpi=150)
    print("Saved plot to experiments/jmlr/exp02_coupled_fou_results.png")

if __name__ == "__main__":
    run_experiment()
