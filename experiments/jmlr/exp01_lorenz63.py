import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from generator_estimator import GeneratorEstimator

def lorenz_drift(state, sigma=10.0, rho=28.0, beta=8.0/3.0):
    x, y, z = state
    return np.array([
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ])

def simulate_lorenz(N, dt, noise_strength=1.0):
    np.random.seed(42)
    X = np.zeros((N, 3))
    # Start near attractor
    X[0] = [1.0, 1.0, 1.0] # Transient?
    # Run burn-in
    burn_in = 1000
    noise_scale = np.sqrt(dt) * noise_strength
    
    current = X[0]
    for _ in range(burn_in):
        f = lorenz_drift(current)
        noise = np.random.randn(3) * noise_scale
        current = current + f * dt + noise
        
    X[0] = current
    
    for i in range(N-1):
        f = lorenz_drift(X[i])
        noise = np.random.randn(3) * noise_scale
        X[i+1] = X[i] + f * dt + noise
        
    return X, noise_strength # Return used params

def run_experiment():
    print("--- Experiment 1: Lorenz-63 (Standard SDE Universality) ---")
    
    # 1. Simulate
    dt = 0.005 # Compromise between 0.001 (good H) and 0.01 (good structure)
    N = 10000 
    noise_strength = 2.0 # Visible diffusion
    
    print(f"Simulating Lorenz-63 (dt={dt}, N={N}, noise={noise_strength})...")
    X, ns = simulate_lorenz(N, dt, noise_strength)
    
    # 2. Fit Estimator
    print("Fitting GeneratorEstimator...")
    # Use 'logsig' backend as standard
    est = GeneratorEstimator(dt=dt, backend='logsig',
                             h_method='variation', h_type='scalar',
                             whiten=True, rank=20)
    est.fit(X)
    
    print(f"Estimated H: {est.H_:.4f}")
    
    # 3. Visualize Drift Field (Z=27 plane)
    print("Visualizing drift field...")
    x = np.linspace(-20, 20, 20)
    y = np.linspace(-25, 25, 20)
    X_grid, Y_grid = np.meshgrid(x, y)
    Z_val = 27.0
    
    grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel(), np.full_like(X_grid.ravel(), Z_val)])
    
    # True drift
    true_drift = np.array([lorenz_drift(p) for p in grid_points])
    
    # Predicted drift
    pred_drift = est.predict_drift(grid_points)
    
    # Reshape for quiver
    U_true, V_true = true_drift[:, 0].reshape(X_grid.shape), true_drift[:, 1].reshape(X_grid.shape)
    U_pred, V_pred = pred_drift[:, 0].reshape(X_grid.shape), pred_drift[:, 1].reshape(X_grid.shape)
    
    # Plot
    fig = plt.figure(figsize=(15, 6))
    
    # Subplot 1: 3D Trajectory
    ax1 = fig.add_subplot(131, projection='3d')
    # Plot a subset for visibility
    ax1.plot(X[:2000, 0], X[:2000, 1], X[:2000, 2], lw=0.5, alpha=0.7, label='Data')
    ax1.set_title(f"Observed Path (H_est={est.H_:.2f})")
    ax1.legend()
    
    # Subplot 2: True Drift Slice
    ax2 = fig.add_subplot(132)
    ax2.quiver(X_grid, Y_grid, U_true, V_true, color='black', alpha=0.5)
    ax2.streamplot(x, y, U_true, V_true, color='blue', density=1.0, linewidth=0.5)
    ax2.set_title("True Drift (z=27)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    
    # Subplot 3: Predicted Drift Slice
    ax3 = fig.add_subplot(133)
    # ax3.quiver(X_grid, Y_grid, U_pred, V_pred, color='red', alpha=0.5)
    # Use streamplot for clearer structure comparison
    strm = ax3.streamplot(x, y, U_pred, V_pred, color='red', density=1.0, linewidth=0.5)
    ax3.set_title("Learned Drift (GeneratorEstimator)")
    ax3.set_xlabel("x")
    
    plt.tight_layout()
    plt.savefig('experiments/jmlr/exp01_lorenz_results.png', dpi=150)
    print("Saved plot to experiments/jmlr/exp01_lorenz_results.png")

if __name__ == "__main__":
    run_experiment()
