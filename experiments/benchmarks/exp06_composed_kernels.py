import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from generator_estimator import GeneratorEstimator

def generate_double_well(n_steps=5000, dt=0.01, sigma=1.0):
    """Generate Double-Well potential data: dX = 4(x - x^3)dt + sigma*dW"""
    np.random.seed(42)
    X = np.zeros(n_steps + 1)
    X[0] = 0.5 # Start near unstable equilibrium
    for i in range(n_steps):
        drift = 4 * (X[i] - X[i]**3)
        X[i+1] = X[i] + drift * dt + sigma * np.sqrt(dt) * np.random.randn()
    return X

def compute_logsig_features_point(X, degree=2):
    """
    Compute 'LogSig' features for a point X (proxy for Polynomial features).
    For scalars X: [X, X^2, ..., X^degree]
    """
    # X shape: (N,)
    features = []
    for d in range(1, degree + 1):
        features.append(X**d)
    return np.column_stack(features)

def linear_logsig_fit(X, dt):
    """Fit Linear model on LogSig features (Linear Regression)."""
    # Y = (X_{t+1} - X_t)/dt
    # features = LogSig(X_t)
    dX = np.diff(X)
    Y = dX / dt
    
    # Features: [x, x^2]
    Phi = compute_logsig_features_point(X[:-1], degree=2)
    
    # Solve Phi * beta = Y
    # Ridge for stability
    lambda_reg = 1e-3
    beta = np.linalg.solve(Phi.T @ Phi + lambda_reg * np.eye(Phi.shape[1]), Phi.T @ Y)
    
    return beta

def predict_linear_logsig(x_test, beta):
    Phi_test = compute_logsig_features_point(x_test, degree=2)
    return Phi_test @ beta

# --- RBF on LogSig ---

class RBFLogSigWrapper:
    def __init__(self, degree=2, n_landmarks=200):
        self.degree = degree
        self.n_landmarks = n_landmarks
        self.est = None
        
    def fit(self, X_traj, dt):
        self.dt = dt
        
        # 1. Lift data to Feature Space Z = [x, x^2]
        self.Z = compute_logsig_features_point(X_traj, self.degree)
        
        # 2. Use GeneratorEstimator with RBF backend, but passing Z as 'trajectory'
        #    Wait, GeneratorEstimator assumes input is state X for drift f(X).
        #    If we pass Z, it learns f(Z). But Z is deterministic function of X.
        #    So f(Z) -> R^dim(Z). We want drift in X-space (1D).
        #    So we can't simply pass Z to GeneratorEstimator as is, because dZ is not what we want to predict.
        #    We want to predict dX using Z.
        
        #    Alternative: Use 'kernel_func' in GeneratorEstimator that computes RBF on LogSig features!
        pass

# Subclass to handle heuristic correctly
class ComposedGeneratorEstimator(GeneratorEstimator):
    def _fit_rbf_backend(self, trajectory, dX, n_iter):
        """Override to compute heuristic on LogSig features."""
        # Standard fit logic but with custom heuristic length_scale
        
        n_obs = len(dX)
        X = trajectory[:-1].reshape(-1, 1)
        
        # 1. Compute LogSig features for heuristic
        # Lifting to Z space
        degree = 2
        Z_list = []
        # Subsample for speed
        idx_heur = np.random.choice(n_obs, min(2000, n_obs), replace=False)
        X_sub = X[idx_heur]
        for d in range(1, degree + 1):
            Z_list.append(X_sub**d)
        Z_sub = np.column_stack(Z_list)
        
        # 2. Compute median distance in Z space
        dists = cdist(Z_sub, Z_sub, 'sqeuclidean')
        # Flatten and take median of non-zero
        dists = dists[dists > 0]
        median_sq_dist = np.median(dists)
        self.length_scale_ = np.sqrt(median_sq_dist)
        print(f"  [Composed] Computed Length Scale on LogSig Features: {self.length_scale_:.4f}")
        
        # 3. Call Nystrom with this length scale and our custom kernel
        # We need to pass the custom kernel
        self._fit_nystrom_backend(trajectory, dX, n_iter, logsig_rbf_kernel, self.length_scale_)

# Define Custom Kernel for GeneratorEstimator
def logsig_rbf_kernel(X1, X2, length_scale):
    # X1, X2 are (N, D) original states (D=1)
    
    # 1. Lift to LogSig (Polynomial)
    # X1 -> Z1 (N, 2)
    d = 2
    Z1_list = [X1**k for k in range(1, d+1)]
    Z1 = np.column_stack(Z1_list)
    
    Z2_list = [X2**k for k in range(1, d+1)]
    Z2 = np.column_stack(Z2_list)
    
    # 2. RBF on Z
    sq_dists = cdist(Z1, Z2, 'sqeuclidean')
    return np.exp(-sq_dists / (2 * length_scale ** 2))

def run_experiment():
    print("Generating Double-Well Data...")
    X_traj = generate_double_well(n_steps=2000)
    dt = 0.01
    
    # Test Data
    x_test = np.linspace(-2, 2, 100)
    true_drift = 4 * (x_test - x_test**3)
    
    # 1. Linear(LogSig_2)
    print("\nRunning Linear(LogSig_2)...")
    beta_lin = linear_logsig_fit(X_traj, dt)
    drift_lin = predict_linear_logsig(x_test, beta_lin)
    mse_lin = np.mean((drift_lin - true_drift)**2)
    print(f"Linear(LogSig_2) MSE: {mse_lin:.4f}")
    
    # 1b. Linear(LogSig_4) - The likely "ksig-KKF" winner
    print("\nRunning Linear(LogSig_4)...")
    # Need to manually call fit with degree 4
    def linear_logsig_fit_deg(X, dt, deg):
        dX = np.diff(X)
        Y = dX / dt
        Phi = compute_logsig_features_point(X[:-1], degree=deg)
        lambda_reg = 1e-3
        beta = np.linalg.solve(Phi.T @ Phi + lambda_reg * np.eye(Phi.shape[1]), Phi.T @ Y)
        return beta
    
    def predict_linear_logsig_deg(x_test, beta, deg):
        Phi_test = compute_logsig_features_point(x_test, degree=deg)
        return Phi_test @ beta

    beta_lin4 = linear_logsig_fit_deg(X_traj, dt, 4)
    drift_lin4 = predict_linear_logsig_deg(x_test, beta_lin4, 4)
    mse_lin4 = np.mean((drift_lin4 - true_drift)**2)
    print(f"Linear(LogSig_4) MSE: {mse_lin4:.4f}")
    
    # 2. RBF(LogSig_2)
    # Using Subclassed Estimator
    print("\nRunning RBF(LogSig_2)...")
    est_rbf = ComposedGeneratorEstimator(
        dt=dt,
        backend='rbf',
        n_landmarks=200,
        reg_param=1e-3
    )
    # We still need to patch _rbf_kernel for prediction?
    # No, _fit_nystrom_backend saves the kernel function?
    # Actually, _fit_nystrom_backend uses 'kernel_func' to compute features, and stores landmarks.
    # But prediction uses 'self._rbf_kernel' by default or needs to know which kernel to use.
    # _predict_drift_rbf currently hardcodes 'self._rbf_kernel'.
    # We need to monkey-patch _rbf_kernel to be logsig_rbf_kernel for prediction to work.
    est_rbf._rbf_kernel = logsig_rbf_kernel
    
    est_rbf.fit(X_traj)
    drift_rbf = est_rbf.predict_drift(x_test)
    mse_rbf = np.mean((drift_rbf - true_drift)**2)
    print(f"RBF(LogSig_2) MSE: {mse_rbf:.4f}")
    
    # 3. Standard RBF(X)
    print("\nRunning Standard RBF(X)...")
    est_std = GeneratorEstimator(dt=dt, backend='rbf', n_landmarks=200, reg_param=1e-3)
    est_std.fit(X_traj)
    drift_std = est_std.predict_drift(x_test)
    mse_std = np.mean((drift_std - true_drift)**2)
    print(f"Standard RBF(X) MSE: {mse_std:.4f}")
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.plot(x_test, true_drift, 'k--', label='True Drift (Cubic)', linewidth=2)
    plt.plot(x_test, drift_lin, 'r-', label=f'Linear(LogSig_2) (MSE={mse_lin:.2f})')
    plt.plot(x_test, drift_lin4, 'm-.', label=f'Linear(LogSig_4) (MSE={mse_lin4:.2f})')
    plt.plot(x_test, drift_rbf, 'g-', label=f'RBF(LogSig_2) (MSE={mse_rbf:.2f})')
    plt.plot(x_test, drift_std, 'b:', label=f'Standard RBF(X) (MSE={mse_std:.2f})')
    plt.title("Composed Kernels: Can RBF(LogSig_2) Learn Cubic Drift?")
    plt.xlabel("x")
    plt.ylabel("Drift f(x)")
    plt.legend()
    plt.grid(True)
    plt.savefig('experiments/benchmarks/results_composed_kernels.png')
    print("Plot saved to experiments/benchmarks/results_composed_kernels.png")
    
    with open('experiments/benchmarks/results_exp06.txt', 'w') as f:
        f.write(f"Linear(LogSig_2) MSE: {mse_lin:.4f}\n")
        f.write(f"Linear(LogSig_4) MSE: {mse_lin4:.4f}\n")
        f.write(f"RBF(LogSig_2) MSE: {mse_rbf:.4f}\n")
        f.write(f"Standard RBF(X) MSE: {mse_std:.4f}\n")

if __name__ == "__main__":
    run_experiment()
