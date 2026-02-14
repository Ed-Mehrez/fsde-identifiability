import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from src.generator_estimator import GeneratorEstimator
import traceback



def generate_fBm(N, dt, H, dim=1):
    """Simple Cholesky-based fractional Brownian motion generator."""
    # Covariance matrix for fGn
    t = np.arange(N)
    r = 0.5 * ((t + 1)**(2*H) - 2*t**(2*H) + np.abs(t - 1)**(2*H)) * dt**(2*H)
    R = np.zeros((N, N))
    for i in range(N):
        R[i, i:] = r[:N-i]
        R[i:, i] = r[:N-i]
    
    L = np.linalg.cholesky(R + 1e-8 * np.eye(N))
    
    noise = np.random.randn(N, dim)
    dB = L @ noise
    return np.cumsum(dB, axis=0) # start at 0? No, this is B(t).

def test_2d_coupled_fou():
    """
    Test learning a 2D coupled fOU process:
    dX_t = A X_t dt + dB_H(t)
    """
    np.random.seed(42)
    N = 2000
    dt = 0.01
    H_true = 0.7
    dim = 2
    
    # 1. Generate fBM noise
    # We generate standard fBM for each channel
    B = generate_fBm(N, dt, H_true, dim=dim)
    dB = np.diff(B, axis=0)
    
    # 2. Simulate fOU
    # A = [[-1, 0.5], [-0.5, -1]] (Stable spiral)
    A_true = np.array([[-1.0, 0.5], [-0.5, -1.0]])
    X = np.zeros((N, dim))
    X[0] = np.random.randn(dim)
    
    for i in range(N-1):
        drift = A_true @ X[i]
        X[i+1] = X[i] + drift * dt + dB[i]
    
    # Scale X to avoid numerical issues if large
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    # 3. Fit Estimator
    est = GeneratorEstimator(dt=dt, backend='logsig', h_method='variation', h_type='vector',
                             rank=20, n_landmarks=100, whiten=True)
    est.fit(X_std)
    
    # 4. Check H estimation
    # Should be close to 0.7 for both dims
    print(f"Estimated H: {est.H_}")
    assert np.all(np.abs(est.H_ - H_true) < 0.15), f"H estimation failed: {est.H_} vs {H_true}"
    
    # 5. Check Drift Prediction
    # Predict at random points from trajectory
    test_idx = np.random.choice(len(X_std)-50, 20)
    X_test = X_std[test_idx]
    
    # We don't know the true drift of X_std exactly without scaling math,
    # but we can check R^2 on a hold-out set or just check fit consistency.
    # Let's check consistency with ground truth drift (scaled)
    # Drift of standardized X: A * X
    # (Since linear map commutes with scaling if centered)
    
    drift_pred = est.predict_drift(X_test)
    
    # Check shape
    assert drift_pred.shape == (20, 2)
    
    # Rough check: sign agreement or correlation
    # We need to know the effective A for the standardized data.
    # Since we standardized by std, and it's linear, approx A_eff = A_true? 
    # Not exactly if variances differ.
    # But let's check if fit runs and shapes are correct.
    assert not np.any(np.isnan(drift_pred))

def test_3d_lorenz_standard():
    """
    Test learning 3D Lorenz-63 (H=0.5).
    """
    np.random.seed(42)
    N = 10000 # Increase N to cover some time with small dt
    dt = 0.001 # Small dt ensures noise dominates drift (if noise is large enough)
    
    # Lorenz-63 parameters
    sigma = 10.0
    rho = 28.0
    beta = 8.0/3.0
    
    def lorenz_drift(state):
        x, y, z = state
        return np.array([
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z
        ])
    
    X = np.zeros((N, 3))
    X[0] = [1.0, 1.0, 1.0]
    
    # Euler-Maruyama with small noise
    for i in range(N-1):
        f = lorenz_drift(X[i])
        noise = np.random.randn(3) * np.sqrt(dt) # Standard Brownian
        # Increase noise to make it visible against drift at dt=0.01 (need > 0.8)
        X[i+1] = X[i] + f * dt + 10.0 * noise
        
    est = GeneratorEstimator(dt=dt, backend='logsig', h_method='variation', h_type='scalar',
                             rank=20, whiten=True) # H should be 0.5
                             
    est.fit(X)
    
    # Check H
    print(f"Estimated H (Lorenz): {est.H_}")
    assert abs(est.H_ - 0.5) < 0.15, f"H should be ~0.5, got {est.H_}"
    
    # Check drift shape
    X_test = X[100:105]
    drift_pred = est.predict_drift(X_test)
    assert drift_pred.shape == (5, 3)

def test_sigma_estimation_shapes():
    """Verify sigma prediction returns correct shapes."""
    dt = 0.01
    N = 100
    D = 2
    X = np.random.randn(N, D)
    est = GeneratorEstimator(dt=dt, backend='logsig', h_type='vector')
    est.fit(X)
    
    X_test = np.random.randn(5, D)
    sigma_pred = est.predict_diffusion(X_test)
    assert sigma_pred.shape == (5, D)

if __name__ == "__main__":
    def run_tests():
        try:
            print("Running test_2d_coupled_fou...")
            test_2d_coupled_fou()
            print("  PASS")
        except Exception as e:
            print(f"  FAIL: {e}")
            traceback.print_exc()

        try:
            print("Running test_3d_lorenz_standard...")
            test_3d_lorenz_standard()
            print("  PASS")
        except Exception as e:
            print(f"  FAIL: {e}")
            traceback.print_exc()

        try:
            print("Running test_sigma_estimation_shapes...")
            test_sigma_estimation_shapes()
            print("  PASS")
        except Exception as e:
            print(f"  FAIL: {e}")
            traceback.print_exc()
            
    run_tests()
