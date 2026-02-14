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
    X[0] = 0.1
    for i in range(N-1):
        f = double_well_drift(X[i])
        noise = np.random.randn() * np.sqrt(dt) * sigma
        X[i+1] = X[i] + f * dt + noise
    return X

def run_comparison():
    print("--- Backend Comparison (Double-Well) ---")
    
    dt = 0.01
    N = 5000 # Use 5000 for speed/memory
    sigma = 1.0 
    
    print(f"Simulating N={N}...")
    X = simulate_double_well(N, dt, sigma)
    X = X.reshape(-1, 1)
    
    # Grid
    x_eval = np.linspace(-2.5, 2.5, 100).reshape(-1, 1)
    drift_true = double_well_drift(x_eval).flatten()
    mask = (x_eval.flatten() > -1.5) & (x_eval.flatten() < 1.5)
    drift_range = np.max(drift_true[mask]) - np.min(drift_true[mask])
    
    backends = ['logsig', 'koopman', 'rbf', 'signature']
    results = {}
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_eval, drift_true, 'k--', label='True Drift', lw=2)
    
    for backend in backends:
        print(f"\nTesting Backend: {backend.upper()}...")
        try:
            start_time = time.time()
            # Use appropriate settings for each
            # Koopman/Logsig: rank=20 is fine
            # RBF: rank doesn't apply (it uses full kernel matrix or Nystrom if implemented, here full K)
            # Signature: uses Nystrom (rank relevant)
            
            est = GeneratorEstimator(dt=dt, backend=backend,
                                     h_method='variation', h_type='scalar',
                                     whiten=True, rank=50, fixed_H=0.5,
                                     reg_param='gcv' if backend != 'koopman' else 1e-3) 
                                     # Koopman gcv might fail if features are collinear
            
            est.fit(X, n_iter=5)
            drift_pred = est.predict_drift(x_eval).flatten()
            
            elapsed = time.time() - start_time
            
            mae = np.mean(np.abs(drift_true[mask] - drift_pred[mask]))
            rmse = np.sqrt(np.mean((drift_true[mask] - drift_pred[mask])**2))
            nrmse = rmse / drift_range
            
            print(f"  MAE: {mae:.4f}")
            print(f"  NRMSE: {nrmse:.4f}")
            print(f"  Time: {elapsed:.2f}s")
            
            results[backend] = {'nrmse': nrmse, 'mae': mae, 'time': elapsed}
            
            plt.plot(x_eval, drift_pred, label=f'{backend} (NRMSE={nrmse:.2f})')
            
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[backend] = {'nrmse': float('inf'), 'mae': float('inf'), 'time': 0}

    plt.title("Backend Comparison: Double-Well Drift Recovery")
    plt.legend()
    plt.ylim([-25, 25])
    plt.grid(True, alpha=0.3)
    plt.savefig('experiments/benchmarks/backend_comparison.png', dpi=150)
    print("\nSaved plot to experiments/benchmarks/backend_comparison.png")
    
    print("\nSummary:")
    print(f"{'Backend':<12} | {'NRMSE':<8} | {'MAE':<8} | {'Time':<8}")
    print("-" * 45)
    for b in backends:
        res = results[b]
        print(f"{b:<12} | {res['nrmse']:.4f}   | {res['mae']:.4f}   | {res['time']:.2f}")

if __name__ == "__main__":
    run_comparison()
