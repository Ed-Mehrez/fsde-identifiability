import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from generator_estimator import GeneratorEstimator

def run_debug():
    dt = 0.01
    N = 1000
    X = np.random.randn(N, 1)
    
    print("Testing Koopman Backend...")
    try:
        est = GeneratorEstimator(dt=dt, backend='koopman', h_method='variation', whiten=True, reg_param=1e-3)
        est.fit(X, n_iter=2)
        print("Success!")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_debug()
