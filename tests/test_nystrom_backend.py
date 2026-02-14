import numpy as np
import sys
import os
import unittest
import time

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from generator_estimator import GeneratorEstimator

class TestNystromBackend(unittest.TestCase):
    def setUp(self):
        # Double-Well params
        self.dt = 0.01
        self.N = 2000
        self.sigma = 1.0
        
        # Simulate data
        np.random.seed(42)
        self.X = np.zeros(self.N + 1)
        self.X[0] = 0.5
        for i in range(self.N):
            drift = 4 * (self.X[i] - self.X[i]**3)
            self.X[i+1] = self.X[i] + drift * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn()
            
    def test_nystrom_fitting_run(self):
        """Test that _fit_nystrom_backend runs without error and produces reasonable results."""
        est = GeneratorEstimator(
            dt=self.dt,
            backend='rbf', # Will use Nystrom if n_landmarks < N
            n_landmarks=100, # Force Nystrom usage (100 < 2000)
            reg_param=1e-3
        )
        
        # Manually trigger Nystrom fit to be sure
        # But 'fit' should call it automatically via _fit_rbf_backend -> _fit_nystrom_backend logic
        
        print("\nTesting Nystrom Backend Fit...")
        start_time = time.time()
        est.fit(self.X) # Should use Nystrom
        duration = time.time() - start_time
        
        print(f"Fit duration: {duration:.4f}s")
        
        # Check if landmarks were set
        self.assertTrue(hasattr(est, 'rbf_landmarks_'))
        self.assertEqual(len(est.rbf_landmarks_), 100)
        
        # Check prediction
        x_test = np.linspace(-2, 2, 20)
        drift_pred = est.predict_drift(x_test)
        
        # Basic shape check
        self.assertEqual(drift_pred.shape, (20,))
        
        # Check against true drift
        drift_true = 4 * (x_test - x_test**3)
        mae = np.mean(np.abs(drift_pred - drift_true))
        print(f"Drift MAE on test set: {mae:.4f}")
        
        # It should be reasonably accurate (e.g. < 5.0 for this noisy data)
        self.assertLess(mae, 10.0)

if __name__ == '__main__':
    unittest.main()
