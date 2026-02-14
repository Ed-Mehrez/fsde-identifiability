"""
Spectral Hurst Estimation via Koopman Operators

Implements the novel method of estimating the Hurst parameter from the
eigenvalue decay rate of the Koopman generator.

Key Theoretical Result:
    For fBm with Hurst parameter H, the eigenvalues λ_k of the Koopman 
    generator satisfy: |λ_k| ~ k^{-(2H+1)}

References:
- Theory document: documentation/spectral_roughness_theory.md
"""

import numpy as np
from scipy.stats import linregress
from typing import Optional, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kgedmd_core import KernelGEDMD
from src.models.kernels import GibbsKernel


class SpectralHurstEstimator:
    """
    Estimate Hurst parameter from Koopman eigenvalue spectrum.
    
    Algorithm:
    1. Embed trajectory in time-delay coordinates
    2. Fit Koopman operator via KernelGEDMD
    3. Extract eigenvalues {λ_k}
    4. Perform log-log regression: log|λ_k| ~ -(2H+1) * log(k)
    5. Extract H from slope
    """
    
    def __init__(
        self,
        kernel: str = 'gibbs',  # Changed default to 'gibbs' per user request
        n_eigen: int = 50,
        embedding_dim: int = 3,
        delay: int = 1,
        bandwidth: Optional[float] = None,
        reg_param: float = 1e-6,
        n_subsample: int = 2000,  # Max centers for Nystrom/subsampling
        max_scale: float = 10.0,   # For Gibbs kernel
        k_neighbors: Optional[int] = None # Manual override for Gibbs
    ):
        """
        Args:
            kernel: Kernel type ('gibbs', 'rbf')
            n_eigen: Number of eigenvalues to use in regression
            embedding_dim: Time-delay embedding dimension
            delay: Time-delay lag
            bandwidth: Base scale for Gibbs or width for RBF
            reg_param: Regularization parameter for KGEDMD
            n_subsample: Max samples for kernel approximation (prevents O(N^3) explosion)
            max_scale: Max length scale for Gibbs kernel
            k_neighbors: Number of neighbors for Gibbs kernel (default: auto)
        """
        self.kernel = kernel
        self.n_eigen = n_eigen
        self.embedding_dim = embedding_dim
        self.delay = delay
        self.bandwidth = bandwidth
        self.reg_param = reg_param
        self.n_subsample = n_subsample
        self.max_scale = max_scale
        self.k_neighbors = k_neighbors
        
        self.kgedmd = None
        self.eigenvalues_ = None
        self.H_ = None
        self.regression_result_ = None
    
    def fit(self, trajectory: np.ndarray) -> 'SpectralHurstEstimator':
        """
        Fit the estimator to a trajectory.
        
        Args:
            trajectory: 1D array of time series data
            
        Returns:
            self
        """
        # 1. Time-delay embedding
        X = self._embed_trajectory(trajectory)
        
        # 2. Auto-select bandwidth if not provided
        if self.bandwidth is None:
            # Silverman's rule adapted for multi-dimensional data
            n, d = X.shape
            sigma = np.std(X, axis=0)
            self.bandwidth = np.mean(sigma) * (n ** (-1.0 / (d + 4)))
            
            if self.kernel == 'gibbs':
                # Gibbs kernel base scale typically smaller than global bandwidth
                self.bandwidth *= 0.5
        
        # 3. Fit Koopman operator
        X_pairs_0 = X[:-1]
        X_pairs_1 = X[1:]
        
        if self.kernel == 'gibbs':
            # Instantiate and fit Gibbs Kernel (KNN-based)
            # Implements Rasmussen & Williams Ch 4.2 approach
            
            # Determine k
            if self.k_neighbors is not None:
                k = self.k_neighbors
            else:
                k = min(10, max(2, len(X)//100))
                
            kernel_obj = GibbsKernel(
                base_scale=self.bandwidth, 
                max_scale=self.max_scale,
                k_neighbors=k
            )
            # Gibbs kernel needs to learn local density from data
            if len(X_pairs_0) > self.n_subsample:
                # Fit kernel on random subsample to save time
                idx = np.random.choice(len(X_pairs_0), self.n_subsample, replace=False)
                kernel_obj.fit(X_pairs_0[idx])
            else:
                kernel_obj.fit(X_pairs_0)
                
            self.kgedmd = KernelGEDMD(
                kernel_type='custom',
                kernel_obj=kernel_obj,
                epsilon=self.reg_param
            )
        else:
            # Standard RBF
            self.kgedmd = KernelGEDMD(
                kernel_type='rbf',
                sigma=self.bandwidth,
                epsilon=self.reg_param
            )
        
        # Use subsampling for large datasets to keep complexity tractable
        n_samples = len(X_pairs_0)
        
        if self.n_subsample and n_samples > self.n_subsample:
            # Perform subsampling here to retain access to Y for mode computation
            # Using random selection for speed (or farthest point could be better but slower)
            idx = np.random.choice(n_samples, self.n_subsample, replace=False)
            X_fit = X_pairs_0[idx]
            Y_fit = (X_pairs_1[idx] - X_pairs_0[idx]) # Increment target assuming dt=1
        else:
            X_fit = X_pairs_0
            Y_fit = X_pairs_1 - X_pairs_0

        # Transpose to (d, m) for KGEDMD if needed? 
        # kgedmd_core checks shape. standard is (d, m).
        # X_fit is (m, d).
        # Let's check kgedmd.fit signature/expectations.
        # It calls gramian_matrix(X). gramian_matrix expects (d, m).
        # But wait. My previous fix 'generator_gram_matrix_vectorized' used X.T in places.
        # kgedmd_core.fit(X, Y) docs say: X (d, m).
        # SpectralHurstEstimator _embed_trajectory returns (N, d).
        # So I must transpose! 
        # Previous code passed (N, d) and it worked?
        # Let's check kgedmd_core.py line 433: "n, d = X.shape; if n < d: ... assume (d, m) ... else assume (m, d)".
        # It attempts to auto-detect. 
        # With d=3, m=2000 => it detects correctly.
        # So I will transpose explicitly to be safe and match internal d3s format.
        
        self.X_fit_ = X_fit.T # (d, m)
        self.Y_fit_ = Y_fit.T # (d, m)
        
        # Fit KGEDMD
        self.kgedmd.fit(self.X_fit_, self.Y_fit_, n_subsample=None) # Already subsampled
        
        # 4. Extract eigenvalues
        self.eigenvalues_ = self.kgedmd.eigenvalues_[:self.n_eigen]
        
        # 5. Compute Koopman Modes (for prediction/validation)
        # Phi = K(X, X) @ Xi
        # V = Phi+ @ Y
        
        # K_fit = G_00 (available in kgedmd)
        # Xi = eigenvectors_ (available in kgedmd)
        if hasattr(self.kgedmd, 'G_00_') and hasattr(self.kgedmd, 'eigenvectors_'):
            Phi = self.kgedmd.G_00_ @ self.kgedmd.eigenvectors_[:, :self.n_eigen]
            # Solve Phi * V = Y.T (since Y is (d, m), Y.T is (m, d))
            # Phi is (m, r). V is (r, d).
            # Y.T is (m, d).
            self.modes_, _, _, _ = np.linalg.lstsq(Phi, self.Y_fit_.T, rcond=None)
        
        return self

    def predict(self, X_start: np.ndarray, steps: int = 1) -> np.ndarray:
        """
        Predict future states using the learned Koopman operator.
        Useful for cross-validation and parameter tuning.
        
        Args:
           X_start: (n_points, d) array of starting states
           steps: Number of steps to predict (currently only 1 supported effectively due to increment model)
           
        Returns:
           X_pred: (n_points, d) array of predicted next states
        """
        if self.kgedmd is None or not hasattr(self, 'modes_'):
            raise RuntimeError("Estimator must be fitted before prediction.")
            
        # 1. Evaluate Eigenfunctions at new points
        # Phi(x) = K(x, X_fit) @ Xi
        # X_start: (N, d). X_fit_: (d, m).
        # kernel.compute expects (m, d) if first arg?
        # GibbsKernel.compute alias calls __call__(X, Y).
        # __call__ expects (N, d).
        
        # We need K(X_start, X_fit_.T) -> (N, m)
        K_eval = self.kgedmd.kernel.compute(X_start, self.X_fit_.T)
        
        # Phi_eval: (N, m) @ (m, r) -> (N, r)
        Phi_eval = K_eval @ self.kgedmd.eigenvectors_[:, :self.n_eigen]
        
        # 2. Predict Drift/Increment: dX = Phi @ Modes
        # (N, r) @ (r, d) -> (N, d)
        dX_pred = Phi_eval @ self.modes_
        
        # 3. Step
        X_pred = X_start + dX_pred
        
        return X_pred
    
    def estimate_hurst(self, method: str = 'regression') -> float:
        """
        Estimate Hurst parameter from eigenvalue spectrum.
        
        Args:
            method: 'regression' for log-log regression
            
        Returns:
            H: Estimated Hurst parameter
        """
        if self.eigenvalues_ is None:
            raise ValueError("Must call fit() before estimate_hurst()")
        
        if method == 'regression':
            H, regression_info = self._estimate_via_regression()
            self.H_ = H
            self.regression_result_ = regression_info
            return H
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _estimate_via_regression(self) -> Tuple[float, dict]:
        """
        Estimate H via log-log regression of eigenvalue decay.
        
        Returns:
            H: Estimated Hurst
            info: Dict with regression diagnostics
        """
        # Take magnitude of eigenvalues
        lambda_mag = np.abs(self.eigenvalues_)
        
        # Remove zero eigenvalues (numerical artifacts)
        nonzero_mask = lambda_mag > 1e-12
        lambda_mag = lambda_mag[nonzero_mask]
        
        # Index k
        k = np.arange(1, len(lambda_mag) + 1)
        
        # Log-log space
        log_k = np.log(k)
        log_lambda = np.log(lambda_mag)
        
        # Linear regression: log|λ_k| = β_0 + β_1 * log(k)
        # Theory: β_1 = -(2H + 1)
        slope, intercept, r_value, p_value, std_err = linregress(log_k, log_lambda)
        
        # Extract Hurst: H = -(slope + 1) / 2
        H = -(slope + 1) / 2
        
        # Clip to valid range
        H = np.clip(H, 0.01, 0.99)
        
        regression_info = {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'std_err': std_err,
            'n_eigenvalues': len(lambda_mag)
        }
        
        return H, regression_info
    
    def _embed_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Time-delay embedding of trajectory.
        
        Args:
            trajectory: 1D array
            
        Returns:
            X: 2D array of shape (n_samples, embedding_dim)
        """
        n = len(trajectory)
        m = self.embedding_dim
        tau = self.delay
        
        # Number of embedded vectors
        n_embed = n - (m - 1) * tau
        
        if n_embed <= 0:
            raise ValueError(f"Trajectory too short for embedding. Need at least {(m-1)*tau + 1} points")
        
        # Create delay embedding
        X = np.zeros((n_embed, m))
        for i in range(m):
            X[:, i] = trajectory[i * tau : i * tau + n_embed]
        
        return X
    
    def get_eigenvalues(self) -> np.ndarray:
        """Return eigenvalues used in estimation."""
        if self.eigenvalues_ is None:
            raise ValueError("Must call fit() first")
        return self.eigenvalues_
    
    def get_regression_diagnostics(self) -> dict:
        """Return linear regression diagnostics."""
        if self.regression_result_ is None:
            raise ValueError("Must call estimate_hurst() first")
        return self.regression_result_
    
    def plot_spectrum(self, ax=None, show_theory: bool = True):
        """
        Plot eigenvalue spectrum in log-log space.
        
        Args:
            ax: Matplotlib axis (creates new if None)
            show_theory: Whether to plot theoretical line
        """
        import matplotlib.pyplot as plt
        
        if self.eigenvalues_ is None:
            raise ValueError("Must call fit() first")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Data
        lambda_mag = np.abs(self.eigenvalues_)
        nonzero_mask = lambda_mag > 1e-12
        lambda_mag = lambda_mag[nonzero_mask]
        k = np.arange(1, len(lambda_mag) + 1)
        
        # Plot empirical eigenvalues
        ax.loglog(k, lambda_mag, 'o', label='Empirical', markersize=6, alpha=0.7)
        
        # Plot theoretical line if H is estimated
        if show_theory and self.H_ is not None:
            k_theory = np.linspace(1, len(k), 100)
            # λ_k ~ k^{-(2H+1)}
            lambda_theory = k_theory ** (-(2 * self.H_ + 1))
            # Scale to match empirical
            scale = lambda_mag[0] / lambda_theory[0]
            lambda_theory *= scale
            
            ax.loglog(
                k_theory, lambda_theory, '--',
                label=f'Theory: $\\lambda_k \\sim k^{{-{2*self.H_+1:.2f}}}$',
                linewidth=2
            )
        
        ax.set_xlabel('Eigenvalue index $k$', fontsize=12)
        ax.set_ylabel('$|\\lambda_k|$', fontsize=12)
        ax.set_title(f'Koopman Eigenvalue Spectrum (H = {self.H_:.3f})', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add R² annotation
        if self.regression_result_ is not None:
            r2 = self.regression_result_['r_squared']
            ax.text(
                0.05, 0.05, f'$R^2 = {r2:.4f}$',
                transform=ax.transAxes,
                fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
        
        return ax


def estimate_hurst_spectral(
    trajectory: np.ndarray,
    kernel: str = 'rbf',
    n_eigen: int = 50,
    **kwargs
) -> Tuple[float, dict]:
    """
    Convenience function for spectral Hurst estimation.
    
    Args:
        trajectory: 1D time series
        kernel: Kernel type
        n_eigen: Number of eigenvalues
        **kwargs: Additional args for SpectralHurstEstimator
        
    Returns:
        H: Estimated Hurst parameter
        diagnostics: Regression diagnostics
    """
    estimator = SpectralHurstEstimator(kernel=kernel, n_eigen=n_eigen, **kwargs)
    estimator.fit(trajectory)
    H = estimator.estimate_hurst()
    diagnostics = estimator.get_regression_diagnostics()
    
    return H, diagnostics


if __name__ == "__main__":
    # Quick test
    print("Testing Spectral Hurst Estimator...")
    
    from rough_paths_generator import generate_fbm
    
    for H_true in [0.2, 0.5, 0.8]:
        print(f"\nTrue H = {H_true}")
        
        # Generate fBm
        path = generate_fbm(n_samples=2000, H=H_true, n_paths=1, seed=42)[0]
        
        # Estimate H
        estimator = SpectralHurstEstimator(n_eigen=30)
        estimator.fit(path)
        H_est = estimator.estimate_hurst()
        
        diag = estimator.get_regression_diagnostics()
        
        print(f"  Estimated H: {H_est:.3f}")
        print(f"  Error: {abs(H_est - H_true):.3f}")
        print(f"  R²: {diag['r_squared']:.4f}")
        print(f"  Slope: {diag['slope']:.3f} (theory: {-(2*H_true+1):.3f})")
