"""
Nyström-accelerated Koopman Eigenfunction Extraction

Uses Nyström approximation for scalable kernel EDMD:
1. Select m landmarks from data
2. Form reduced kernel matrices K_mm, K_nm
3. Project dynamics into landmark feature space
4. Eigendecompose to get Koopman eigenfunctions

Works with both RBF and signature kernels.
"""

import numpy as np
from scipy.linalg import eig, sqrtm, svd
from typing import Optional, Tuple, List, Callable
import iisignature


class NystromKoopman:
    """
    Nyström-accelerated Koopman eigenfunction extraction.

    Given trajectory data X_t, X_{t+dt}, finds eigenfunctions φ such that:
        E[φ(X_{t+dt}) | X_t] = λ φ(X_t)

    The Nyström approximation projects into a low-dimensional feature space
    defined by m landmark points, making eigendecomposition O(m³) instead of O(n³).
    """

    def __init__(self,
                 n_landmarks: int = 100,
                 kernel: str = 'rbf',  # 'rbf', 'signature', or callable
                 bandwidth: float = None,  # For RBF; auto-tune if None
                 sig_depth: int = 3,  # For signature kernel
                 reg: float = 1e-6,
                 landmark_selection: str = 'random'):  # 'random' or 'kmeans'
        """
        Args:
            n_landmarks: Number of Nyström landmarks
            kernel: Kernel type ('rbf', 'signature') or custom callable.
                    If callable, should have signature: kernel(X1, X2) -> np.ndarray
                    where X1, X2 are arrays/lists and output is (len(X1), len(X2))
            bandwidth: RBF bandwidth (auto-tuned if None)
            sig_depth: Signature truncation depth
            reg: Regularization for matrix inversions
            landmark_selection: How to select landmarks
        """
        self.n_landmarks = n_landmarks
        self.bandwidth = bandwidth
        self.sig_depth = sig_depth
        self.reg = reg
        self.landmark_selection = landmark_selection

        # Handle custom kernel functions
        if callable(kernel):
            self.kernel_type = 'custom'
            self._custom_kernel = kernel
        else:
            self.kernel_type = kernel
            self._custom_kernel = None

        # Will be set during fit
        self.landmarks_X = None  # Landmarks at time t
        self.landmarks_Y = None  # Landmarks at time t+dt (same indices)
        self.K_mm = None  # Landmark kernel matrix
        self.K_mm_inv_sqrt = None  # K_mm^{-1/2}
        self.eigenvalues = None
        self.eigenvectors = None  # In landmark feature space
        self._sig_prep = None

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF kernel between state vectors or paths."""
        # Handle both 1D states and paths
        if X1.ndim == 1:
            X1 = X1.reshape(1, -1)
        if X2.ndim == 1:
            X2 = X2.reshape(1, -1)

        # For paths, flatten to vectors
        if X1.ndim == 3:
            X1 = X1.reshape(X1.shape[0], -1)
        if X2.ndim == 3:
            X2 = X2.reshape(X2.shape[0], -1)

        sq_dists = np.sum(X1**2, axis=1, keepdims=True) + \
                   np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
        return np.exp(-sq_dists / (2 * self.bandwidth**2))

    def _signature_kernel(self, paths1: List[np.ndarray],
                          paths2: List[np.ndarray]) -> np.ndarray:
        """Truncated signature kernel between paths."""
        n1, n2 = len(paths1), len(paths2)
        K = np.zeros((n1, n2))

        # Compute signatures
        sigs1 = []
        sigs2 = []

        for p in paths1:
            if self._sig_prep is None:
                self._sig_prep = iisignature.prepare(p.shape[1], self.sig_depth)
            sigs1.append(iisignature.sig(p, self.sig_depth))

        for p in paths2:
            sigs2.append(iisignature.sig(p, self.sig_depth))

        sigs1 = np.array(sigs1)
        sigs2 = np.array(sigs2)

        # Normalize and compute inner products
        norms1 = np.linalg.norm(sigs1, axis=1, keepdims=True) + 1e-10
        norms2 = np.linalg.norm(sigs2, axis=1, keepdims=True) + 1e-10
        sigs1_norm = sigs1 / norms1
        sigs2_norm = sigs2 / norms2

        K = sigs1_norm @ sigs2_norm.T
        return K

    def _compute_kernel(self, data1, data2) -> np.ndarray:
        """Compute kernel matrix based on kernel type."""
        if self.kernel_type == 'rbf':
            return self._rbf_kernel(data1, data2)
        elif self.kernel_type == 'signature':
            return self._signature_kernel(data1, data2)
        elif self.kernel_type == 'custom':
            return self._custom_kernel(data1, data2)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel_type}")

    def _select_landmarks(self, X: np.ndarray, Y: np.ndarray,
                          n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select landmark indices."""
        n_samples = len(X)

        if self.landmark_selection == 'random':
            indices = np.random.choice(n_samples, min(n, n_samples), replace=False)
        elif self.landmark_selection == 'kmeans':
            # Simple k-means for landmark selection
            from scipy.cluster.vq import kmeans2
            if isinstance(X, list):
                # For paths, use flattened representation
                X_flat = np.array([x.flatten() for x in X])
            else:
                X_flat = X
            _, labels = kmeans2(X_flat, min(n, n_samples), minit='++')
            # Select one point per cluster
            indices = []
            for i in range(min(n, n_samples)):
                cluster_points = np.where(labels == i)[0]
                if len(cluster_points) > 0:
                    indices.append(cluster_points[0])
            indices = np.array(indices)
        else:
            raise ValueError(f"Unknown landmark selection: {self.landmark_selection}")

        return indices

    def _auto_bandwidth(self, X: np.ndarray) -> float:
        """Auto-tune RBF bandwidth using median heuristic."""
        if isinstance(X, list):
            X_flat = np.array([x.flatten() for x in X])
        else:
            X_flat = X if X.ndim == 2 else X.reshape(len(X), -1)

        # Sample pairwise distances
        n = min(500, len(X_flat))
        idx = np.random.choice(len(X_flat), n, replace=False)
        sample = X_flat[idx]

        dists = []
        for i in range(n):
            for j in range(i+1, n):
                dists.append(np.linalg.norm(sample[i] - sample[j]))

        return np.median(dists) + 1e-6

    def fit(self, X: np.ndarray, Y: np.ndarray) -> 'NystromKoopman':
        """
        Fit Nyström-Koopman model.

        Args:
            X: States/paths at time t. Shape (n_samples, dim) or list of paths
            Y: States/paths at time t+dt. Same shape as X.

        Returns:
            self
        """
        n_samples = len(X)

        # Auto-tune bandwidth if needed
        if self.kernel_type == 'rbf' and self.bandwidth is None:
            self.bandwidth = self._auto_bandwidth(X)
            print(f"Auto-tuned bandwidth: {self.bandwidth:.4f}")

        # Select landmarks
        indices = self._select_landmarks(X, Y, self.n_landmarks)
        m = len(indices)

        if isinstance(X, list):
            self.landmarks_X = [X[i] for i in indices]
            self.landmarks_Y = [Y[i] for i in indices]
        else:
            self.landmarks_X = X[indices]
            self.landmarks_Y = Y[indices]

        # Compute landmark kernel matrix K_mm
        self.K_mm = self._compute_kernel(self.landmarks_X, self.landmarks_X)
        self.K_mm += self.reg * np.eye(m)  # Regularize

        # Compute K_mm^{-1/2}
        eigvals, eigvecs = np.linalg.eigh(self.K_mm)
        eigvals = np.maximum(eigvals, 1e-10)
        self.K_mm_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        # Compute kernel matrices for EDMD
        # Following Williams et al. kernel EDMD formulation:
        # G_XX = K(X, X), G_XY = K(X, Y)
        # Then solve: G_XY v = λ G_XX v

        # Nyström approximation: G ≈ K_nm @ K_mm^{-1} @ K_mn
        # So G_XX ≈ K_nm @ K_mm^{-1} @ K_mn where landmarks are from X
        # And G_XY ≈ K_nm @ K_mm^{-1} @ K_mY where K_mY = K(landmarks_X, Y)

        K_nm = self._compute_kernel(X, self.landmarks_X)  # (n, m)
        K_mY = self._compute_kernel(self.landmarks_X, Y)  # (m, n)

        # Nyström feature representation: Φ(x) = K_mm^{-1/2} k_m(x)
        Phi_X = K_nm @ self.K_mm_inv_sqrt  # (n, m)

        # For Y, we use the SAME landmark basis (K_mm from X-landmarks)
        # but compute kernel with Y: K(landmarks_X, Y)
        # Φ_Y^T = K_mm^{-1/2} @ K_mY  -> Φ_Y = K_mY^T @ K_mm^{-1/2}
        Phi_Y = K_mY.T @ self.K_mm_inv_sqrt  # (n, m)

        # DMD-style Koopman operator: find A such that Φ_Y ≈ Φ_X @ A^T
        # i.e., A^T = (Φ_X^T Φ_X + λI)^{-1} Φ_X^T Φ_Y
        # So A = Φ_Y^T Φ_X (Φ_X^T Φ_X + λI)^{-1}

        # Using covariance matrices:
        C_XX = Phi_X.T @ Phi_X / n_samples  # (m, m)
        C_XY = Phi_X.T @ Phi_Y / n_samples  # (m, m)

        # Regularize
        C_XX_reg = C_XX + self.reg * np.eye(m)

        # Koopman operator in feature space: A^T = C_XX^{-1} @ C_XY
        # The eigenvalue problem for Koopman eigenfunctions:
        # φ is eigenfunction with eigenvalue λ means: E[φ(Y)|X] = λ φ(X)
        # In features: A^T v = λ v, where φ(x) = v^T Φ(x)
        #
        # So we solve: C_XX^{-1} C_XY v = λ v
        # Or equivalently: C_XY v = λ C_XX v

        # Solve generalized eigenvalue problem
        eigenvalues, eigenvectors = eig(C_XY, C_XX_reg)

        # Sort by eigenvalue magnitude (descending)
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        self.eigenvalues = eigenvalues[idx]
        self.eigenvectors = eigenvectors[:, idx]

        # Normalize eigenvectors
        for i in range(self.eigenvectors.shape[1]):
            self.eigenvectors[:, i] /= np.linalg.norm(self.eigenvectors[:, i]) + 1e-10

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Koopman eigenfunction values at new points.

        Args:
            X: States/paths to evaluate. Shape (n_samples, dim) or list.

        Returns:
            Eigenfunction values, shape (n_samples, n_eigenfunctions)
        """
        if self.K_mm_inv_sqrt is None:
            raise ValueError("Must fit before transform")

        # Compute kernel with landmarks
        K_nm = self._compute_kernel(X, self.landmarks_X)

        # Project to Nyström feature space
        Phi = K_nm @ self.K_mm_inv_sqrt

        # Apply eigenvector transformation
        return Phi @ self.eigenvectors

    def eigenfunction(self, X: np.ndarray, idx: int = 0) -> np.ndarray:
        """Get values of a single eigenfunction."""
        return self.transform(X)[:, idx]

    def predict_forward(self, X: np.ndarray, n_steps: int = 1) -> np.ndarray:
        """
        Predict future states using Koopman eigenfunctions.

        This uses the spectral decomposition:
        E[f(X_{t+k})|X_t] = Σ_i λ_i^k <f, φ_i> φ_i(X_t)
        """
        phi_X = self.transform(X)  # (n, m) eigenfunction values

        # Apply eigenvalues k times
        Lambda_k = np.diag(self.eigenvalues ** n_steps)

        # Predicted eigenfunction values at future time
        phi_future = phi_X @ Lambda_k

        return phi_future


def test_on_fou():
    """Test Nyström-Koopman on fractional OU process."""
    import sys
    sys.path.insert(0, '/home/ed/SynologyDrive/Documents/Research/PE_Research/rkhs_kronic/src')
    from rough_paths_generator import FractionalBrownianMotion

    np.random.seed(42)

    # Parameters
    H = 0.3
    kappa = 2.0
    theta = 0.5
    sigma = 0.1
    dt = 0.01
    n_steps = 2000
    n_paths = 50

    print("=" * 70)
    print("Testing Nyström-Koopman on fOU")
    print("=" * 70)
    print(f"H={H}, κ={kappa}, θ={theta}, σ={sigma}")
    print(f"n_steps={n_steps}, n_paths={n_paths}")

    # Generate fOU trajectories
    def generate_fou(seed):
        np.random.seed(seed)
        fbm = FractionalBrownianMotion(H=H, dt=dt, seed=seed)
        path = fbm.generate(n_samples=n_steps+1, n_paths=1)[0]
        fgn = np.diff(path)

        x = np.zeros(n_steps + 1)
        x[0] = theta
        for i in range(n_steps):
            x[i+1] = x[i] + kappa * (theta - x[i]) * dt + sigma * fgn[i]
        return x

    # Generate data pairs (X_t, X_{t+1})
    all_X = []
    all_Y = []

    for seed in range(n_paths):
        traj = generate_fou(seed)
        # Use states at each time step
        for i in range(len(traj) - 1):
            all_X.append(traj[i])
            all_Y.append(traj[i+1])

    X = np.array(all_X).reshape(-1, 1)
    Y = np.array(all_Y).reshape(-1, 1)

    print(f"\nTotal data pairs: {len(X)}")

    # Test with RBF kernel
    print("\n--- RBF Kernel ---")
    nk_rbf = NystromKoopman(n_landmarks=200, kernel='rbf', reg=1e-4)
    nk_rbf.fit(X, Y)

    print(f"Top 6 eigenvalues: {nk_rbf.eigenvalues[:6]}")
    print(f"Expected: λ=1 (constant), λ≈{np.exp(-kappa*dt):.4f} (linear)")

    # Check multiple eigenfunctions
    x_test = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    x_centered = x_test.flatten() - theta  # Center around equilibrium

    print("\nEigenfunction correlations with (x - θ):")
    for i in range(5):
        phi_i = nk_rbf.eigenfunction(x_test, idx=i)
        corr = np.corrcoef(x_centered, phi_i.real)[0, 1]
        # Also check if eigenfunction is approximately constant
        std = np.std(phi_i.real)
        print(f"  φ_{i}: λ={nk_rbf.eigenvalues[i].real:.4f}, corr={(corr):.4f}, std={std:.4f}")

    # The linear eigenfunction should have high correlation with x-θ
    # Find which eigenfunction is most correlated
    best_corr = 0
    best_idx = 0
    for i in range(min(10, len(nk_rbf.eigenvalues))):
        phi_i = nk_rbf.eigenfunction(x_test, idx=i)
        corr = abs(np.corrcoef(x_centered, phi_i.real)[0, 1])
        if corr > best_corr:
            best_corr = corr
            best_idx = i

    print(f"\nBest linear eigenfunction: φ_{best_idx}")
    print(f"  λ = {nk_rbf.eigenvalues[best_idx].real:.4f}")
    print(f"  |corr with x-θ| = {best_corr:.4f}")

    # Test with signature kernel (using short path windows)
    print("\n--- Signature Kernel ---")

    # Create path windows
    window_len = 10
    paths_X = []
    paths_Y = []

    for seed in range(n_paths):
        traj = generate_fou(seed)
        t = np.arange(len(traj)) * dt

        for i in range(window_len, len(traj) - 1):
            # Path window ending at time i
            path_X = np.column_stack([t[i-window_len:i+1] - t[i-window_len],
                                      traj[i-window_len:i+1]])
            # Path window ending at time i+1
            path_Y = np.column_stack([t[i-window_len+1:i+2] - t[i-window_len+1],
                                      traj[i-window_len+1:i+2]])
            paths_X.append(path_X)
            paths_Y.append(path_Y)

    print(f"Total path pairs: {len(paths_X)}")

    nk_sig = NystromKoopman(n_landmarks=200, kernel='signature',
                            sig_depth=3, reg=1e-4)
    nk_sig.fit(paths_X, paths_Y)

    print(f"Top 6 eigenvalues: {nk_sig.eigenvalues[:6]}")

    # Extract terminal state from each path for correlation check
    terminal_states = np.array([p[-1, 1] for p in paths_X[:2000]]) - theta

    print("\nEigenfunction correlations with terminal state - θ:")
    best_corr_sig = 0
    best_idx_sig = 0
    for i in range(5):
        phi_i = nk_sig.eigenfunction(paths_X[:2000], idx=i)
        corr = np.corrcoef(terminal_states, phi_i.real)[0, 1]
        std = np.std(phi_i.real)
        print(f"  φ_{i}: λ={nk_sig.eigenvalues[i].real:.4f}, corr={corr:.4f}, std={std:.4f}")
        if abs(corr) > best_corr_sig:
            best_corr_sig = abs(corr)
            best_idx_sig = i

    print(f"\nBest linear eigenfunction: φ_{best_idx_sig}")
    print(f"  λ = {nk_sig.eigenvalues[best_idx_sig].real:.4f}")
    print(f"  |corr with x-θ| = {best_corr_sig:.4f}")

    # Verify eigenvalue interpretation
    print("\n" + "=" * 70)
    print("EIGENVALUE INTERPRETATION")
    print("=" * 70)
    print(f"For OU: λ_linear = 1 - κ*dt = 1 - {kappa}*{dt} = {1 - kappa*dt:.4f}")
    print(f"Or: λ_linear = exp(-κ*dt) = {np.exp(-kappa*dt):.4f}")
    print(f"RBF found: λ = {nk_rbf.eigenvalues[1].real:.4f} (φ with corr={best_corr:.2f})")

    # Check conditioning
    print("\n--- Conditioning Analysis ---")
    K_mm_cond = np.linalg.cond(nk_rbf.K_mm)
    print(f"K_mm condition number: {K_mm_cond:.2e}")

    return nk_rbf, nk_sig


def test_bandwidth_sweep():
    """Sweep bandwidth to find optimal for eigenfunction recovery."""
    import sys
    sys.path.insert(0, '/home/ed/SynologyDrive/Documents/Research/PE_Research/rkhs_kronic/src')
    from rough_paths_generator import FractionalBrownianMotion

    np.random.seed(42)

    H, kappa, theta, sigma, dt = 0.3, 2.0, 0.5, 0.1, 0.01
    n_steps, n_paths = 1000, 30

    def generate_fou(seed):
        np.random.seed(seed)
        fbm = FractionalBrownianMotion(H=H, dt=dt, seed=seed)
        path = fbm.generate(n_samples=n_steps+1, n_paths=1)[0]
        fgn = np.diff(path)
        x = np.zeros(n_steps + 1)
        x[0] = theta
        for i in range(n_steps):
            x[i+1] = x[i] + kappa * (theta - x[i]) * dt + sigma * fgn[i]
        return x

    # Generate data
    all_X, all_Y = [], []
    for seed in range(n_paths):
        traj = generate_fou(seed)
        for i in range(len(traj) - 1):
            all_X.append(traj[i])
            all_Y.append(traj[i+1])
    X = np.array(all_X).reshape(-1, 1)
    Y = np.array(all_Y).reshape(-1, 1)

    x_test = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    x_centered = x_test.flatten() - theta

    print("=" * 70)
    print("Bandwidth Sweep for RBF Kernel")
    print("=" * 70)
    print(f"Expected λ_linear = {1 - kappa*dt:.4f}")
    print()

    bandwidths = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

    print(f"{'Bandwidth':>10s} {'λ_1':>10s} {'λ_2':>10s} {'Best Corr':>10s} {'Best λ':>10s}")
    print("-" * 55)

    for bw in bandwidths:
        nk = NystromKoopman(n_landmarks=150, kernel='rbf', bandwidth=bw, reg=1e-4)
        nk.fit(X, Y)

        # Find best eigenfunction
        best_corr, best_idx = 0, 0
        for i in range(min(5, len(nk.eigenvalues))):
            phi_i = nk.eigenfunction(x_test, idx=i)
            corr = abs(np.corrcoef(x_centered, phi_i.real)[0, 1])
            if corr > best_corr:
                best_corr, best_idx = corr, i

        print(f"{bw:>10.3f} {nk.eigenvalues[0].real:>10.4f} {nk.eigenvalues[1].real:>10.4f} "
              f"{best_corr:>10.4f} {nk.eigenvalues[best_idx].real:>10.4f}")


if __name__ == "__main__":
    nk_rbf, nk_sig = test_on_fou()
    print("\n")
    test_bandwidth_sweep()
