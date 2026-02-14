import torch
import numpy as np
import sigkernel

class NystromFeatures:
    """
    Approximates the Infinite Signature Kernel using Nystrom method.
    Maps a path X to a finite-dimensional vector psi(X) such that:
       <psi(X), psi(Y)> approx K(X, Y)
       
    This effectively performs a Tensor Decomposition of the feature space
    onto the subspace spanned by the landmarks.
    """
    def __init__(self, landmarks, dyadic_order=1, rank=None, static_kernel='linear', sigma=1.0):
        """
        Args:
            landmarks (torch.Tensor): (N_landmarks, Length, Dim)
            dyadic_order (int): Precision of SigKernel PDE solver.
            rank (int): Target rank (dimension of output features). If None, uses N_landmarks.
            static_kernel (str): 'linear' or 'rbf'.
            sigma (float): RBF width (irrelevant for linear).
        """
        self.landmarks = landmarks
        self.rank = rank if rank is not None else len(landmarks)
        
        # Initialize Kernel
        if static_kernel == 'linear':
            static = sigkernel.LinearKernel()
        elif static_kernel == 'rbf':
            static = sigkernel.RBFKernel(sigma=sigma)
        else:
            raise ValueError(f"Unknown static kernel: {static_kernel}")
            
        self.sk = sigkernel.SigKernel(static, dyadic_order=dyadic_order)
        
        # Fit Nystrom Map
        self._fit()
        
    def _fit(self):
        print(f"Fitting Nystrom Features on {len(self.landmarks)} landmarks...")
        # Compute K_ZZ (N x N)
        # Using GPU if available?
        # For safety/compatibility, stick to CPU or check device.
        device = self.landmarks.device
        
        # Compute Gram Matrix
        # SigKernel takes (N, L, D) -> (N, N)
        K_zz = self.sk.compute_Gram(self.landmarks, self.landmarks, sym=True)
        
        # Eigendecomposition
        # K_zz = U S U^T
        # We want Map: x -> K_xZ @ U @ S^{-1/2}
        # SVD is robust.
        
        K_zz_np = K_zz.cpu().detach().numpy()
        # Add slight jitter for stability
        K_zz_np += 1e-6 * np.eye(len(K_zz_np))
        
        U, S, Vh = np.linalg.svd(K_zz_np)
        
        # Truncate to Rank
        if self.rank < len(S):
            U = U[:, :self.rank]
            S = S[:self.rank]
            
        self.U = torch.tensor(U, dtype=torch.float64).to(device)
        self.S_inv_sqrt = torch.tensor(np.diag(1.0 / np.sqrt(S)), dtype=torch.float64).to(device)
        
        # Precompute the Projector: M = U @ S^{-1/2}
        self.Projector = self.U @ self.S_inv_sqrt
        
        print(f"Nystrom Fit Complete. Effective Rank: {len(S)}")
        
    def transform(self, X):
        """
        Args:
            X (torch.Tensor): (Batch, Length, Dim)
        Returns:
            Features (torch.Tensor): (Batch, Rank)
        """
        # Compute K_xZ (Batch x N_landmarks)
        K_xz = self.sk.compute_Gram(X, self.landmarks, sym=False)
        
        # Project
        # Psi = K_xz @ Projector
        Psi = K_xz @ self.Projector
        return Psi

if __name__ == "__main__":
    # Simple Test
    print("Testing NystromFeatures...")
    # Generate random paths
    paths = torch.randn(50, 20, 2) # 50 paths, length 20, 2D
    landmarks = paths[:10]
    
    nf = NystromFeatures(landmarks, rank=5, static_kernel='linear')
    feats = nf.transform(paths)
    print(f"Input Shape: {paths.shape}")
    print(f"Output Shape: {feats.shape}")
    
    # Check Kernel Approximation
    # <Psi(x), Psi(y)> vs K(x,y)
    K_true = nf.sk.compute_Gram(paths[:5], paths[:5], sym=True)
    K_approx = feats[:5] @ feats[:5].T
    
    diff = torch.norm(K_true - K_approx) / torch.norm(K_true)
    print(f"Approximation Error (Rel Norm): {diff.item():.4f}")
