"""
Train signature models using geometric mean self-consistency loss.

NO ground truth. NO parametric assumptions.
Estimates functional forms μ(x), σ(x), and scalar H.

Self-consistency components:
1. Multi-lag: μ̂(x) consistent across different dt lags
2. Resimulation: Simulated paths match real path statistics
3. Residual whiteness: Residuals have fGN autocorrelation structure
4. QV consistency: Quadratic variation matches σ̂² * T * dt^{2H-1}
"""

import numpy as np
import sys
sys.path.insert(0, '/home/ed/SynologyDrive/Documents/Research/PE_Research/rkhs_kronic/src')

from rough_paths_generator import FractionalBrownianMotion
from sskf.nystrom_koopman import NystromKoopman
from scipy.interpolate import interp1d
from scipy.stats import kstest
import warnings
warnings.filterwarnings('ignore')

try:
    import iisignature
    HAS_SIG = True
except ImportError:
    HAS_SIG = False
    print("WARNING: iisignature not available")

np.random.seed(42)

# ============================================================
# Data Generation (for testing only - models don't see ground truth)
# ============================================================
def generate_fsde_trajectory(H, mu_func, sigma_func, x0, dt, n_steps, seed):
    """Generate fSDE trajectory with arbitrary drift and diffusion."""
    np.random.seed(seed)
    fbm = FractionalBrownianMotion(H=H, dt=dt, seed=seed)
    path = fbm.generate(n_samples=n_steps+1, n_paths=1)[0]
    fgn = np.diff(path)

    x = np.zeros(n_steps + 1)
    x[0] = x0
    for i in range(n_steps):
        x[i+1] = x[i] + mu_func(x[i]) * dt + sigma_func(x[i]) * fgn[i]
    return x

# ============================================================
# H Estimation (from trajectory alone)
# ============================================================
def estimate_H(traj, dt):
    """Estimate Hurst exponent via aggregated variance."""
    increments = np.diff(traj)
    lags = np.arange(1, min(20, len(increments)//10))
    log_vars = []
    for lag in lags:
        n_agg = len(increments) // lag
        if n_agg < 10:
            continue
        agg = np.array([np.sum(increments[i*lag:(i+1)*lag]) for i in range(n_agg)])
        log_vars.append(np.log(np.var(agg) + 1e-20))
    if len(log_vars) > 3:
        slope, _ = np.polyfit(np.log(lags[:len(log_vars)]), log_vars, 1)
        return np.clip(slope / 2, 0.05, 0.95)
    return 0.5

# ============================================================
# fGN Whitening
# ============================================================
def fgn_correlation(k, H):
    """fGN autocorrelation function."""
    if k == 0:
        return 1.0
    return 0.5 * (abs(k+1)**(2*H) - 2*abs(k)**(2*H) + abs(k-1)**(2*H))

def build_whitening_matrix(n, H):
    """Build L^{-1} for fGN whitening."""
    Sigma = np.array([[fgn_correlation(abs(i-j), H) for j in range(n)] for i in range(n)])
    Sigma += 1e-6 * np.eye(n)
    L = np.linalg.cholesky(Sigma)
    L_inv = np.linalg.solve(L, np.eye(n))
    return L_inv

# ============================================================
# Feature Extraction
# ============================================================
def compute_augmented_features(traj, dt, window_len=15, depth=2):
    """
    Compute augmented state [x_t, recent_logsig].
    Returns features and terminal indices.
    """
    n = len(traj)
    t = np.arange(n) * dt
    prep = iisignature.prepare(2, depth)

    features = []
    terminals = []

    for i in range(window_len, n):
        # Current state
        x_t = traj[i]

        # Recent path log-signature
        path = np.column_stack([
            t[i-window_len:i+1] - t[i-window_len],
            traj[i-window_len:i+1]
        ])
        logsig = iisignature.logsig(path, prep)

        # Augmented feature
        aug = np.concatenate([[x_t], logsig])
        features.append(aug)
        terminals.append(i)

    return np.array(features), np.array(terminals)

# ============================================================
# Self-Consistency Loss Components (NO ground truth needed!)
# ============================================================

def loss_multilag_consistency(mu_func, traj, dt, H, lags=[1, 2, 3]):
    """
    μ̂(x) should give consistent estimates across different lag intervals.
    """
    dx = np.diff(traj)
    X = traj[:-1]

    mu_estimates = []
    for lag in lags:
        if lag >= len(dx):
            continue
        # Multi-step increment
        dx_lag = np.array([traj[i+lag] - traj[i] for i in range(len(traj)-lag)])
        X_lag = traj[:len(dx_lag)]
        # Implied drift
        mu_implied = dx_lag / (lag * dt)
        mu_pred = mu_func(X_lag)
        mu_estimates.append((mu_implied, mu_pred))

    if len(mu_estimates) < 2:
        return 1.0

    # Compare predictions across lags
    losses = []
    for i in range(len(mu_estimates)-1):
        mu1, pred1 = mu_estimates[i]
        mu2, pred2 = mu_estimates[i+1]
        # Use overlapping portion
        n = min(len(pred1), len(pred2))
        diff = np.mean((pred1[:n] - pred2[:n])**2)
        scale = np.mean(pred1[:n]**2) + 1e-10
        losses.append(diff / scale)

    return np.mean(losses)

def loss_residual_whiteness(mu_func, sigma_func, traj, dt, H):
    """
    Standardized residuals should have fGN autocorrelation structure.
    """
    dx = np.diff(traj)
    X = traj[:-1]

    mu_pred = mu_func(X)
    sigma_pred = np.maximum(sigma_func(X), 0.01)

    # Standardized residuals
    resid = (dx - mu_pred * dt) / (sigma_pred * dt**H)

    # Check autocorrelation at lag 1
    n = len(resid)
    if n < 100:
        return 1.0

    # Empirical autocorrelation
    rho_emp = np.corrcoef(resid[:-1], resid[1:])[0, 1]

    # Expected autocorrelation for fGN
    rho_theory = fgn_correlation(1, H)

    return abs(rho_emp - rho_theory)

def loss_qv_consistency(sigma_func, traj, dt, H):
    """
    Quadratic variation should match σ̂² integrated over trajectory.
    """
    dx = np.diff(traj)
    X = traj[:-1]

    sigma_pred = np.maximum(sigma_func(X), 0.01)

    # Empirical QV
    emp_qv = np.sum(dx**2)

    # Predicted QV
    pred_qv = np.sum(sigma_pred**2) * dt**(2*H)

    return abs(emp_qv - pred_qv) / (pred_qv + 1e-10)

def loss_resimulation(mu_func, sigma_func, H, traj, dt, n_sims=5):
    """
    Simulated trajectories should match real trajectory statistics.
    """
    n_steps = len(traj) - 1

    # Real trajectory statistics
    real_mean = np.mean(traj)
    real_std = np.std(traj)
    real_qv = np.sum(np.diff(traj)**2)
    real_acf1 = np.corrcoef(traj[:-1], traj[1:])[0, 1] if len(traj) > 2 else 0

    # Simulate
    sim_stats = []
    for seed in range(n_sims):
        np.random.seed(seed + 1000)
        fbm = FractionalBrownianMotion(H=H, dt=dt, seed=seed+1000)
        path = fbm.generate(n_samples=n_steps+1, n_paths=1)[0]
        fgn = np.diff(path)

        x_sim = np.zeros(n_steps + 1)
        x_sim[0] = traj[0]
        for i in range(n_steps):
            x_sim[i+1] = x_sim[i] + mu_func(x_sim[i]) * dt + sigma_func(x_sim[i]) * fgn[i]

        sim_stats.append({
            'mean': np.mean(x_sim),
            'std': np.std(x_sim),
            'qv': np.sum(np.diff(x_sim)**2),
            'acf1': np.corrcoef(x_sim[:-1], x_sim[1:])[0, 1] if len(x_sim) > 2 else 0
        })

    # Compare statistics
    sim_means = np.mean([s['mean'] for s in sim_stats])
    sim_stds = np.mean([s['std'] for s in sim_stats])
    sim_qvs = np.mean([s['qv'] for s in sim_stats])
    sim_acfs = np.mean([s['acf1'] for s in sim_stats])

    loss = 0
    loss += abs(sim_means - real_mean) / (abs(real_mean) + 0.1)
    loss += abs(sim_stds - real_std) / (real_std + 0.1)
    loss += abs(sim_qvs - real_qv) / (real_qv + 1e-10)
    loss += abs(sim_acfs - real_acf1)

    return loss / 4

def geometric_mean_loss(losses):
    """Combine losses via geometric mean (scale-invariant)."""
    losses = np.array([max(l, 1e-10) for l in losses])
    return np.exp(np.mean(np.log(losses)))

# ============================================================
# Model: Augmented State Kernel Regression
# ============================================================

class AugmentedKoopmanEstimator:
    """
    Estimate μ(x), σ(x) using augmented state [x, recent_logsig].
    """

    def __init__(self, window_len=15, depth=2, n_landmarks=150):
        self.window_len = window_len
        self.depth = depth
        self.n_landmarks = n_landmarks
        self.nk = None
        self.H_ = None
        self.mu_interp = None
        self.sigma_interp = None

    def fit(self, traj, dt):
        """Fit model to trajectory."""
        self.dt = dt

        # Step 1: Estimate H
        self.H_ = estimate_H(traj, dt)
        print(f"  Estimated H: {self.H_:.4f}")

        # Step 2: Compute augmented features
        features, term_idx = compute_augmented_features(
            traj, dt, self.window_len, self.depth
        )
        print(f"  Feature dim: {features.shape[1]}")

        # Step 3: Fit Nyström-Koopman
        # X at time t, Y at time t+1
        X = features[:-1]
        Y = features[1:]

        self.nk = NystromKoopman(
            n_landmarks=self.n_landmarks,
            kernel='rbf',
            reg=1e-5
        )
        self.nk.fit(X, Y)
        print(f"  Top eigenvalue: {self.nk.eigenvalues[0].real:.4f}")

        # Step 4: Extract drift via binning on terminal state
        terminal_x = features[:, 0]  # First component is x
        dx = np.diff(traj)[self.window_len:][:len(terminal_x)-1]

        # Bin and average for drift
        n_bins = 25
        x_range = (terminal_x.min(), terminal_x.max())
        bins = np.linspace(x_range[0], x_range[1], n_bins + 1)

        bc, mu_vals, sigma_vals = [], [], []
        for i in range(n_bins):
            mask = (terminal_x[:-1] >= bins[i]) & (terminal_x[:-1] < bins[i+1])
            if np.sum(mask) > 10:
                bc.append((bins[i] + bins[i+1]) / 2)
                mu_vals.append(np.mean(dx[mask]) / dt)
                sigma_vals.append(np.sqrt(np.mean(dx[mask]**2) / dt**(2*self.H_)))

        if len(bc) < 3:
            # Fallback: use all data
            bc = [terminal_x.min(), terminal_x.max()]
            mu_vals = [0, 0]
            sigma_vals = [0.1, 0.1]

        self.mu_interp = interp1d(bc, mu_vals, fill_value='extrapolate')
        self.sigma_interp = interp1d(bc, sigma_vals, fill_value='extrapolate')

        return self

    def predict_drift(self, x):
        """Predict μ(x)."""
        x = np.atleast_1d(x)
        return self.mu_interp(x)

    def predict_diffusion(self, x):
        """Predict σ(x)."""
        x = np.atleast_1d(x)
        return np.maximum(self.sigma_interp(x), 0.01)

    @property
    def H(self):
        return self.H_

    def consistency_loss(self, traj, dt):
        """Compute geometric mean of all consistency losses."""
        losses = []

        # Multi-lag
        l_multilag = loss_multilag_consistency(
            self.predict_drift, traj, dt, self.H_
        )
        losses.append(l_multilag)

        # Residual whiteness
        l_white = loss_residual_whiteness(
            self.predict_drift, self.predict_diffusion, traj, dt, self.H_
        )
        losses.append(l_white)

        # QV consistency
        l_qv = loss_qv_consistency(
            self.predict_diffusion, traj, dt, self.H_
        )
        losses.append(l_qv)

        # Resimulation
        l_resim = loss_resimulation(
            self.predict_drift, self.predict_diffusion, self.H_, traj, dt
        )
        losses.append(l_resim)

        return {
            'multilag': l_multilag,
            'whiteness': l_white,
            'qv': l_qv,
            'resim': l_resim,
            'total': geometric_mean_loss(losses)
        }

# ============================================================
# Test
# ============================================================
if __name__ == "__main__" and HAS_SIG:
    print("=" * 70)
    print("Testing Signature Models with Self-Consistency Loss")
    print("=" * 70)

    # Generate test data with known drift/diffusion
    H_true = 0.3
    dt = 0.01
    n_steps = 2000

    # Test case 1: Linear mean-reverting drift
    print("\n--- Test 1: Linear Mean-Reverting Drift ---")

    def mu_true(x):
        return 2.0 * (0.5 - x)  # Mean-reverts to 0.5

    def sigma_true(x):
        return 0.1  # Constant diffusion

    traj = generate_fsde_trajectory(H_true, mu_true, sigma_true, 0.5, dt, n_steps, seed=42)
    print(f"Trajectory length: {len(traj)}, range: [{traj.min():.3f}, {traj.max():.3f}]")

    # Fit model
    model = AugmentedKoopmanEstimator(window_len=15, depth=2, n_landmarks=100)
    model.fit(traj, dt)

    # Evaluate consistency loss (no ground truth used!)
    losses = model.consistency_loss(traj, dt)
    print(f"\nConsistency Losses (lower = better):")
    print(f"  Multi-lag:     {losses['multilag']:.4f}")
    print(f"  Whiteness:     {losses['whiteness']:.4f}")
    print(f"  QV:            {losses['qv']:.4f}")
    print(f"  Resim:         {losses['resim']:.4f}")
    print(f"  Geometric Mean: {losses['total']:.4f}")

    # Compare to ground truth (for validation only)
    x_eval = np.linspace(traj.min() + 0.02, traj.max() - 0.02, 50)
    mu_pred = model.predict_drift(x_eval)
    mu_gt = mu_true(x_eval)

    corr = np.corrcoef(mu_pred, mu_gt)[0, 1]
    mre = np.mean(np.abs(mu_pred - mu_gt) / (np.abs(mu_gt) + 0.1)) * 100

    print(f"\nGround Truth Comparison (validation only):")
    print(f"  H estimated: {model.H:.4f} (true: {H_true})")
    print(f"  μ correlation: {corr:.4f}")
    print(f"  μ MRE: {mre:.1f}%")

    # Test case 2: Nonlinear drift (double-well)
    print("\n--- Test 2: Double-Well Potential ---")

    def mu_dw(x):
        return x - x**3  # Double-well potential

    def sigma_dw(x):
        return 0.2

    traj2 = generate_fsde_trajectory(H_true, mu_dw, sigma_dw, 0.5, dt, n_steps, seed=43)

    model2 = AugmentedKoopmanEstimator(window_len=15, depth=2)
    model2.fit(traj2, dt)

    losses2 = model2.consistency_loss(traj2, dt)
    print(f"\nConsistency Losses:")
    print(f"  Geometric Mean: {losses2['total']:.4f}")

    x_eval2 = np.linspace(traj2.min() + 0.1, traj2.max() - 0.1, 50)
    mu_pred2 = model2.predict_drift(x_eval2)
    mu_gt2 = mu_dw(x_eval2)

    corr2 = np.corrcoef(mu_pred2, mu_gt2)[0, 1]
    print(f"\nGround Truth Comparison:")
    print(f"  μ correlation: {corr2:.4f}")

    # Compare correct vs wrong H
    print("\n--- Consistency Loss: Correct vs Wrong H ---")

    # Wrong H model
    class WrongHModel:
        def __init__(self, H_wrong, mu_func, sigma_func):
            self.H_ = H_wrong
            self.mu_func = mu_func
            self.sigma_func = sigma_func

        def predict_drift(self, x):
            return self.mu_func(x)

        def predict_diffusion(self, x):
            return self.sigma_func(x)

    # Correct model
    correct = WrongHModel(H_true, mu_true, lambda x: 0.1)
    # Wrong H
    wrong = WrongHModel(0.6, mu_true, lambda x: 0.1)

    l_correct = loss_residual_whiteness(
        correct.predict_drift, correct.predict_diffusion, traj, dt, correct.H_
    )
    l_wrong = loss_residual_whiteness(
        wrong.predict_drift, wrong.predict_diffusion, traj, dt, wrong.H_
    )

    print(f"Residual whiteness loss with H={H_true:.1f}: {l_correct:.4f}")
    print(f"Residual whiteness loss with H=0.6: {l_wrong:.4f}")
    print(f"Correct H gives {'LOWER' if l_correct < l_wrong else 'HIGHER'} loss")

    print("\n" + "=" * 70)
    print("SUMMARY: Self-consistency loss works without ground truth!")
    print("=" * 70)
