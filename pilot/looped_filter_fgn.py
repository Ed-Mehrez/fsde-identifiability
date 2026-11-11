"""Preston Rozwood, "Looped Reasoning as Closed-Loop Filtering" (draft v0), Prop. 4 with
long-memory candidate noise.

Prop. 4 gives the steady-state variance of a damped loop at a solution, along a real block
mode mu, for white candidate noise: V(gain) ~ gain at small gains. Here the candidate noise
is unit fGn(H) instead; the variance is computed exactly as gain^2 * w^T T w with
w_j = a^j, a = 1 - gain (1 - mu), and T the fGn autocovariance. Prediction:
V ~ gain^(2 - 2H) at small gains.

    python pilot/looped_filter_fgn.py
"""
# Generalise Preston's Prop. 4 to fGn(H) candidate noise (exact, no simulation).
# Scalar mode: delta_t = a delta_{t-1} + g xi_t, a = 1 - g (1 - mu), xi ~ unit fGn(H).
# Stationary variance V(g) = g^2 * w' T w,  w_j = a^j,  T Toeplitz fGn autocovariance.
import numpy as np, sys
sys.path.insert(0, '.')
from fsde import fgn_autocov
from scipy.linalg import toeplitz
mu = -1.8; c = 1 - mu            # Preston's toy mode, stability needs g < 2/c = 0.714
gs = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6])
print("V(gain) / V(0.2) for candidate noise fGn(H); local slope d log V / d log g at small gains")
for H in [0.3, 0.5, 0.7, 0.9]:
    V = []
    for g in gs:
        a = 1 - g * c
        J = int(min(20000, np.ceil(np.log(1e-10) / np.log(abs(a)))))
        w = a ** np.arange(J)
        T = toeplitz(fgn_autocov(J, H))
        V.append(g * g * w @ T @ w)
    V = np.array(V)
    sl = np.polyfit(np.log(gs[:3]), np.log(V[:3]), 1)[0]
    print(f"H={H}: " + "  ".join(f"g={g:.2f}:{v / V[gs == 0.2][0]:.3f}" for g, v in zip(gs, V)) +
          f"   slope {sl:.2f} (theory 2-2H = {2 - 2 * H:.2f})")
