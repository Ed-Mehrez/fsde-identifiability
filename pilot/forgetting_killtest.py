"""Direction 2 kill test: how much does an H-aware forgetting kernel beat a tuned EMA?

Target theta_t = fBm(H) (unit-step increment sd 1), observed as y_t = theta_t + eps_t,
eps ~ N(0, sigma^2) white. Estimate theta_T from y_1..y_T (causal filtering), T = 1000.
All errors are exact (Gaussian covariances), not simulated.

Compared:
  ema_best   - single EMA, decay tuned to minimise MSE (grid over effective window)
  bank_K     - optimal linear combination of K EMAs with log-spaced windows
               (1, 4, 16, 64, ...), i.e. a K-factor Markov lift
  optimal    - exact best linear causal filter E[theta_T | y_1..y_T]

Reports MSE(ema_best) / MSE(optimal) and MSE(bank_K) / MSE(optimal).

    python pilot/forgetting_killtest.py
"""

import numpy as np
import pandas as pd

T = 1000


def fbm_cov(H, T):
    t = np.arange(1, T + 1, dtype=float)
    return 0.5 * (t[:, None] ** (2 * H) + t[None, :] ** (2 * H) - np.abs(t[:, None] - t[None, :]) ** (2 * H))


def ema_weights(window, T):
    beta = 1 - 1 / window
    w = (1 - beta) * beta ** np.arange(T)  # weight on y_{T-j}
    w = w / w.sum()
    return w[::-1]  # aligned with y_1..y_T


def mse_linear(w, K, sigma):
    """MSE of theta_T - w @ y for weights w over y_1..y_T."""
    # Var(theta_T) - 2 w.Cov(y, theta_T) + w' Cov(y) w
    return K[-1, -1] - 2 * w @ K[:, -1] + w @ K @ w + sigma ** 2 * w @ w


def best_combo(Ws, K, sigma):
    """Optimal linear combination of basis weight vectors (rows of Ws), free coefficients."""
    G = Ws @ K @ Ws.T + sigma ** 2 * Ws @ Ws.T
    b = Ws @ K[:, -1]
    c = np.linalg.solve(G + 1e-12 * np.eye(len(G)), b)
    return mse_linear(c @ Ws, K, sigma)


rows = []
windows = np.unique(np.round(np.logspace(0, np.log10(500), 200)).astype(int))
for H in [0.2, 0.3, 0.5, 0.7, 0.8, 0.9]:
    K = fbm_cov(H, T)
    for sigma in [0.3, 1.0, 3.0, 10.0, 30.0]:
        Sy = K + sigma ** 2 * np.eye(T)
        k = K[:, -1]
        opt = K[-1, -1] - k @ np.linalg.solve(Sy, k)
        ema = min(mse_linear(ema_weights(w, T), K, sigma) for w in windows)
        r = dict(H=H, sigma=sigma, mse_opt=opt, ema_best=ema / opt)
        for nb in [2, 3, 4]:
            Ws = np.stack([ema_weights(4 ** j, T) for j in range(nb)])
            r[f"bank_{nb}"] = best_combo(Ws, K, sigma) / opt
        rows.append(r)
        print(f"H={H} sigma={sigma:>4}: EMA/opt {r['ema_best']:.3f}  bank2 {r['bank_2']:.3f}  "
              f"bank3 {r['bank_3']:.3f}  bank4 {r['bank_4']:.3f}", flush=True)

df = pd.DataFrame(rows)
print("\nMSE of best single EMA / MSE of optimal filter")
print(df.pivot(index="H", columns="sigma", values="ema_best").round(3))
print("\nMSE of optimal 3-EMA bank (windows 1, 4, 16) / optimal")
print(df.pivot(index="H", columns="sigma", values="bank_3").round(3))


# ---- fairness check: tuned two-EMA combination (Holt / double-EMA family) ----
print("\nMSE of best tuned 2-EMA combination (free windows and coefficients) / optimal")
grid = np.unique(np.round(np.logspace(0, np.log10(500), 20)).astype(int))
rows2 = []
for H in [0.2, 0.5, 0.7, 0.8, 0.9]:
    K = fbm_cov(H, T)
    E = np.stack([ema_weights(w, T) for w in grid])
    EK = E @ K
    for sigma in [1.0, 3.0, 10.0]:
        k = K[:, -1]
        opt = K[-1, -1] - k @ np.linalg.solve(K + sigma ** 2 * np.eye(T), k)
        G_all = EK @ E.T + sigma ** 2 * E @ E.T
        b_all = E @ k
        best = np.inf
        for i in range(len(grid)):
            for j in range(i + 1, len(grid)):
                G = G_all[np.ix_([i, j], [i, j])]
                b = b_all[[i, j]]
                c = np.linalg.solve(G + 1e-12 * np.eye(2), b)
                best = min(best, K[-1, -1] - b @ c)
        rows2.append(dict(H=H, sigma=sigma, two_ema=best / opt))
print(pd.DataFrame(rows2).pivot(index="H", columns="sigma", values="two_ema").round(3))
