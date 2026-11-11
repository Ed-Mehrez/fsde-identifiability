"""Direction 1 test: forecasting temporal totals of long-memory rate series.

Pre-registered design (fixed before the first run, 2026-11-11)
---------------------------------------------------------------
Data: fGn "rate" series (think daily inflow or load), H in {0.3, 0.5, 0.7, 0.8, 0.9},
300 series per H, context 512, horizon 64, random scale and offset per series.

Target: the total S_n = sum of the next n values, n in {8, 16, 32, 64}.

Forecast models give per-horizon quantiles (marginals). Rules to turn them into a
forecast of S_n, all using the context only:
  indep       - per-horizon Gaussian (median, sd from the 80% width), independent
  comonotone  - sum of same-level quantiles (perfect dependence)
  fgn_copula  - per-horizon median and sd from the model, correlation = the
                conditional correlation of fGn(H_hat) future values given 512 past
                values, H_hat estimated from the context with fsde
Reference forecasts:
  fsde_plugin - Gaussian conditional forecast of fGn(H_hat) with sample mean and sd
                (no foundation model at all)
  oracle      - exact conditional law of S_n given the context

"Marginal sources": chronos-2, timesfm-2.5, and the oracle's own per-horizon marginals
(isolates the error due to the dependence assumption alone).

Added after the first run (2026-11-11), prompted by prior art, not pre-registered:
  ar1_copula  - per-horizon median and sd from the model, AR(1) correlation rho^|i-j|
                with rho = lag-1 autocorrelation of the context. This is the copula
                used by existing work on foundation-model marginals (Baron et al. 2025,
                arXiv 2510.02224; Choi et al. 2026, arXiv 2609.25980).

Primary outcome: at n = 64, coverage of the 80% interval for S_n and pinball loss of
S_n relative to the oracle. Hypotheses: indep under-covers increasingly with H > 0.5
and over-covers at H = 0.3; comonotone over-covers; fgn_copula is closest to nominal
and has the lowest pinball loss of the three combination rules.

    HF_HUB_DISABLE_XET=1 python pilot/aggregate_fgn.py --out results/aggregate_fgn
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import fsde  # noqa: E402
from pilot.gate1_fbm import MODELS, Q, Z, pinball  # noqa: E402

Z80 = Z[-1]  # norm.ppf(0.9)
NS = [8, 16, 32, 64]
H_GRID = np.round(np.arange(0.02, 0.99, 0.01), 2)


def fgn_conditional(H, L, h):
    """A (h x L) and C (h x h): future | past for unit fGn with known zero mean."""
    g = fsde.fgn_autocov(L + h, H)
    idx = np.abs(np.arange(L + h)[:, None] - np.arange(L + h)[None, :])
    G = g[idx]
    Gcc, Gfc, Gff = G[:L, :L], G[L:, :L], G[L:, L:]
    A = np.linalg.solve(Gcc, Gfc.T).T
    return A, Gff - A @ Gfc.T


def gaussian_q(mean, sd):
    return mean[..., None] + sd[..., None] * Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-series", type=int, default=300)
    ap.add_argument("--H", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.8, 0.9])
    ap.add_argument("--models", nargs="+", default=["chronos-2", "timesfm-2.5"])
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--out", default="results/aggregate_fgn")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(os.cpu_count() or 4)
    os.makedirs(args.out, exist_ok=True)
    L, h, N = args.context, args.horizon, args.n_series
    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    grid = {H: fgn_conditional(H, L, h) for H in H_GRID}
    print(f"conditional-law grid: {time.time() - t0:.0f}s", flush=True)

    # ---- data, oracle, H_hat ------------------------------------------------
    D = []
    for H in args.H:
        x = fsde.sample_fgn(L + h, H, size=N, rng=rng)
        scale = 10 ** rng.uniform(-1, 1, N)
        offset = rng.normal(0, 3, N) * scale
        A, C = fgn_conditional(H, L, h)
        ctx_raw, fut_raw = x[:, :L], x[:, L:]
        cm = ctx_raw - ctx_raw.mean(1, keepdims=True)
        H_hat = np.array([fsde.hurst_second_diff(np.cumsum(c)) for c in cm])
        D.append(dict(H=H, ctx=offset[:, None] + scale[:, None] * ctx_raw,
                      y=offset[:, None] + scale[:, None] * fut_raw,
                      o_mean=offset[:, None] + scale[:, None] * (ctx_raw @ A.T),
                      o_cov=C, scale=scale, H_hat=H_hat))
        print(f"H={H}: H_hat {H_hat.mean():.3f} ± {H_hat.std():.3f}", flush=True)

    # ---- model marginals ----------------------------------------------------
    marg = {}
    for name in args.models:
        t1 = time.time()
        model = MODELS[name](L, h)
        marg[name] = [np.sort(model(d["ctx"], h), axis=-1) for d in D]
        del model
        print(f"[{name}] {time.time() - t1:.0f}s", flush=True)
    # oracle's own marginals (exact per-horizon Gaussian quantiles)
    marg["oracle-marginals"] = [gaussian_q(d["o_mean"], d["scale"][:, None] * np.sqrt(np.diag(d["o_cov"]))[None, :])
                                for d in D]

    # ---- aggregate forecasts and scores --------------------------------------
    rows = []
    for di, d in enumerate(D):
        y, H = d["y"], d["H"]
        Hg = H_GRID[np.clip(np.round((np.clip(d["H_hat"], 0.02, 0.98) - 0.02) / 0.01).astype(int), 0, len(H_GRID) - 1)]
        for n in NS:
            S = y[:, :n].sum(1)
            one = np.ones(n)
            # oracle
            o_mean = d["o_mean"][:, :n].sum(1)
            o_sd = d["scale"] * np.sqrt(one @ d["o_cov"][:n, :n] @ one)
            q_or = gaussian_q(o_mean, o_sd)
            pb_or = pinball(q_or, S)
            forecasts = {("oracle", "oracle"): q_or}
            # fsde plug-in (no foundation model)
            ctx = d["ctx"]
            mu, sd = ctx.mean(1), ctx.std(1)
            pm = np.empty(N); ps = np.empty(N)
            for i in range(N):
                A_, C_ = grid[Hg[i]]
                pm[i] = mu[i] * n + sd[i] * (A_[:n] @ ((ctx[i] - mu[i]) / sd[i])).sum()
                ps[i] = sd[i] * np.sqrt(one @ C_[:n, :n] @ one)
            forecasts[("fsde_plugin", "fsde_plugin")] = gaussian_q(pm, ps)
            # combination rules on each marginal source
            for src, qs in marg.items():
                q = qs[di][:, :n]
                med = q[..., 4]
                s_k = (q[..., -1] - q[..., 0]) / (2 * Z80)
                forecasts[(src, "indep")] = gaussian_q(med.sum(1), np.sqrt((s_k ** 2).sum(1)))
                forecasts[(src, "comonotone")] = q.sum(1)
                cop_sd = np.empty(N)
                for i in range(N):
                    C_ = grid[Hg[i]][1][:n, :n]
                    R = C_ / np.sqrt(np.outer(np.diag(C_), np.diag(C_)))
                    cop_sd[i] = np.sqrt(s_k[i] @ R @ s_k[i])
                forecasts[(src, "fgn_copula")] = gaussian_q(med.sum(1), cop_sd)
                ar_sd = np.empty(N)
                lag = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
                for i in range(N):
                    c = d["ctx"][i]
                    rho = np.clip(np.corrcoef(c[:-1], c[1:])[0, 1], -0.99, 0.99)
                    ar_sd[i] = np.sqrt(s_k[i] @ (rho ** lag) @ s_k[i])
                forecasts[(src, "ar1_copula")] = gaussian_q(med.sum(1), ar_sd)
            for (src, rule), qf in forecasts.items():
                qf = np.sort(qf, axis=-1)
                rows.append(dict(H=H, n=n, source=src, rule=rule,
                                 coverage80=float(((qf[:, 0] <= S) & (S <= qf[:, -1])).mean()),
                                 lower=float((S < qf[:, 0]).mean()), upper=float((S > qf[:, -1]).mean()),
                                 width_vs_oracle=float(np.median((qf[:, -1] - qf[:, 0]) / (q_or[:, -1] - q_or[:, 0]))),
                                 loss_vs_oracle=float(pinball(qf, S).sum() / pb_or.sum())))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(args.out, "results.csv"), index=False)
    prim = R[R.n == 64]
    print("\n== n = 64: coverage of 80% interval for the total (target 0.80)")
    print(prim.pivot_table(index=["source", "rule"], columns="H", values="coverage80").round(3).to_string())
    print("\n== n = 64: pinball loss of the total / oracle")
    print(prim.pivot_table(index=["source", "rule"], columns="H", values="loss_vs_oracle").round(3).to_string())
    print("\n== n = 64: median interval width / oracle")
    print(prim.pivot_table(index=["source", "rule"], columns="H", values="width_vs_oracle").round(2).to_string())


if __name__ == "__main__":
    main()
