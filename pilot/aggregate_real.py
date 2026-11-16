"""Direction 1 on real data: forecasting 64-day totals of long-memory rate series.

PRE-REGISTERED DESIGN (committed before the first run, 2026-11-16)
------------------------------------------------------------------
All forecast targets fall in 2026, after every model's weights were committed
(Chronos-2 2025-10-30, TimesFM-2.5 2025-10-01). Weekly origins from 2026-01-05,
context 512 days, horizon h = 64 days.

Domains (rate series x_t; target = the total S = sum of the next 64 values):
  crypto_rv        daily realised variance of 172 Binance perpetuals (model input: log RV;
                   marginal quantiles mapped back with exp). Total = 64-day integrated variance.
  entsoe_load      daily mean load (MW) of ENTSO-E bidding zones. Total = energy.
  entsoe_price     daily mean day-ahead price (EUR/MWh). Total = 64 x baseload average price.
  reservoir_inflow daily change in stored volume (ML/day) of Australian reservoirs.
                   Total = change in storage over 64 days.
  crypto_ret       daily log returns of the same 172 perpetuals (CONTROL, H ~ 1/2).
                   Total = 64-day log return.

Per-horizon marginals: Chronos-2 and TimesFM-2.5, 9 quantiles (0.1..0.9).
Combination rules (label-free, context only):
  indep        Gaussian copula with identity correlation
  ar1_copula   Gaussian copula, correlation rho^|i-j|, rho = lag-1 autocorrelation of the
               context after removing day-of-week means (prior art: Baron et al. 2025,
               Choi et al. 2026 use AR(1) correlation from lag-1 autocorrelation)
  fgn_copula   Gaussian copula, correlation = conditional correlation of fGn(H_hat) future
               values given 512 past values; H_hat = fsde.hurst_second_diff of the cumulative
               sum of the context after removing day-of-week means and centring
  comonotone   sum of same-level quantiles
Reference (no foundation model):
  fsde_plugin  Gaussian conditional path of fGn(H_hat) on the deseasonalised context,
               day-of-week means added back, mapped to the target scale, summed
Copula sampling: 1000 common random draws per window; per-horizon inverse CDF = linear
interpolation of the 9 quantiles on the model-input scale, split-normal tails beyond
0.1 / 0.9, then mapped to the target scale (exp for crypto_rv) and summed.

Scores per window for the total S: inside the 80% interval [q0.1, q0.9]; lower / upper
exceedance; pinball loss over the 9 levels divided by (64 x sd of the target-scale context)
so it is scale-free.

Inference: weekly origins overlap (64-day targets), so 2026 has 3-4 independent periods
per domain. Report per-origin differences, their values at non-overlapping origins
(every 10th week) and a Newey-West (lag 9) t-statistic.

Registered predictions (long-memory domains = crypto_rv, entsoe_load, entsoe_price,
reservoir_inflow):
  P1  For both models, |coverage - 0.80| of the 64-day total is smaller for fgn_copula
      than for ar1_copula, in each long-memory domain.
  P2  For both models, mean scaled pinball of the total is lower for fgn_copula than for
      ar1_copula (ratio < 1), in each long-memory domain.
  P3  Control (crypto_ret): |coverage(fgn) - coverage(ar1)| < 0.03 and pinball ratio
      fgn/ar1 within [0.98, 1.02], for both models.
  P4  indep under-covers the total (coverage < 0.70) in at least 3 of the 4 long-memory
      domains, for both models.
Decision rule: direction 1 passes on real data if P1 and P2 hold in at least 3 of the 4
long-memory domains for both models, AND at the non-overlapping origins the sign of
|cov_err(fgn)| - |cov_err(ar1)| is negative in at least 3 of 4 periods in those domains.
Otherwise it shrinks to a short note.

    HF_HUB_DISABLE_XET=1 python pilot/aggregate_real.py \
        --daily series_daily.csv --water series_water.csv --entsoe zone_daily.parquet \
        --out results/aggregate_real --cache quantiles_real.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri
from scipy.stats import t as tdist

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import fsde  # noqa: E402
from pilot.aggregate_fgn import fgn_conditional  # noqa: E402
from pilot.gate1_fbm import MODELS, Q, pinball  # noqa: E402
from pilot.gate1_real import build_windows  # noqa: E402

L, HZ = 512, 64
H_GRID = np.round(np.arange(0.02, 0.99, 0.01), 2)
Z10 = ndtri(0.1)  # -1.2816
LONG = ["crypto_rv", "entsoe_load", "entsoe_price", "reservoir_inflow"]
TO_TARGET = {"crypto_rv": np.exp}


# --------------------------------------------------------------------- data ----

def load_domains(daily, water, entsoe):
    parts = []
    d = pd.read_csv(daily, parse_dates=["date"])
    rv = d[d.domain == "crypto_log_rv"].assign(domain="crypto_rv")
    parts.append(rv[["domain", "series", "date", "value"]])
    for s, g in d[d.domain == "crypto_log_price"].groupby("series"):
        p = g.set_index("date").value.sort_index().asfreq("D")
        r = p.diff().dropna()
        parts.append(pd.DataFrame(dict(domain="crypto_ret", series=s, date=r.index, value=r.values)))
    pr = d[d.domain == "entsoe_price"].assign(domain="entsoe_price")
    parts.append(pr[["domain", "series", "date", "value"]])
    z = pd.read_parquet(entsoe)
    z = z[(z.load_coverage >= 0.95) & z.load_mw_mean.notna()]
    parts.append(pd.DataFrame(dict(domain="entsoe_load", series=z.zone_key.values,
                                   date=pd.to_datetime(z.local_date).values, value=z.load_mw_mean.values)))
    w = pd.read_csv(water, parse_dates=["date"])
    w = w[w.date >= "2024-06-01"]
    for s, g in w.groupby("series"):
        v = np.exp(g.set_index("date").value.sort_index())
        v = v[~v.index.duplicated()].asfreq("D")
        f = v.diff().dropna()
        parts.append(pd.DataFrame(dict(domain="reservoir_inflow", series=s, date=f.index, value=f.values)))
    return pd.concat(parts, ignore_index=True)


def deseason(ctx, origin):
    """Remove day-of-week means (estimated on the context). Returns (x_d, dow_mean, dow_ctx)."""
    days = pd.date_range(origin - pd.Timedelta(days=L), periods=L, freq="D")
    dow = days.dayofweek.values
    means = np.array([ctx[dow == k].mean() for k in range(7)])
    return ctx - means[dow], means, dow


# ----------------------------------------------------------------- sampling ----

def inv_cdf(q, U):
    """q: (n, h, 9) quantiles at 0.1..0.9 (input scale); U: (n, M, h) uniforms -> (n, M, h)."""
    qq = np.transpose(q, (0, 1, 2))[:, None, :, :]  # (n, 1, h, 9)
    pos = (U - 0.1) / 0.1
    i = np.clip(np.floor(pos), 0, 7).astype(int)
    w = np.clip(pos - i, 0, 1)
    lo = np.take_along_axis(np.broadcast_to(qq, U.shape + (9,)), i[..., None], -1)[..., 0]
    hi = np.take_along_axis(np.broadcast_to(qq, U.shape + (9,)), (i + 1)[..., None], -1)[..., 0]
    x = lo + w * (hi - lo)
    q10, q50, q90 = q[:, None, :, 0], q[:, None, :, 4], q[:, None, :, 8]
    zu = ndtri(np.clip(U, 1e-9, 1 - 1e-9))
    x = np.where(U < 0.1, q50 + (q50 - q10) / -Z10 * zu, x)
    x = np.where(U > 0.9, q50 + (q90 - q50) / -Z10 * zu, x)
    return x


def total_quantiles(q, chol, eps, to_target):
    """Gaussian-copula totals. q (n,h,9) input scale; chol (n,h,h) or (h,h); eps (M,h)."""
    Z = np.einsum("nij,mj->nmi", chol, eps) if chol.ndim == 3 else eps @ chol.T
    if Z.ndim == 2:
        Z = np.broadcast_to(Z, (q.shape[0],) + Z.shape)
    X = inv_cdf(q, ndtr(Z))
    S = to_target(X).sum(-1)  # (n, M)
    return np.quantile(S, Q, axis=1).T  # (n, 9)


def safe_chol(R):
    return np.linalg.cholesky(R + 1e-8 * np.eye(R.shape[-1]))


# --------------------------------------------------------------------- main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily", required=True)
    ap.add_argument("--water", required=True)
    ap.add_argument("--entsoe", required=True)
    ap.add_argument("--models", nargs="+", default=["chronos-2", "timesfm-2.5"])
    ap.add_argument("--out", default="results/aggregate_real")
    ap.add_argument("--cache", default="quantiles_real.npz")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--step", type=int, default=7)
    ap.add_argument("--start", default="2026-01-05", help="first origin (smoke tests use 2025)")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(os.cpu_count() or 4)
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)

    t0 = time.time()
    df = load_domains(args.daily, args.water, args.entsoe)
    W = build_windows(df, L, HZ, args.step, args.start)
    meta = pd.DataFrame([{k: w[k] for k in ("domain", "series", "origin")} for w in W])
    ctx = np.stack([w["ctx"] for w in W]).astype(float)
    y = np.stack([w["y"] for w in W]).astype(float)
    print(meta.groupby("domain").agg(windows=("series", "size"), series=("series", "nunique"),
                                     first=("origin", "min"), last=("origin", "max")), flush=True)
    print(f"windows built in {time.time() - t0:.0f}s", flush=True)

    # model inputs: the stored crypto_rv series is already log RV
    is_rv = (meta.domain == "crypto_rv").values
    x_in = ctx.copy()
    y_target = y.copy()
    y_target[is_rv] = np.exp(y[is_rv])
    ctx_target = ctx.copy()
    ctx_target[is_rv] = np.exp(ctx[is_rv])

    # H_hat, rho, deseasonalised quantities (context only)
    H_hat = np.empty(len(W)); rho = np.empty(len(W)); mu = np.empty(len(W)); sd = np.empty(len(W))
    dowm = np.empty((len(W), 7)); fut_dow = np.empty((len(W), HZ), int)
    for i, (c, w) in enumerate(zip(x_in, W)):
        xd, means, _ = deseason(c, w["origin"])
        dowm[i] = means
        fut_dow[i] = pd.date_range(w["origin"], periods=HZ, freq="D").dayofweek.values
        xc = xd - xd.mean()
        mu[i], sd[i] = xd.mean(), xd.std()
        H_hat[i] = fsde.hurst_second_diff(np.cumsum(xc))
        rho[i] = np.clip(np.corrcoef(xd[:-1], xd[1:])[0, 1], -0.99, 0.99)
    meta["H_hat"], meta["rho"] = H_hat, rho
    Hg = H_GRID[np.clip(np.round((np.clip(H_hat, 0.02, 0.98) - 0.02) / 0.01).astype(int), 0, len(H_GRID) - 1)]
    print(meta.groupby("domain")[["H_hat", "rho"]].median().round(3), flush=True)

    grid = {H: fgn_conditional(H, L, HZ) for H in H_GRID}
    fgn_chol = {}
    for H, (A, C) in grid.items():
        R = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
        fgn_chol[H] = safe_chol(R)

    # model quantiles (cached)
    store = dict(np.load(args.cache)) if os.path.exists(args.cache) else {}
    for name in args.models:
        if name in store:
            continue
        t1 = time.time()
        model = MODELS[name](L, HZ)
        store[name] = np.sort(model(x_in, HZ), axis=-1).astype(np.float32)
        del model
        np.savez_compressed(args.cache, **store)
        print(f"[{name}] forecast {time.time() - t1:.0f}s", flush=True)

    # drop windows with non-finite inputs or forecasts (reported, not silently used)
    ok = np.isfinite(H_hat) & np.isfinite(rho) & np.isfinite(x_in).all(1) & np.isfinite(y_target).all(1)
    for name in args.models:
        ok &= np.isfinite(store[name]).all(axis=(1, 2))
    print(f"dropping {int((~ok).sum())} of {len(ok)} windows with non-finite values", flush=True)
    keep = np.where(ok)[0]
    meta = meta.iloc[keep].reset_index(drop=True)
    W = [W[i] for i in keep]
    x_in, y_target, ctx_target = x_in[keep], y_target[keep], ctx_target[keep]
    H_hat, rho, mu, sd, dowm, fut_dow, Hg = (a[keep] for a in (H_hat, rho, mu, sd, dowm, fut_dow, Hg))
    store = {k: v[keep] for k, v in store.items()}

    eps = rng.standard_normal((args.draws, HZ))
    S_true = y_target.sum(1)
    scale = HZ * ctx_target.std(1)
    lag = np.abs(np.arange(HZ)[:, None] - np.arange(HZ)[None, :])
    rows = []

    def score(name, rule, qS, idx):
        r = meta.iloc[idx].copy()
        r["model"], r["rule"] = name, rule
        S = S_true[idx]
        r["inside"] = (qS[:, 0] <= S) & (S <= qS[:, -1])
        r["lower"] = S < qS[:, 0]
        r["upper"] = S > qS[:, -1]
        r["pinball"] = pinball(qS, S) / scale[idx]
        rows.append(r)

    t2 = time.time()
    chunk = 150
    for s in range(0, len(W), chunk):
        idx = np.arange(s, min(s + chunk, len(W)))
        dom = meta.domain.values[idx]
        # plug-in (no foundation model): Gaussian conditional path of fGn(H_hat)
        qp = np.empty((len(idx), 9))
        for j, i in enumerate(idx):
            A, C = grid[Hg[i]]
            xd = x_in[i] - dowm[i][pd.date_range(W[i]["origin"] - pd.Timedelta(days=L), periods=L,
                                                   freq="D").dayofweek.values]
            mean = mu[i] + dowm[i][fut_dow[i]] + A @ (xd - mu[i])
            paths = mean[None, :] + sd[i] * (eps @ fgn_chol[Hg[i]].T) * np.sqrt(np.diag(C))[None, :]
            tt = TO_TARGET.get(dom[j], lambda v: v)
            qp[j] = np.quantile(tt(paths).sum(1), Q)
        score("fsde", "fsde_plugin", qp, idx)
        for name in args.models:
            q = store[name][idx].astype(float)
            tt_mask = dom == "crypto_rv"
            for rule in ["indep", "ar1_copula", "fgn_copula", "comonotone"]:
                qS = np.empty((len(idx), 9))
                for sub, tt in [(tt_mask, np.exp), (~tt_mask, lambda v: v)]:
                    if not sub.any():
                        continue
                    qs = q[sub]
                    if rule == "comonotone":
                        qS[sub] = tt(qs).sum(1)
                        continue
                    if rule == "indep":
                        chol = np.eye(HZ)
                    elif rule == "ar1_copula":
                        chol = safe_chol(rho[idx][sub][:, None, None] ** lag[None])
                    else:
                        chol = np.stack([fgn_chol[h] for h in Hg[idx][sub]])
                    qS[sub] = total_quantiles(qs, chol, eps, tt)
                score(name, rule, qS, idx)
        if s // chunk % 20 == 0:
            print(f"  scored {idx[-1] + 1}/{len(W)} windows [{time.time() - t2:.0f}s]", flush=True)

    R = pd.concat(rows, ignore_index=True)
    R.to_csv(os.path.join(args.out, "windows.csv.gz"), index=False)

    # ---- summaries -------------------------------------------------------------
    summ = R.groupby(["domain", "model", "rule"]).agg(
        n=("inside", "size"), coverage=("inside", "mean"), lower=("lower", "mean"),
        upper=("upper", "mean"), pinball=("pinball", "mean")).reset_index()
    ref = summ[summ.rule == "ar1_copula"].set_index(["domain", "model"]).pinball
    summ["pinball_vs_ar1"] = [p / ref.get((d, m), np.nan) for d, m, p in
                              zip(summ.domain, summ.model, summ.pinball)]
    summ.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    print(summ.round(3).to_string(index=False))

    # ---- per-origin inference, non-overlapping periods ---------------------------
    def hac_t(x, lag=9):
        x = np.asarray(x, float); n = len(x); e = x - x.mean()
        s = e @ e / n
        for l in range(1, min(lag, n - 1) + 1):
            s += 2 * (1 - l / (lag + 1)) * (e[l:] @ e[:-l]) / n
        tt = x.mean() / np.sqrt(s / n)
        return float(tt), float(2 * tdist.sf(abs(tt), n - 1))

    stats = {}
    for (d, m), g in R[R.model.isin(args.models)].groupby(["domain", "model"]):
        po = g.groupby(["origin", "rule"]).agg(cov=("inside", "mean"), pb=("pinball", "mean")).unstack("rule")
        dcov = (po["cov"]["fgn_copula"] - 0.8).abs() - (po["cov"]["ar1_copula"] - 0.8).abs()
        lratio = po["pb"]["fgn_copula"] / po["pb"]["ar1_copula"]
        t_c, p_c = hac_t(dcov.values)
        t_l, p_l = hac_t(np.log(lratio.values))
        stats[f"{d} | {m}"] = dict(
            n_origins=len(po),
            dcoverr_mean=float(dcov.mean()), dcoverr_nw_t=t_c, dcoverr_nw_p=p_c,
            dcoverr_nonoverlap=[round(float(v), 4) for v in dcov.values[::10]],
            logloss_ratio_mean=float(np.log(lratio).mean()), logloss_nw_t=t_l, logloss_nw_p=p_l,
            loss_ratio_nonoverlap=[round(float(v), 4) for v in lratio.values[::10]])
    with open(os.path.join(args.out, "stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    for k, v in stats.items():
        print(f"{k:32s} d|coverr| {v['dcoverr_mean']:+.3f} (NW p {v['dcoverr_nw_p']:.2g}) "
              f"periods {v['dcoverr_nonoverlap']}  | loss ratio periods {v['loss_ratio_nonoverlap']}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
