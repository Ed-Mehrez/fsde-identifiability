"""Gate 1 pilot, stage B: real series with forecast targets entirely in 2026.

Every model evaluated was released before 2026 (TimesFM-2.5 2025-09-02,
Chronos-2 2025-10-30, Chronos-Bolt 2024-11-25), so no forecast target can be
in its training data. Context windows may predate release; the outcomes cannot.

For each series and each origin (every ``--step`` days from 2026-01-05 while
the 64-day target window is observed) we forecast from the previous 512 days
and record:

- coverage of the 80% interval [q0.1, q0.9] at every horizon, and the
  lower and upper exceedance rates (target 0.10 each);
- forecast-implied H: slope of log(80% width) against log horizon;
- context-implied H: half the slope of log E[(X_{t+k} - X_t)^2] against log k,
  over the same k = 1..64, estimated on the context only (label-free);
- context roughness H from second differences (``fsde.hurst_second_diff``);
- pinball loss relative to an H-agnostic baseline: last value plus the
  empirical quantiles of the context's own k-step changes.

Usage:
    HF_HUB_DISABLE_XET=1 python pilot/gate1_real.py --data series_daily.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import fsde  # noqa: E402
from pilot.gate1_fbm import MODELS, Q, implied_H, pinball  # noqa: E402

KS = np.arange(1, 65)


def context_H(x: np.ndarray) -> float:
    V = [np.mean((x[k:] - x[:-k]) ** 2) for k in KS]
    return float(np.polyfit(np.log(KS), np.log(V), 1)[0] / 2)


def empirical_baseline(x: np.ndarray, h: int) -> np.ndarray:
    """Last value + empirical quantiles of the context's k-step changes."""
    out = np.empty((h, len(Q)))
    for k in range(1, h + 1):
        out[k - 1] = x[-1] + np.quantile(x[k:] - x[:-k], Q)
    return out


def build_windows(df: pd.DataFrame, L: int, h: int, step: int, start: str, max_gap: int = 3):
    rows = []
    for (dom, ser), g in df.groupby(["domain", "series"]):
        s = g.set_index("date").value.sort_index().asfreq("D")
        origin = pd.Timestamp(start)
        while origin + pd.Timedelta(days=h - 1) <= s.index[-1]:
            ctx = s[origin - pd.Timedelta(days=L):origin - pd.Timedelta(days=1)]
            fut = s[origin:origin + pd.Timedelta(days=h - 1)]
            if (len(ctx) == L and len(fut) == h and fut.notna().all()
                    and ctx.isna().sum() <= max_gap and ctx.iloc[[0, -1]].notna().all()):
                rows.append(dict(domain=dom, series=ser, origin=origin,
                                 ctx=ctx.interpolate().values, y=fut.values))
            origin += pd.Timedelta(days=step)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--step", type=int, default=14)
    ap.add_argument("--start", default="2026-01-05")
    ap.add_argument("--models", nargs="+", default=["chronos-2", "timesfm-2.5", "chronos-bolt-small"])
    ap.add_argument("--out", default="results/pilot_gate1_real")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(os.cpu_count() or 4)
    os.makedirs(args.out, exist_ok=True)
    L, h = args.context, args.horizon
    long_from = h // 2

    df = pd.read_csv(args.data, parse_dates=["date"])
    W = build_windows(df, L, h, args.step, args.start)
    meta = pd.DataFrame([{k: w[k] for k in ("domain", "series", "origin")} for w in W])
    ctx = np.stack([w["ctx"] for w in W])
    y = np.stack([w["y"] for w in W])
    print(meta.groupby("domain").agg(windows=("series", "size"), series=("series", "nunique"),
                                     first=("origin", "min"), last=("origin", "max")), flush=True)

    meta["H_ctx_var"] = [context_H(c) for c in ctx]
    meta["H_ctx_rough"] = [fsde.hurst_second_diff(c) for c in ctx]
    base = np.stack([empirical_baseline(c, h) for c in ctx])
    pb_base = pinball(base, y)  # (n, h)

    rows = []
    for name in ["baseline"] + args.models:
        t0 = time.time()
        if name == "baseline":
            qm = base
        else:
            model = MODELS[name](L, h)
            qm = np.sort(model(ctx, h), axis=-1)
            del model
        lo, hi = qm[..., 0], qm[..., -1]
        inside = (lo <= y) & (y <= hi)
        width = hi - lo
        res = meta.copy()
        res["model"] = name
        res["H_fc"] = implied_H(np.maximum(width, 1e-12), KS[:h])
        res["cov_short"] = inside[:, :8].mean(axis=1)
        res["cov_long"] = inside[:, long_from:].mean(axis=1)
        res["lower_long"] = (y < lo)[:, long_from:].mean(axis=1)
        res["upper_long"] = (y > hi)[:, long_from:].mean(axis=1)
        pb = pinball(qm, y)
        res["pb_long"] = pb[:, long_from:].mean(axis=1)
        res["pb_base_long"] = pb_base[:, long_from:].mean(axis=1)
        rows.append(res)
        print(f"[{name}] {time.time() - t0:.0f}s", flush=True)

    R = pd.concat(rows, ignore_index=True)
    R["gap"] = R.H_ctx_var - R.H_fc
    R.to_csv(f"{args.out}/windows.csv", index=False)

    # summaries per domain x model
    summ = R.groupby(["domain", "model"]).agg(
        windows=("series", "size"), H_ctx_var=("H_ctx_var", "mean"), H_fc=("H_fc", "mean"),
        cov_short=("cov_short", "mean"), cov_long=("cov_long", "mean"),
        lower_long=("lower_long", "mean"), upper_long=("upper_long", "mean"),
        pb_long=("pb_long", "sum"), pb_base_long=("pb_base_long", "sum")).reset_index()
    summ["loss_vs_baseline"] = summ.pb_long / summ.pb_base_long
    summ = summ.drop(columns=["pb_long", "pb_base_long"])
    summ.to_csv(f"{args.out}/summary.csv", index=False)
    print(summ.round(3).to_string(index=False))

    # Gate 1: does the label-free gap predict long-horizon miscalibration?
    # Aggregate per series (mean over origins) to reduce noise from overlapping windows.
    stats = {}
    S = R.groupby(["domain", "model", "series"]).agg(
        gap=("gap", "mean"), cov_long=("cov_long", "mean"), pb_long=("pb_long", "sum"),
        pb_base_long=("pb_base_long", "sum")).reset_index()
    S["miscal"] = (S.cov_long - 0.8).abs()
    S["rel_loss"] = S.pb_long / S.pb_base_long
    for (dom, m), g in S.groupby(["domain", "model"]):
        if m == "baseline":
            continue
        r1 = spearmanr(g.gap, g.cov_long - 0.8)
        r2 = spearmanr(g.gap.abs(), g.miscal)
        r3 = spearmanr(g.gap.abs(), g.rel_loss)
        stats[f"{dom} | {m}"] = dict(
            n_series=len(g),
            gap_vs_cov_gap=dict(rho=r1.statistic, p=r1.pvalue),
            absgap_vs_abs_miscal=dict(rho=r2.statistic, p=r2.pvalue),
            absgap_vs_loss_vs_baseline=dict(rho=r3.statistic, p=r3.pvalue))
    # Cross-sectional version robust to common shocks: Spearman across series
    # within each origin, then a t-test of the mean over origins.
    from scipy.stats import ttest_1samp

    for (dom, m), g in R.groupby(["domain", "model"]):
        if m == "baseline":
            continue
        rhos = []
        for _, go in g.groupby("origin"):
            if len(go) >= 10 and go.gap.std() > 0 and go.cov_long.std() > 0:
                rhos.append(spearmanr(go.gap, go.cov_long).statistic)
        rhos = np.array([r for r in rhos if np.isfinite(r)])
        if len(rhos) >= 3:
            t = ttest_1samp(rhos, 0.0)
            stats[f"{dom} | {m}"]["within_origin_gap_vs_cov"] = dict(
                mean_rho=float(rhos.mean()), n_origins=len(rhos), t=float(t.statistic), p=float(t.pvalue))
            print(f"{dom:18s} {m:20s} within-origin gap~cov rho {rhos.mean():+.3f} over {len(rhos)} origins "
                  f"(t {t.statistic:+.2f}, p {t.pvalue:.1e})")
    with open(f"{args.out}/gate1_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2, default=float)
    for k, v in stats.items():
        print(f"{k:40s} n={v['n_series']:3d}  gap~cov gap rho {v['gap_vs_cov_gap']['rho']:+.2f} "
              f"(p {v['gap_vs_cov_gap']['p']:.1e})  |gap|~|miscal| rho {v['absgap_vs_abs_miscal']['rho']:+.2f} "
              f"(p {v['absgap_vs_abs_miscal']['p']:.1e})  |gap|~loss rho {v['absgap_vs_loss_vs_baseline']['rho']:+.2f}")


if __name__ == "__main__":
    main()
