"""Label-free H-rescaling of foundation-model forecast intervals.

Stages A and B found that foundation models compress the growth of their
forecast uncertainty with horizon on persistent series (forecast-implied H
below the series' own H), which costs long-horizon coverage. The fix uses
only the context and the forecast itself:

    q'_tau(k) = median(k) + (q_tau(k) - median(k)) * k ** clip(H_ctx - H_fc, -c, c)

where H_ctx is the context's variogram exponent over k = 1..64 and H_fc is
the forecast's 80%-width scaling exponent over the same horizons. The
factor is 1 at k = 1, so short horizons, which are already calibrated, are
unchanged. The median is unchanged, so point accuracy is unchanged.

Variants compared (all label-free):
- ``model``: the raw forecast;
- ``h_rescaled``: the fix above;
- ``hybrid``: model median plus the empirical baseline's spread around its
  own median (borrows width from the context, ignores the model's spread);
- ``baseline``: last value plus empirical quantiles of the context's k-step changes.
On fBm the exact ``oracle`` is added.

Usage:
    HF_HUB_DISABLE_XET=1 python pilot/h_rescale.py --data series_water.csv --out results/h_rescale_reservoir
    HF_HUB_DISABLE_XET=1 python pilot/h_rescale.py --fbm --out results/h_rescale_fbm
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
from pilot.gate1_fbm import MODELS, Q, Z, implied_H, oracle, pinball  # noqa: E402
from pilot.gate1_real import KS, build_windows, context_H, empirical_baseline  # noqa: E402

MED = 4  # index of q0.5 in Q


def h_rescale(q, H_ctx, H_fc, clip=0.5):
    h = q.shape[1]
    d = np.clip(H_ctx - H_fc, -clip, clip)
    s = np.arange(1, h + 1)[None, :] ** d[:, None]  # (n, h)
    med = q[..., MED:MED + 1]
    return med + (q - med) * s[..., None]


def hybrid(q_model, q_base):
    return q_model[..., MED:MED + 1] + (q_base - q_base[..., MED:MED + 1])


def score(q, y, long_from):
    lo, hi = q[..., 0], q[..., -1]
    inside = (lo <= y) & (y <= hi)
    pb = pinball(q, y)
    return dict(cov_short=inside[:, :8].mean(1), cov_long=inside[:, long_from:].mean(1),
                lower_long=(y < lo)[:, long_from:].mean(1), upper_long=(y > hi)[:, long_from:].mean(1),
                pb_long=pb[:, long_from:].mean(1))


def load_real(path, L, h, step, start):
    df = pd.read_csv(path, parse_dates=["date"])
    W = build_windows(df, L, h, step, start)
    meta = pd.DataFrame([{k: w[k] for k in ("domain", "series", "origin")} for w in W])
    return meta, np.stack([w["ctx"] for w in W]), np.stack([w["y"] for w in W]), None


def load_fbm(L, h, n, Hs, seed):
    rng = np.random.default_rng(seed)
    metas, C, Y, O = [], [], [], []
    for H in Hs:
        x = fsde.fbm(L + h, H, 1.0, size=n, rng=rng)[:, 1:]
        A, sd = oracle(H, L, h)
        C.append(x[:, :L]); Y.append(x[:, L:])
        O.append((x[:, :L] @ A.T)[..., None] + sd[None, :, None] * Z)
        metas.append(pd.DataFrame(dict(domain="fbm", series=np.arange(n), origin=0, H_true=H)))
    return pd.concat(metas, ignore_index=True), np.concatenate(C), np.concatenate(Y), np.concatenate(O)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--fbm", action="store_true")
    ap.add_argument("--n-series", type=int, default=300)
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--step", type=int, default=7)
    ap.add_argument("--start", default="2026-01-05")
    ap.add_argument("--models", nargs="+", default=["chronos-2", "timesfm-2.5", "chronos-bolt-small"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", help="npz cache of model quantiles (default: <out>/quantiles.npz)")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(os.cpu_count() or 4)
    os.makedirs(args.out, exist_ok=True)
    L, h = args.context, args.horizon
    long_from = h // 2
    if args.fbm:
        meta, ctx, y, q_or = load_fbm(L, h, args.n_series, [0.1, 0.3, 0.5, 0.7, 0.8, 0.9], 0)
    else:
        meta, ctx, y, q_or = load_real(args.data, L, h, args.step, args.start)
    print(f"{len(meta)} windows", flush=True)

    meta["H_ctx"] = [context_H(c) for c in ctx]
    q_base = np.stack([empirical_baseline(c, h) for c in ctx])

    cache = args.cache or os.path.join(args.out, "quantiles.npz")
    store = dict(np.load(cache)) if os.path.exists(cache) else {}
    for name in args.models:
        if name in store:
            continue
        t0 = time.time()
        model = MODELS[name](L, h)
        store[name] = np.sort(model(ctx, h), axis=-1).astype(np.float32)
        del model
        np.savez_compressed(cache, **store)
        print(f"[{name}] forecast {time.time() - t0:.0f}s", flush=True)

    rows = []

    def add(name, variant, q):
        r = meta.copy()
        r["model"], r["variant"] = name, variant
        r["H_fc_after"] = implied_H(np.maximum(q[..., -1] - q[..., 0], 1e-12), KS[:h])
        for k, v in score(q, y, long_from).items():
            r[k] = v
        rows.append(r)

    add("baseline", "baseline", q_base)
    if q_or is not None:
        add("oracle", "oracle", q_or)
    for name in args.models:
        q = store[name].astype(float)
        H_fc = implied_H(np.maximum(q[..., -1] - q[..., 0], 1e-12), KS[:h])
        add(name, "model", q)
        add(name, "h_rescaled", h_rescale(q, meta.H_ctx.values, H_fc))
        add(name, "hybrid", hybrid(q, q_base))
    R = pd.concat(rows, ignore_index=True)
    R.to_csv(os.path.join(args.out, "windows.csv"), index=False)

    # pinball relative to the raw model of the same window (accuracy cost of the fix)
    by = ["H_true"] if args.fbm else []
    if not args.fbm:
        R["H_bin"] = pd.cut(R.H_ctx, [-1, 0.3, 0.5, 0.7, 0.85, 2],
                            labels=["<0.3", "0.3-0.5", "0.5-0.7", "0.7-0.85", ">0.85"])
        by = ["H_bin"]
    for grouping in ([], by):
        g = R.groupby(grouping + ["model", "variant"], observed=True).agg(
            n=("series", "size"), cov_short=("cov_short", "mean"), cov_long=("cov_long", "mean"),
            lower_long=("lower_long", "mean"), upper_long=("upper_long", "mean"),
            H_fc_after=("H_fc_after", "mean"), pb_long=("pb_long", "sum")).reset_index()
        ref = R[R.variant == "baseline"].groupby(grouping, observed=True).pb_long.sum() if grouping \
            else R[R.variant == "baseline"].pb_long.sum()
        g["loss_vs_baseline"] = g.pb_long / (g[grouping].merge(ref.rename("ref").reset_index(), on=grouping)["ref"].values
                                             if grouping else ref)
        g = g.drop(columns="pb_long")
        tag = "by_" + by[0] if grouping else "overall"
        g.to_csv(os.path.join(args.out, f"summary_{tag}.csv"), index=False)
        print(f"\n== {tag}")
        print(g.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
