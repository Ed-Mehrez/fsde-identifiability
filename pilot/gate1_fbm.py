"""Gate 1 pilot, stage A: do time-series foundation models get long memory right on fBm?

Fractional Brownian motion is Gaussian with known covariance, so the exact
optimal forecast distribution given the context is available in closed form
(the "oracle"). For each Hurst exponent H and each model we measure:

- calibration: coverage of the 80% interval [q0.1, q0.9] at every horizon k;
- forecast-implied H: slope of log(interval width) against log k. The oracle's
  prediction-error sd grows like k^H, so this reads off the H a model is
  effectively using for its uncertainty;
- skill: pinball (quantile) loss relative to the oracle (1 = optimal);
- a label-free diagnostic: H estimated from the context alone (``fsde``) minus
  the forecast-implied H. It needs no future data, so it could flag
  miscalibration in deployment.

Usage:
    HF_HUB_DISABLE_XET=1 python pilot/gate1_fbm.py --n-series 300
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import fsde  # noqa: E402

Q = np.round(np.arange(0.1, 0.91, 0.1), 2)  # 9 quantile levels
Z = norm.ppf(Q)
Z80 = norm.ppf(0.9)


# ----------------------------------------------------------------- oracle ----

def fbm_cov(t: np.ndarray, H: float) -> np.ndarray:
    s, u = t[:, None], t[None, :]
    return 0.5 * (s ** (2 * H) + u ** (2 * H) - np.abs(s - u) ** (2 * H))


def oracle(H: float, L: int, h: int):
    """Exact conditional law of B_{L+1..L+h} given B_{1..L} (B_0 = 0, unit step).
    Returns (A, sd): mean = A @ context, sd per horizon."""
    t = np.arange(1, L + h + 1, dtype=float)
    S = fbm_cov(t, H)
    Scc, Sfc, Sff = S[:L, :L], S[L:, :L], S[L:, L:]
    A = np.linalg.solve(Scc, Sfc.T).T
    var = np.diag(Sff - A @ Sfc.T)
    return A, np.sqrt(np.maximum(var, 0))


# ----------------------------------------------------------------- models ----

class ChronosModel:
    def __init__(self, repo):
        import torch
        from chronos import BaseChronosPipeline

        self.torch = torch
        self.pipe = BaseChronosPipeline.from_pretrained(repo, device_map="cpu", dtype=torch.float32)

    def __call__(self, ctx: np.ndarray, h: int, batch: int = 64) -> np.ndarray:
        out = []
        for s in range(0, len(ctx), batch):
            inp = [self.torch.tensor(r, dtype=self.torch.float32) for r in ctx[s:s + batch]]
            q, _ = self.pipe.predict_quantiles(inp, prediction_length=h, quantile_levels=list(Q))
            q = q if isinstance(q, self.torch.Tensor) else self.torch.stack([qi.reshape(h, len(Q)) for qi in q])
            out.append(q.reshape(-1, h, len(Q)).cpu().numpy())
        return np.concatenate(out)


class TimesFMModel:
    def __init__(self, L, h):
        import timesfm

        self.m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch",
                                                                 torch_compile=False)
        self.m.compile(timesfm.ForecastConfig(max_context=L, max_horizon=h, normalize_inputs=True,
                                              use_continuous_quantile_head=True,
                                              fix_quantile_crossing=True, per_core_batch_size=32))

    def __call__(self, ctx: np.ndarray, h: int) -> np.ndarray:
        _, q = self.m.forecast(horizon=h, inputs=list(ctx))
        return np.asarray(q)[:, :h, 1:10]  # column 0 is the mean; 1..9 are q0.1..q0.9


MODELS = {
    "oracle": None,  # exact Gaussian conditional; sanity check (coverage 0.8, rel. pinball 1)
    "chronos-2": lambda L, h: ChronosModel("amazon/chronos-2"),
    "chronos-bolt-small": lambda L, h: ChronosModel("amazon/chronos-bolt-small"),
    "timesfm-2.5": lambda L, h: TimesFMModel(L, h),
}


# ---------------------------------------------------------------- metrics ----

def pinball(q: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mean pinball loss over the 9 levels; q (..., 9), y (...)."""
    d = y[..., None] - q
    return np.mean(np.maximum(Q * d, (Q - 1) * d), axis=-1)


def implied_H(width: np.ndarray, ks: np.ndarray) -> np.ndarray:
    """Slope of log width vs log k, row-wise."""
    lk = np.log(ks)
    lw = np.log(np.maximum(width, 1e-12))
    lkc = lk - lk.mean()
    return (lw - lw.mean(axis=-1, keepdims=True)) @ lkc / (lkc @ lkc)


def evaluate(qm, y, scale, oracle_mean, sd, ctx_H, long_from):
    """Per-series metrics for one (model, H) cell."""
    h = y.shape[1]
    ks = np.arange(1, h + 1)
    lo, hi, med = qm[..., 0], qm[..., -1], qm[..., 4]
    inside = (lo <= y) & (y <= hi)
    width = (hi - lo) / scale[:, None]
    q_or = oracle_mean[..., None] + scale[:, None, None] * sd[None, :, None] * Z
    pb_m, pb_o = pinball(qm, y), pinball(q_or, y)
    H_fc = implied_H(width, ks)
    # realised squared error over model-implied variance at long horizons (1 if calibrated)
    implied_sd = (hi - lo) / (2 * Z80)
    disp = np.mean(((y - med) / np.maximum(implied_sd, 1e-12))[:, long_from:] ** 2, axis=1)
    return dict(
        cov_by_k=inside.mean(axis=0),
        lower_exceed_by_k=(y < lo).mean(axis=0),
        width_ratio_by_k=width.mean(axis=0) / (2 * Z80 * sd),
        rel_pinball_by_k=pb_m.mean(axis=0) / pb_o.mean(axis=0),
        series=pd.DataFrame(dict(H_fc=H_fc, H_ctx=ctx_H, disp_long=disp,
                                 cov_long=inside[:, long_from:].mean(axis=1),
                                 rel_pinball=pb_m.mean(axis=1) / pb_o.mean(axis=1))),
    )


# ------------------------------------------------------------------- main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-series", type=int, default=300)
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--H", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--out", default="results/pilot_gate1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(os.cpu_count() or 4)
    os.makedirs(args.out, exist_ok=True)
    L, h, N = args.context, args.horizon, args.n_series
    long_from = h // 2
    rng = np.random.default_rng(args.seed)

    # data and oracle, shared by all models
    data = {}
    for H in args.H:
        x = fsde.fbm(L + h, H, 1.0, size=N, rng=rng)[:, 1:]  # drop B_0 = 0
        scale = 10 ** rng.uniform(-1, 1, N)  # random affine map per series
        offset = rng.normal(0, 3, N) * scale
        A, sd = oracle(H, L, h)
        ctx_raw, fut_raw = x[:, :L], x[:, L:]
        data[H] = dict(
            ctx=offset[:, None] + scale[:, None] * ctx_raw,
            y=offset[:, None] + scale[:, None] * fut_raw,
            scale=scale, sd=sd,
            oracle_mean=offset[:, None] + scale[:, None] * (ctx_raw @ A.T),
            ctx_H=np.array([fsde.hurst_second_diff(c) for c in ctx_raw]),
            oracle_H_fc=float(implied_H(2 * Z80 * sd, np.arange(1, h + 1))),
        )

    curves, series, cells = [], [], []
    for name in args.models:
        t0 = time.time()
        model = MODELS[name](L, h) if MODELS[name] is not None else None
        print(f"[{name}] loaded in {time.time() - t0:.0f}s", flush=True)
        for H in args.H:
            d = data[H]
            t1 = time.time()
            if model is None:
                qm = d["oracle_mean"][..., None] + d["scale"][:, None, None] * d["sd"][None, :, None] * Z
            else:
                qm = np.sort(model(d["ctx"], h), axis=-1)
            r = evaluate(qm, d["y"], d["scale"], d["oracle_mean"], d["sd"], d["ctx_H"], long_from)
            for k in range(h):
                curves.append(dict(model=name, H=H, k=k + 1, coverage80=r["cov_by_k"][k],
                                   lower_exceed=r["lower_exceed_by_k"][k],
                                   width_ratio=r["width_ratio_by_k"][k],
                                   rel_pinball=r["rel_pinball_by_k"][k]))
            s = r["series"].assign(model=name, H=H)
            series.append(s)
            cell = dict(
                model=name, H=H,
                oracle_H_fc=d["oracle_H_fc"],
                H_fc=float(s.H_fc.mean()), H_fc_sd=float(s.H_fc.std()),
                H_ctx=float(s.H_ctx.mean()),
                cov80_short=float(r["cov_by_k"][:8].mean()),
                cov80_long=float(r["cov_by_k"][long_from:].mean()),
                lower_exceed_long=float(r["lower_exceed_by_k"][long_from:].mean()),
                width_ratio_k1=float(r["width_ratio_by_k"][0]),
                width_ratio_k64=float(r["width_ratio_by_k"][-1]),
                rel_pinball=float(np.mean(r["rel_pinball_by_k"])),
                rel_pinball_long=float(np.mean(r["rel_pinball_by_k"][long_from:])),
            )
            cells.append(cell)
            print(f"[{name}] H={H:.1f}  H_fc {cell['H_fc']:.3f} (oracle {cell['oracle_H_fc']:.3f})  "
                  f"cov80 short {cell['cov80_short']:.2f} long {cell['cov80_long']:.2f}  "
                  f"width k64 x{cell['width_ratio_k64']:.2f}  rel pinball {cell['rel_pinball']:.3f}  "
                  f"[{time.time() - t1:.0f}s]", flush=True)
        del model

    curves = pd.DataFrame(curves)
    series = pd.concat(series, ignore_index=True)
    cells = pd.DataFrame(cells)
    cells["dH_fc"] = cells.H_fc - cells.oracle_H_fc
    cells["cov_gap_long"] = cells.cov80_long - 0.8
    series["gap_ctx_fc"] = series.H_ctx - series.H_fc
    curves.to_csv(f"{args.out}/curves.csv", index=False)
    series.to_csv(f"{args.out}/series.csv", index=False)
    cells.to_csv(f"{args.out}/cells.csv", index=False)

    # Gate 1 statistics
    stats = {}
    rho, p = spearmanr(cells.dH_fc, cells.cov_gap_long)
    stats["cell: dH_fc vs long coverage gap"] = dict(spearman=rho, p=p, n=len(cells))
    rho, p = spearmanr(cells.H_ctx - cells.H_fc, cells.cov_gap_long)
    stats["cell: (H_ctx - H_fc) vs long coverage gap"] = dict(spearman=rho, p=p, n=len(cells))
    for name, g in series.groupby("model"):
        rho, p = spearmanr(g.gap_ctx_fc, np.log(g.disp_long))
        stats[f"series[{name}]: (H_ctx - H_fc) vs log long-horizon dispersion"] = dict(
            spearman=rho, p=p, n=len(g))
    with open(f"{args.out}/gate1_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2, default=float)
    print(json.dumps(stats, indent=2, default=float))


if __name__ == "__main__":
    main()
