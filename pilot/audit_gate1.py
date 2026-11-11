"""Audit of the Gate 1 statistics (2026-11-11).

1. Within-origin rho(gap, coverage): naive t-test vs Newey-West (lag 9, since weekly
   origins with 64-day horizons overlap), and the rhos at non-overlapping origins only.
2. Reservoir context-H bins vs second-difference roughness (trend confound).
3. Stage A: label-free gap vs coverage within each (model, H) cell.

    python pilot/audit_gate1.py
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr, t as tdist

def hac_t(x, lag):
    """Newey-West t-stat for mean of x with Bartlett kernel."""
    x = np.asarray(x, float); n = len(x); e = x - x.mean()
    s = e @ e / n
    for l in range(1, lag + 1):
        s += 2 * (1 - l / (lag + 1)) * (e[l:] @ e[:-l]) / n
    se = np.sqrt(s / n); tt = x.mean() / se
    return tt, 2 * tdist.sf(abs(tt), df=n - 1)

print("== (1) within-origin rho(gap, cov_long): naive t vs Newey-West (lag 9 ~ 64-day overlap)")
for f in ["results/pilot_gate1_real/windows.csv", "results/pilot_gate1_reservoir/windows.csv"]:
    R = pd.read_csv(f)
    for (dom, m), g in R[R.model != "baseline"].groupby(["domain", "model"]):
        rh = g.groupby("origin").apply(lambda go: spearmanr(go.gap, go.cov_long).statistic, include_groups=False).dropna().sort_index()
        n = len(rh); tn = rh.mean() / (rh.std(ddof=1) / np.sqrt(n)); pn = 2 * tdist.sf(abs(tn), n - 1)
        th, ph = hac_t(rh.values, 9)
        # non-overlapping origins only (every 10th week)
        sub = rh.values[::10]
        print(f"  {dom:24s} {m:19s} mean rho {rh.mean():+.3f}  naive p {pn:.1e}  NW p {ph:.1e}  | non-overlap origins: {len(sub)} rhos {np.round(sub,2)}")

print("\n== (2) reservoir H_ctx bins: how many windows are near-linear trends?")
R = pd.read_csv("results/pilot_gate1_reservoir/windows.csv")
base = R[R.model == "baseline"].copy()
# reconstruct contexts to measure trendiness: need data; approximate via H_ctx_rough (second-diff) and H_ctx_var
base["bin"] = pd.cut(base.H_ctx_var, [-1, .3, .5, .7, .85, 2])
print(base.groupby("bin", observed=True).agg(n=("series", "size"), H_rough_med=("H_ctx_rough", "median"),
      share_rough_gt_0p95=("H_ctx_rough", lambda s: (s > 0.95).mean())).round(3))

print("\n== (3) stage A: does the label-free gap add info beyond true H? per-series Spearman WITHIN each (model, H) cell")
S = pd.read_csv("results/pilot_gate1/series.csv")
S = S[S.model != "oracle"]
rows = []
for (m, H), g in S.groupby(["model", "H"]):
    rows.append((m, H, spearmanr(g.gap_ctx_fc, np.log(g.disp_long)).statistic, spearmanr(g.gap_ctx_fc, g.cov_long).statistic))
W = pd.DataFrame(rows, columns=["model", "H", "rho_gap_logdisp", "rho_gap_cov"])
print(W.pivot(index="H", columns="model", values="rho_gap_cov").round(2))
print("mean within-cell rho(gap, cov_long):", round(W.rho_gap_cov.mean(), 3))
