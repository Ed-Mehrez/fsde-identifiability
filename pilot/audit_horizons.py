"""Audit (2026-11-11): reservoir smoothness and coverage / loss by horizon block, overall and by context-H bin.

    python pilot/audit_horizons.py --data series_water.csv --cache quantiles.npz
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '.')
from pilot.gate1_real import build_windows, empirical_baseline, context_H
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True, help='series_water.csv from prepare_real.py')
ap.add_argument('--cache', required=True, help='quantiles.npz written by h_rescale.py for the same data')
args = ap.parse_args()
df = pd.read_csv(args.data, parse_dates=['date'])
W = build_windows(df, 512, 64, 7, '2026-01-05')
ctx = np.stack([w['ctx'] for w in W]); y = np.stack([w['y'] for w in W])
store = dict(np.load(args.cache))
assert store['chronos-2'].shape[0] == len(W), (store['chronos-2'].shape, len(W))

# (a) interpolation / piecewise-linear check: share of (near-)zero second differences in the context
d2 = np.abs(np.diff(ctx, n=2, axis=1)); d1 = np.abs(np.diff(ctx, axis=1))
zero2 = (d2 < 1e-9).mean(1); zero1 = (d1 < 1e-9).mean(1)
Hc = np.array([context_H(c) for c in ctx])
bins = pd.cut(Hc, [-1, .3, .5, .7, .85, 2])
print('== reservoir contexts: share of exactly-zero 1st / 2nd differences, by H bin')
print(pd.DataFrame(dict(bin=bins, zero_d1=zero1, zero_d2=zero2)).groupby('bin', observed=True).median().round(3))

# (b) coverage and loss vs baseline by horizon block, for every model, overall and by H bin
base = np.stack([empirical_baseline(c, 64) for c in ctx])
Qs = np.round(np.arange(.1, .91, .1), 2)
def pin(q, yy):
    d = yy[..., None] - q; return np.mean(np.maximum(Qs * d, (Qs - 1) * d), -1)
blocks = [(1, 8), (9, 16), (17, 32), (33, 64)]
rows = []
for name, q in [('baseline', base)] + [(k, v.astype(float)) for k, v in store.items()]:
    inside = (q[..., 0] <= y) & (y <= q[..., -1]); pb = pin(q, y); pbb = pin(base, y)
    for b in ['all'] + list(bins.categories):
        m = np.ones(len(W), bool) if b == 'all' else (bins == b)
        for a, z in blocks:
            sl = slice(a - 1, z)
            rows.append(dict(model=name, H_bin=str(b), horizon=f'{a}-{z}', cov=inside[m, sl].mean(),
                             loss_vs_base=pb[m, sl].sum() / pbb[m, sl].sum()))
T = pd.DataFrame(rows)
print('\n== coverage of 80% interval by horizon block (all windows)')
print(T[T.H_bin == 'all'].pivot(index='model', columns='horizon', values='cov')[[f'{a}-{z}' for a, z in blocks]].round(3))
print('\n== loss / baseline by horizon block (all windows)')
print(T[T.H_bin == 'all'].pivot(index='model', columns='horizon', values='loss_vs_base')[[f'{a}-{z}' for a, z in blocks]].round(3))
print('\n== coverage by horizon block, H bin (0.85, 2.0] vs (-1, 0.3]')
for b in [str(bins.categories[-1]), str(bins.categories[0])]:
    print(b); print(T[T.H_bin == b].pivot(index='model', columns='horizon', values='cov')[[f'{a}-{z}' for a, z in blocks]].round(3))
