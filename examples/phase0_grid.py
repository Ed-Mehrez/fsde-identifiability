"""Phase 0 benchmark: accuracy of the audited estimators, everything estimated.

For each model and H, simulate R independent paths (dt = 0.02, T = 1000) and
fit (H, sigma(x), mu(x)) with nothing known. Errors are relative L2 on the
5-95% quantile range of X. Writes results/phase0_grid.md.

    python examples/phase0_grid.py [--reps 8]
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import fsde  # noqa: E402

DT, N, BURN = 0.02, 50000, 200
MODELS = {
    "fOU (kappa=1, sigma=0.7)": (fsde.fou(1.0, 0.7), dict(n_basis=6)),
    "double-well (a=1, sigma=0.7)": (fsde.double_well(1.0, 0.7), dict(n_basis=8)),
    "hyperbolic (kappa=1, s=0.5)": (fsde.hyperbolic(1.0, 0.5), dict(n_basis=6)),
}
HS = [0.2, 0.35, 0.5, 0.65, 0.8]


def rel_l2(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2) / np.mean(b ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []
    t0 = time.time()
    for name, (m, kw) in MODELS.items():
        for H in HS:
            eh, es, em, cov = [], [], [], []
            for _ in range(args.reps):
                x = fsde.simulate_fsde(m, H, DT, N, burn_in=BURN, rng=rng)
                g = np.linspace(*np.quantile(x, [0.05, 0.95]), 60)
                f = fsde.fit_fsde(x, DT, **kw)
                eh.append(f.H - H)
                es.append(rel_l2(f.sigma(g), m.sigma(g)))
                em.append(rel_l2(f.mu(g), m.mu(g)))
                cov.append(np.mean(np.abs(f.mu(g) - m.mu(g)) <= 1.96 * f.mu_se(g)))
            rows.append((name, H, np.mean(eh), np.std(eh), np.mean(es), np.mean(em), np.std(em), np.mean(cov)))
            print(f"{name:32s} H={H:.2f}  H err {rows[-1][2]:+.4f}±{rows[-1][3]:.4f}  "
                  f"sigma {rows[-1][4]:.3f}  mu {rows[-1][5]:.3f}±{rows[-1][6]:.3f}  "
                  f"band cov {rows[-1][7]:.2f}  [{time.time() - t0:.0f}s]", flush=True)

    os.makedirs("results", exist_ok=True)
    with open("results/phase0_grid.md", "w") as fh:
        fh.write(f"# Phase 0 grid\n\ndt = {DT}, T = {N * DT:.0f}, {args.reps} replicates per cell, "
                 f"seed {args.seed}; everything estimated (H, sigma(x), mu(x)).\n"
                 "Errors are relative L2 on the 5-95% quantile range of X. Band coverage is the share of "
                 "evaluation points inside the pointwise 95% band for mu, which conditions on H_hat and "
                 "sigma_hat and ignores sieve bias, so values under 0.95 are expected.\n\n")
        fh.write("| Model | H | H_hat - H (mean ± sd) | sigma rel. L2 | mu rel. L2 (mean ± sd) | mu band coverage |\n")
        fh.write("| --- | --- | --- | --- | --- | --- |\n")
        for r in rows:
            fh.write(f"| {r[0]} | {r[1]:.2f} | {r[2]:+.4f} ± {r[3]:.4f} | {r[4]:.3f} | "
                     f"{r[5]:.3f} ± {r[6]:.3f} | {r[7]:.2f} |\n")
    print("wrote results/phase0_grid.md")


if __name__ == "__main__":
    main()
