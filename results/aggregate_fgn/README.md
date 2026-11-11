# Direction 1 test: forecasting temporal totals of long-memory series (fGn)

**Finding.** Foundation models give per-horizon quantiles. Turning them into a forecast of the *total* over the horizon needs an assumption about how the horizons depend on each other, and both standard assumptions fail badly under long memory:

- **Independence** gives intervals for the 64-step total that are 0.26–0.46× the correct width at H = 0.7–0.9, so they cover only 23–43% instead of 80%. At H = 0.3 they are 2.4× too wide.
- **Comonotonicity** (all horizons moving in lockstep) gives intervals 2–20× too wide.
- **An fGn copula with H estimated from the context** gives widths within 2% of the oracle at every H.

The design and hypotheses were fixed in the script docstring before the first run (`pilot/aggregate_fgn.py`).

## Setup

- **Data:** fGn "rate" series (for example daily inflow or load), H ∈ {0.3, 0.5, 0.7, 0.8, 0.9}, 300 series per H. Context 512, horizon 64. Random scale and offset per series.
- **Target:** S_n, the sum of the next n values, for n = 8, 16, 32, 64. The tables below are n = 64.
- **Per-horizon marginals:**
  - Chronos-2 and TimesFM-2.5, zero-shot;
  - the oracle's own exact per-horizon marginals, which isolate the effect of the dependence assumption.
- **Rules for combining marginals into a forecast of S_n,** all using the context only:
  - *indep:* Gaussian per horizon (median, with sd from the 80% width), assumed independent;
  - *comonotone:* sum of same-level quantiles;
  - *fgn_copula:* the model's per-horizon median and sd, with the conditional correlation of fGn(Ĥ) future values given the 512 past values. Ĥ comes from `fsde.hurst_second_diff` on the cumulative context (mean 0.300 / 0.796 at H = 0.3 / 0.8).
- **References:**
  - *fsde_plugin:* the Gaussian conditional forecast of fGn(Ĥ) with sample mean and sd. It uses no foundation model.
  - *oracle:* the exact conditional law of S_n.

## Results (n = 64)

**Coverage of the 80% interval for S_64**

| Source / rule | H = 0.3 | 0.5 | 0.7 | 0.8 | 0.9 |
| --- | --- | --- | --- | --- | --- |
| Chronos-2, indep | 1.00 | 0.75 | 0.41 | 0.29 | 0.23 |
| Chronos-2, comonotone | 1.00 | 1.00 | 1.00 | 1.00 | 0.96 |
| Chronos-2, fgn_copula | 0.75 | 0.76 | 0.78 | 0.77 | 0.71 |
| TimesFM-2.5, indep | 1.00 | 0.77 | 0.43 | 0.28 | 0.26 |
| TimesFM-2.5, fgn_copula | 0.71 | 0.77 | 0.78 | 0.78 | 0.76 |
| Oracle marginals, indep | 1.00 | 0.83 | 0.45 | 0.32 | 0.27 |
| Oracle marginals, fgn_copula | 0.79 | 0.83 | 0.83 | 0.82 | 0.81 |
| fsde_plugin (no foundation model) | 0.78 | 0.78 | 0.81 | 0.77 | 0.68 |
| Oracle | 0.80 | 0.83 | 0.84 | 0.81 | 0.80 |

**Pinball loss of S_64 relative to the oracle**

| Source / rule | H = 0.3 | 0.5 | 0.7 | 0.8 | 0.9 |
| --- | --- | --- | --- | --- | --- |
| Chronos-2, indep | 1.36 | 1.12 | 1.19 | 1.11 | 1.36 |
| Chronos-2, comonotone | 7.68 | 3.72 | 1.75 | 1.40 | 1.33 |
| Chronos-2, fgn_copula | 1.20 | 1.13 | 1.09 | 1.01 | 1.19 |
| TimesFM-2.5, indep | 1.37 | 1.12 | 1.19 | 1.18 | 1.26 |
| TimesFM-2.5, fgn_copula | 1.22 | 1.13 | 1.09 | 1.04 | 1.09 |
| Oracle marginals, fgn_copula | 1.01 | 1.01 | 1.01 | 1.00 | 1.00 |
| fsde_plugin | 1.10 | 1.10 | 1.01 | 1.00 | 1.05 |

**Median interval width ÷ oracle**

| Rule (Chronos-2 / TimesFM-2.5) | H = 0.3 | 0.5 | 0.7 | 0.8 | 0.9 |
| --- | --- | --- | --- | --- | --- |
| indep | 2.46 / 2.44 | 1.00 / 1.00 | 0.46 / 0.48 | 0.34 / 0.35 | 0.26 / 0.26 |
| comonotone | 19.7 / 19.5 | 8.0 / 8.0 | 3.7 / 3.8 | 2.7 / 2.8 | 2.1 / 2.1 |
| fgn_copula | 1.00 / 1.00 | 1.00 / 1.00 | 0.98 / 1.01 | 0.98 / 1.02 | 0.98 / 1.02 |

The coverage standard error is about 0.023 per cell (300 independent series). All n are in `results.csv`.

## Pre-registered hypotheses

| Hypothesis | Outcome |
| --- | --- |
| indep under-covers increasingly for H > 0.5 | Yes: 0.75 → 0.41 → 0.29 → 0.23 (Chronos-2) |
| indep over-covers at H = 0.3 | Yes: 1.00, 2.4× too wide |
| comonotone over-covers | Yes: 0.96–1.00, 2–20× too wide |
| fgn_copula closest to nominal | Yes: 0.71–0.78, against 0.23–1.00 for the others |
| fgn_copula has the lowest pinball loss of the three rules | Yes, except H = 0.5, where the three rules coincide in theory and the noise in Ĥ costs a hair (1.127 vs 1.122; TimesFM 1.126 vs 1.117) |

## Caveats

- **The copula family is exactly right here** because the data are fGn. On real rate series it is misspecified. That test (daily load totals, 64-day integrated variance from crypto RV, reservoir net inflow, all with 2026 targets) is the real one, and it would face the same 3–4-independent-period limit as stage B.
- **The fGn copula's remaining shortfall** (coverage 0.71–0.78) comes from the models' medians and marginals: the same copula on the oracle's marginals is nominal.
- **On fGn, the pure fSDE plug-in matches or beats foundation model plus copula.** That is expected on data from its own model family and says nothing yet about real data.
