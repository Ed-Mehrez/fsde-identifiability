# Direction 1 on real data: 64-day totals of long-memory rate series (2026 targets)

**Verdict by the pre-registered decision rule: does not pass.** Direction 1 shrinks to a short note.

- **P1 (coverage) holds** in 3 of 4 long-memory domains for both models.
- **P2 (pinball loss) holds** in only 2 of 4.
- **The control (P3)** misses its coverage tolerance narrowly.

The coverage finding is strong and consistent across independent periods. The prior-art AR(1) copula badly under-covers 64-day totals of persistent series, and the fGn copula recovers much of that coverage. Proper-score (pinball) gains are small (0–6%), and on real data stronger dependence of any kind helps, not specifically fGn.

The design, predictions and decision rule were committed before the full run, in the docstring of `pilot/aggregate_real.py` (commit "Pre-register direction 1 on real data").

**Disclosure.** A smoke test used to debug the script (2025 start, 91-day steps, Chronos-2 only, 200 draws) included 3 origins in 2026, so a partial summary was seen before that commit. The predictions were written before any run and were not changed.

## Setup

**Domains and data:**

| Domain | Series | Windows | Median context Ĥ | Median lag-1 ρ |
| --- | --- | --- | --- | --- |
| Crypto daily realised variance (total = 64-day integrated variance) | 172 Binance perpetuals | 6,079 | 0.89 | 0.63 |
| ENTSO-E daily mean load (total = energy) | 41 zones | 1,189 | 1.00 | 0.95 |
| ENTSO-E daily mean price (total = 64 × baseload average) | 44 zones | 1,367 | 0.99 | 0.78 |
| Reservoir daily net inflow (total = change in storage) | 266 reservoirs | 7,569 | 0.94 | 0.58 |
| **Control:** crypto daily log returns (total = 64-day return) | 172 | 6,079 | 0.48 | −0.01 |

Both Ĥ and ρ are computed on the context after removing day-of-week means.

**Protocol:**
- Weekly origins from 2026-01-05, context 512 days, horizon 64, so every target is in 2026.
- Marginals come from Chronos-2 and TimesFM-2.5 at 9 quantiles each.
- Totals are formed by Gaussian-copula sampling over the models' own quantiles, with 1,000 common draws per window.
- Pinball loss of the total is divided by 64 × the context sd, which makes it scale-free.

## Results

**Coverage of the 80% interval for the 64-day total** (Chronos-2 / TimesFM-2.5)

| Domain | indep | AR(1) copula (prior art) | fGn copula | comonotone | fsde plug-in (no model) |
| --- | --- | --- | --- | --- | --- |
| Crypto RV | 0.24 / 0.25 | 0.43 / 0.42 | **0.64 / 0.65** | 0.88 / 0.86 | 0.57 |
| Electricity load | 0.20 / 0.16 | **0.79 / 0.71** | 0.76 / 0.65 | 0.95 / 0.90 | 0.26 |
| Electricity price | 0.16 / 0.13 | 0.42 / 0.37 | 0.61 / 0.54 | **0.87 / 0.81** | 0.46 |
| Reservoir inflow | 0.40 / 0.36 | 0.61 / 0.58 | **0.73 / 0.71** | 0.92 / 0.91 | 0.75 |
| Control: crypto returns | 0.69 / 0.67 | 0.68 / 0.66 | 0.65 / 0.63 | 1.00 / 1.00 | 0.83 |

**Pinball loss of the total, relative to the AR(1) copula** (mean over windows; Chronos-2 / TimesFM-2.5)

| Domain | indep | fGn copula | comonotone |
| --- | --- | --- | --- |
| Crypto RV | 1.09 / 1.06 | **0.94 / 0.97** | 0.93 / 1.02 |
| Electricity load | 1.16 / 1.17 | 0.99 / 1.00 | 1.03 / 0.99 |
| Electricity price | 1.10 / 1.09 | 0.95 / 0.95 | **0.95 / 0.92** |
| Reservoir inflow | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| Control: crypto returns | 1.00 / 1.00 | 1.01 / 1.01 | 2.27 / 2.37 |

Reservoir pinball is identical across rules to three decimals. A few windows with very large changes in storage dominate the mean.

**Per-period consistency** (`stats.json`). This is |coverage error| of fGn minus that of AR(1) at non-overlapping origins; negative means fGn is better.

- **Crypto RV:** −0.13 to −0.31 in all 4 periods, for both models.
- **Electricity price:** −0.03 to −0.31 in all 4 periods, for both models.
- **Reservoir inflow:** −0.09 to −0.16 in all 3 periods, for both models.
- **Electricity load:** 0 to +0.08, so no improvement. There, AR(1) with ρ ≈ 0.95 already carries strong dependence.

## Registered predictions

| Prediction | Outcome | Verdict |
| --- | --- | --- |
| P1: fGn coverage error < AR(1), both models, each long-memory domain | RV, price, reservoir yes; load no (0.040 vs 0.013 and 0.152 vs 0.086) | Holds in 3 of 4 |
| P2: fGn pinball < AR(1), both models, each long-memory domain | RV yes (0.94, 0.97); price yes (0.95, 0.95); load 0.99 / 1.00; reservoir 1.000 / 1.000 | Holds in 2 of 4: **fails the ≥ 3 requirement** |
| P3: control, coverage difference < 0.03 and pinball ratio in [0.98, 1.02] | Coverage difference 0.032 / 0.035; pinball 1.012 / 1.013 | Coverage part fails narrowly; pinball part holds |
| P4: indep coverage < 0.70 in ≥ 3 long-memory domains | 0.13–0.40 in all 4 | Holds |
| Decision rule: P1 and P2 in ≥ 3 domains, and the sign consistent across periods | P2 fails | **Does not pass** |

## Post-hoc robustness (not pre-registered)

**Median per-window pinball ratio, fGn / AR(1):**

| Domain | Chronos-2 | TimesFM-2.5 |
| --- | --- | --- |
| Crypto RV | 0.967 | 0.975 |
| Electricity load | 0.982 | 0.995 |
| Electricity price | 0.958 | 0.953 |
| Reservoir inflow | 1.012 | 1.007 |
| Control | 1.006 | 1.007 |

By the median, comonotone / AR(1) is worse than AR(1) in RV (1.06, 1.12), load (1.11, 1.05) and reservoirs (1.39, 1.35). It is better only for price (0.96, 0.92). Its mean advantage in RV comes from a few windows.

## Interpretation

1. **Under long memory, the prior-art AR(1) copula is the wrong default.** It under-covers 64-day totals at 0.37–0.61 where 0.80 is targeted, in three of four persistent domains, consistently across independent periods. Long-memory dependence fixes most of the gap.
2. **The fix shows up in coverage, not much in proper scores.** Median errors dominate pinball loss of totals, so gains are 2–6% at best and nil for reservoirs.
3. **On real data, the specific fGn form is not uniquely right.**
   - Ĥ is near 1 in three domains, which suggests trends or level shifts rather than stationary fGn.
   - Comonotone dependence is competitive for electricity prices.
   - The fGn copula still under-covers (0.54–0.73), so the models' marginals are also too narrow.
4. **The pure fSDE plug-in** (no foundation model) has the lowest mean pinball for crypto RV and for the control. That is consistent with the rough-volatility literature, where fractional models forecast realised variance well, so it is not new.
5. **Caveats:**
   - 2026 provides only 3–4 independent periods per domain.
   - Absolute coverage reflects a few 2026 market and weather paths.
   - The reservoir pinball comparison is uninformative because of outliers.
