# Label-free H-rescaling of forecast intervals: negative result

**Result: the fix does not work.** Rescaling a model's interval widths by k^(Ĥ_context − H_forecast) restores long-horizon coverage on persistent fBm. But it does not improve quantile loss there, and on real reservoirs it makes both coverage and loss worse. By the proper scoring rule (pinball loss), the raw foundation-model forecasts remain the best option on real data. The method is in `pilot/h_rescale.py`; the fBm run is in `../h_rescale_fbm/`.

## Variants (all label-free)

- **model:** raw forecast.
- **h_rescaled:** median kept; distance of each quantile from the median at horizon k multiplied by k^clip(Ĥ_ctx − H_fc, ±0.5). Ĥ_ctx is the context variogram exponent over lags 1–64; H_fc is the forecast's 80%-width scaling exponent. The factor is 1 at k = 1.
- **hybrid:** model median plus the empirical baseline's spread.
- **baseline:** last value plus empirical quantiles of the context's k-step changes.

## fBm (300 series per H; exact oracle available)

| H | Model | Long coverage 0.80: model → h_rescaled | Loss ÷ baseline: model → h_rescaled |
| --- | --- | --- | --- |
| 0.3 | Chronos-2 | 0.82 → 0.71 | 0.96 → 0.95 |
| 0.3 | Chronos-Bolt | 0.81 → 0.64 | 0.99 → 1.01 |
| 0.8 | Chronos-2 | 0.64 → 0.72 | 1.01 → 1.03 |
| 0.8 | TimesFM-2.5 | 0.64 → 0.75 | 0.93 → 0.94 |
| 0.9 | Chronos-2 | 0.60 → 0.77 | 0.99 → 1.07 |
| 0.9 | TimesFM-2.5 | 0.66 → 0.78 | 0.90 → 1.03 |
| 0.9 | Chronos-Bolt | 0.68 → 0.89 | 0.99 → 1.31 |

For reference, the oracle's loss ÷ baseline is 0.79–0.96.

Coverage improves where H ≥ 0.7, but loss does not. The models' horizon-1 width and their median are also off, and fixing only the growth exponent overshoots for models whose short-horizon intervals are already wide (Bolt). It undershoots at low H.

## Reservoirs (267 series, 7,688 windows, all targets in 2026)

| Model | Variant | Long coverage (0.80) | Lower / upper tail (0.10) | Loss ÷ baseline |
| --- | --- | --- | --- | --- |
| Chronos-2 | model | 0.74 | 0.13 / 0.13 | **0.79** |
| | h_rescaled | 0.60 | 0.22 / 0.18 | 0.85 |
| | hybrid | 0.81 | 0.11 / 0.08 | 0.89 |
| TimesFM-2.5 | model | 0.73 | 0.13 / 0.14 | **0.84** |
| | h_rescaled | 0.63 | 0.19 / 0.19 | 0.87 |
| | hybrid | 0.81 | 0.11 / 0.09 | 0.90 |
| Chronos-Bolt | model | 0.73 | 0.14 / 0.13 | **0.82** |
| | h_rescaled | 0.74 | 0.14 / 0.12 | 0.84 |
| | hybrid | 0.80 | 0.12 / 0.09 | 0.88 |

On the persistent subset (context H > 0.85), h_rescaled moves coverage toward or past nominal: Chronos-2 0.75 → 0.77, TimesFM 0.70 → 0.74, Bolt 0.69 → 0.92. In every case it raises the loss (0.58 → 0.60, 0.72 → 0.73, 0.62 → 0.73).

**Why it fails on real data.** The context variogram exponent (mean 0.56) is *below* the models' forecast exponents (0.63–0.67). So the rescaling narrows intervals that were not too wide. Unconditional variogram scaling is the wrong reference for how conditional forecast uncertainty should grow on real, non-self-similar series. The two coincide only for exactly self-similar processes like fBm.

**The hybrid** reaches nominal coverage, but costs 6–13% in pinball loss against the raw model: it buys calibration with sharpness.

## Implication for the project

Combined with stage B, the evidence does not support Paper B's central claim that H mismatch causes failures that matter in practice:

- **On fBm:** the models compress H, and that costs coverage.
- **On real 2026 data:** the effect is second-order; the label-free gap is a weak predictor; and the obvious H-based correction hurts.

By the plan's Gate 1 rule, Paper B should shrink to a short note: "foundation models compress long memory; on real data it costs some long-horizon coverage but not proper-score accuracy". The effort should move to Paper A.
