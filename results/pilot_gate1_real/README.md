# Gate 1 pilot, stage B: real series, all forecast targets in 2026

> **Audit correction (2026-11-11), see `../AUDIT_2026-11-11.md`.** The p-values below treat overlapping weekly origins as independent. 2026 contains only 3–4 non-overlapping 64-day target periods per domain, so they are overstated; read the per-period sign consistency instead. The high-H reservoir windows are trend-dominated (second-difference roughness ≈ 1), not fractional long memory. The verdict is downgraded from "weak pass" to **inconclusive**.

**Original verdict: weak pass.** The fBm failure (stage A) shows up on real data in direction but not in size:

- **Persistent reservoirs:** the models under-cover by 5–12 points at long horizons, while an H-agnostic baseline stays calibrated.
- **The no-future-data gap check:** it points the right way in 3 of 4 domains, but is weak (within-date |ρ| ≤ 0.21, against 0.82 on fBm).
- **Accuracy is good:** the models beat the baseline by 16–32% wherever there is structure to learn.

## Setup

- **Contamination control:** every model was released before 2026: Chronos-Bolt on 2024-11-25, TimesFM-2.5 on 2025-09-02, Chronos-2 on 2025-10-30. Forecast origins are weekly from 2026-01-05, and every 64-day target lies in 2026. Context windows, the previous 512 days, may predate release.
- **Data:** all from Hugging Face, built by `pilot/prepare_real.py`.

| Domain | Source | Series | Windows | Context H, lags 1–64 (mean) |
| --- | --- | --- | --- | --- |
| Crypto log price (control) | `rogerdehe/klines-binance`, 5-minute USDT perpetuals | 172 | 6,079 | 0.48 |
| Crypto log realised variance | same; daily sum of squared 5-minute returns | 172 | 6,079 | 0.09 (rough, H ≈ 0.20 from second differences) |
| Electricity day-ahead price | `hsilvosa/entsoe-observatory`, daily mean per bidding zone | 44 | 1,367 | 0.10 |
| Reservoir log storage volume | `National-Digital/au-water-storage-levels` (Bureau of Meteorology) | 267 | 7,688 | 0.56 (spread 0.04–1.0) |

- **Baseline:** last value plus the empirical quantiles of the context's own k-step changes. It knows nothing about H, but its interval width scales with the series' own historical variogram.
- **Gap (label-free):** context-implied H (variogram over lags 1–64) minus forecast-implied H (80%-width scaling over horizons 1–64).
- **Statistics:** computed within each origin (Spearman across series), then t-tested across origins. This stops common shocks, such as a crypto market move hitting every asset at once, from counting as independent evidence.

## Results

Long-horizon means average horizons 33–64. "Loss ÷ baseline" is pinball loss at long horizons relative to the baseline.

| Domain | Model | Coverage 80%, short (1–8) | Long | Lower / upper tail (target 0.10) | Loss ÷ baseline | Within-origin ρ(gap, coverage) |
| --- | --- | --- | --- | --- | --- | --- |
| Crypto price | baseline | 0.86 | 0.77 | 0.05 / 0.18 | 1.00 | |
| | Chronos-2 | 0.82 | 0.74 | 0.13 / 0.13 | 1.00 | −0.07 (p = 0.001) |
| | TimesFM-2.5 | 0.83 | 0.74 | 0.15 / 0.11 | 1.01 | −0.13 (p = 5 × 10⁻⁷) |
| | Chronos-Bolt | 0.83 | 0.69 | 0.20 / 0.12 | 1.00 | −0.17 (p = 2 × 10⁻¹³) |
| Crypto log RV | baseline | 0.79 | 0.75 | 0.10 / 0.15 | 1.00 | |
| | Chronos-2 | 0.77 | 0.75 | 0.13 / 0.11 | 0.74 | +0.05 (p = 0.03) |
| | TimesFM-2.5 | 0.78 | 0.76 | 0.10 / 0.14 | 0.75 | +0.12 (p = 2 × 10⁻⁶) |
| | Chronos-Bolt | 0.78 | 0.76 | 0.13 / 0.11 | 0.74 | +0.03 (p = 0.17) |
| Electricity price | baseline | 0.66 | 0.63 | 0.06 / 0.31 | 1.00 | |
| | Chronos-2 | 0.73 | 0.69 | 0.06 / 0.26 | 0.68 | −0.21 (p = 8 × 10⁻⁶) |
| | TimesFM-2.5 | 0.73 | 0.64 | 0.04 / 0.32 | 0.78 | −0.07 (p = 0.10) |
| | Chronos-Bolt | 0.71 | 0.72 | 0.05 / 0.23 | 0.69 | −0.10 (p = 0.01) |
| Reservoirs | baseline | 0.76 | 0.75 | 0.13 / 0.12 | 1.00 | |
| | Chronos-2 | 0.79 | 0.74 | 0.13 / 0.13 | 0.79 | −0.10 (p = 1 × 10⁻⁷) |
| | TimesFM-2.5 | 0.84 | 0.73 | 0.13 / 0.14 | 0.84 | −0.13 (p = 6 × 10⁻⁸) |
| | Chronos-Bolt | 0.85 | 0.73 | 0.14 / 0.13 | 0.82 | −0.06 (p = 0.003) |

A negative ρ is the stage-A direction: when the series' own H exceeds the forecast's, coverage falls.

**The persistent subset is the closest real-data analogue of stage A.** Reservoirs split by their context H:

| Context H | Baseline coverage | Chronos-2 | TimesFM-2.5 | Chronos-Bolt | Forecast-implied H (baseline / C-2 / TFM / Bolt) | Loss ÷ baseline (C-2 / TFM / Bolt) |
| --- | --- | --- | --- | --- | --- | --- |
| < 0.3 | 0.76 | 0.77 | 0.78 | 0.79 | 0.34 / 0.33 / 0.31 / 0.28 | 0.79 / 0.79 / 0.80 |
| 0.3–0.5 | 0.71 | 0.77 | 0.74 | 0.76 | 0.72 / 0.64 / 0.59 / 0.49 | 0.85 / 0.88 / 0.86 |
| 0.5–0.7 | 0.75 | 0.72 | 0.74 | 0.71 | 0.95 / 0.74 / 0.66 / 0.54 | 0.82 / 0.90 / 0.88 |
| 0.7–0.85 | 0.68 | 0.69 | 0.68 | 0.68 | 0.95 / 0.85 / 0.81 / 0.66 | 0.77 / 0.84 / 0.83 |
| > 0.85 | **0.81** | 0.75 | 0.70 | **0.69** | 0.98 / 0.87 / 0.85 / 0.65 | 0.58 / 0.72 / 0.62 |

The coverage standard error is at most 0.021, clustered by origin. In the most persistent fifth of windows, every model under-covers relative to the baseline, by 6–12 points. The model whose forecast scaling is most compressed (Bolt, implied H 0.65) is the most over-confident, yet all models are far more accurate in the median.

## Interpretation

1. **The fBm mechanism exists on real data, but it is second-order.** The models compress uncertainty growth on persistent series and lose coverage there. On real data that effect is much smaller than the gains from a better median.
2. **The no-future-data gap check is not yet a usable diagnostic.** It points the right way on prices, electricity and reservoirs, but explains little variance per window, and on realised volatility it points slightly the wrong way. Stage A's ρ = −0.82 was helped by fBm's clean, mechanical link between width scaling and coverage.
3. **Paper B's strongest framing is "accurate but over-confident at long horizons where memory is strongest",** with H as the explanatory variable, rather than "H mismatch predicts failure".

## Suggested next test (cheap)

A post-hoc, no-future-data fix: rescale each model's interval half-widths at horizon k by k^(Ĥ_context − H_forecast), keeping the median. If this restores coverage on the persistent reservoirs without losing the accuracy advantage, it is a practical, publishable intervention, the Phase 3 "H-matching fix", obtained cheaply. It needs a re-run that saves quantiles, about 15 minutes for the reservoirs.

## Reproduce

```bash
python pilot/prepare_real.py --binance DIR/futures/5m --entsoe zone_daily.parquet --out series_daily.csv
python pilot/prepare_real.py --water data.parquet --out series_water.csv
HF_HUB_DISABLE_XET=1 python pilot/gate1_real.py --data series_daily.csv --step 7 --out results/pilot_gate1_real
HF_HUB_DISABLE_XET=1 python pilot/gate1_real.py --data series_water.csv --step 7 --out results/pilot_gate1_reservoir
```

Window-level results: `windows.csv` here and in `../pilot_gate1_reservoir/`.
