# Direction 2 kill test: power-law forgetting for tracking long-memory drift

**Result: negative.** When the quantity being tracked drifts like fBm, the best *single* EMA is within 0–10% of the optimal linear filter for rough or random-walk drift (H ≤ 0.5). It loses 11–72% for persistent drift (H = 0.8–0.9). However, a tuned combination of **two** EMAs (the Holt / AdEMAMix family) is within **0.5–1.3%** of optimal at every H and noise level tested. Power-law kernels and log-spaced EMA banks add essentially nothing for this problem.

All numbers are exact Gaussian error variances, not simulations (`pilot/forgetting_killtest.py`, full output in `output.txt`). The setup: target θ_t = fBm(H) with unit step sd, observations y_t = θ_t + N(0, σ²), and the goal is to estimate θ_T from y_1..y_T with T = 1000.

**Error of the best single EMA ÷ error of the optimal filter**

| H | σ = 0.3 | 1 | 3 | 10 | 30 |
| --- | --- | --- | --- | --- | --- |
| 0.2 | 1.11 | 1.07 | 1.06 | 1.04 | 1.03 |
| 0.5 | 1.08 | 1.08 | 1.01 | 1.00 | 1.00 |
| 0.7 | 1.09 | 1.29 | 1.06 | 1.04 | 1.04 |
| 0.8 | 1.11 | 1.51 | 1.15 | 1.11 | 1.11 |
| 0.9 | 1.16 | 1.72 | 1.27 | 1.29 | 1.30 |

**Error of the best tuned two-EMA combination ÷ optimal:** 1.000–1.013 across H ∈ {0.2, 0.5, 0.7, 0.8, 0.9} and σ ∈ {1, 3, 10}.

**Interpretation.**
- Tracking the *current* state uses only a narrow band of timescales, set by the signal-to-noise ratio. Two timescales cover that band. The gain at high H is trend extrapolation, which a two-EMA difference (Holt's method) already captures.
- Power-law memory in the drift therefore does **not** call for power-law forgetting in the tracker.
- This rules out the "optimal forgetting under long-memory drift" framing of direction 2.
- It does not rule out problems that need *many* timescales at once, such as retaining information across all past ages (memory capacity or replay). Those are a different objective from tracking.
