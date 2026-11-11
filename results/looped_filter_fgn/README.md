# Link to "Looped Reasoning as Closed-Loop Filtering" (Rozwood, draft v0)

The draft's §5 treats candidate noise along the loop axis as white and the drifting target as a random walk. Under those assumptions the optimal decay is a Kalman gain, which is a single EMA. Both assumptions are the H = 1/2 case of fractional processes. Two exact toy calculations show what changes when the loop-axis processes have long memory.

## 1. Proposition 4 with fGn(H) candidate noise

Exact, from `pilot/looped_filter_fgn.py`. The setup is the draft's toy mode μ = −1.8, whose stability limit is gain < 0.714. The steady-state error variance at a solution scales as **V ∝ γ^(2−2H)** at small gains, not ∝ γ.

| H of candidate errors | V(0.02) / V(0.2): a 10× slower filter | Fitted slope (theory 2 − 2H) |
| --- | --- | --- |
| 0.3 (anti-persistent, e.g. flip-mode oscillation) | 0.028 | 1.45 (1.40) |
| 0.5 (white: the draft's case) | 0.074 | 1.04 (1.00) |
| 0.7 | 0.207 | 0.62 (0.60) |
| 0.9 (persistent errors) | 0.590 | 0.21 (0.20) |

When candidate errors persist across iterations, slower forgetting buys little variance reduction. At H = 0.9, a 10× slower filter cuts variance only 1.7×, against 13.5× for white noise. Slower forgetting still costs search speed. When errors anti-persist, as with the two-step flip oscillations of §4.2, averaging is far more effective than the white-noise theory predicts.

## 2. Proposition 5 with an fBm(H) drifting target

This uses `pilot/forgetting_killtest.py` (`../forgetting_killtest/`). For a target drifting as fBm(H), observed through white noise:

- The single-decay filter (the draft's Kalman gain, InfiLoop's form) is optimal at H = 1/2, recovering Prop. 5.
- It loses 11–72% in error variance at H = 0.8–0.9.
- A tuned two-decay filter is within 0.5–1.3% of optimal at every H.

## Testable predictions for trained looped reasoners

1. The innovations z_t − h_{t−1} along the loop have a measurable Hurst exponent. Persistent errors (H > 1/2) would explain why InfiLoop's decay matters for *finding* solutions (§10) and not for holding them.
2. Search trajectories have an MSD exponent, E‖h_{t+k} − h_t‖² ∝ k^(2H). It should separate solved, lost (overthinking TRM) and wandering puzzles, and give a label-free halting signal.
3. If the drift during search is persistent, a two-decay outer filter for InfiLoop finds more solutions than one decay. This can be tested at inference time, as with the decay sweep in Fig. 7b.
