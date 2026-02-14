# Principled Drift Discovery (Stage 2)

## The Concept: Covariance Subtraction

We established in Stage 1 that we can accurately recover $\sigma$ and $H$ using a high-frequency (low stride) pass.
Now, for Stage 2 (Drift), we use a coarse bandwidth (large stride, e.g., $S=20$ or $50$).
At this scale, the observed feature covariance is a mix of Drift dynamics and integrated Fractional Noise.

$$ \Sigma*{obs} \approx \Sigma*{drift} + \Sigma\_{noise}(\sigma, H) $$

Since we know $(\sigma, H)$, we can explicitly estimate $\Sigma_{noise}$ via Monte Carlo simulation (generating pure fBM paths) and subtract it.

$$ \Sigma*{clean} = \Sigma*{obs} - \Sigma\_{noise} $$

Then, we perform Total Least Squares (TLS) on the cleaned covariance matrix to find the drift operator $A$.

## Algorithm

1.  **Stage 1 Result**: Assume we have $\hat{\sigma} \approx \sigma_{true}$ and $\hat{H} \approx H_{true}$.
2.  **Noise Calibration**:
    - Generate $K$ paths of pure fBM with $(\hat{\sigma}, \hat{H})$.
    - Compute features $\Psi_{noise}$.
    - Compute Covariance matrices $C_{XX}^{noise}, C_{XY}^{noise}, C_{YY}^{noise}$.
3.  **Data Processing**:
    - Compute features $\Psi_{data}$ from observed $X$.
    - Compute Covariance matrices $C_{XX}^{data}, C_{XY}^{data}, C_{YY}^{data}$.
4.  **Subtraction**:
    - $C_{XX}^{clean} = C_{XX}^{data} - C_{XX}^{noise}$ (and for XY, YY).
5.  **Solved**:
    - Construct Block Matrix $Z_{clean} = [[C_{XX}, C_{XY}], [C_{YX}, C_{YY}]]$.
    - Solve TLS SVD on $Z_{clean}$.
    - Extract $\theta$.

## Phase 2b: Refinement & Convergence

The user asked: "How can we improve upon it? More samples?"
We will investigate two axes of improvement:

1.  **Ensemble Size ($K$)**: Does averaging more noise proxies reduce the variance of the subtraction?
    - Sweep $K \in [10, 50, 100]$.
    - Script: `examples/test_drift_convergence.py`.
2.  **Iterative Refinement (Optional Phase 3)**:
    - If linear subtraction hits a limit (due to nonlinear cross-terms), we can use the estimated $\hat{\theta}$ to generate a _new_ proxy with drift, and iteratively minimize the distance between Proxy Covariance and Data Covariance.

## script: `examples/test_drift_discovery.py` & `examples/test_drift_convergence.py`

- `discovery` validates the concept.
- `convergence` tests the limit of accuracy.

## Why this is Principled

It relies on the **additivity of covariance** (assuming Signal and Noise are largely uncorrelated over long windows) and the **accuracy of the Stage 1 noise estimate**. It replaces a "Black Box" lookup table with an explicit noise model.
