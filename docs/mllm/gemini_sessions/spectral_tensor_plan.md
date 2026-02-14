# Final Plan: The Spectral Tensor Estimator

## Philosophy: "Principled and Optimized"

We have derived the necessary components. Now we integrate them into a single, rigorous Estimator.

### Why Prior Methods Failed

- **Naive Kernel**: Overfit Roughness noise (Principled but not Robust).
- **Pure Filtering**: Good state estimate, but biased Eigenvalues (Optimized but not Principled for ID).
- **Raw Stride**: Good for $\theta$, but noisy (Principled but High Variance).

### The Solution: Closed-Loop Spectral Tensor Sig-KKF

We combine them:

1.  **Tensor Sig-KKF** (Optimized State Estimator):
    - Uses Nyström/Tensor features to filter $X_t \to Z_t$ (Latent State).
    - Removes Roughness noise efficiently.
2.  **Spectral Calibration** (Principled Regularization):
    - Uses Filter Residuals to estimate $H$.
    - Sets the **Physics-Informed Stride** $S \sim 1/H$.
3.  **Latent Identification** (Robust Recovery):
    - Performs Linear Identification on the **Clean Latent Path** $Z_t$, not the rough observations.

---

## Implementation Specifications

### Target File: `src/sskf/spectral_tensor_estimator.py`

### Class `SpectralTensorEstimator`

#### 1. `fit(X, dt)`

Automates the calibration loop:

1.  **Coarse Pass (H-Estimation)**:
    - Train a small Rank-20 Tensor Sig-KKF (Stride=1).
    - Compute **Residual Trace** $\tau$.
    - Map $\tau \to \hat{H}$ (using our empirical log-linear law).
    - Determine Optimal Stride $S = \text{int}(C / \hat{H})$.

2.  **Fine Pass (Drift-Estimation)**:
    - Train full Rank-50 **Strided Tensor Sig-KKF** (Stride=$S$) on data.
    - Run the Filter to generate clean latent trajectory $Z_{path} = [\hat{z}_0, \hat{z}_1, \dots]$.

3.  **System ID**:
    - Solve $Z_{t+1} \approx \Phi Z_t$ (or just use the internal operator $A$).
    - Wait, internal $A$ was biased.
    - **Better**: Use the **Predicted Observables** $\hat{X}_t$ (Output of filter).
    - Fit fODE to clean $\hat{X}_t$: $\hat{X}_{t+S} - \hat{X}_t \approx -\theta \hat{X}_t (S \Delta t)$.
    - This is **Denoised RLS**.

#### 2. `predict(X)`

- Runs the calibrated filter online.

## Verification Experiment (`examples/final_demo.py`)

- **System**: Double Well (Cubic Drift) + Rough Noise ($H=0.3$).
- **Metric 1**: Drift Recovery Error (Target $< 10\%$).
- **Metric 2**: State Tracking RMSE (Target $< 0.2$).
