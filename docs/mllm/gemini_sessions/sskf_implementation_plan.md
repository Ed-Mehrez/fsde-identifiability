# Implementation Plan: Unified Signature-Space Kalman Filter (SSKF)

## Objective

Unify Sig-GMM and Sig-KKF into a single, explicit **Signature-Space Kalman Filter** that operates directly on signature features.

## Theoretical Foundation

### State-Space Model in Signature Space

The signature transform $S(X)_{0,t}$ lifts fBM into a **Markov process** in the tensor algebra $T((V))$.

**State**: Truncated Signature $s_t = S^{(N)}(X)_{t-w,t}$ (rolling window of depth $N$).
**Dynamics**:
$$s_{t+1} = s_t \otimes \Delta X_t = s_t \otimes \left( -\theta X_t \Delta t + \sigma \Delta B^H_t \right)$$
This is **linear** in the state $s_t$ when $\theta, \sigma$ are known.

**Observation**: $y_t = X_t$ (the first-level signature term).

### Scientific Rationale: Why Hybrid?

Our investigation revealed a fundamental dichotomy:

1.  **Roughness ($H$) is Nonlinear**: It is encoded in the fine variations and covariance structure of the path. The **Signature Kernel** is ideal for this (via eigenvalue decay).
2.  **Drift ($\theta$) is Linear**: For fOU, the drift operator $L=-\theta I$ is unbounded and linear.
    - **RBF Kernels** (normalized or unnormalized) fail here: In rough regimes ($H<0.5$), they either overfit noise or regularize towards zero due to the smoothness prior.
    - **Linear Kernels** are numerically unstable due to infinite feature dimension.
    - **Solution**: **Linear RLS** on the state space is stable and imposes the correct bias.

### The Validated Online Architecture: **HS-RLS** (H-Adaptive Spectral RLS)

Our experiments proved that kernel regression consistently underestimates drift due to regularization bias ($X_{t+1} \approx X_t \implies L \approx 0$).
The robust solution is a **Hybrid Architecture**:

1.  **Online H-Estimation (Sig-GMM)**:
    - Use rolling window Signature Kernel Gram matrix.
    - Extract $H_t$ from eigenvalue decay.
    - **Status:** Validated (Bias $\approx -0.1$, consistent ordering).

2.  **Spectral Preconditioning**:
    - Apply Low-Pass Filter with cutoff $f_c \propto 1/H_t$.
    - This removes high-frequency noise that confuses the estimator.

3.  **Recursive Least Squares (RLS)**:
    - Apply RLS to the _filtered_ increments: $d\tilde{X}_t = -\theta \tilde{X}_t dt$.
    - **Why:** RLS imposes the correct linear inductive bias, unlike the kernel.
    - **Adaptivity:** Use forgetting factor $\lambda \approx 0.995$ to track changing $\theta$.

### Algorithm Steps

1.  **Initialize**: Buffer $B$, RLS state $(P, \hat{\theta})$.
2.  **On new $x_t$**:
    - Update buffer.
    - If $t \% \text{stride} == 0$: Update $H_t$ via Sig-GMM.
    - **Filter/Stride**: Apply stride $S \approx 10/H_t$ (decimation). This acts as a spectral filter to remove high-frequency fractional noise.
    - **RLS Step**:
      - Use strided increments: $\Delta \tilde{x}_t = x_t - x_{t-S}$.
      - $\Delta t_{eff} = S \cdot \Delta t$.
      - Update $\hat{\theta}$ using $\Delta \tilde{x}_t \approx -\hat{\theta} x_{t-S} \Delta t_{eff}$.
    - Return $\hat{\theta}$.

3.  **Auxiliary: Method of Moments (Sanity Check)**
    - Calculate global/windowed variance $V_X$ and quadratic variation $\hat{\sigma}^2$.
    - Invert stationary variance formula: $\hat{\theta}_{mom} \approx \left( \frac{\hat{\sigma}^2 \Gamma(2H+1)}{2 V_X} \right)^{\frac{1}{2H}}$
    - **Use Case**: Stabilizes estimates when drift is strong (where RLS bias is high).

### Limitations

- **High Drift ($\theta \gg 1$)**: Striding effectively increases $\Delta t$. If $\theta \cdot S \Delta t \sim 1$, linear approximation fails.
- **Solution**: The Hierarchical Gate detects high drift as "smoothness" ($H > 0.4$) and switches to standard (non-strided) experts.

---

## Implementation Steps

### Phase 1: Signature Feature Extraction

- [ ] Implement truncated signature computation (depth 3-5).
- [ ] Use `signatory` library for efficient computation.
- [ ] Define rolling window size $w$ (e.g., 20 steps).

### Phase 2: Signature Dynamics Model

- [ ] Derive the state transition matrix $F(\theta, \sigma)$ for $s_{t+1} = F \cdot s_t + \text{noise}$.
  - This is the tensor product operator acting on truncated signatures.
- [ ] Compute the process noise covariance $Q(H)$ from fGn properties.

### Phase 3: Kalman Filter Implementation

- [ ] Implement KF with:
  - State dimension: $\dim(S^{(N)}) = \sum_{k=0}^{N} d^k$ (e.g., $d=2, N=3$ → 15 dims).
  - Observation dimension: 1 (just $X_t$).
- [ ] Run KF to estimate the signature state.
- [ ] Extract $X_t$ prediction from signature state.

### Phase 4: Joint Parameter Estimation

- [ ] Use EM-style updates:
  - E-step: Run KF with current $(\theta, H)$ estimates.
  - M-step: Update $(\theta, H)$ from likelihood / residuals.
- [ ] Alternative: Bayesian filtering with parameter priors.

---

## Expected Outcome

A unified **Signature Koopman Kalman Filter** that:

1. Operates explicitly in signature space (not kernel trick).
2. Has provable Markovian dynamics.
3. Jointly estimates $(H, \theta, \sigma)$ online.

## Complexity

- **State dimension** grows exponentially with depth $N$.
- **Mitigation**: Use Random Fourier Features for signature kernel, or Nyström approximation.
