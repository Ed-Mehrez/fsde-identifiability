# Implementation Plan: Robust Koopman Solver (TLS-DMD)

## The Problem

We observed that standard Ridge Regression yields Koopman operators with eigenvalues $\lambda \approx 1$ (Continuous $\theta \approx 0$).

- **Cause**: Ridge Regression solves $\min \|Y - XA\|^2$. It assumes $X$ is noiseless. When $X$ has noise (Roughness), Ridge "shrinks" the operator to reduce variance, causing the estimated dynamics to decay too fast (or in this case, persist as Identity if noise dominates).
- **User Insight**: "Use SOTA tools for eigenvectors."

## The Solution: Total Least Squares (TLS)

TLS assumes errors in _both_ inputs and outputs.
For the system $Y \approx X A$, we form the augmented matrix $Z = [X; Y]$.
We perform SVD on $Z$ to find the subspace that best approximates the data trajectory.

### Algorithm (TLS-DMD on Tensor Features)

1.  Construct Data Matrices:
    - $\Psi_X$: Features at time $t$.
    - $\Psi_Y$: Features at time $t+S$.
2.  Augment: $Z = [\Psi_X; \Psi_Y]$.
3.  SVD: $Z = U \Sigma V^T$.
4.  Partition $V$ into quadrants: $V = \begin{bmatrix} V_{11} & V_{12} \\ V_{21} & V_{22} \end{bmatrix}$.
5.  Solution: $A_{TLS} = -V_{12} V_{22}^{-1}$ (Classic TLS formula).

### Enhancements

- **Truncation**: Use the "OptSpace" or hard threshold on Singular Values of $Z$ to denoise before determining $A$.
- **Stabilization**: If $V_{22}$ is ill-conditioned, revert to Ridge or use pseudo-inverse.

## Target Implementation

Modify `TensorSigKKF.fit()` to accept `solver='tls'`.

```python
def fit(self, X, dt, stride, solver='tls'):
    if solver == 'tls':
        # ... TLS implementation ...
        self.A = ...
    else:
        # ... Ridge implementation ...
```

## Validation

Re-run `final_demo.py`.
Success condition: The **Eigenvalues** of $A_{TLS}$ yield $\hat{\theta} \approx 1.0$ directly.
