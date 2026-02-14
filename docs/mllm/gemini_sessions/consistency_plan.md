# Experiment: Spectral Consistency Analysis

## Goal

Identify the physical drift parameter $\theta$ by exploiting its invariance to the multiresolution time-scale (Stride).

## Rationale

- **Noise Modes**: Arise from overfitting roughness. Their effective decay rate depends on the sampling frequency (Stride).
- **Physical Modes**: Represent the underlying differential equation. Their continuous-time rate $\theta$ should be intrinsic and independent of Stride (as long as Stride captures the dynamics).

## Protocol

1.  **Data**: fOU Process ($H=0.3, \theta=1.0$).
2.  **Sweep**: Stride $S = [15, 20, 25, 30, 35, 40, 45, 50]$.
3.  **Model**: Tensor Sig-KKF (Rank 50, Solver TLS).
4.  **Metric**: Continuous Eigenvalues $\mu = -\log(\lambda) / (S \cdot dt)$.
5.  **Analysis**:
    - Cluster eigenvalues across strides.
    - Compute Variance of each cluster center.
    - Select cluster with min Variance.

## Implementation

Target: `examples/test_spectral_consistency.py`
