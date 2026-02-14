# Optimization: Fast Spectral Consistency

## The Bottleneck

The current `test_spectral_consistency.py` re-runs `TensorSigKKF.fit()` for every stride.

- `fit()` calls `NystromFeatures.transform()`.
- `transform()` computes the Signature Kernel matrix (O(N*M*L)).
- This is done $K$ times for $K$ strides.

## The Fix: Precomputation

Since the features $\Psi(X_{t-w:t})$ depend only on the window, not the prediction horizon (Stride), we can:

1.  Compute matrix $\Psi_{all}$ of shape $(T, \text{Rank})$ **once**.
2.  For Stride $S$:
    - Construct $X = \Psi_{all}[0:-S]$
    - Construct $Y = \Psi_{all}[S:]$
    - Run TLS-SVD on $[X, Y]$.

## Expected Speedup

- Kernel Eval: ~60 seconds (once).
- SVD: ~0.1 seconds (per stride).
- Total for 10 strides: ~60.1s (vs ~600s). **10x-100x Speedup.**

## improving Accuracy (Rank Sweep)

With the speedup, we can afford to check if the bias ($0.65$ vs $1.0$) is due to low Rank.
We will sweep `Rank` in an outer loop.
