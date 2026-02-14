# Implementation Plan: Signature Koopman Estimation

## Objective

Extract $H$ and $\theta$ directly from the **Koopman Operator** constructed in the Signature RKHS, rather than using ad-hoc regression.

## Theoretical Foundation

The infinitesimal generator $\mathcal{L}$ of the fOU process acts on test functions $f$ as:
$$ \mathcal{L}f(x) = -\theta x f'(x) + \frac{1}{2} \sigma^2 \dots $$
For the coordinate function $f(x) = x$:
$$ \mathcal{L}x = -\theta x $$
Thus, if we learn the Koopman operator $\mathcal{K}_\tau \approx e^{\tau \mathcal{L}}$, we can approximate the generator:
$$ \mathcal{L} \approx \frac{\mathcal{K}\_\tau - I}{\tau} $$
Then apply this to the state $x$:
$$ \mathcal{L}x \approx -\theta x \implies \theta \approx -\frac{\mathcal{L}x}{x} $$

## Implementation Steps

### 1. Construct Koopman Operator $\mathcal{K}_\tau$ in RKHS

- Use `sigkernel` to build the Gram matrix $G_{XX}$ and cross-covariance $G_{XY}$ (where $Y$ is time-shifted $X$).
- $K_\tau = G_{XX}^{-1} G_{XY}$ (implicitly).

### 2. Extract Drift $\theta$

- Identify the coordinate projection in RKHS (the function that maps path to current value).
- Apply $K_\tau$ to this function.
- Compare result to original state.

### 3. Extract $H$ from Operator Spectrum

- The spectrum of $\mathcal{L}$ for fOU is discrete: $-\theta n$ for $n=0,1,\dots$ independent of $H$. [Wait, actually H affects the eigenfunctions, not eigenvalues for classical OU, but for fOU the noise color interacts].
- **Alternate H Extraction**: The _roughness_ is encoded in the decay of the kernel matrix eigenvalues (as we established with Sig-GMM). We keep Sig-GMM for H.

## Why This Works Online

- We can update the operator estimate $K_\tau$ online (rank-1 updates to inverse covariance).
- It separates the _dynamics learning_ (operator) from _parameter extraction_ (interrogating the operator).
