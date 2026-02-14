# Implementation Plan: Tensor-Accelerated Signature Kalman Filter (Tensor Sig-KKF)

## Goal

Implement a robust, computationally efficient **Tensor Sig-KKF** to solve the "Bad Parameter Recovery" problem.
By operating in a low-rank tensor subspace of the infinite signature feature space, we can:

1.  **Regularize** consistently (truncating small singular values = truncation of noise).
2.  **Filter** effectively (Kalman Update on low-rank state).
3.  **Invert** accurately (System ID on clean latent state).

## Theoretical Architecture

The "Infinite" Signature State $Z_t$ lives in a Hilbert Space $\mathcal{H}$.
The Koopman Operator $\mathcal{K}: \mathcal{H} \to \mathcal{H}$ is linear.

### 1. Tensor Representation (Approximation)

We approximate the operator $\mathcal{K}$ and covariance $P$ using **Low-Rank Factorizations** (Tensor Decomposition in Feature Space).
$$ \mathcal{K} \approx U \Sigma V^T $$
$$ P_t \approx L_t L_t^T $$
Where $U, V, L \in \mathbb{R}^{D \times r}$ ($r \ll D$ is rank).

Since $D$ (signature dim) is huge/infinite, we use the **Kernel Logic**:

- We never calculate $Z_t$ explicitly.
- We work with **Kernel Gram Matrices** $K(X, X')$.
- "Tensor Acceleration" here means using **Nyström Approximation** or **Random Fourier Features (RFF)** to create explicit low-dim feature maps $\phi(x) \in \mathbb{R}^r$ that approximate the tensor product kernel.

### 2. The Filter (K-KF)

We implement the **Kernel Kalman Filter (KKF)** using the low-rank tensor features.

- **State**: Weights $w_t$ in the feature space $\phi(\cdot)$.
- **Predict**: $w_{t+1} = A w_t$ (Transition learned via Regression).
- **Update**: $w_{t+1} = w_{t+1} + K_{gain} (y - C w_{t+1})$.
- **Inversion**: $\hat{\theta} \approx \text{Eigenvalues}(A)$.

## Implementation Roadmap

### Step 1: Tensor Feature Map (`src/sskf/tensor_features.py`)

- Implement `NystromFeatures`:
  - Select $M$ landmarks $\{z_j\}$.
  - Compute empirical kernel map $\psi(x) = K(x, z_j) \cdot \Lambda^{-1/2}$.
  - This creates an effective finite-dimensional proxy for the tensor signature state.

### Step 2: Tensor Sig-KKF Class (`src/sskf/tensor_filter.py`)

- **Fit (Offline/Batch)**:
  - Learn Transition $A$ via Ride Regression on Tensor Features: $\min || \Psi_{next} - \Psi A ||^2$.
  - Learn Output $C$ via Ridge Regression: $\min || Y_{next} - \Psi C ||^2$.
- **Filter (Online)**:
  - Standard KF recursion on the $r$-dimensional feature state.
  - $O(r^3)$ per step (very fast).

### Step 3: Parameter Inversion (`examples/test_tensor_kkf.py`)

- Train Tensor Sig-KKF on fOU data.
- Extract eigenvalues of $A$.
- Compare $\hat{\theta}_{tensor}$ vs $\hat{\theta}_{naive}$.
- Expectation: Nyström regularization keeps the eigenvalues stable.

## Why this solves the user's problem

- **Naive Kernel**: Ill-conditioned, overfits noise.
- **Tensor (Nyström) Kernel**: Explicitly limits rank. By selecting rank $r$ carefully (using the Spectral $H$-criterion), we force the filter to ignore the "Roughness Noise" subspace.
