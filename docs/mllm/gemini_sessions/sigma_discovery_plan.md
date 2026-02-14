# Principled Sigma Discovery Plan

## The User's Insight

"Lookup tables are crap." -> We need to determine parameters from the _data structure_, not a patch.
"Learn H, then optimize sampling to find Sigma."

## The Physics of Scale

The observed signal increments are:
$$ \Delta X \approx \underbrace{-\theta X \Delta t}_{\text{Drift}} + \underbrace{\sigma \Delta B_H}_{\text{Diffusion}} $$

- Drift Power: $P_{drift} \propto \theta^2 \Delta t^2$
- Diffusion Power: $P_{diff} \propto \sigma^2 \Delta t^{2H}$

Ratio (Signal-to-Noise of Drift):
$$ SNR*{drift} = \frac{P*{drift}}{P\_{diff}} \propto \Delta t^{2-2H} $$

## Optimization Strategy

To find **Sigma ($\sigma$)**:
We want **minimum Drift contamination**.
$$ \text{Minimize } \Delta t^{2-2H} $$
Since $H < 1$, the exponent is positive.
**Conclusion**: Smallest possible Stride ($S=1$) is optimal for $\sigma$.
**Algorithm**:

1. Fix Stride $S=1$.
2. Target $Y = (\Delta X)^2$.
3. Train Sig-KKF -> Learn $\mathbb{E}[(\Delta X)^2 | X]$.
4. Result $/ \Delta t^{2H} = \sigma^2$.

To find **Theta ($\theta$)**:
We want **maximum Drift Signal**.
**Conclusion**: Largest possible Stride ($S_{max}$) before nonlinearity breaks.
But we know "Attenuation Bias" exists.
If we know $\sigma$ from Step 1, we can we use it?
Maybe simply: The "Bias" is just the un-accounted Diffusion energy?
We will verify Step 1 first.

## Script: `examples/test_sigma_discovery.py`

1. Generate fOU.
2. Asssume H known (e.g. 0.3).
3. Sweep Stride $S$.
4. Train `TensorSigKKF` to predict Squared Increment.
5. Estimate $\hat{\sigma}$.
6. Show that Error minimizes at low $S$.
