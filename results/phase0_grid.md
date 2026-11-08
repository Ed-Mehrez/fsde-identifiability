# Phase 0 grid

dt = 0.02, T = 1000, 8 replicates per cell, seed 0; everything estimated (H, sigma(x), mu(x)).
Errors are relative L2 on the 5-95% quantile range of X. Band coverage is the share of evaluation points inside the pointwise 95% band for mu, which conditions on H_hat and sigma_hat and ignores sieve bias, so values under 0.95 are expected.

| Model | H | H_hat - H (mean ± sd) | sigma rel. L2 | mu rel. L2 (mean ± sd) | mu band coverage |
| --- | --- | --- | --- | --- | --- |
| fOU (kappa=1, sigma=0.7) | 0.20 | -0.0031 ± 0.0029 | 0.014 | 0.177 ± 0.058 | 0.81 |
| fOU (kappa=1, sigma=0.7) | 0.35 | +0.0018 ± 0.0033 | 0.015 | 0.102 ± 0.024 | 0.98 |
| fOU (kappa=1, sigma=0.7) | 0.50 | +0.0010 ± 0.0046 | 0.020 | 0.084 ± 0.039 | 0.94 |
| fOU (kappa=1, sigma=0.7) | 0.65 | -0.0023 ± 0.0028 | 0.018 | 0.124 ± 0.032 | 1.00 |
| fOU (kappa=1, sigma=0.7) | 0.80 | -0.0018 ± 0.0032 | 0.023 | 0.202 ± 0.139 | 0.98 |
| double-well (a=1, sigma=0.7) | 0.20 | -0.0010 ± 0.0029 | 0.050 | 0.234 ± 0.078 | 0.89 |
| double-well (a=1, sigma=0.7) | 0.35 | -0.0014 ± 0.0036 | 0.026 | 0.147 ± 0.041 | 0.96 |
| double-well (a=1, sigma=0.7) | 0.50 | -0.0016 ± 0.0063 | 0.030 | 0.181 ± 0.053 | 0.90 |
| double-well (a=1, sigma=0.7) | 0.65 | -0.0034 ± 0.0053 | 0.026 | 0.173 ± 0.058 | 0.96 |
| double-well (a=1, sigma=0.7) | 0.80 | -0.0050 ± 0.0041 | 0.031 | 0.418 ± 0.207 | 0.95 |
| hyperbolic (kappa=1, s=0.5) | 0.20 | -0.0008 ± 0.0045 | 0.031 | 0.216 ± 0.068 | 0.70 |
| hyperbolic (kappa=1, s=0.5) | 0.35 | -0.0003 ± 0.0044 | 0.021 | 0.107 ± 0.040 | 0.90 |
| hyperbolic (kappa=1, s=0.5) | 0.50 | -0.0019 ± 0.0062 | 0.029 | 0.097 ± 0.049 | 0.90 |
| hyperbolic (kappa=1, s=0.5) | 0.65 | -0.0008 ± 0.0037 | 0.022 | 0.160 ± 0.094 | 0.79 |
| hyperbolic (kappa=1, s=0.5) | 0.80 | +0.0018 ± 0.0087 | 0.163 | 0.244 ± 0.127 | 0.94 |
