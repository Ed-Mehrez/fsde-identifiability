"""
Baseline Hurst Parameter Estimators

Implements standard methods for Hurst parameter estimation:
1. Detrended Fluctuation Analysis (DFA)
2. Log-Periodogram Regression
3. Variogram Method
4. R/S Analysis (original Hurst method)

Used as baselines to compare against spectral Koopman method.

References:
- Peng et al. (1994). "Mosaic organization of DNA nucleotides"
- Geweke & Porter-Hudak (1983). "Estimation of long memory"
- Hubert & Veraart (2019). "Estimation of integrated variance"
"""

import numpy as np
from scipy.stats import linregress
from typing import Tuple, Optional


def estimate_hurst_dfa(
    trajectory: np.ndarray,
    min_box: int = 4,
    max_box: Optional[int] = None,
    n_boxes: int = 20
) -> Tuple[float, dict]:
    """
    Detrended Fluctuation Analysis (DFA).
    
    Estimates Hurst parameter from the scaling of detrended fluctuations.
    
    Algorithm:
    1. Integrate the series: Y(i) = Σ(X(j) - mean)
    2. For each box size n:
       - Divide Y into non-overlapping segments
       - Fit polynomial trend in each segment
       - Compute RMS fluctuation F(n)
    3. Hurst from: F(n) ~ n^H
    
    Args:
        trajectory: 1D time series
        min_box: Minimum box size
        max_box: Maximum box size (default: len//4)
        n_boxes: Number of box sizes to test
        
    Returns:
        H: Estimated Hurst parameter
        info: Dict with diagnostics
    """
    X = np.asarray(trajectory)
    N = len(X)
    
    if max_box is None:
        max_box = N // 4
    
    # 1. Integrate (cumulative sum of deviations)
    Y = np.cumsum(X - np.mean(X))
    
    # 2. Box sizes (logarithmically spaced)
    box_sizes = np.unique(np.logspace(
        np.log10(min_box), np.log10(max_box), n_boxes
    ).astype(int))
    
    fluctuations = []
    
    for n in box_sizes:
        # Number of complete boxes
        n_segments = N // n
        
        if n_segments == 0:
            continue
        
        segment_fluctuations = []
        
        for i in range(n_segments):
            # Extract segment
            segment = Y[i*n : (i+1)*n]
            
            # Fit linear trend (degree 1 polynomial)
            t = np.arange(n)
            coeffs = np.polyfit(t, segment, deg=1)
            trend = np.polyval(coeffs, t)
            
            # Detrended segment
            detrended = segment - trend
            
            # RMS fluctuation
            F = np.sqrt(np.mean(detrended**2))
            segment_fluctuations.append(F)
        
        # Average fluctuation for this box size
        fluctuations.append(np.mean(segment_fluctuations))
    
    fluctuations = np.array(fluctuations)
    box_sizes = box_sizes[:len(fluctuations)]
    
    # 3. Log-log regression: log(F(n)) = H * log(n) + const
    log_n = np.log(box_sizes)
    log_F = np.log(fluctuations)
    
    slope, intercept, r_value, p_value, std_err = linregress(log_n, log_F)
    
    H = slope
    H = np.clip(H, 0.01, 0.99)
    
    info = {
        'box_sizes': box_sizes,
        'fluctuations': fluctuations,
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'p_value': p_value,
        'std_err': std_err
    }
    
    return H, info


def estimate_hurst_periodogram(
    trajectory: np.ndarray,
    trim_frac: float = 0.1
) -> Tuple[float, dict]:
    """
    Log-Periodogram Regression (Geweke-Porter-Hudak estimator).
    
    Estimates Hurst from the spectral density slope at low frequencies.
    
    For fBm, the spectral density S(f) ~ f^{-(2H+1)}
    
    Args:
        trajectory: 1D time series
        trim_frac: Fraction of low frequencies to use (default: 0.1)
        
    Returns:
        H: Estimated Hurst parameter
        info: Dict with diagnostics
    """
    X = np.asarray(trajectory)
    N = len(X)
    
    # Compute periodogram
    fft_vals = np.fft.fft(X - np.mean(X))
    periodogram = (np.abs(fft_vals)**2) / N
    
    # Frequencies
    freqs = np.fft.fftfreq(N)
    
    # Use only positive frequencies
    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    periodogram = periodogram[pos_mask]
    
    # Use low frequencies (trim_frac of spectrum)
    n_freqs = int(len(freqs) * trim_frac)
    n_freqs = max(10, n_freqs)  # At least 10 frequencies
    
    freqs_low = freqs[:n_freqs]
    periodogram_low = periodogram[:n_freqs]
    
    # Log-log regression: log(S(f)) = -(2H+1) * log(f) + const
    log_freq = np.log(freqs_low)
    log_period = np.log(periodogram_low)
    
    slope, intercept, r_value, p_value, std_err = linregress(log_freq, log_period)
    
    # Extract H: slope = -(2H+1)
    H = -(slope + 1) / 2
    H = np.clip(H, 0.01, 0.99)
    
    info = {
        'freqs': freqs_low,
        'periodogram': periodogram_low,
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'p_value': p_value,
        'std_err': std_err,
        'n_freqs_used': n_freqs
    }
    
    return H, info


def estimate_hurst_variogram(
    trajectory: np.ndarray,
    max_lag: Optional[int] = None,
    n_lags: int = 50
) -> Tuple[float, dict]:
    """
    Variogram Method.
    
    Estimates H from the scaling of the variogram:
        Var(X_{t+h} - X_t) ~ h^{2H}
    
    Args:
        trajectory: 1D time series
        max_lag: Maximum lag (default: len//4)
        n_lags: Number of lags to test
        
    Returns:
        H: Estimated Hurst parameter
        info: Dict with diagnostics
    """
    X = np.asarray(trajectory)
    N = len(X)
    
    if max_lag is None:
        max_lag = N // 4
    
    lags = np.unique(np.logspace(0, np.log10(max_lag), n_lags).astype(int))
    
    variogram = []
    
    for h in lags:
        if h >= N:
            continue
        
        # Compute increments at lag h
        increments = X[h:] - X[:-h]
        
        # Variogram = 0.5 * E[(X_{t+h} - X_t)^2]
        v = 0.5 * np.mean(increments**2)
        variogram.append(v)
    
    variogram = np.array(variogram)
    lags = lags[:len(variogram)]
    
    # Log-log regression: log(V(h)) = 2H * log(h) + const
    log_lag = np.log(lags)
    log_var = np.log(variogram)
    
    slope, intercept, r_value, p_value, std_err = linregress(log_lag, log_var)
    
    H = slope / 2
    H = np.clip(H, 0.01, 0.99)
    
    info = {
        'lags': lags,
        'variogram': variogram,
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'p_value': p_value,
        'std_err': std_err
    }
    
    return H, info


def estimate_hurst_rs(
    trajectory: np.ndarray,
    min_size: int = 10,
    max_size: Optional[int] = None,
    n_sizes: int = 20
) -> Tuple[float, dict]:
    """
    R/S Analysis (Rescaled Range Analysis).
    
    Original method by Hurst (1951). Estimates H from the scaling of
    the rescaled range statistic:
        R/S ~ n^H
    
    Args:
        trajectory: 1D time series
        min_size: Minimum segment size
        max_size: Maximum segment size (default: len//2)
        n_sizes: Number of segment sizes
        
    Returns:
        H: Estimated Hurst parameter
        info: Dict with diagnostics
    """
    X = np.asarray(trajectory)
    N = len(X)
    
    if max_size is None:
        max_size = N // 2
    
    sizes = np.unique(np.logspace(
        np.log10(min_size), np.log10(max_size), n_sizes
    ).astype(int))
    
    RS_values = []
    
    for n in sizes:
        n_segments = N // n
        
        if n_segments == 0:
            continue
        
        rs_segments = []
        
        for i in range(n_segments):
            segment = X[i*n : (i+1)*n]
            
            # Mean-adjusted cumulative sum
            Y = np.cumsum(segment - np.mean(segment))
            
            # Range
            R = np.max(Y) - np.min(Y)
            
            # Standard deviation
            S = np.std(segment)
            
            if S > 0:
                rs_segments.append(R / S)
        
        if rs_segments:
            RS_values.append(np.mean(rs_segments))
    
    RS_values = np.array(RS_values)
    sizes = sizes[:len(RS_values)]
    
    # Log-log regression: log(R/S) = H * log(n) + const
    log_n = np.log(sizes)
    log_RS = np.log(RS_values)
    
    slope, intercept, r_value, p_value, std_err = linregress(log_n, log_RS)
    
    H = slope
    H = np.clip(H, 0.01, 0.99)
    
    info = {
        'sizes': sizes,
        'RS_values': RS_values,
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'p_value': p_value,
        'std_err': std_err
    }
    
    return H, info


if __name__ == "__main__":
    # Quick test
    print("Testing baseline Hurst estimators...")
    
    from rough_paths_generator import generate_fbm
    
    methods = {
        'DFA': estimate_hurst_dfa,
        'Periodogram': estimate_hurst_periodogram,
        'Variogram': estimate_hurst_variogram,
        'R/S': estimate_hurst_rs
    }
    
    for H_true in [0.2, 0.5, 0.8]:
        print(f"\nTrue H = {H_true}")
        
        # Generate fBm
        path = generate_fbm(n_samples=5000, H=H_true, n_paths=1, seed=42)[0]
        
        for name, method in methods.items():
            H_est, info = method(path)
            error = abs(H_est - H_true)
            r2 = info['r_squared']
            
            print(f"  {name:12s}: H={H_est:.3f}, Error={error:.3f}, R²={r2:.4f}")
