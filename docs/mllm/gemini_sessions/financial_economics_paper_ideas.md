# Financial Economics Paper Ideas for RKHS-KRONIC

This document outlines strong applications and theoretical contributions for developing rigorous academic papers geared toward financial economists using the RKHS-KRONIC framework.

---

## Paper 1: Dynamic Portfolio Hedging under Model Uncertainty

### Application

Multi-asset portfolio hedging with stochastic volatility and jumps

### Strong Points

- **Empirical validation** using real equity options data (SPX, VIX)
- Demonstrate superior out-of-sample hedging compared to Black-Scholes and Heston
- Show robustness to model misspecification (e.g., when true process has jumps but you hedge with diffusion model)

### Theory to Develop

1. **Nonparametric Hedge Ratio Estimation Theorem**: Prove convergence rates for signature-based delta estimation without parametric assumptions
2. **Robust Hedging Bounds**: Derive worst-case hedging error bounds when true volatility process deviates from assumed model
3. **Kernel Bandwidth Selection for Financial Time Series**: Principled bandwidth choice based on realized variance estimators

### Why Financial Economists Care

Model risk is THE central problem in derivatives pricing. Your framework provides a data-driven alternative to parametric calibration.

### Practical Implementation Guide

#### Data Sources (Free/Low-Cost)

**Problem**: SPX/VIX options data is expensive from Bloomberg/WRDS

**Solutions**:

1. **CBOE DataShop** (~$50-200/month): End-of-day options data for SPX, VIX
2. **Polygon.io** ($200/month): Options historical data with minute-level granularity
3. **Deribit API** (FREE for historical): Bitcoin/Ethereum options with excellent liquidity
   - **Best choice for research**: Free historical data, high-frequency tick data available
   - Liquid markets with tight spreads (comparable to equity options)
   - Well-documented API for both REST and WebSocket
4. **QuantConnect** (FREE tier): Daily options data for research purposes
5. **ORATS** ($50/month): Processed options data with implied vol surfaces

**Recommendation**: **Start with Deribit crypto options**

- Free historical data via API
- Plenty of data points even at daily frequency (BTC/ETH options trade 24/7)
- Panel structure: multiple strikes × multiple expirations = thousands of daily observations
- High volatility environment makes hedging errors easier to detect

**Granularity**: Daily OHLC is **sufficient** for:

- Delta hedging validation (daily rebalancing is realistic for academic papers)
- Regime detection (volatility regimes persist for weeks/months)
- Panel data analysis leverages cross-section (100s of strikes), not just time series

#### Baselines for Validation

- **Black-Scholes Delta**: Analytical benchmark (will underperform in stochastic vol)
- **Heston Semi-Closed Form**: Carr-Madan FFT pricing + finite difference Greeks
- **Practitioner's Delta**: Implied volatility surface + sticky-delta assumption
- **SABR Model**: Industry standard for vol surface (Hagan et al. 2002)

#### Success Metric

- 20-30% reduction in out-of-sample hedging variance vs Black-Scholes
- 10-15% improvement vs calibrated Heston (shows value of model-free approach)

---

## Paper 2: High-Dimensional Portfolio Optimization with Transaction Costs

### Application

Optimal execution and portfolio rebalancing for 50-500 assets

### Strong Points

- Standard HJB methods fail beyond ~5 assets
- Your kernel methods scale linearly in number of data points, not exponentially in dimensions
- Include realistic frictions: transaction costs, market impact, discrete rebalancing

### Theory to Develop

1. **Curse of Dimensionality Escape Theorem**: Prove that kernel-based Koopman approximation has sample complexity **polynomial** in dimension vs **exponential** for grid-based PDE
2. **Transaction Cost Regularization**: Show that L1/L2 penalties on control correspond to proportional/quadratic transaction costs
3. **Concentration Inequalities for Signature Features**: Finite-sample bounds on estimation error for high-dimensional signatures

### Why Financial Economists Care

Institutional investors (pension funds, endowments) NEED scalable portfolio optimization. Current methods don't work beyond toy examples.

### Practical Implementation Guide

#### How to Show "Good Enough" Performance

**Validation Strategy**: Show you approach theoretical optimum as dimension increases

**Baselines for Comparison**:

1. **Mean-Variance Optimization (Markowitz)**: Closed-form solution exists
   - Works up to ~20 assets with sample covariance matrix
   - Shows you match analytical solution in low dimensions
2. **1/N Naive Portfolio**: DeMiguel et al. (2009) showed this beats most methods out-of-sample
   - If you beat 1/N, you're doing well
3. **Risk Parity**: Industry standard (Bridgewater's All-Weather)
   - Weights proportional to inverse volatility
4. **Shrinkage Estimators**: Ledoit-Wolf covariance shrinkage
   - State-of-art for estimation error reduction

**Approximate Optimality Test**:

- **Toy Problem (N=5)**: Solve HJB numerically on grid → compare your policy
  - Should match within 5% of optimal Sharpe ratio
- **Medium Problem (N=20)**: Compare against Markowitz with perfect covariance
  - Your advantage: robust to estimation error
- **Large Problem (N=100)**: HJB impossible, compare against:
  - **Certainty Equivalent Loss**: $CEQ = \mu^T w - \frac{\gamma}{2} w^T \Sigma w$
  - Higher CEQ = better (can compute even if optimal policy unknown)

**Success Metrics**:

- Out-of-sample Sharpe ratio 15-25% higher than 1/N
- Turnover 30-50% lower than unconstrained Markowitz (transaction cost savings)
- Scales to N=100+ assets where competitors fail

---

## Paper 3: Regime-Switching Volatility and Path-Dependent Trading

### Application

Volatility timing and risk parity strategies that adapt to market regimes

### Strong Points

- Use signature features to detect early regime shifts (volatility clustering, crisis transitions)
- Compare against standard Markov regime-switching models (Hamilton 1989)
- Show that path-dependent features (moving averages encoded in signatures) predict regime changes

### Theory to Develop

1. **Path Signature as Sufficient Statistic for Hidden States**: Prove that under general non-Markovian dynamics, truncated signatures provide complete information for optimal control
2. **Regime Detection Bounds**: Derive minimax optimal detection delays for regime shifts using signature-based filters
3. **Memory Kernel Theory**: Characterize which path functionals (volatility memory, correlation memory) are captured by each signature level

### Why Financial Economists Care

Asset allocation critically depends on volatility regimes. Your method provides a principled way to incorporate path history beyond AR/GARCH.

### Practical Implementation Guide

#### Handling Hidden States: Validation Strategy

**The Challenge**: True regimes are unobservable, so how do we prove we're better?

**Solution**: Two-pronged validation approach

**1. Synthetic Data with Known Regimes**

- Simulate 2-state Markov regime-switching model (Hamilton 1989)
- State 1: Low vol (σ=10%), State 2: High vol (σ=40%)
- **Ground truth available**: You know true regime at each time
- **Test**: Does your signature-based detector identify regime BEFORE standard methods?
  - Metric: Average detection delay (in days)
  - Baseline: GARCH(1,1), MS-GARCH, Hidden Markov Model
  - Success: 20-30% faster detection than HMM

**2. Real Data with Ex-Post Labeling**

- Use NBER recession dates, VIX>30 periods, or COVID crash as "ground truth" regimes
- **Test**: Does portfolio rebalance BEFORE the regime fully materializes?
  - Example: Did you reduce equity exposure in Feb 2020 vs crash in March 2020?
  - Metric: Drawdown during known crisis periods
  - Baseline: Static allocation, GARCH-based signals

**Baselines for Comparison**:

1. **GARCH(1,1)**: Industry standard for volatility forecasting
2. **MS-GARCH**: Markov-Switching GARCH (Gray 1996)
3. **Hidden Markov Model**: Baum-Welch filter for regime probabilities
4. **Exponential Smoothing**: Simple moving averages of realized vol

**Why Signatures Win**:

- **Path integrals** capture momentum/acceleration of volatility changes
- Example: Signature level 2 contains $\int_0^t \sigma_s ds$ (cumulative vol) and $\int_0^t s \, d\sigma_s$ (vol trend)
- GARCH only uses current $\sigma_t^2$ and past $\sigma_{t-1}^2$

**Success Metrics**:

- 15-25% reduction in portfolio volatility during high-vol regimes
- 3-5 days earlier regime detection vs HMM (measurable in synthetic data)

---

## Paper 4: Intertemporal Hedging with Recursive Preferences (Epstein-Zin)

### Application

Long-horizon investors (endowments, sovereign wealth funds) with hedging demands

### Strong Points

- Disentangle risk aversion from intertemporal substitution (as in Epstein-Zin preferences)
- Show your bilinear controller **automatically learns** the hedging component without solving nonlinear PDE
- Validate on Campbell-Shiller stock-bond hedging portfolios

### Theory to Develop

1. **Automatic Hedging Demand Discovery**: Prove that the Riccati matrix $P$ in your bilinear regulator converges to the analytic hedging coefficients from Duffie-Epstein HJB
2. **Utility Gradient Decomposition**: Show signature features separate myopic vs hedging components in optimal portfolio
3. **Long-Run Risk Approximation**: Characterize bias when using finite-order signatures for infinite-horizon problems

### Why Financial Economists Care

Recursive preferences are THE standard for long-term investors (Campbell-Cochrane, Bansal-Yaron models). Solving the PDE is notoriously hard; a data-driven approach is revolutionary.

### Practical Implementation Guide

#### Validation in Toy Settings: Campbell-Shiller Hedging Portfolio

**The Baseline Exists!** Campbell & Viceira (2002) solved Epstein-Zin portfolio choice analytically for **simple settings**:

**Toy Setting That Has Closed-Form Solution**:

- **2 assets**: Stock + Bond
- **1 state variable**: Dividend yield or expected return predictor
- **Dynamics**: VAR(1) for returns and predictor
- **Known solution**: Myopic demand + Hedging demand (analytical formulas in Campbell-Viceira Ch. 7)

**Validation Protocol**:

1. **Simulate** the Campbell-Viceira economy (calibrated to US data)
2. **Compute** their analytical hedging portfolio weights
3. **Train** your signature-based bilinear controller on same simulated data
4. **Compare**: Do your learned weights match their analytical hedging term?
   - Should match within 5-10% (your advantage: no need to know the true VAR parameters)

**What to Show**:

- **Table**: Learned weights vs Analytical weights across different risk aversion γ
- **Decomposition**: Your Riccati matrix $P$ should separate into:
  - Diagonal: Myopic term (matches Merton)
  - Off-diagonal: Hedging term (matches Campbell-Viceira)

**Alternative Toy Problem**: Kim & Omberg (1996)

- **Same structure**, explicit solution for Gaussian case
- Validate that your method learns the "hedging slope" coefficient

**Baselines**:

1. **Myopic-only** (Merton solution ignoring hedging): You should beat this
2. **Campbell-Viceira analytical**: You should match this in toy case
3. **Misspecified VAR**: Estimate VAR, solve with wrong params → your robustness advantage

**Success Metric**:

- Match analytical solution within 5-10% in toy setting
- Outperform misspecified model-based solution by 20-30% in regime-switching extensions

---

## Paper 5: Option Market Making and Inventory Management

### Application

Real-time delta hedging for options market makers under inventory constraints

### Strong Points

- Incorporate bid-ask spreads, inventory limits, adverse selection
- Compare against Avellaneda-Stoikov and Guéant-Lehalle frameworks
- Use online learning (your recursive Koopman) for intraday adaptation

### Theory to Develop

1. **Online Regret Bounds for Hedging**: Prove sublinear regret for online signature-based hedging vs clairvoyant oracle
2. **Inventory Penalty Kernel Design**: Characterize optimal RKHS for inventory-constrained control (quadratic or asymmetric penalties)
3. **High-Frequency Signature Asymptotics**: Derive limiting behavior of signatures at tick-level (Hayashi-Yoshida type results)

### Why Financial Economists Care

Market making is a $100B+ industry. Practical, low-latency algorithms with theoretical guarantees are extremely valuable.

### Practical Implementation Guide

#### Framework Applicability: Avellaneda-Stoikov for Options?

**Short Answer**: Not directly, but easy to adapt

**Avellaneda-Stoikov (2008)**: Designed for **equity/FX market making**

- Sets bid-ask spreads to balance inventory risk vs adverse selection
- **Not for options** because they don't model delta hedging

**Guéant-Lehalle-Fernandez-Tapia (2013)**: Extends to derivatives

- Incorporates **delta hedging** + inventory management
- Solves stochastic control for optimal quotes GIVEN hedge ratio
- **Key assumption**: Delta is known (Black-Scholes or given)

**Your Contribution**: Learn the delta AND inventory policy jointly

- Their framework ASSUMES you know how to hedge → focuses on quoting
- Your framework LEARNS optimal hedge ratio from data → can plug into their quoting model

**Implementation**:

1. Use your signature-based delta estimator (replaces Black-Scholes delta)
2. Plug into Guéant-Lehalle framework for bid-ask optimization
3. Compare against:
   - **Baseline 1**: Black-Scholes delta + Guéant-Lehalle quotes
   - **Baseline 2**: Heston delta + Guéant-Lehalle quotes
   - **Your method**: Learned delta + Guéant-Lehalle quotes

**Baselines for Options Market Making**:

1. **Black-Scholes Delta Hedging**: Naive benchmark
2. **Stochastic Vol Aware**: Heston or SABR deltas
3. **Almgren-Chriss** (2000): Optimal execution (for closing positions)

**Success Metric**:

- 10-20% lower inventory variance (fewer blowups)
- 5-10% higher PnL per contract quoted

---

## Paper 6: Rough Volatility and Fractional Control

### Application

Option pricing and hedging when volatility is rough (Hurst < 0.5)

### Strong Points

- Rough volatility models (Gatheral et al. 2018) are empirically superior but analytically intractable
- Your signature methods naturally handle rough paths (via rough path theory integration)
- Show hedging performance outperforms misspecified smooth Heston

### Theory to Develop

1. **Rough Koopman Operator Theory**: Extend Koopman-RKHS framework to fractional Brownian motion and rough volatility
2. **Signature Truncation for Rough Paths**: Optimal signature order as function of Hurst parameter $H$
3. **Non-Markovian Hedging Optimality**: Prove path-dependent strategies are strictly optimal (not just approximately) under rough volatility

### Why Financial Economists Care

Rough volatility is the hottest topic in derivatives research (2018-2025). A tractable computational method would be **highly impactful**.

### Practical Implementation Guide

#### Data Sources for Rough Volatility Calibration

**Realized Volatility Data** (needed to estimate Hurst parameter):

1. **Oxford-Man Realized Library** (FREE): High-frequency realized measures
   - Daily realized variance for equities, FX, commodities
   - 5-minute returns → compute quadratic variation
2. **Deribit** (FREE): Crypto tick data
   - Estimate realized vol at multiple frequencies
   - Test rough vs smooth volatility hypotheses
3. **CBOE VIX Futures**: Forward variance curve
   - Implied term structure reveals roughness

**Estimating Hurst Parameter**:

**Traditional Methods** (for comparison baselines):

- **Log-periodogram regression** (Hubert & Veraart 2019)
- **Detrended Fluctuation Analysis** (Gatheral et al. 2018)
- Expected range: H ≈ 0.1 for SPX (very rough)

**YOUR NOVEL CONTRIBUTION: Spectral Koopman Estimation**

**Key Theoretical Result**: The Koopman operator eigenvalue spectrum directly encodes roughness:

$$\lambda_k \sim k^{-(2H+1)}$$

**Why This is Elegant**:

- **Model-free**: No need to specify fBm structure a priori
- **Direct extraction**: Hurst parameter from spectral decay rate
- **One framework**: Same Koopman operator used for hedging learns roughness

**Implementation**:

```python
# Fit Koopman operator
kgedmd = KernelGEDMD(kernel='rbf')
kgedmd.fit(volatility_trajectory)

# Extract eigenvalues
λ = kgedmd.eigenvalues()

# Estimate Hurst from spectral decay
# log(λ_k) ~ -(2H+1) * log(k)
slope = linregress(log(k), log(|λ_k|))
H_estimated = -(slope + 1) / 2
```

**NEW Theoretical Contribution**:

- **"Spectral Roughness Theorem"**: Prove KGEDMD eigenvalue decay recovers Hurst parameter
- **Convergence rates**: Sample complexity bounds for spectral estimator
- **Advantages over DFA**: Can handle non-stationary rough processes

#### Baselines for Comparison

**Smooth Models (Will Fail)**:

1. **Heston Model**: Assumes H=0.5 (standard Brownian)
   - Calibrate to same data → compare hedging P&L
2. **SABR Model**: CEV stochastic vol (smooth)
   - Industry standard for vol surface

**Rough Models (Competitors)**:

1. **Bayer-Friz-Gatheral (2016)**: Rough Heston simulation
   - Hybrid scheme for fractional processes (slow but accurate)
   - Your advantage: No need to simulate, learn from data
2. **McCrickerd-Pakkanen (2018)**: Turbocharging Monte Carlo
   - Fast rough vol simulation
   - Still requires knowing H, κ, ξ parameters

#### Validation Strategy

**1. Synthetic Rough Heston**: Simulate with known H=0.1

- Does your signature method capture roughness without knowing H?
- Baseline: Misspecified smooth Heston hedging
- Success: 30-50% reduction in hedging variance vs smooth model

**2. Real SPX Options Data**:

- Use Deribit crypto (free) or CBOE options
- Literature shows crypto vol is VERY rough (H≈0.05-0.15)
- Compare hedge P&L against smooth models

#### Why Signatures Win with Rough Paths

**Theoretical Advantage**:

- Signatures are DEFINED for rough paths (Lyons 1998)
- Work for any Hölder regularity α > 1/3 (covers all rough vol with H>0)
- Standard methods (Itô calculus) break down when H<0.5

**Practical Test**:

- **Feature comparison**:
  - GARCH uses {r_t, r_t-1} (Markovian, assumes smoothness)
  - Signatures use $\{S^1, S^{11}, S^{12}, ...\}$ (captures rough increments)
  - Show signatures encode fractional integration automatically

#### Success Metrics

- Hedging variance 30-50% lower than Heston when true data is rough
- No need to pre-specify Hurst parameter (learned implicitly from signatures)
- Scales to multidimensional rough vol (multiple assets with rough correlation)

#### **RECOMMENDED: Two-Paper Strategy**

**Paper 6a (Theory): "Spectral Estimation of Path Roughness via Koopman Operators"**

- **Scope**: Spectral Roughness Theorem + synthetic fBm validation
- **Contribution**: Novel Hurst estimator from Koopman eigenvalue decay
- **Validation**: Monte Carlo on synthetic rough Heston (known H)
- **Audience**: Mathematical finance (SIAM, Math Finance)
- **Timeline**: 3-4 months
- **Advantage**: Clean result, no data quality battles with reviewers

**Paper 6b (Empirical): "Model-Free Hedging under Rough Volatility"**

- **Scope**: Apply spectral method to real crypto/equity options
- **Contribution**: Show practical hedging gains over smooth models
- **Validation**: Deribit crypto options + SPX options (if budget allows)
- **Audience**: Finance journals (JFE, RFS, Journal of Finance)
- **Timeline**: 8-12 months (after 6a is published)
- **Advantage**: Cites Paper 6a for theory, focuses on empirical performance

**Why Split is Better**:

1. **Faster first publication**: Theory-only is cleaner, fewer reviewer concerns
2. **Two papers > one paper**: Better for CV/academic impact
3. **Risk mitigation**: Real data messiness doesn't contaminate theoretical result
4. **Cross-citations**: Paper 6b cites 6a, boosting both impact factors

---

## Recommended Starting Point

### Best First Paper: Paper 2 (High-Dimensional Portfolio) OR Paper 3 (Regime-Switching)

**Reasons**:

1. You already have Heston infrastructure → extend to multiple assets
2. Strong empirical differentiation vs existing methods (PDE solvers fail, GARCH misses path dependence)
3. Clear metric for success: Sharpe ratio, turnover, out-of-sample performance
4. Both theory and experiments are achievable in 3-6 months

**For Paper 2**:

- Use sector ETFs (10-20 assets)
- Add transaction costs
- Compare against mean-variance optimization and risk parity

**For Paper 3**:

- Use VIX/SPX data with known regime shifts (2008 crisis, COVID crash)
- Show your signature features "see" the regime before it fully transitions

---

## Core Theoretical Contributions (ALL Papers)

For **all** papers, emphasize these theoretical novelties that financial economists will value:

1. **Sample Complexity Bounds**: How much data do you need? Prove $O(n^{-1/2})$ rates

2. **Model-Free Optimality**: What class of dynamics can you handle without parametric assumptions?

3. **Computational Complexity**: Polynomial vs exponential scaling proofs

4. **Asymptotic Consistency**: Convergence to HJB solution as bandwidth → 0 and sample → ∞

5. **Robust Performance Guarantees**: Worst-case bounds under model misspecification

---

## Next Steps

I can help you:

1. **Develop a detailed research plan** for one of these papers
2. **Write the theoretical proofs** for specific results
3. **Design empirical experiments** with real data
4. **Draft paper outlines** in financial economics journal style (e.g., Journal of Finance, Review of Financial Studies)
