# BTC Uncertain Volatility Backbone — Research Plan

## Goal

Replicate the SPX uncertain-vol backbone analysis for BTC options: fit a regime-dependent 3-state displaced-diffusion mixture to ATM IV, extend to OTM via additive/affine shifts, and produce the same set of plots and diagnostics.

---

## Context: What Was Done for SPX

The SPX pipeline (in `Vol Backbone Research Extended__3.ipynb` and `Uncertain Vol - OTM Backbone Fits.ipynb`) proceeds as follows:

1. **Raw data**: SPX options CSVs with `datetime, strike, call_bid, call_ask, put_bid, put_ask, underlying, expiry`
2. **Cleaning**: `call_mid > 0`, bid-ask spread `< 25%` of mid, `tau > 1/365`, US holiday expiry rollback
3. **IV computation**: BS call IV from `call_mid` (Brentq, sigma ∈ [1e-6, 5.0]) with arbitrage checks
4. **ATM IV extraction**: Per (date, expiry), select strikes with `|log-moneyness| < 0.05`, require ≥3 points, linear interpolation at k=0
5. **Maturity interpolation**: Interpolate to 30-day maturity in **total-variance space** (`σ² × τ`), requiring bracketing of TARGET_TAU
6. **OTM IV extraction**: Puts for k<0, calls for k≥0, same total-variance interpolation per moneyness level
7. **Regimes**: Year-based — Low Vol (2015-17), Transition (2018-19), COVID (2020), Post-COVID (2021+)
8. **Model**: 3-state displaced-diffusion mixture, 9 params (3 softmax weights, 3 betas, 3 sigmas_ln), MSE loss, L-BFGS-B optimizer
9. **Fitting**: β bounded to (0.1, 0.95) to force negative skew; σ_ln bounded to (0.01, 2.0); σ_n anchored to S_ref = mean(S_data)
10. **OTM extension**: Additive shift δ_k = mean(OTM IV) − mean(ATM model IV), justified by β ≈ 1 in affine regression

Key findings from SPX:
- States 1 & 2 are "dead" (σ_ln at lower bound 0.01) — model reduces to near-single-state
- β ≈ 1 for active states validates additive shift for OTM
- Bachelier mixture rejected as degenerate at ATM
- Additive shift limitations: assumes constant OTM-ATM spread across spot range

---

## Phase 1: Data Acquisition & Processing

### 1.1 Source BTC Options Data

**Primary source**: Deribit (~90% BTC options volume) via `RiveChen/deribit-historical-data` scraper

| Source | Cost | Historical Depth | Notes |
|--------|------|-----------------|-------|
| Deribit API (via scraper) | Free | Full historical **trade** data (not order book) | Trade records include `mark_price` and `iv` per trade; ~1-2 hrs for BTC options, ~10GB |
| Tardis.dev | Paid | Full historical order book + trades + mark IV | Best data quality; includes continuous bid/ask snapshots and mark_iv time-series. Use if trade-only data has insufficient OTM coverage |
| Deribit API (direct) | Free | Recent trade snapshots only | `get_historical_volatility` gives realized vol (NOT per-strike IV); `get_last_trades_by_instrument_and_time` gives per-trade data including `iv` field |
| CME BTC options | Free via CME Datamine | From ~2022 | Institutional, far less liquid than Deribit |

**What the Deribit trade data provides per trade record**:
- `timestamp`, `price` (trade price in USD), `amount` (size in BTC/USD)
- `mark_price` (Deribit's mark price at trade time)
- `iv` (implied volatility computed by Deribit using the correct BTC-settled formula)
- `index_price` (underlying BTC index price)
- `instrument_name` (e.g., `BTC-28MAR26-100000-C`)

**What is NOT available historically**:
- Bid/ask time-series (only current snapshots via `public/get_order_book`)
- Mark IV time-series (only current via `public/ticker`)
- Continuous IV surface snapshots (must be self-collected or reconstructed from trade data)

**Data acquisition strategy**:
1. Use `RiveChen/deribit-historical-data` to download all BTC option trades
2. Aggregate trades into daily snapshots at 08:00 UTC (see §1.3)
3. Use Deribit's `iv` field as primary IV source (correctly computed for BTC-settled options)
4. If OTM coverage is insufficient from trade data alone, consider:
   a. Supplementing with Tardis.dev order book data (paid)
   b. Polling Deribit API going forward for daily mark_iv snapshots
   c. Computing IV ourselves using bid/ask mid prices and the BTC-settled formula (§1.4)

**Time span**: 2020–present. Data quality before 2020 is too sparse (Deribit BTC options liquidity was thin pre-2020). Use 2020-01-01 onwards.

### 1.2 BTC Option Settlement & IV Computation

**Critical difference from SPX**: Deribit BTC options are European-style, **cash-settled in BTC** with payoff:

```
Call payoff (in BTC) = max(S_T − K, 0) / S_T
Put payoff (in BTC)  = max(K − S_T, 0) / S_T
```

where S_T is the BTC/USD index price at expiry (30-min TWAP ending at 08:00 UTC on expiry day).

The division by S_T makes these **quanto-style** options. The correct BTC-denominated pricing formulas are:

```python
def BlackScholesLognormalCall_BTC(S, K, r, sigma, T):
    """BTC-settled call option price (in BTC units)."""
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return norm.cdf(d1) - (K / S) * np.exp(-r * T) * norm.cdf(d2)

def BlackScholesLognormalPut_BTC(S, K, r, sigma, T):
    """BTC-settled put option price (in BTC units)."""
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return (K / S) * np.exp(-r * T) * norm.cdf(-d2) - norm.cdf(-d1)
```

Note: `BlackScholesLognormalCall_BTC(S, K, r, sigma, T) = BlackScholesLognormalCall(S, K, r, sigma, T) / S`. The standard USD-settled BS call price divided by spot gives the BTC-settled price. However, the IV that produces the correct price for the BTC-settled formula is **not** the same as the IV for the standard BS formula — the brentq root-finding must use the BTC-settled formula.

**IV computation strategy (two-tier)**:

| Priority | Method | When to use |
|----------|--------|-------------|
| **Primary** | Use Deribit's `iv` field from trade records | Always, when available. Deribit computes IV using the correct BTC-settled pricing model internally. This is the most reliable source. |
| **Validation/Fallback** | `impliedVolatility_BTC_settled()` in `option_pricers.py` | When Deribit `iv` is not available (e.g., gaps in trade data, or we want to verify Deribit's IV). Requires bid/ask mid price in BTC. |

The fallback IV solver:

```python
def impliedVolatility_BTC_settled(S, K, r, price_btc, T, payoff):
    """Implied vol for BTC-settled option given price in BTC units."""
    try:
        if payoff.lower() == 'call':
            return brentq(
                lambda sigma: price_btc -
                    BlackScholesLognormalCall_BTC(S, K, r, sigma, T),
                1e-6, 10.0)
        elif payoff.lower() == 'put':
            return brentq(
                lambda sigma: price_btc -
                    BlackScholesLognormalPut_BTC(S, K, r, sigma, T),
                1e-6, 10.0)
    except Exception:
        return np.nan
```

**Important note on the model**: The uncertain-vol backbone model (`uncertain_backbone`) must also use BTC-settled formulas. We need:

```python
def DisplacedDiffusionCall_BTC(S, K, r, sigma_n, sigma_ln, beta, T):
    """BTC-settled displaced-diffusion call price."""
    S_ = S + sigma_n * (1 - beta) / (sigma_ln * beta)
    K_ = K + sigma_n * (1 - beta) / (sigma_ln * beta)
    sigma_ = sigma_ln * beta
    return BlackScholesLognormalCall_BTC(S_, K_, r, sigma_, T)

def DisplacedDiffusionPut_BTC(S, K, r, sigma_n, sigma_ln, beta, T):
    """BTC-settled displaced-diffusion put price."""
    S_ = S + sigma_n * (1 - beta) / (sigma_ln * beta)
    K_ = K + sigma_n * (1 - beta) / (sigma_ln * beta)
    sigma_ = sigma_ln * beta
    return BlackScholesLognormalPut_BTC(S_, K_, r, sigma_, T)

def uncertain_backbone_BTC(S, K, r, T, weights, betas, sigmas_n, sigmas_ln):
    """BTC-settled uncertain-vol backbone."""
    prices = [DisplacedDiffusionCall_BTC(S, K, r, sigma_n, sigma_ln, beta, T)
              for beta, sigma_n, sigma_ln in zip(betas, sigmas_n, sigmas_ln)]
    model_price = np.dot(weights, prices)
    model_vol = impliedVolatility_BTC_settled(S, K, r, model_price, T, 'call')
    return model_vol
```

### 1.3 Daily Snapshot Construction

**Observation time**: **08:00 UTC** (aligned with Deribit's daily settlement TWAP window, minimizing time-of-day effects)

**Tau computation**: Exact timestamp-based, not calendar-day approximation.

```python
from datetime import datetime, timezone, timedelta, time

DERIBIT_EXPIRY_TIME = time(8, 0)  # 08:00 UTC — Deribit settlement time
OBSERVATION_TIME = time(8, 0)      # Daily snapshot time

# For each trade record:
observation_ts = datetime.combine(trade_date, OBSERVATION_TIME, tzinfo=timezone.utc)
expiry_ts = datetime.combine(expiry_date, DERIBIT_EXPIRY_TIME, tzinfo=timezone.utc)
tau = (expiry_ts - observation_ts).total_seconds() / (365.0 * 86400)
```

This is critical for short-dated BTC options where a 1-day error in tau dramatically affects IV. The SPX pipeline's calendar-day approximation (`(expiry - datetime).dt.days / 365`) is insufficient for BTC because:
- Deribit options expire at a precise time (08:00 UTC), not "end of business day"
- The 24/7 market has no single "close" time
- Short-dated BTC options are extremely sensitive to tau errors

**Daily aggregation strategy**: For each (date, instrument) pair, take the trade(s) closest to 08:00 UTC. If no trade within ±30 minutes, skip that day for that instrument. Use Deribit's `iv` field from the closest trade as the daily IV observation.

### 1.4 Data Cleaning (Adapting SPX Pipeline)

| SPX Filter | BTC Adaptation | Rationale |
|-----------|---------------|-----------|
| `call_mid > 0` + `bid-ask < 25%` | Use Deribit `iv` field directly when available; for fallback IV computation, require `mid_price_btc > 0` and `spread_btc / mid_btc < 30-40%` | Deribit IV is pre-computed correctly; wider spreads expected for BTC |
| `tau > 1/365` | `tau > 3/365` (tightened from SPX) | BTC daily expiries are very noisy; microstructure issues irreparable at 1-day tenor |
| US holiday expiry rollback | **Remove entirely** | BTC trades 24/7; no holiday calendar needed |
| Timestamp at 16:15 ET | **08:00 UTC** (see §1.3) | 24/7 market needs a consistent daily anchor; 08:00 UTC aligns with Deribit settlement |
| `r = 0.02, q = 0.015` | Derive `r - q` from futures basis; use `r ≈ 0` as first approximation for short-dated options | No dividends; forward pricing is fundamentally different (see §1.5) |
| Calls for k≥0, puts for k<0 | **Use whichever side has tighter bid-ask or more trades**; puts dominate on Deribit | BTC put market is more liquid across moneyness |
| `implied_vol_call()` for IV | Use Deribit `iv` field as primary; `impliedVolatility_BTC_settled()` as fallback | BTC-settled options require a different pricing formula |

**BTC-specific considerations**:
- **Settlement**: BTC options are cash-settled in BTC with a 30-min TWAP settlement price (see §1.2)
- **Strike spacing**: Deribit BTC strikes are spaced at $500–$10,000 intervals depending on expiry and BTC level. Coarser than SPX — may need wider ATM window (0.10 vs 0.05)
- **Liquidity concentration**: BTC options heavily concentrated in short-dated tenors and near-ATM strikes. Deep OTM may have very sparse trade data.
- **Instrument naming**: Deribit uses `BTC-DDMMMYY-STRIKE-C/P` format (e.g., `BTC-28MAR26-100000-C`). Parse strike and expiry from this.

### 1.5 Forward Price & Cost of Carry

Deribit BTC options are priced against the **forward price** implied by the BTC futures/perpetual swap curve, not against a fixed dividend yield.

**Approach** (in order of preference):

1. **Use Deribit's `index_price` and `mark_price` fields** — these already incorporate the correct forward pricing. The `iv` field is computed using the implied forward.
2. **Derive from futures data**: If separate forward rates are needed, download BTC futures prices from Deribit and compute:
   ```
   r - q = ln(F/S) / T
   ```
   where F = BTC futures price for expiry T, S = BTC spot.
3. **Simple approximation**: For short-dated options (T < 60 days), use `r = 0`. This is standard practice for BTC options pricing and introduces negligible error at short tenors.

For the backbone fitting, the displaced-diffusion model uses `r` primarily in the BS pricing formula. Since `r = 0` and `T = 30/365 ≈ 0.082`, the discount factor `exp(-rT)` is very close to 1 for typical USD interest rates. The model is **much more sensitive to the vol parameters** than to `r`.

**Recommendation**: Use `r = 0` for the backbone fitting (Phase 3). This eliminates the need to specify `q` separately. Validate by comparing with Deribit's `iv` values at a sample of points.

### 1.6 ATM IV Extraction

Same method as SPX, with BTC adaptations:

```python
ATM_MONEYNESS_WINDOW = 0.10  # wider than SPX (0.05) due to coarser strikes
MIN_ATM_POINTS = 3

for (date, expiry), g in df.groupby(["date", "expiry"]):
    g = g.sort_values("log_moneyness")
    near = g[np.abs(g["log_moneyness"]) < ATM_MONEYNESS_WINDOW]
    if len(near) < MIN_ATM_POINTS:
        continue

    # Use Deribit iv field if available; otherwise compute IV
    if "iv_deribit" in near.columns and near["iv_deribit"].notna().all():
        iv_source = near["iv_deribit"]
    else:
        iv_source = near["iv_computed"]

    f = interp1d(near["log_moneyness"], iv_source,
                 kind="linear", fill_value="extrapolate")
    atm_list.append({"date": date, "expiry": expiry,
                     "tau": g["tau"].iloc[0], "spot": g["underlying"].iloc[0],
                     "atm_iv": float(f(0.0))})
```

Then **interpolate to 30-day maturity in total-variance space** (critical — not in vol space):

```python
TARGET_TAU = 30 / 365

for date, g in atm_df.groupby("date"):
    g = g.sort_values("tau")
    if g["tau"].min() > TARGET_TAU or g["tau"].max() < TARGET_TAU:
        continue
    total_var = g["atm_iv"] ** 2 * g["tau"]
    f = interp1d(g["tau"], total_var, kind="linear")
    tv_1m = float(f(TARGET_TAU))
    iv_1m = np.sqrt(tv_1m / TARGET_TAU)
```

**Output**: `processed_data/btc_1M_ATM.csv` with columns `date, spot, atm_iv_1m, year`

### 1.7 OTM IV Extraction

- **Target moneyness levels**: `k = -0.30, -0.20, -0.10, 0, +0.10, +0.20, +0.30` — wider than SPX (`±0.20`) since BTC has fatter tails and more OTM activity
- **Put/call selection**: Use whichever side has more trades/lower spread at each moneyness level. On Deribit, puts dominate for k < 0 and may also be more liquid for k ≥ 0.
- Same total-variance interpolation to 30 days per (date, k) group
- Assign regime labels (Phase 2)
- **OTM coverage caveat**: Trade data may be sparse for deep OTM strikes. Consider:
  - Requiring minimum trade count per (date, k) bucket
  - Using wider time windows (e.g., ±2 hours around 08:00 UTC) for OTM strikes
  - Accepting gaps in the data rather than forcing interpolation from thin data

**Output**: `processed_data/btc_iv1m_k_pc.csv` with columns `date, spot, k, iv_1m, year, regime`

### 1.8 Data Quality Checks

**Filters to apply** (mirroring SPX pipeline, with BTC adaptations):

| Filter | Condition | Rationale |
|--------|-----------|-----------|
| Valid Deribit IV | `iv_deribit > 0` when available | Remove records where Deribit couldn't compute IV |
| Mid price > 0 (for fallback IV) | `price_btc > 0` | Remove zero-priced records |
| Bid-ask spread threshold | `spread_btc / mid_btc < 30-40%` | BTC options are less liquid |
| Minimum tau | `tau > 3/365` | Avoid ultra-short-dated noise; BTC daily expiries are very noisy |
| Minimum ATM neighbors | `len(near) >= 3` | Need 3+ points for interpolation |
| Bracket 30-day maturity | `tau.min <= 30/365 <= tau.max` | Required for total-variance interpolation |
| IV bounds check | `0 < iv < 5.0` (wider than SPX's 3.0) | BTC vol can exceed 200% |
| Non-negative total variance | `tv_1m > 0` | Arbitrage constraint |
| Minimum trade proximity to 08:00 UTC | Trade within ±30 min of 08:00 UTC | Ensure representative daily snapshot |
| Minimum trade count per bucket | ≥1 trade per (date, expiry, strike) bucket | Avoid stale prices |

---

## Phase 2: Regime Identification (Data-Driven)

### 2.1 Exploratory Analysis

- Plot BTC 1M ATM IV time series
- Compute descriptive stats: mean, median, std, percentiles by year
- Visualize volatility clustering
- Compare with SPX ATM IV time series

### 2.2 Change-Point Detection

**Preferred method**: PELT (Pruned Exact Linear Time) via `ruptures` library

```python
import ruptures as rpt

signal = btc_atm_df["atm_iv_1m"].values
algo = rpt.Pelt(custom_cost=rpt.costs.CostL2()).fit(signal)
breakpoints = algo.predict(pen=10)  # penalty to be tuned via BIC
```

Alternative methods to compare:
- **Quantile-based**: Split IV distribution into terciles or quartiles
- **K-means**: Cluster on (IV level, IV change, IV curvature) features
- **Hidden Markov Model**: Fit a 2-3 state HMM on IV levels

### 2.3 Likely BTC Regimes (To Be Confirmed by Change-Point Detection)

| Tentative Regime | Period | BTC Context |
|-----------------|--------|-------------|
| Low Vol Pre-COVID | Jan 2019 – Feb 2020 | BTC $3k–$10k, relatively calm |
| COVID Crash & Recovery | Mar 2020 – Dec 2020 | BTC $4k→$29k, vol spike then recovery |
| Bull Market | Jan 2021 – Nov 2021 | BTC $30k→$69k→$47k, high vol |
| Bear/FTX | Dec 2021 – Dec 2022 | BTC $47k→$16k, FTX collapse Nov 2022 |
| Recovery/ETF | Jan 2023 – present | BTC $16k→$100k+, declining vol, ETF approval Jan 2024 |

**Note**: BTC regime transitions are faster than SPX. Expect more change-points. Let the data (PELT) decide, not fixed year boundaries.

---

## Phase 3: Model Fitting

### 3.1 Model Specification

**Same structure**: 3-state displaced-diffusion mixture, but using BTC-settled pricing formulas.

```python
# 9 parameters: weights (3), betas (3), sigmas_ln (3)
# Price = sum(w_i * DisplacedDiffusionCall_BTC(S, K, r, sigma_n_i, sigma_ln_i, beta_i, T))
# sigma_n_i = sigma_ln_i * S_ref  (anchored to mean spot)
# Weights via softmax: w = exp(w_raw) / sum(exp(w_raw))
#
# KEY: Must use BTC-settled pricing formulas (§1.4) and BTC-settled IV solver
#      for the model backbone. Standard BS formulas will produce biased IVs.
```

**BTC-adapted parameter settings**:

| Parameter | SPX Setting | BTC Adaptation | Rationale |
|-----------|------------|----------------|-----------|
| β bounds | `(0.1, 0.95)` | **`(0.05, 2.0)`** | BTC smile is more symmetric; β > 1 may be needed for positive skew regions; lower bound relaxed to allow symmetric-like behavior |
| σ_ln initial | `[0.2, 0.5, 1.0]` | **`[0.5, 1.0, 2.0]`** | BTC vol is 50-150% vs SPX 15-50% |
| σ_ln bounds | `(0.01, 2.0)` | **`(0.05, 5.0)`** | Need wider range for BTC extreme vol |
| r (risk-free rate) | `0.02` | **`0.0`** (initially) | Short-dated BTC options insensitive to r; see §1.5 |
| q (carry cost) | `0.015` | **`0.0`** (with r=0) | No dividends; when r=0, q is irrelevant |
| TARGET_TAU | `30/365` | Same | Standardize to 1M |
| Weights init | `[0.3, 0.3, 0.4]` | Same | Neutral start |
| Weight bounds | `(None, None)` pre-softmax | Same | Softmax handles normalization |

**Critical note on β bounds**: SPX bounds `(0.1, 0.95)` force β < 1, which generates negative skew (the characteristic equity smirk). BTC often has a more symmetric "smile" rather than "smirk." Relaxing the upper β bound to 2.0 allows β > 1, which generates positive skew (fatter right tail). The model should be free to discover the BTC skew direction. The lower bound is relaxed to 0.05 (vs 0.1 for SPX) to avoid forcing extreme skew in either direction.

**Critical note on BTC-settled vs standard model**: The backbone fitting must use `uncertain_backbone_BTC()` which internally calls the BTC-settled displaced-diffusion pricers and the BTC-settled IV solver. Using the standard (USD-settled) model would produce systematically biased IVs for BTC options.

### 3.2 Loss Function

**MSE** (same as SPX, but using BTC-settled backbone):

```python
def loss_mse_btc(params, S_data, iv_data, r, T):
    weights = params[0:3]
    betas = params[3:6]
    sigmas_ln = params[6:9]
    weights = np.exp(weights) / np.sum(np.exp(weights))  # softmax
    S_ref = np.mean(S_data)
    sigmas_n = list(np.array(sigmas_ln) * S_ref)  # anchor to mean spot
    if np.any(betas <= 0) or np.any(sigmas_ln <= 0):
        return 1e6
    model_ivs = [op.uncertain_backbone_BTC(S, S, r, T, weights, betas, sigmas_n, sigmas_ln)
                 for S in S_data]
    return np.mean((np.array(model_ivs) - np.array(iv_data))**2)
```

**MSE justified for BTC** (same reasons as SPX):
1. Smoothly differentiable → better L-BFGS-B convergence
2. Penalizes large deviations (BTC has even fatter tails than SPX)
3. MLE-equivalent under Gaussian errors

### 3.3 Number of States

- Start with 3 states (same as SPX)
- Also fit 2-state and 4-state models
- Compare using **BIC** (Bayesian Information Criterion): BIC = n·ln(MSE) + k·ln(n)
- For n states: parameters = 3n - 1 ( (n-1) free weights + n betas + n sigmas_ln )
- BTC may need fewer states (less skew) or more (more complex dynamics) than SPX

### 3.4 Fitting per Regime

```python
for reg in REGIMES:
    reg_data = btc_atm_df[btc_atm_df['regime'] == reg]
    S_data = reg_data['spot'].values
    iv_data = reg_data['atm_iv_1m'].values

    res = minimize(loss_mse_btc, init, args=(S_data, iv_data, r_btc, TARGET_TAU),
                   method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000})

    btc_fits[reg] = {'params': res.x, 'success': res.success, 'fun': res.fun}
```

**Output**: `processed_data/btc_fits_regime_mse.pkl`

---

## Phase 4: OTM Extension & Visualization

### 4.1 Additive Shift (Initial Approach)

Same formula as SPX:

```
δ_k = mean(IV_k_obs for regime) − mean(IV_ATM_model for regime)
IV_OTM_model(S) = IV_ATM_model(S) + δ_k
```

### 4.2 Affine Shift Investigation (BTC-Specific)

BTC's more symmetric smile may mean β ≠ 1 in the OTM-vs-ATM regression. Explicitly compare three approaches:

| Approach | Formula | Parameters | When Appropriate |
|----------|---------|------------|------------------|
| Additive | `IV_OTM = IV_ATM + δ` | 1 per (regime, k) | β ≈ 1 in regression |
| Proportional | `IV_OTM = λ · IV_ATM` | 1 per (regime, k) | Multiplicative relationship |
| Affine | `IV_OTM = α + β · IV_ATM` | 2 per (regime, k) | General case; captures both level and shape shift |

**Diagnostic**: For each regime and moneyness level, regress `IV_k_obs` on `IV_ATM_model`. If slope ≈ 1, additive is sufficient. If slope ≈ 0.8 or 1.2, affine is needed.

### 4.3 Plots (Matching SPX Format)

1. **ATM backbone fitted plot by regime** — model curve overlaid on scatter
2. **Volatility smile & backbone** — at multiple spot levels within a regime
3. **OTM panels** — (−10% | ATM | +10%), (−20% | ATM | +20%), and (−30% | ATM | +30%) with shift annotations
4. **Shift diagnostics table** — δ_k by regime and moneyness
5. **Affine regression table** — slope and R² for each (regime, k) pair

---

## Phase 5: SPX vs BTC Comparison

| Aspect | SPX | BTC (Expected) |
|--------|-----|----------------|
| ATM IV level | 15-50% | 50-150% |
| Skew shape | Negative smirk | More symmetric smile |
| Number of active states | 1 (states 1&2 dead) | TBD — may be more |
| β for active state | 0.1-0.9 | May span wider range |
| Additive shift δ_k | +0.09 to +0.21 for OTM puts | Likely larger absolute shifts |
| Regime transitions | Slow (year-based) | Faster, more frequent |
| Data density | 2405 ATM obs | TBD (may be fewer daily obs due to trade data gaps) |
| Liquidity | Deep across strikes | Concentrated near ATM; sparse at deep OTM |
| Option settlement | USD-settled | BTC-settled (quanto-style) |
| IV source | Computed from bid/ask mid using standard BS | Deribit-provided IV (primary); computed from BTC-settled BS (fallback) |

**Deliverable**: `btc_vs_spx_comparison.md`

---

## Phase 6: Code Organization

| File | Purpose | Status |
|------|---------|--------|
| `option_pricers.py` | **Extend** — add BTC-settled pricing + IV solver (§1.2) alongside existing standard-BS functions | Needs modification |
| `btc_data_download.py` | New: Download BTC option trade data from Deribit via `RiveChen/deribit-historical-data` scraper, parse instrument names, aggregate to daily snapshots | New |
| `btc_data_processing.py` | New: BTC data cleaning, ATM/OTM extraction, tau computation with exact timestamps | New |
| `BTC Uncertain Vol - Backbone Fits.ipynb` | New: Main analysis notebook (mirrors SPX OTM notebook structure) | New |
| `./data/raw/BTC_Options/btc_trades_raw/` | Raw downloaded trade data from Deribit (local cache) | New |
| `processed_data/btc_daily_snapshots.csv` | Daily aggregated trade data with IV | New |
| `processed_data/btc_1M_ATM.csv` | BTC ATM IV data | New |
| `processed_data/btc_iv1m_k_pc.csv` | BTC OTM IV data by moneyness | New |
| `processed_data/btc_fits_regime_mse.pkl` | Saved regime fits | New |

---

## Resolved Questions

1. **Data source**: Deribit API via `RiveChen/deribit-historical-data` scraper. Primary IV source is Deribit's per-trade `iv` field. Fallback is our own BTC-settled IV solver. Tardis.dev remains optional upgrade for order book data.
2. **Time period**: 2020–present (data quality improves significantly from 2020 onward; pre-2020 is very sparse).
3. **Moneyness definition**: Start with `k = ln(K/S)` for SPX consistency. May explore `k = ln(K/F)` (forward moneyness) if results are poor, since BTC forward prices can diverge significantly from spot.
4. **Risk-free rate / cost of carry**: Use `r = 0, q = 0` for backbone fitting (short-dated BTC options are insensitive to r). Validate against Deribit's mark IV. If needed, derive `r - q` from futures basis.
5. **β bounds**: Start with `(0.05, 2.0)` — allows both positive and negative skew, lets the data speak.
6. **Number of states**: Start with 3, compare with 2 and 4 via BIC.
7. **Regime count**: Let PELT change-point detection decide; use tentative regimes (§2.3) as priors.
8. **Observation timestamp**: **08:00 UTC** daily snapshot.
9. **Tau computation**: Exact timestamp-based with expiry at 08:00 UTC.
10. **IV computation**: Primary = Deribit `iv` field. Validation/fallback = BTC-settled IV solver.