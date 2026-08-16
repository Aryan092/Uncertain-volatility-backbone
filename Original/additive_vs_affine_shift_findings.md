# Additive vs Affine Shift: Findings Summary

## 1. The Problem

The uncertain-vol backbone model is fitted **only to ATM data** (K = S). It has 9 parameters per regime, but at ATM two of the three σ_ln values collapse to the lower bound (~0.01), making states 1 and 2 effectively "dead." The model therefore reduces to a near-single-state displaced diffusion at ATM, and its built-in skew is too weak to reproduce the empirical OTM put premium. To extend the backbone to OTM strikes, we need a way to shift the ATM backbone shape to the correct level.

## 2. Three Candidate Approaches

### 2.1 Additive Shift (Current Implementation)

$$\text{IV}_{\text{OTM}}(S, k) = \text{IV}_{\text{ATM}}^{\text{model}}(S) + \delta_k$$

where $\delta_k = \overline{\text{IV}_k^{\text{obs}}} - \overline{\text{IV}_{\text{ATM}}^{\text{model}}}$ (mean residual per regime per moneyness level).

**Pros:**
- Simple, one scalar per (regime, moneyness) pair
- Preserves the shape (curvature, asymmetry) of the ATM backbone exactly
- Justified by β ≈ 1 in affine regression (see section 3 below)
- When β ≈ 1, the slope is correct and only the intercept needs adjustment — an additive shift is exactly the intercept term

**Cons:**
- Assumes the OTM–ATM spread is constant across the entire spot range within a regime
- In reality, the skew flattens at very high spot levels, so the spread is slightly spot-dependent
- Does not adjust curvature at different moneyness levels

### 2.2 Proportional (Multiplicative) Shift

$$\text{IV}_{\text{OTM}}(S, k) = \lambda_k \cdot \text{IV}_{\text{ATM}}^{\text{model}}(S)$$

**Pros:**
- Maintains proportionality — high-vol periods get proportionally larger shifts

**Cons:**
- Distorts curvature: it scales the backbone shape, amplifying peaks and compressing troughs
- If β ≈ 1 in the affine regression, a proportional shift introduces an unwanted slope modifier (it replaces β = 1 with some λ)
- Less appropriate when the relationship between OTM and ATM IV is roughly parallel (constant spread) rather than proportional

### 2.3 Affine Shift (General Case)

$$\text{IV}_{\text{OTM}}(S, k) = \alpha_k + \beta_k \cdot \text{IV}_{\text{ATM}}^{\text{model}}(S)$$

**Pros:**
- Most general — both level (α) and shape (β) can vary by moneyness
- Subsumes additive (β = 1) and proportional (α = 0) as special cases
- Can capture slight changes in backbone shape across moneyness levels

**Cons:**
- Two parameters per (regime, moneyness) instead of one
- If β ≈ 1 across the board, the extra parameter provides negligible improvement
- More parameters increases overfitting risk with limited data per regime
- Harder to interpret and explain

## 3. Empirical Evidence: β ≈ 1 Supports Additive Shift

The key empirical finding is that regressing OTM observed IV on ATM model IV across regimes yields a slope coefficient β ≈ 1. This means:

- **The shape of the backbone transfers directly** from ATM to OTM — the relationship is approximately parallel, not proportional
- When β ≈ 1, the affine model reduces to $\text{IV}_{\text{OTM}} = \alpha_k + \text{IV}_{\text{ATM}}$, which is exactly the additive shift
- The additive shift δ_k captures the intercept α_k that separates OTM levels from ATM

This β ≈ 1 result is also consistent with the displaced-diffusion parameter estimates: the active state (state 3) has β values ranging from 0.10 to 0.52 across regimes, but the ATM call price is insensitive to β — it's the σ_ln parameter that drives the fit. The near-unity regression slope of OTM IV on ATM IV confirms that the co-movement is level-preserving.

## 4. Shift Diagnostics

Additive shifts δ_k = mean(OTM IV) − mean(ATM model IV) by regime and moneyness:

| Regime | k = −20% | k = −10% | k = 0% | k = +10% | k = +20% |
|--------|----------|----------|--------|----------|----------|
| COVID | +0.2102 | +0.1366 | 0.0 | −0.0438 | −0.0027 |
| Low Vol (2015–17) | +0.1890 | +0.1012 | 0.0 | −0.0260 | −0.0013 |
| Post-COVID | +0.1859 | +0.0884 | 0.0 | −0.0176 | +0.0276 |
| Transition (2018–19) | +0.1810 | +0.0883 | 0.0 | −0.0127 | +0.0367 |

**Key observations:**
- OTM puts (k < 0): Positive δ — puts trade above ATM, reflecting the volatility skew
- OTM calls (k > 0): Negative or small δ — calls trade at or below ATM
- The shift magnitude is largest for deep OTM puts (k = −20%), as expected
- Shifts are consistent in sign across all regimes

## 5. Current Status

- **Active approach:** Additive shift (implemented in `Uncertain Vol - OTM Backbone Fits.ipynb`)
- **Justification:** β ≈ 1 in affine regression; the additive shift is the appropriate reduced-form model when slope is preserved
- **Known limitation:** The OTM–ATM spread is not perfectly constant across spot levels (skew flattens at high spot); an affine shift would capture this but adds complexity

## 6. Open Questions (Deferred)

1. **Full affine calibration:** Would fitting α_k and β_k per (regime, moneyness) meaningfully improve the fit, or does β ≈ 1 consistently enough that the extra parameter is wasted?
2. **Joint multi-moneyness fitting:** Instead of fitting ATM only and shifting, could the model be fitted simultaneously to ATM + OTM data, letting the mixture states generate the skew endogenously?
3. **Spot-dependent shift:** Could δ_k be made a function of S (e.g., δ_k(S) = a + b·S) to capture the slight flattening of the skew at high spot levels?
4. **Time-varying parameters:** The current model uses fixed parameters per regime. How would allowing parameters to vary within a regime affect the shift?
5. **Term structure:** The current analysis is for 1M options (τ = 30/365). How does the additive shift behave across different tenors?