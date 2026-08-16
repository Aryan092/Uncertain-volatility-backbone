# Uncertain Volatility Backbone

Modelling the SPX implied-volatility backbone — the dependence of ATM and OTM implied volatility on the underlying spot price — with a 3-state uncertain-volatility (displaced-diffusion mixture) framework. A self-contained research project spanning model derivation, fitting, robustness checks, an applied BTC extension, and a written paper.

## Highlights

- **Joint ATM–OTM calibration** of a 9-parameter, 3-state displaced-diffusion mixture per regime. Putting OTM strikes inside the loss reproduces the OTM put premium endogenously and retires the ad hoc additive shift required by ATM-only fits.
- **Held-out interpolation test** at the intermediate strikes $k = \pm 0.10$ exposes a small, consistent asymmetric smile bias (under-priced put wing, over-priced call wing) — the signature of a smile that is too linear between trained strikes.
- **Three robustness studies** — additive shift, percentile-pinned $\sigma$, and a Bachelier price-level mixture — each shown to relocate, rather than resolve, the dead-state degeneracy that motivates the joint approach.
- **Empirical span** 2015–2025 across four regimes (Low Vol, Transition, COVID, Post-COVID), plus a Deribit BTC-options replication pipeline.
- **Full write-up** in LaTeX: [`The paper/`](The paper/Uncertain Volatility Backbone.pdf).

## Motivation

The SPX volatility backbone describes how implied vol varies with the index level. Saner regimes show a gentle negative slope (vol rises as spot drops); crisis regimes steepen dramatically. A single-parameter model cannot capture this regime-dependent shape. We therefore fit a **mixture of displaced-diffusion states**, which produces a rich backbone shape — from nearly flat to strongly downward-sloping — while remaining parsimonious enough to estimate from cross-sectional data.

## Data

- **Source:** SPX option chains, 2015–2025 (daily close)
- **Moneyness grid:** k ∈ {0.0, −0.10, −0.20, +0.10, +0.20} in log-moneyness
- **Maturity:** Interpolated to a fixed 30/365 term via total-variance interpolation
- **OTM puts:** Put IVs computed from `put_mid` for negative moneyness; call IVs for positive
- **Regimes:** Low Vol (2015–17), Transition (2018–19), COVID (2020), Post-COVID (2021–25)

## Model

Each state *i* in the mixture follows a **displaced-diffusion** process:

$$S_i' = S + \frac{\sigma_{n,i}(1 - \beta_i)}{\sigma_{ln,i} \cdot \beta_i}, \quad K_i' = K + \frac{\sigma_{n,i}(1 - \beta_i)}{\sigma_{ln,i} \cdot \beta_i}, \quad \sigma_i' = \sigma_{ln,i} \cdot \beta_i$$

The call price under state *i* is then the standard Black–Scholes price on (S', K', σ'). The mixture price is:

$$C_{mix} = \sum_{i=1}^{3} w_i \cdot C_{BS}(S_i', K_i', r, \sigma_i', T)$$

and the model ATM implied volatility is obtained by inverting the mixture price back through Black–Scholes.

**Parameters per state:** weight w_i, displacement β_i, log-normal vol σ_{ln,i}. Normal vol is anchored: σ_{n,i} = σ_{ln,i} · S_ref.

**Total per regime:** 9 parameters (3 weights via softmax, 3 βs, 3 σ_{ln}s).

## Key Results

### Regime Fits (MSE loss)

| Regime | w₁ | w₂ | w₃ | β₁ | β₂ | β₃ | σ_ln₁ | σ_ln₂ | σ_ln₃ | MSE |
|--------|------|------|------|------|------|------|--------|--------|--------|---------|
| Low Vol (2015–17) | 0.39 | 0.34 | 0.27 | 0.90 | 0.90 | 0.10 | 0.010 | 0.010 | 0.343 | 0.000795 |
| Transition (2018–19) | 0.38 | 0.34 | 0.28 | 0.88 | 0.89 | 0.10 | 0.010 | 0.010 | 0.401 | 0.000907 |
| COVID (2020) | 0.25 | 0.26 | 0.49 | 0.14 | 0.10 | 0.10 | 0.010 | 0.010 | 0.236 | 0.005632 |
| Post-COVID (2021–25) | 0.38 | 0.33 | 0.29 | 0.91 | 0.87 | 0.18 | 0.010 | 0.010 | 0.517 | 0.002463 |

### Structural Findings

1. **States 1 and 2 are "dead":** σ_ln₁ ≈ σ_ln₂ ≈ 0.01 (at the lower bound) for all regimes. The real dynamics are carried by state 3 alone. The ATM backbone is effectively a single-state model.

2. **β controls skew at OTM strikes:** The displacement parameter β is what distinguishes the states at the money. When β ≈ 1, the state is nearly log-normal (low skew contribution); when β ≪ 1, it is nearly normal (high skew).

3. **COVID inverts the structure:** During COVID, all three βs collapse to ≈ 0.1 and weight shifts to state 3. The backbone steepens dramatically because the single active state is almost purely normal.

4. **OTM backbone = shifted ATM shape:** Off-ATM (k ≠ 0) backbones share the same functional form as the ATM backbone, requiring only a vertical shift per moneyness level. The ATM shape generalises well.

### Price-Level vs Log-Moneyness Comparison

We also fitted a **pure Bachelier (normal) mixture** — 3 states with weights and σ_N only (6 parameters, no β). Key finding: the Bachelier mixture is **degenerate at ATM**. For 3 of 4 regimes, the optimiser collapses weights to (≈0, ≈0, 1.0), reducing the 3-state model to a single state. The ATM Bachelier call price is proportional to Σ w_i σ_{N,i}, so individual states are not identifiable. The β parameter in the displaced-diffusion model is what prevents this degeneracy. **The Bachelier mixture is not a viable alternative for the backbone.** Full results in [`Original/price_level_vs_log_moneyness_results.md`](Original/price_level_vs_log_moneyness_results.md).

## Repository Structure

```
.
├── option_pricers.py                            # BS, displaced-diffusion, BTC-settled pricers, IV solvers
├── Original/                                    # SPX ATM backbone + OTM + price-level studies
│   ├── Vol Backbone Research Extended__3.ipynb  # Main ATM backbone notebook
│   ├── Uncertain Vol - OTM Backbone Fits.ipynb  # OTM extension & additive shift
│   ├── Uncertain Vol - Price Level vs Log Moneyness.ipynb  # Bachelier comparison
│   ├── additive_vs_affine_shift_findings.md
│   └── price_level_vs_log_moneyness_results.{md,pdf}
├── Joint/                                       # Joint ATM+OTM calibration
│   └── Uncertain Vol - Joint ATM OTM Fit.ipynb
├── Percentile/                                  # Percentile-pinned σ robustness variants
│   ├── Uncertain Vol - Percentile Pinned 6p.ipynb
│   └── Uncertain Vol - Percentile Pinned 6p Tight Beta.ipynb
├── BTC/                                         # Replication on Deribit BTC options
│   ├── btc_data_download.py                     # Deribit API downloader → daily snapshots
│   ├── btc_data_processing.py                   # ATM/OTM extraction, 30-day interpolation
│   ├── btc_uncertain_vol_backbone_plan.md        # Methodology & plan notes
│   └── BTC Uncertain Vol - Backbone Fits.ipynb
├── The paper/                                   # Written-up research
│   ├── Uncertain Volatility Backbone.tex        # LaTeX source
│   ├── Uncertain Volatility Backbone.pdf         # Compiled PDF
│   └── refs.bib
├── figures/                                     # Paper figures
│   ├── generate_robustness_figures.py
│   └── *.png
├── processed_data/                              # Saved fits & small figures (committed)
│   ├── fits_*.pkl                               # Pickled regime-fit parameters
│   ├── fits_joint_atm_pm20_summary.csv
│   └── *.png
└── README.md
```

Raw SPX option CSVs, raw BTC trade data, large intermediates, and the externally-cited research-papers folder are not tracked (see `.gitignore`). Fitted parameters and final figures are committed so the results can be inspected without re-running anything.

## The Paper

The full write-up is [`The paper/Uncertain Volatility Backbone.pdf`](The paper/Uncertain Volatility Backbone.pdf). It documents the joint ATM–OTM calibration, the held-out interpolation test, the three robustness checks, the structural findings, and the open questions. The LaTeX source is alongside it (`Uncertain Volatility Backbone.tex` + `refs.bib`).

## Reproducing the Results

1. **Requirements:** Python 3.12+, numpy, pandas, matplotlib, scipy, statsmodels (optional: `ruptures` for PELT regime detection).
2. **SPX data:** Place raw SPX option CSVs in `./SPX_Options/` (not tracked). Expected schema: `datetime, strike, call_bid, call_ask, put_bid, put_ask, underlying, expiry`.
3. **Main ATM backbone:** Run `Original/Vol Backbone Research Extended__3.ipynb` top-to-bottom. It writes intermediates to `processed_data/`.
4. **OTM extension:** Run `Original/Uncertain Vol - OTM Backbone Fits.ipynb`.
5. **Joint ATM+OTM:** Run `Joint/Uncertain Vol - Joint ATM OTM Fit.ipynb`.
6. **Robustness:** Run the two notebooks in `Percentile/` and `Original/Uncertain Vol - Price Level vs Log Moneyness.ipynb`.
7. **BTC replication:** `python BTC/btc_data_download.py --start 2020-01-01 --end <date>`, then `python BTC/btc_data_processing.py`, then the BTC notebook.
8. **Figures:** `python figures/generate_robustness_figures.py` regenerates the robustness figures used in the paper.

## Open Questions

- **Richer mixture:** a 4-state extension may absorb the asymmetric interpolation bias at $k = \pm 0.10$.
- **Data-driven regimes:** PELT changepoint detection in place of year-based regimes.
- **Time-varying parameters:** parameters are static per regime; a rolling-window or Kalman-filter estimator could capture intra-regime drift.
- **Term structure:** only the 1M maturity is modelled — the backbone slope varies with maturity, so a 3D smile surface is the natural extension.
- **Bitcoin backbone:** complete the BTC estimation once sufficient cross-sectional OTM data is available.