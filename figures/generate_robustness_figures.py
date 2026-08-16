"""Generate robustness-check figures for the paper (PNG, high-DPI).

Outputs to the figures/ directory next to this file. Loads processed data
and saved fits from ../processed_data/. Reproduces the inline-only plots
that live in the Original/ and Percentile/ notebooks so the paper can
embed them.

Figures produced:
  robust_additive_shift_triptych.png   (k=-10% | ATM | k=+10%)
  robust_additive_shift_deep.png       (k=-20% | ATM | k=+20%)
  robust_price_level_degeneracy.png    (log-moneyness vs Bachelier, refit)
  robust_percentile_6p_overshoot.png   (6p direct smile vs 9p shifted)
  robust_percentile_6p_tight_failure.png (tight-beta collapse / NaN at -20%)
"""
import os, sys, pickle
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDATA = os.path.join(ROOT, "processed_data")
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)

R, TAU = 0.02, 30/365
REGIMES = ["Low Vol (2015-17)", "Transition (2018-19)", "COVID", "Post-COVID"]
REGIME_SHORT = {"Low Vol (2015-17)": "Low Vol", "Transition (2018-19)": "Transition",
                "COVID": "COVID", "Post-COVID": "Post-COVID"}
RCOLOR = {"Low Vol (2015-17)": "#1f77b4", "Transition (2018-19)": "#ff7f0e",
          "COVID": "#d62728", "Post-COVID": "#2ca02c"}

# ---------------------------------------------------------------------------
# Vectorised pricers
# ---------------------------------------------------------------------------
def bs_call(S, K, r, sigma, T):
    sigma = np.maximum(sigma, 1e-9)
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def dd_call(S, K, r, sigma_n, sigma_ln, beta, T):
    shift = sigma_n*(1-beta)/(np.maximum(sigma_ln,1e-12)*np.maximum(beta,1e-6))
    S_ = S + shift; K_ = K + shift; sigma_ = sigma_ln*beta
    return bs_call(S_, K_, r, sigma_, T)

def bachelier_call(S, K, r, sigma_n, T):
    d = (S-K)/(np.maximum(sigma_n,1e-9)*np.sqrt(T))
    return np.exp(-r*T)*((S-K)*norm.cdf(d) + sigma_n*np.sqrt(T)*norm.pdf(d))

def iv_from_price(S, K, r, price, T, lo=1e-6, hi=5.0, niter=100):
    price = np.asarray(price, dtype=float); S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    lo = np.full_like(price, lo, dtype=float); hi = np.full_like(price, hi, dtype=float)
    for _ in range(niter):
        mid = 0.5*(lo+hi)
        p = bs_call(S, K, r, mid, T)
        hi = np.where(p > price, mid, hi)
        lo = np.where(p < price, mid, lo)
    return 0.5*(lo+hi)

def softmax(x):
    x = np.asarray(x, dtype=float)
    e = np.exp(x - x.max())
    return e/e.sum()

def backbone_iv(S, K, weights, betas, sigmas_n, sigmas_ln, r=R, T=TAU):
    price = np.zeros_like(np.asarray(S, dtype=float))
    for w, b, sn, sl in zip(weights, betas, sigmas_n, sigmas_ln):
        price = price + w*dd_call(S, K, r, sn, sl, b, T)
    return iv_from_price(S, K, r, price, T)

def backbone_iv_bachelier(S, K, weights, sigmas_n, r=R, T=TAU):
    price = np.zeros_like(np.asarray(S, dtype=float))
    for w, sn in zip(weights, sigmas_n):
        price = price + w*bachelier_call(S, K, r, sn, T)
    return iv_from_price(S, K, r, price, T)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
atm = pd.read_csv(os.path.join(PDATA, "1M_ATM.csv"))
atm["regime"] = np.where(atm["year"] <= 2017, "Low Vol (2015-17)",
                np.where(atm["year"] <= 2019, "Transition (2018-19)",
                np.where(atm["year"] == 2020, "COVID", "Post-COVID")))
kpc = pd.read_csv(os.path.join(PDATA, "iv1m_k_pc.csv"))

# ---------------------------------------------------------------------------
# Fit loaders
# ---------------------------------------------------------------------------
def load_9p():
    d = pickle.load(open(os.path.join(PDATA, "fits_regime_mse.pkl"), "rb"))
    out = {}
    for reg, v in d.items():
        raw_w = v["params"][:3]; betas = v["params"][3:6]; sln = v["params"][6:9]
        w = softmax(raw_w)
        S_ref = atm[atm["regime"] == reg]["spot"].mean()
        sn = np.array(sln)*S_ref
        out[reg] = dict(w=w, betas=betas, sigmas_n=sn, sigmas_ln=sln, S_ref=S_ref,
                        fun=v["fun"])
    return out

def load_joint():
    d = pickle.load(open(os.path.join(PDATA, "fits_joint_atm_pm20.pkl"), "rb"))
    out = {}
    for reg, v in d.items():
        raw_w = v["params"][:3]; betas = v["params"][3:6]; sln = v["params"][6:9]
        w = softmax(raw_w); S_ref = v["S_ref"]
        sn = np.array(sln)*S_ref
        out[reg] = dict(w=w, betas=betas, sigmas_n=sn, sigmas_ln=sln, S_ref=S_ref,
                        fun=v["fun"])
    return out

def load_6p(pkl):
    d = pickle.load(open(os.path.join(PDATA, pkl), "rb"))
    out = {}
    for reg, v in d.items():
        raw_w = v["params"][:3]; betas = v["params"][3:6]
        w = softmax(raw_w); S_ref = v["S_ref"]
        sln = v["sigma_ln_pinned"]; sn = np.array(sln)*S_ref
        out[reg] = dict(w=w, betas=betas, sigmas_n=sn, sigmas_ln=sln, S_ref=S_ref,
                        fun=v["fun"])
    return out

f9 = load_9p()
fj = load_joint()
f6 = load_6p("fits_regime_mse_6p.pkl")
f6t = load_6p("fits_regime_mse_6p_tight.pkl")
kpp = pickle.load(open(os.path.join(PDATA, "fits_regime_mse.pkl"), "rb"))

# ---------------------------------------------------------------------------
# Figure R1/R2: additive-shift triptychs (9p fit + delta_k)
# ---------------------------------------------------------------------------
def additive_shifts(reg):
    fit = f9[reg]
    sub = atm[atm["regime"] == reg]
    Sobs = sub["spot"].values
    model_atm = backbone_iv(Sobs, Sobs, fit["w"], fit["betas"],
                            fit["sigmas_n"], fit["sigmas_ln"])
    deltas = {}
    for k in [-0.2, -0.1, 0.0, 0.1, 0.2]:
        obs_k = kpc[(kpc["regime"] == reg) & (kpc["k"] == k)]
        if k == 0.0:
            deltas[k] = 0.0; continue
        S_k = obs_k["spot"].values
        model_atm_at_k = backbone_iv(S_k, S_k, fit["w"], fit["betas"],
                                     fit["sigmas_n"], fit["sigmas_ln"])
        deltas[k] = obs_k["iv_1m"].mean() - model_atm_at_k.mean()
    return deltas

def shift_interp(reg, k_arr):
    """Linear interpolation of additive shift over the 5 anchor k values."""
    ks = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
    ds = additive_shifts(reg)
    dv = np.array([ds[k] for k in ks])
    return np.interp(k_arr, ks, dv)

def plot_triptych(ks, fname, suptitle):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, k in zip(axes, ks):
        for reg in REGIMES:
            fit = f9[reg]
            sub = atm[atm["regime"] == reg]
            ax.scatter(sub["spot"], sub["atm_iv_1m"], s=4, alpha=0.25,
                       color=RCOLOR[reg], label=REGIME_SHORT[reg], edgecolors="none")
            Sgrid = np.linspace(sub["spot"].min(), sub["spot"].max(), 120)
            if k == 0.0:
                iv = backbone_iv(Sgrid, Sgrid, fit["w"], fit["betas"],
                                fit["sigmas_n"], fit["sigmas_ln"])
            else:
                K = Sgrid*np.exp(k)
                iv = backbone_iv(Sgrid, K, fit["w"], fit["betas"],
                                 fit["sigmas_n"], fit["sigmas_ln"]) + additive_shifts(reg)[k]
                obs_k = kpc[(kpc["regime"] == reg) & (kpc["k"] == k)]
                ax.scatter(obs_k["spot"], obs_k["iv_1m"], s=4, alpha=0.25,
                           color=RCOLOR[reg], marker="x")
            ax.plot(Sgrid, iv, color=RCOLOR[reg], lw=1.8)
        ax.set_title(f"k = {k:+.0%}" + (" (ATM)" if k == 0.0 else ""))
        ax.set_xlabel("SPX spot")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("1M implied volatility")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT, fname), dpi=200)
    plt.close(fig)
    print("wrote", fname)

plot_triptych([-0.1, 0.0, 0.1], "robust_additive_shift_triptych.png",
              "ATM-only fit + additive shift to OTM: $k=-10\\%$ | ATM | $k=+10\\%$")
plot_triptych([-0.2, 0.0, 0.2], "robust_additive_shift_deep.png",
              "Additive shift to deep OTM: $k=-20\\%$ | ATM | $k=+20\\%$")

# ---------------------------------------------------------------------------
# Figure R3: price-level (Bachelier) degeneracy -- refit 6p Bachelier
# ---------------------------------------------------------------------------
def fit_bachelier(reg):
    sub = atm[atm["regime"] == reg]
    S = sub["spot"].values; y = sub["atm_iv_1m"].values
    def loss(p):
        w = softmax(p[:3]); sn = p[3:6]
        price = sum(w[i]*bachelier_call(S, S, R, sn[i], TAU) for i in range(3))
        iv = iv_from_price(S, S, R, price, TAU)
        m = np.isfinite(iv) & (iv > 0)
        if m.sum() < 10: return 1e6
        return np.mean((iv[m]-y[m])**2)
    best = None
    for _ in range(8):
        x0 = np.concatenate([np.random.default_rng(_).normal(0,0.8,3),
                             np.array([50.0,150.0,400.0])])
        r = minimize(loss, x0, method="L-BFGS-B",
                     bounds=[(None,None)]*3 + [(1,1000)]*3, options={"maxiter":400})
        if best is None or r.fun < best.fun: best = r
    w = softmax(best.x[:3]); sn = best.x[3:6]
    return dict(w=w, sigmas_n=sn, fun=best.fun)

print("Fitting Bachelier price-level model...")
fbach = {reg: fit_bachelier(reg) for reg in REGIMES}
for reg in REGIMES:
    print(" ", reg, "w=", np.round(fbach[reg]["w"],4),
          "sigma_N=", np.round(fbach[reg]["sigmas_n"],2), "MSE=", fbach[reg]["fun"])

fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
for ax, reg in zip(axes, REGIMES):
    sub = atm[atm["regime"] == reg]
    ax.scatter(sub["spot"], sub["atm_iv_1m"], s=4, alpha=0.3, color="gray", edgecolors="none")
    Sgrid = np.linspace(sub["spot"].min(), sub["spot"].max(), 120)
    f = f9[reg]
    iv_lm = backbone_iv(Sgrid, Sgrid, f["w"], f["betas"], f["sigmas_n"], f["sigmas_ln"])
    fb = fbach[reg]
    iv_pl = backbone_iv_bachelier(Sgrid, Sgrid, fb["w"], fb["sigmas_n"])
    ax.plot(Sgrid, iv_lm, color="#1f77b4", lw=2, label="Log-moneyness (9p)")
    ax.plot(Sgrid, iv_pl, "--", color="#d62728", lw=2, label="Price-level (Bachelier 6p)")
    deg = "degenerate" if fb["w"].max() > 0.97 else "non-degenerate"
    ax.set_title(f"{REGIME_SHORT[reg]}\nmax $w$={fb['w'].max():.2f} ({deg})")
    ax.set_xlabel("SPX spot"); ax.grid(alpha=0.3)
axes[0].set_ylabel("1M ATM IV"); axes[0].legend(fontsize=8)
fig.suptitle("Log-moneyness (displaced-diffusion) vs price-level (Bachelier mixture) ATM fits. "
             "Bachelier weights collapse to a single state in 3/4 regimes.", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(OUT, "robust_price_level_degeneracy.png"), dpi=200)
plt.close(fig)
print("wrote robust_price_level_degeneracy.png")

# ---------------------------------------------------------------------------
# Figure R4: percentile 6p direct OTM smile vs 9p additive-shifted
# ---------------------------------------------------------------------------
def model_smile(fit, S, k_grid, put_wing=True):
    K = S*np.exp(k_grid)
    return backbone_iv(np.full_like(k_grid, S), K, fit["w"], fit["betas"],
                       fit["sigmas_n"], fit["sigmas_ln"])

fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
k_grid = np.linspace(-0.22, 0.22, 120)
for ax, reg in zip(axes, REGIMES):
    fit9 = f9[reg]; fit6 = f6[reg]
    S_ref = fit6["S_ref"]
    smile6 = model_smile(fit6, S_ref, k_grid)
    smile9 = model_smile(fit9, S_ref, k_grid)
    shifts = additive_shifts(reg)
    smile9_shift = smile9 + shift_interp(reg, k_grid)
    obs_means = kpc[kpc["regime"] == reg].groupby("k")["iv_1m"].mean()
    ax.plot(k_grid, smile6, color="#d62728", lw=2, label="6p percentile-pinned (direct)")
    ax.plot(k_grid, smile9_shift, "--", color="#1f77b4", lw=2, label="9p ATM + additive shift")
    ax.scatter(obs_means.index, obs_means.values, color="k", s=28, zorder=5, label="observed mean")
    ax.axhline(0, color="0.7", lw=0.5)
    ax.set_title(REGIME_SHORT[reg]); ax.set_xlabel("log-moneyness $k$"); ax.grid(alpha=0.3)
    ax.set_xlim(-0.22, 0.22)
axes[0].set_ylabel("1M implied vol"); axes[0].legend(fontsize=7, loc="upper right")
fig.suptitle("Percentile-pinned 6p direct OTM smile vs 9p additive-shift. "
             "6p puts massive weight on the low-vol state and over-shoots the put wing.", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(OUT, "robust_percentile_6p_overshoot.png"), dpi=200)
plt.close(fig)
print("wrote robust_percentile_6p_overshoot.png")

# ---------------------------------------------------------------------------
# Figure R5: 6p-tight failure (beta collapse to 0.5 floor; NaN at -20%)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
for ax, reg in zip(axes, REGIMES):
    fit9 = f9[reg]; fit6t = f6t[reg]
    S_ref = fit6t["S_ref"]
    smile6t = model_smile(fit6t, S_ref, k_grid)
    smile9 = model_smile(fit9, S_ref, k_grid)
    shifts = additive_shifts(reg)
    smile9_shift = smile9 + shift_interp(reg, k_grid)
    obs_means = kpc[kpc["regime"] == reg].groupby("k")["iv_1m"].mean()
    ax.plot(k_grid, smile6t, color="#9467bd", lw=2,
            label="6p-tight $\\beta\\in[0.5,0.95]$ (direct)")
    ax.plot(k_grid, smile9_shift, "--", color="#1f77b4", lw=2, label="9p + additive shift")
    ax.scatter(obs_means.index, obs_means.values, color="k", s=28, zorder=5)
    nanmask = ~np.isfinite(smile6t)
    if nanmask.any():
        ax.axvspan(k_grid[nanmask].min(), k_grid[nanmask].max(), color="red", alpha=0.12)
        ax.text(0.02, 0.96, "NaN (IV inversion\nfailed)", transform=ax.transAxes,
                va="top", fontsize=7, color="red")
    ax.set_title(f"{REGIME_SHORT[reg]}\n$\\beta$="+f"{np.round(fit6t['betas'],2)}")
    ax.set_xlabel("log-moneyness $k$"); ax.grid(alpha=0.3); ax.set_xlim(-0.22, 0.22)
axes[0].set_ylabel("1M implied vol"); axes[0].legend(fontsize=7, loc="upper right")
fig.suptitle("Tight-$\\beta$ percentile-pinned 6p: $\\beta$ collapses to the 0.5 floor in every regime; "
             "IV inversion fails on the deep OTM put wing ($k=-20\\%$).", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(OUT, "robust_percentile_6p_tight_failure.png"), dpi=200)
plt.close(fig)
print("wrote robust_percentile_6p_tight_failure.png")

print("done.")