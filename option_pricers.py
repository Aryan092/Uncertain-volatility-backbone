import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pylab as plt


def BlackScholesLognormalCall(S, K, r, sigma, T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)


def BlackScholesLognormalPut(S, K, r, sigma, T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)


def BlackScholesLognormalStraddle(S, K, r, sigma, T):
    call = BlackScholesLognormalCall(S, K, r, sigma, T)
    put = BlackScholesLognormalPut(S, K, r, sigma, T)
    return call + put


def uncertain_volatility(S, K, r, T, weights, sigmas):
    prices = [BlackScholesLognormalCall(S, K, r, sigma, T)
              for sigma in sigmas]
    model_price = np.dot(weights, prices)
    model_vol = impliedVolatility(S, K, r, model_price, T, 'call')
    return model_vol


def uncertain_backbone(S, K, r, T, weights, betas, sigmas_n, sigmas_ln):
    prices = [DisplacedDiffusionCall(S, K, r, sigma_n, sigma_ln, beta, T)
              for beta, sigma_n, sigma_ln in zip(betas, sigmas_n, sigmas_ln)]
    model_price = np.dot(weights, prices)
    model_vol = impliedVolatility(S, K, r, model_price, T, 'call')
    return model_vol


def uncertain_backbone_put(S, K, r, T, weights, betas, sigmas_n, sigmas_ln):
    prices = [DisplacedDiffusionPut(S, K, r, sigma_n, sigma_ln, beta, T)
              for beta, sigma_n, sigma_ln in zip(betas, sigmas_n, sigmas_ln)]
    model_price = np.dot(weights, prices)
    model_vol = impliedVolatility(S, K, r, model_price, T, 'put')
    return model_vol


def DisplacedDiffusionCall(S, K, r, sigma_n, sigma_ln, beta, T):
    S_ = S + sigma_n*(1-beta)/(sigma_ln*beta)
    K_ = K + sigma_n*(1-beta)/(sigma_ln*beta)
    sigma_ = sigma_ln * beta
    return BlackScholesLognormalCall(S_, K_, r, sigma_, T)


def DisplacedDiffusionPut(S, K, r, sigma_n, sigma_ln, beta, T):
    S_ = S + sigma_n*(1-beta)/(sigma_ln*beta)
    K_ = K + sigma_n*(1-beta)/(sigma_ln*beta)
    sigma_ = sigma_ln * beta
    return BlackScholesLognormalPut(S_, K_, r, sigma_, T)


def impliedVolatility(S, K, r, price, T, payoff):
    try:
        if (payoff.lower() == 'call') or (payoff.lower() == 'payer'):
            impliedVol = brentq(lambda x: price -
                                BlackScholesLognormalCall(S, K, r, x, T),
                                1e-12, 10.0)
        elif (payoff.lower() == 'put') or (payoff.lower() == 'receiver'):
            impliedVol = brentq(lambda x: price -
                                BlackScholesLognormalPut(S, K, r, x, T),
                                1e-12, 10.0)
        elif payoff.lower() == 'straddle':
            impliedVol = brentq(lambda x: price -
                                BlackScholesLognormalStraddle(S, K, r, x, T),
                                1e-12, 10.0)
        else:
            raise NameError('Payoff type not recognized')
    except Exception:
        impliedVol = np.nan

    return impliedVol


# ---------------------------------------------------------------------------
# BTC-settled option pricing and IV functions
#
# Deribit BTC options are cash-settled in BTC with payoff:
#   Call payoff (BTC) = max(S_T - K, 0) / S_T
#   Put  payoff (BTC) = max(K - S_T, 0) / S_T
#
# The BTC-denominated BS prices are:
#   c_BTC = N(d1) - (K/S) * exp(-rT) * N(d2)
#   p_BTC = (K/S) * exp(-rT) * N(-d2) - N(-d1)
#
# Note: c_BTC = c_USD / S  and  p_BTC = p_USD / S
# However, the IV that solves c_BTC = market_price_btc is different
# from the IV that solves c_USD = market_price_usd.
# The brentq root-finding must use the BTC-settled formula.
# ---------------------------------------------------------------------------


def BlackScholesLognormalCall_BTC(S, K, r, sigma, T):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return norm.cdf(d1) - (K / S) * np.exp(-r * T) * norm.cdf(d2)


def BlackScholesLognormalPut_BTC(S, K, r, sigma, T):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return (K / S) * np.exp(-r * T) * norm.cdf(-d2) - norm.cdf(-d1)


def BlackScholesLognormalStraddle_BTC(S, K, r, sigma, T):
    call = BlackScholesLognormalCall_BTC(S, K, r, sigma, T)
    put = BlackScholesLognormalPut_BTC(S, K, r, sigma, T)
    return call + put


def DisplacedDiffusionCall_BTC(S, K, r, sigma_n, sigma_ln, beta, T):
    S_ = S + sigma_n * (1 - beta) / (sigma_ln * beta)
    K_ = K + sigma_n * (1 - beta) / (sigma_ln * beta)
    sigma_ = sigma_ln * beta
    return BlackScholesLognormalCall_BTC(S_, K_, r, sigma_, T)


def DisplacedDiffusionPut_BTC(S, K, r, sigma_n, sigma_ln, beta, T):
    S_ = S + sigma_n * (1 - beta) / (sigma_ln * beta)
    K_ = K + sigma_n * (1 - beta) / (sigma_ln * beta)
    sigma_ = sigma_ln * beta
    return BlackScholesLognormalPut_BTC(S_, K_, r, sigma_, T)


def impliedVolatility_BTC_settled(S, K, r, price_btc, T, payoff):
    try:
        if (payoff.lower() == 'call') or (payoff.lower() == 'payer'):
            return brentq(
                lambda x: price_btc -
                    BlackScholesLognormalCall_BTC(S, K, r, x, T),
                1e-12, 10.0)
        elif (payoff.lower() == 'put') or (payoff.lower() == 'receiver'):
            return brentq(
                lambda x: price_btc -
                    BlackScholesLognormalPut_BTC(S, K, r, x, T),
                1e-12, 10.0)
        elif payoff.lower() == 'straddle':
            return brentq(
                lambda x: price_btc -
                    BlackScholesLognormalStraddle_BTC(S, K, r, x, T),
                1e-12, 10.0)
        else:
            raise NameError('Payoff type not recognized')
    except Exception:
        return np.nan


def uncertain_backbone_BTC(S, K, r, T, weights, betas, sigmas_n, sigmas_ln):
    prices = [DisplacedDiffusionCall_BTC(S, K, r, sigma_n, sigma_ln, beta, T)
              for beta, sigma_n, sigma_ln in zip(betas, sigmas_n, sigmas_ln)]
    model_price = np.dot(weights, prices)
    model_vol = impliedVolatility_BTC_settled(S, K, r, model_price, T, 'call')
    return model_vol


def uncertain_backbone_put_BTC(S, K, r, T, weights, betas, sigmas_n, sigmas_ln):
    prices = [DisplacedDiffusionPut_BTC(S, K, r, sigma_n, sigma_ln, beta, T)
              for beta, sigma_n, sigma_ln in zip(betas, sigmas_n, sigmas_ln)]
    model_price = np.dot(weights, prices)
    model_vol = impliedVolatility_BTC_settled(S, K, r, model_price, T, 'put')
    return model_vol


# S = 5000.0
# r = 0.02
# T = 0.25
# weights = [0.3, 0.3, 0.4]
# sigmas = [0.1, 0.3, 0.5]
# strikes = np.linspace(3000, 6000, 100)
# plt.figure(tight_layout=True, figsize=(8, 6))
# for S in [3500, 4000, 4500, 5000, 5500]:
#     impliedvols = [uncertain_volatility(S, K, r, T, weights, sigmas)
#                    for K in strikes]
#     plt.plot(strikes, impliedvols, label=f'{S=}')

# plt.legend()
# plt.show()

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--test-btc':
        print("=== BTC-settled pricing verification ===")
        S, K, r, sigma, T = 65000.0, 70000.0, 0.0, 0.80, 30/365

        c_usd = BlackScholesLognormalCall(S, K, r, sigma, T)
        c_btc = BlackScholesLognormalCall_BTC(S, K, r, sigma, T)
        p_usd = BlackScholesLognormalPut(S, K, r, sigma, T)
        p_btc = BlackScholesLognormalPut_BTC(S, K, r, sigma, T)

        print(f"S={S}, K={K}, r={r}, sigma={sigma}, T={T:.6f}")
        print(f"Call USD: ${c_usd:.4f}  Call BTC: {c_btc:.8f} BTC  Ratio: {c_usd/c_btc:.2f} (should be S={S:.0f})")
        print(f"Put  USD: ${p_usd:.4f}  Put  BTC: {p_btc:.8f} BTC  Ratio: {p_usd/p_btc:.2f} (should be S={S:.0f})")

        iv_call = impliedVolatility(S, K, r, c_usd, T, 'call')
        iv_call_btc = impliedVolatility_BTC_settled(S, K, r, c_btc, T, 'call')
        print(f"IV (USD-settled):  {iv_call:.6f}")
        print(f"IV (BTC-settled):  {iv_call_btc:.6f}")
        print(f"Both should recover sigma={sigma}")

        print()
        print("=== BTC-settled displaced-diffusion backbone ===")
        sigmas_ln = [0.5, 1.0, 2.0]
        sigmas_n = list(np.array(sigmas_ln) * S)
        weights = [0.3, 0.3, 0.4]
        betas = [1.0, 1.2, 0.8]

        strikes = np.linspace(S * 0.8, S * 1.2, 50)

        iv_usd = [uncertain_backbone(S, K, r, T, weights, betas,
                                     sigmas_n, sigmas_ln) for K in strikes]
        iv_btc = [uncertain_backbone_BTC(S, K, r, T, weights, betas,
                                         sigmas_n, sigmas_ln) for K in strikes]

        plt.figure(tight_layout=True, figsize=(8, 6))
        plt.plot(strikes, iv_usd, label='USD-settled', linewidth=2)
        plt.plot(strikes, iv_btc, '--', label='BTC-settled', linewidth=2)
        plt.xlabel('Strike')
        plt.ylabel('Implied Volatility')
        plt.title('Uncertain-Vol Backbone: USD vs BTC Settlement')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    else:
        S = 5000.0
        r = 0.02
        T = 0.25
        sigmas_ln = [0.1, .5, .8]
        sigmas_n = list(np.array(sigmas_ln)*S)
        weights = [0.5, 0.2, 0.3]
        betas = [1.5, 1.1, 1.25]
        strikes = np.linspace(3000, 6000, 100)
        plt.figure(tight_layout=True, figsize=(8, 6))
        for S in [3500, 4000, 4500, 5000, 5500]:
            impliedvols = [uncertain_backbone(S, K, r, T, weights, betas,
                                              sigmas_n, sigmas_ln)
                           for K in strikes]
            plt.plot(strikes, impliedvols, label=f'{S=}')

        backbones = []
        backbones = [uncertain_backbone(S, S, r, T, weights, betas,
                                        sigmas_n, sigmas_ln)
                     for S in strikes]
        plt.plot(strikes, backbones, linewidth=3.0, color='k', label='Backbone')

        plt.legend()
        plt.show()
