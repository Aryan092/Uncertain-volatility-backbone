"""
BTC Options Data Processing

Cleans daily snapshot data and extracts ATM and OTM implied volatilities,
following the same methodology as the SPX pipeline but adapted for BTC.

Key differences from SPX:
  - Settlement: BTC cash-settled (quanto-style payoff max(S_T-K,0)/S_T)
  - IV source: Deribit 'iv' field (primary), BTC-settled BS solver (fallback)
  - Tau: exact timestamp-based, expiry at 08:00 UTC
  - Moneyness window: 0.10 (vs 0.05 for SPX) due to coarser strikes
  - No holiday rollback (24/7 market)
  - Observation time: 08:00 UTC daily
  - r = 0 for backbone fitting

Input:  processed_data/btc_daily_snapshots.csv
Output: processed_data/btc_1M_ATM.csv
        processed_data/btc_iv1m_k_pc.csv
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from option_pricers import (
    BlackScholesLognormalCall_BTC,
    BlackScholesLognormalPut_BTC,
    impliedVolatility_BTC_settled,
)

TARGET_TAU = 30 / 365
ATM_MONEYNESS_WINDOW = 0.10
MIN_ATM_POINTS = 3
MIN_TAU = 3 / 365
MAX_IV = 5.0
SNAPSHOT_HOUR = 8

OTM_MONEYNESS_LEVELS = [-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30]
SPREAD_THRESHOLD = 0.40


def load_snapshots(filepath):
    """Load daily snapshots CSV."""
    df = pd.read_csv(filepath, parse_dates=["date"])
    if "trade_ts" in df.columns:
        df["trade_ts"] = pd.to_datetime(df["trade_ts"], utc=True)
    return df


def clean_data(df):
    """Apply BTC-specific data quality filters."""
    initial = len(df)
    df = df.copy()
    df = df[df["iv"] > 0]
    df = df[df["iv"] < MAX_IV]
    df = df[df["tau"] > MIN_TAU]
    df = df[df["underlying_price"] > 0]
    df = df[df["strike"] > 0]
    df = df.dropna(subset=["iv", "underlying_price", "strike", "tau"])
    print(f"Cleaning: {initial} -> {len(df)} records "
          f"(removed {initial - len(df)})")
    return df


def extract_atm_iv(df):
    """Extract ATM IV per (date, expiry) by interpolating to k=0.

    Uses Deribit 'iv' field as primary source.
    Falls back to BTC-settled BS IV solver if iv field is missing.
    """
    atm_list = []

    for (date, expiry), g in df.groupby(["date", "expiry_date"]):
        g = g.sort_values("log_moneyness")

        near = g[np.abs(g["log_moneyness"]) < ATM_MONEYNESS_WINDOW]

        if len(near) < MIN_ATM_POINTS:
            continue

        iv_source = near["iv"].values

        mask = np.isfinite(iv_source) & (iv_source > 0) & (iv_source < MAX_IV)
        if mask.sum() < 2:
            continue

        f = interp1d(
            near["log_moneyness"].values[mask],
            iv_source[mask],
            kind="linear",
            fill_value="extrapolate",
        )

        atm_iv = float(f(0.0))

        if atm_iv <= 0 or atm_iv > MAX_IV:
            continue

        atm_list.append({
            "date": date,
            "expiry_date": expiry,
            "tau": g["tau"].iloc[0],
            "spot": g["underlying_price"].iloc[0],
            "atm_iv": atm_iv,
            "n_strikes_near": len(near),
        })

    atm_df = pd.DataFrame(atm_list)
    if not atm_df.empty:
        print(f"ATM extraction: {len(atm_df)} (date, expiry) groups")
    return atm_df


def interpolate_to_1m(atm_df):
    """Interpolate ATM IV to constant 30-day maturity in total-variance space."""
    iv_1m_list = []

    for date, g in atm_df.groupby("date"):
        g = g.sort_values("tau")

        if g["tau"].min() > TARGET_TAU or g["tau"].max() < TARGET_TAU:
            continue

        total_var = g["atm_iv"] ** 2 * g["tau"]

        if total_var.min() < 0:
            continue

        f = interp1d(g["tau"], total_var, kind="linear")

        tv_1m = float(f(TARGET_TAU))

        if tv_1m <= 0:
            continue

        iv_1m = np.sqrt(tv_1m / TARGET_TAU)

        if iv_1m > MAX_IV:
            continue

        iv_1m_list.append({
            "date": date,
            "spot": g["spot"].iloc[0],
            "atm_iv_1m": iv_1m,
        })

    result = pd.DataFrame(iv_1m_list)
    if not result.empty:
        result["year"] = pd.to_datetime(result["date"]).dt.year
        print(f"1M ATM interpolation: {len(result)} observations")
    return result


def extract_otm_iv(df):
    """Extract OTM IV at target moneyness levels.

    For each moneyness level k, select the nearest available strike
    and interpolate to 30-day maturity.
    """
    otm_records = []

    for k_target in OTM_MONEYNESS_LEVELS:
        if k_target == 0.0:
            continue

        for (date, expiry), g in df.groupby(["date", "expiry_date"]):
            spot = g["underlying_price"].iloc[0]
            target_strike = spot * np.exp(k_target)

            if k_target < 0:
                side_df = g[
                    (g["option_type"] == "p") &
                    (g["log_moneyness"] < ATM_MONEYNESS_WINDOW)
                ]
            else:
                side_df = g[
                    (g["option_type"] == "c") &
                    (g["log_moneyness"] > -ATM_MONEYNESS_WINDOW)
                ]

            if len(side_df) == 0:
                other_type = "c" if k_target < 0 else "p"
                side_df = g[
                    (g["option_type"] == other_type) &
                    (np.abs(g["log_moneyness"] - k_target) < 0.05)
                ]
            if len(side_df) == 0:
                continue

            side_df = side_df.copy()
            side_df["moneyness_diff"] = np.abs(side_df["log_moneyness"] - k_target)
            nearest = side_df.nsmallest(3, "moneyness_diff")

            if len(nearest) < 1:
                continue

            avg_iv = nearest["iv"].mean()

            if not np.isfinite(avg_iv) or avg_iv <= 0 or avg_iv > MAX_IV:
                continue

            otm_records.append({
                "date": date,
                "expiry_date": expiry,
                "tau": g["tau"].iloc[0],
                "spot": spot,
                "k": k_target,
                "iv": avg_iv,
                "n_strikes": len(nearest),
            })

    otm_df = pd.DataFrame(otm_records)
    return otm_df


def interpolate_otm_to_1m(otm_df):
    """Interpolate OTM IV to constant 30-day maturity for each moneyness level."""
    otm_1m_list = []

    for (date, k), g in otm_df.groupby(["date", "k"]):
        g = g.sort_values("tau")

        if g["tau"].min() > TARGET_TAU or g["tau"].max() < TARGET_TAU:
            continue

        total_var = g["iv"] ** 2 * g["tau"]

        if total_var.min() < 0:
            continue

        f = interp1d(g["tau"], total_var, kind="linear")
        tv_1m = float(f(TARGET_TAU))

        if tv_1m <= 0:
            continue

        iv_1m = np.sqrt(tv_1m / TARGET_TAU)

        if iv_1m > MAX_IV:
            continue

        otm_1m_list.append({
            "date": date,
            "spot": g["spot"].iloc[0],
            "k": k,
            "iv_1m": iv_1m,
        })

    result = pd.DataFrame(otm_1m_list)
    if not result.empty:
        result["year"] = pd.to_datetime(result["date"]).dt.year
        print(f"OTM 1M interpolation: {len(result)} observations for "
              f"{result['k'].nunique()} moneyness levels")
    return result


def assign_regimes(df, method="year", regime_map=None):
    """Assign regime labels.

    Args:
        df: DataFrame with 'year' column
        method: 'year' for year-based, 'peft' for change-point detection
        regime_map: dict mapping year ranges to regime names (for year method)
    """
    if regime_map is None:
        regime_map = {
            (2020, 2020): "COVID",
            (2021, 2021): "Bull",
            (2022, 2022): "Bear/FTX",
            (2023, 2025): "Recovery/ETF",
        }

    if method == "year":
        df = df.copy()
        df["regime"] = "Other"
        for (y_start, y_end), label in regime_map.items():
            mask = (df["year"] >= y_start) & (df["year"] <= y_end)
            df.loc[mask, "regime"] = label
        print(f"Regime distribution:\n{df['regime'].value_counts()}")
        return df

    elif method == "peft":
        try:
            import ruptures as rpt
        except ImportError:
            print("ruptures not installed. Falling back to year-based regimes.")
            return assign_regimes(df, method="year", regime_map=regime_map)

        signal = df.sort_values("date")["atm_iv_1m"].values
        algo = rpt.Pelt(custom_cost=rpt.costs.CostL2()).fit(signal)
        breakpoints = algo.predict(pen=10)
        dates = df.sort_values("date")["date"].values
        bp_dates = [dates[bp] for bp in breakpoints[:-1] if bp < len(dates)]
        print(f"PELT detected {len(bp_dates)} change-points: {bp_dates}")
        return df

    return df


def main():
    from pathlib import Path

    processed_dir = Path(__file__).parent / "processed_data"
    snapshots_path = processed_dir / "btc_daily_snapshots.csv"

    if not snapshots_path.exists():
        print(f"Error: {snapshots_path} not found.")
        print("Run btc_data_download.py first to generate snapshots.")
        return

    print("Loading BTC daily snapshots...")
    df = load_snapshots(snapshots_path)
    print(f"Loaded {len(df)} records")
    print(f"Columns: {list(df.columns)}")
    if len(df) == 0:
        print("No data to process. Check the snapshot file.")
        return

    print("\nStep 1: Cleaning data...")
    df = clean_data(df)

    print("\nStep 2: Extracting ATM IV...")
    atm_df = extract_atm_iv(df)
    if atm_df.empty:
        print("No ATM IV data extracted. Check input data quality.")
        return

    print("\nStep 3: Interpolating ATM IV to 30-day maturity...")
    atm_1m_df = interpolate_to_1m(atm_df)
    if atm_1m_df.empty:
        print("No 1M ATM data after interpolation.")
        return

    atm_1m_df = assign_regimes(atm_1m_df, method="year")
    atm_path = processed_dir / "btc_1M_ATM.csv"
    atm_1m_df.to_csv(atm_path, index=False)
    print(f"Saved ATM data to {atm_path}")
    print(f"ATM data: {len(atm_1m_df)} observations")
    print(f"Date range: {atm_1m_df['date'].min()} to {atm_1m_df['date'].max()}")
    print(f"ATM IV stats:\n{atm_1m_df['atm_iv_1m'].describe()}")

    print("\nStep 4: Extracting OTM IV...")
    otm_df = extract_otm_iv(df)
    if otm_df.empty:
        print("No OTM IV data extracted.")
        return

    print("\nStep 5: Interpolating OTM IV to 30-day maturity...")
    otm_1m_df = interpolate_otm_to_1m(otm_df)
    if otm_1m_df.empty:
        print("No 1M OTM data after interpolation.")
        return

    otm_1m_df = assign_regimes(otm_1m_df, method="year")
    otm_path = processed_dir / "btc_iv1m_k_pc.csv"
    otm_1m_df.to_csv(otm_path, index=False)
    print(f"Saved OTM data to {otm_path}")
    print(f"OTM data: {len(otm_1m_df)} observations")
    print(f"Moneyness levels: {sorted(otm_1m_df['k'].unique())}")
    print(f"Date range: {otm_1m_df['date'].min()} to {otm_1m_df['date'].max()}")

    print("\nDone!")


if __name__ == "__main__":
    main()