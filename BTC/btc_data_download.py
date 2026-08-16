"""
BTC Options Data Download from Deribit API

Downloads historical option trade data from Deribit, aggregates into daily
snapshots at 08:00 UTC, and saves processed data for the backbone analysis.

Uses Deribit's public API v2:
  - get_instruments: enumerate all BTC option contracts
  - get_last_trades_by_instrument_and_time: download historical trades

Data flow:
  1. Fetch instrument list (active + expired)
  2. For each instrument, download all historical trades
  3. Parse instrument names to extract strike, expiry, option_type
  4. Aggregate to daily snapshots at 08:00 UTC
  5. Save to processed_data/

Usage:
  python btc_data_download.py --start 2020-01-01 --end 2025-05-01
  python btc_data_download.py --reset          # re-download everything
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone, timedelta, time as dt_time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DERIBIT_API = "https://www.deribit.com/api/v2/public"
SNAPSHOT_TIME = dt_time(8, 0)
SNAPSHOT_WINDOW_MIN = 30
MIN_TAU = 3 / 365
RAW_DATA_DIR = Path(__file__).parent / "data" / "raw" / "BTC_Options"

DATA_DIR = RAW_DATA_DIR / "btc_trades_raw"

PROCESSED_DIR = Path(__file__).parent / "processed_data"
RATE_LIMIT_DELAY = 0.15


def api_get(endpoint, params, max_retries=3):
    for attempt in range(max_retries):
        try:
            r = requests.get(f"{DERIBIT_API}/{endpoint}", params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                raise ValueError(f"API error: {data['error']}")
            return data.get("result", data)
        except (requests.RequestException, ValueError) as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt * RATE_LIMIT_DELAY * 10)
            else:
                raise


def parse_instrument_name(name):
    """Parse 'BTC-28MAR25-100000-C' into (expiry_str, strike, option_type)."""
    parts = name.split("-")
    if len(parts) != 4 or parts[0] != "BTC":
        return None, None, None
    expiry_str = parts[1]
    try:
        strike = float(parts[2])
    except ValueError:
        return None, None, None
    option_type = parts[3].lower()
    if option_type not in ("c", "p"):
        return None, None, None
    try:
        expiry_ts = pd.Timestamp(expiry_str, tz="UTC")
    except Exception:
        try:
            expiry_dt = datetime.strptime(expiry_str, "%d%b%y")
            expiry_ts = pd.Timestamp(expiry_dt, tz="UTC")
        except ValueError:
            return None, None, None
    return expiry_ts, strike, option_type


def get_all_instruments():
    """Fetch all BTC option instruments (active + expired)."""
    instruments = []
    for expired in [False, True]:
        result = api_get("get_instruments", {
            "currency": "BTC",
            "kind": "option",
            "expired": str(expired).lower(),
        })
        if isinstance(result, list):
            instruments.extend(result)
        time.sleep(RATE_LIMIT_DELAY)
    print(f"Found {len(instruments)} BTC option instruments")
    return instruments


def get_trades_for_instrument(instrument_name, start_ts, end_ts):
    """Download all trades for a single instrument in a time range."""
    all_trades = []
    current_start = start_ts
    while True:
        result = api_get("get_last_trades_by_instrument_and_time", {
            "instrument_name": instrument_name,
            "start_timestamp": str(current_start),
            "end_timestamp": str(end_ts),
            "count": "1000",
            "include_old": "true",
            "sorting": "asc",
        })
        trades = result.get("trades", [])
        if not trades:
            break
        all_trades.extend(trades)
        if not result.get("has_more", False):
            break
        last_ts = max(t[" timestamp"] if " timestamp" in t else t.get("timestamp", end_ts)
                      for t in trades)
        current_start = last_ts + 1
        time.sleep(RATE_LIMIT_DELAY)
        if len(all_trades) >= 50000:
            break
    return all_trades


def download_instruments_trades(instruments, start_date, end_date, cache_dir):
    """Download trades for all instruments, with per-instrument caching."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    start_ms = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)
    all_records = []
    skipped = 0
    no_trades = 0
    total = len(instruments)
    for i, inst in enumerate(instruments):
        name = inst["instrument_name"]
        expiry_ts, strike, opt_type = parse_instrument_name(name)
        if expiry_ts is None or strike is None:
            skipped += 1
            continue
        cache_file = cache_dir / f"{name}.parquet"
        if cache_file.exists():
            try:
                df_inst = pd.read_parquet(cache_file)
                all_records.append(df_inst)
                continue
            except Exception:
                pass
        if (i + 1) % 100 == 0 or i < 5:
            print(f"  [{i+1}/{total}] Downloading {name}...", end=" ")
        trades = get_trades_for_instrument(name, start_ms, end_ms)
        if (i + 1) % 100 == 0 or i < 5:
            print(f"got {len(trades)} trades")
        if not trades:
            no_trades += 1
            continue
        df_inst = pd.DataFrame(trades)
        df_inst["instrument_name"] = name
        df_inst["strike"] = strike
        df_inst["option_type"] = opt_type
        df_inst["expiry_ts"] = expiry_ts
        df_inst["expiry_date"] = expiry_ts.date() if hasattr(expiry_ts, 'date') else expiry_ts
        try:
            df_inst.to_parquet(cache_file, index=False)
        except Exception:
            pass
        all_records.append(df_inst)
        time.sleep(RATE_LIMIT_DELAY)
    print(f"Downloaded {len(all_records)} instruments, {no_trades} with no trades, {skipped} skipped")
    if not all_records:
        return pd.DataFrame()
    return pd.concat(all_records, ignore_index=True)


def build_daily_snapshots(df_trades, snapshot_hour=8, window_minutes=30):
    """Aggregate raw trades into daily snapshots at 08:00 UTC."""
    if df_trades.empty:
        return pd.DataFrame()
    if "timestamp" in df_trades.columns:
        df_trades["trade_ts"] = pd.to_datetime(df_trades["timestamp"], unit="ms", utc=True)
    elif "timestamp" in df_trades.columns:
        df_trades["trade_ts"] = pd.to_datetime(df_trades["timestamp"], unit="ms", utc=True)
    df_trades["trade_date"] = df_trades["trade_ts"].dt.date
    df_trades["snapshot_time"] = df_trades["trade_ts"].apply(
        lambda ts: dt_time(ts.hour, ts.minute)
    )
    start_window = dt_time(snapshot_hour, 0) - timedelta(minutes=window_minutes)
    end_window = dt_time(snapshot_hour, 0) + timedelta(minutes=window_minutes)
    start_window = dt_time(start_window.hour, start_window.minute)
    end_window = dt_time(end_window.hour, end_window.minute)
    if start_window <= end_window:
        in_window = (
            (df_trades["snapshot_time"] >= start_window) &
            (df_trades["snapshot_time"] <= end_window)
        )
    else:
        in_window = (
            (df_trades["snapshot_time"] >= start_window) |
            (df_trades["snapshot_time"] <= end_window)
        )
    df_window = df_trades[in_window].copy()
    df_window["time_diff"] = abs(
        df_window["trade_ts"].dt.hour * 60 + df_window["trade_ts"].dt.minute
        - snapshot_hour * 60
    )
    deduped = []
    for (date, instrument), g in df_window.groupby(["trade_date", "instrument_name"]):
        closest = g.loc[g["time_diff"].idxmin()]
        record = {}
        record["date"] = date
        record["instrument_name"] = instrument
        record["strike"] = closest["strike"]
        record["option_type"] = closest["option_type"]
        record["expiry_date"] = closest.get("expiry_date", closest.get("expiry_ts"))
        record["trade_ts"] = closest["trade_ts"]
        record["price"] = closest.get("price", np.nan)
        record["mark_price"] = closest.get("mark_price", np.nan)
        record["iv"] = closest.get("iv", np.nan)
        record["index_price"] = closest.get("index_price", np.nan)
        record["underlying_price"] = closest.get("index_price", np.nan)
        record["amount"] = closest.get("amount", np.nan)
        deduped.append(record)
    return pd.DataFrame(deduped)


def compute_derived_fields(df):
    """Compute tau, log_moneyness, and other derived fields."""
    if df.empty:
        return df
    expiry_ts = pd.to_datetime(df["expiry_date"], utc=True)
    expiry_ts = expiry_ts.dt.normalize() + pd.Timedelta(hours=8)
    snapshot_ts = pd.to_datetime(df["date"], utc=True).dt.normalize() + pd.Timedelta(hours=8)
    df["tau"] = (expiry_ts - snapshot_ts).dt.total_seconds() / (365.0 * 86400)
    df["tau"] = df["tau"].where(df["tau"] > MIN_TAU, np.nan)
    df["log_moneyness"] = np.log(df["strike"] / df["underlying_price"])
    df = df.dropna(subset=["tau", "iv", "underlying_price"])
    df = df[df["tau"] > 0]
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    return df


def main():
    parser = argparse.ArgumentParser(description="Download BTC options data from Deribit")
    parser.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-06-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--reset", action="store_true", help="Clear cache and re-download")
    parser.add_argument("--skip-download", action="store_true", help="Skip download, process cached data only")
    args = parser.parse_args()
    start_date = pd.Timestamp(args.start, tz="UTC")
    end_date = pd.Timestamp(args.end, tz="UTC")
    if args.reset and DATA_DIR.exists():
        import shutil
        shutil.rmtree(DATA_DIR)
    if not args.skip_download:
        print("Step 1: Fetching instrument list...")
        instruments = get_all_instruments()
        print(f"Step 2: Downloading trades for {len(instruments)} instruments...")
        df_trades = download_instruments_trades(instruments, start_date, end_date, DATA_DIR)
        if df_trades.empty:
            print("No trade data downloaded. Check API access.")
            return
        print(f"Downloaded {len(df_trades)} total trade records")
        raw_path = PROCESSED_DIR / "btc_trades_raw.parquet"
        df_trades.to_parquet(raw_path, index=False)
        print(f"Saved raw trades to {raw_path}")
    else:
        raw_path = PROCESSED_DIR / "btc_trades_raw.parquet"
        if not raw_path.exists():
            print(f"No cached data at {raw_path}. Run without --skip-download first.")
            return
        df_trades = pd.read_parquet(raw_path)
        print(f"Loaded {len(df_trades)} cached trade records")
    print("Step 3: Building daily snapshots at 08:00 UTC...")
    df_snapshots = build_daily_snapshots(df_trades)
    print(f"Built {len(df_snapshots)} daily snapshot records")
    print("Step 4: Computing derived fields (tau, log_moneyness)...")
    df_snapshots = compute_derived_fields(df_snapshots)
    print(f"After filtering: {len(df_snapshots)} records")
    snapshot_path = PROCESSED_DIR / "btc_daily_snapshots.csv"
    df_snapshots.to_csv(snapshot_path, index=False)
    print(f"Saved daily snapshots to {snapshot_path}")
    print(f"Date range: {df_snapshots['date'].min()} to {df_snapshots['date'].max()}")
    print(f"Unique instruments: {df_snapshots['instrument_name'].nunique()}")
    print(f"Unique dates: {df_snapshots['date'].nunique()}")


if __name__ == "__main__":
    main()