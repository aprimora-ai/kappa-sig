# -*- coding: utf-8 -*-
"""
Kappa Sentinel — Price Downloader
===================================
Downloads OHLCV data for all ETFs via yfinance.
Caches locally to avoid repeated API calls.

David Ohio | odavidohio@gmail.com | March 2026
"""
import os, sys, json
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
from config.universe import ALL_UNIVERSES, get_all_unique_tickers

try:
    import yfinance as yf
except ImportError:
    print("ERROR: pip install yfinance")
    sys.exit(1)

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "prices"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def download_all(start: str = "2020-01-01", end: str = None,
                 force: bool = False) -> dict:
    """
    Download OHLCV for all unique tickers across all universes.

    Args:
        start: Start date (YYYY-MM-DD). Default: 2020-01-01 (~5 years)
        end: End date. Default: today
        force: If True, re-download even if cached

    Returns:
        dict of {ticker: filepath} for successfully downloaded tickers
    """
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    tickers = get_all_unique_tickers()
    print(f"[Downloader] {len(tickers)} unique tickers, {start} to {end}")

    results = {}
    failed = []

    for i, ticker in enumerate(tickers):
        fpath = CACHE_DIR / f"{ticker}.csv"

        # Skip if cached and not forced
        if not force and fpath.exists():
            # Check if cache is recent (< 1 day old)
            mtime = datetime.fromtimestamp(fpath.stat().st_mtime)
            if (datetime.now() - mtime).total_seconds() < 86400:
                results[ticker] = str(fpath)
                continue

        # Download
        try:
            tag = f"[{i+1}/{len(tickers)}]"
            print(f"  {tag} {ticker}...", end=" ", flush=True)
            df = yf.download(ticker, start=start, end=end,
                             progress=False, auto_adjust=True)
            if df is None or df.empty:
                print("EMPTY")
                failed.append(ticker)
                continue

            df.to_csv(fpath)
            results[ticker] = str(fpath)
            n_rows = len(df)
            print(f"OK ({n_rows} rows)")

        except Exception as e:
            print(f"FAIL ({e})")
            failed.append(ticker)

    print(f"\n[Downloader] Done: {len(results)} ok, {len(failed)} failed")
    if failed:
        print(f"  Failed: {', '.join(failed)}")

    return results


def load_universe(universe_id: str) -> "pd.DataFrame":
    """
    Load cached prices for a specific universe.
    Returns DataFrame with tickers as columns, date as index.
    """
    import pandas as pd

    universe = ALL_UNIVERSES.get(universe_id)
    if not universe:
        raise ValueError(f"Unknown universe: {universe_id}")

    tickers = universe["tickers"]
    frames = {}

    for ticker in tickers:
        fpath = CACHE_DIR / f"{ticker}.csv"
        if fpath.exists():
            df = pd.read_csv(fpath, index_col=0, parse_dates=True)
            if "Close" in df.columns:
                frames[ticker] = df["Close"]
            elif "Adj Close" in df.columns:
                frames[ticker] = df["Adj Close"]

    if not frames:
        raise RuntimeError(f"No data for universe {universe_id}")

    combined = pd.DataFrame(frames).dropna(how="all")
    print(f"[Load] {universe_id}: {len(frames)}/{len(tickers)} tickers, "
          f"{len(combined)} trading days")
    return combined


def update_daily():
    """
    Daily update: download only the latest data for all tickers.
    Uses 30-day window to ensure overlap with cached data.
    """
    start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    return download_all(start=start, force=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kappa Sentinel Downloader")
    parser.add_argument("--full", action="store_true",
                        help="Full historical download (5 years)")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.full:
        download_all(start=args.start, force=args.force)
    else:
        update_daily()
