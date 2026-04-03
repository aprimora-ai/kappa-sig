#!/usr/bin/env python3
"""
save_study_data.py — Download and version-lock the price data used in the study.

Saves prices as CSV with SHA-256 checksum so any future reproduction can
verify they are using identical input data.

Usage:
    python scripts/save_study_data.py
    python scripts/save_study_data.py --out data/results/

Author: David Ohio <odavidohio@gmail.com>
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
except ImportError as e:
    sys.exit(f"Missing dependency: {e}. Run: pip install -r requirements.txt")

STUDIES = {
    "gfc2008": {
        "tickers": ["SPY","QQQ","IWM","DIA","XLF","XLE","XLK","XLV","TLT","IEF","LQD","GLD"],
        "start": "2006-01-01",
        "end": "2010-07-01",
        "note": "GFC 2008. HYG excluded (launched Apr 2007). USO excluded (insufficient pre-crisis history)."
    },
    "covid2020": {
        "tickers": ["SPY","QQQ","IWM","XLF","XLE","XLK","XLV","TLT","LQD","HYG","GLD","USO"],
        "start": "2018-01-01",
        "end": "2021-12-31",
        "note": "COVID-19 crash 2020."
    },
    "rates2022": {
        "tickers": ["SPY","QQQ","IWM","XLF","TLT","IEF","LQD","HYG","GLD","USO"],
        "start": "2021-01-01",
        "end": "2023-12-31",
        "note": "Fed rate hike cycle / bond crisis 2022."
    },
}

def sha256_of_csv(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def download_and_save(name: str, study: dict, outdir: str) -> dict:
    print(f"\n[{name}] Downloading {len(study['tickers'])} tickers "
          f"{study['start']} → {study['end']} ...")

    data = yf.download(
        tickers=study["tickers"],
        start=study["start"],
        end=study["end"],
        auto_adjust=True,
        progress=False,
    )

    if isinstance(data.columns, pd.MultiIndex):
        close = data.xs("Close", axis=1, level=0)
    else:
        close = data[["Close"]]

    close = close.dropna(how="all").ffill()
    before = list(close.columns)
    close = close.dropna(axis=1, how="any")
    removed = [c for c in before if c not in close.columns]
    if removed:
        print(f"  [WARN] Removed (NaN after ffill): {removed}")
    close = close.dropna()

    csv_path = os.path.join(outdir, f"prices_{name}.csv")
    close.to_csv(csv_path)

    checksum = sha256_of_csv(csv_path)
    n_rows, n_cols = close.shape
    tickers_final = list(close.columns)

    print(f"  Saved:    {csv_path}")
    print(f"  Shape:    {n_rows} rows × {n_cols} tickers")
    print(f"  Tickers:  {tickers_final}")
    print(f"  SHA-256:  {checksum}")

    return {
        "file": os.path.basename(csv_path),
        "study": name,
        "start": study["start"],
        "end": study["end"],
        "tickers_requested": study["tickers"],
        "tickers_final": tickers_final,
        "tickers_removed": removed,
        "n_rows": n_rows,
        "n_tickers": n_cols,
        "sha256": checksum,
        "downloaded_at": datetime.utcnow().isoformat() + "Z",
        "note": study.get("note", ""),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/results", help="Output directory")
    p.add_argument("--studies", nargs="+", default=list(STUDIES.keys()),
                   choices=list(STUDIES.keys()), help="Which studies to save")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    manifest = []
    for name in args.studies:
        info = download_and_save(name, STUDIES[name], args.out)
        manifest.append(info)

    manifest_path = os.path.join(args.out, "data_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[OK] Manifest saved: {manifest_path}")
    print("\nTo verify data integrity in a future reproduction:")
    print("  python scripts/verify_data.py")

if __name__ == "__main__":
    main()
