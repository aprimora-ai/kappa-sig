#!/usr/bin/env python3
"""
verify_data.py — Verify SHA-256 checksums of study price data.

Usage:
    python scripts/verify_data.py
    python scripts/verify_data.py --manifest data/results/data_manifest.json

Author: David Ohio <odavidohio@gmail.com>
"""

import argparse
import hashlib
import json
import os
import sys

def sha256_of_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/results/data_manifest.json")
    args = p.parse_args()

    if not os.path.exists(args.manifest):
        sys.exit(f"Manifest not found: {args.manifest}\nRun: python scripts/save_study_data.py")

    with open(args.manifest) as f:
        manifest = json.load(f)

    base_dir = os.path.dirname(args.manifest)
    all_ok = True

    print(f"Verifying {len(manifest)} dataset(s)...\n")
    for entry in manifest:
        path = os.path.join(base_dir, entry["file"])
        if not os.path.exists(path):
            print(f"  [MISSING] {entry['file']}")
            all_ok = False
            continue
        actual = sha256_of_file(path)
        expected = entry["sha256"]
        status = "OK" if actual == expected else "MISMATCH"
        print(f"  [{status}] {entry['file']}")
        print(f"    Study:    {entry['study']} ({entry['start']} → {entry['end']})")
        print(f"    Tickers:  {entry['tickers_final']}")
        print(f"    Expected: {expected}")
        if actual != expected:
            print(f"    Actual:   {actual}")
            all_ok = False

    print()
    if all_ok:
        print("✅ All checksums match — data integrity verified.")
    else:
        print("❌ Checksum mismatch detected — data may have changed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
