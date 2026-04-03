#!/usr/bin/env python3
"""
run_gfc2008.py — Kappa-FIN: 2008 Global Financial Crisis
=========================================================
14 ETFs covering equities, sectors, fixed income, and commodities.
Period: 2006-01-01 to 2010-07-01 (calm / build-up / crash / recovery)

Usage:
    python scripts/run_gfc2008.py
    python scripts/run_gfc2008.py --out results/gfc2008

Author: David Ohio <odavidohio@gmail.com>
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kappa_fin.engine import Config, run

# 12 tickers with full coverage from 2006-01-01.
# HYG (iShares HY Corporate Bond) launched April 2007 — excluded to preserve
# a clean pre-crisis window starting Jan 2006.
# USO (US Oil Fund) has late-start NaN rows that shift dates[0] to May 2006,
# which breaks the CALM scan (504-day window needs data from Feb 2006).
# Both limitations should be noted in the paper as data availability constraints.
TICKERS_GFC = [
    "SPY", "QQQ", "IWM", "DIA",          # broad equities
    "XLF", "XLE", "XLK", "XLV",          # sectors
    "TLT", "IEF",                          # treasuries
    "LQD",                                 # investment-grade credit
    "GLD",                                 # gold
    # Excluded: HYG (launched Apr 2007), USO (breaks CALM window start)
]

def main():
    p = argparse.ArgumentParser(description="Kappa-FIN: GFC 2008 analysis")
    p.add_argument("--out", type=str, default="./out_gfc2008")
    p.add_argument("--calm_search_to", type=str, default="2007-06-30",
                   help="Search CALM period up to this date (default: mid-2007)")
    p.add_argument("--calm_length_days", type=int, default=504,
                   help="CALM segment length in calendar days (default: 504 ~ 2 trading years)")
    p.add_argument("--delta", type=float, default=0.08,
                   help="Damage sensitivity (default: 0.08)")
    p.add_argument("--gamma", type=float, default=0.97,
                   help="Memory decay (default: 0.97)")
    p.add_argument("--pre_q", type=float, default=0.968,
                   help="Quantile for pre-crossing Oh baseline (default: 0.968)")
    args = p.parse_args()

    cfg = Config(
        tickers=TICKERS_GFC,
        start="2006-01-01",
        end="2010-07-01",
        window=22,
        k=5,
        dep_method="spearman",
        shrink_lambda=0.05,
        dist_mode="corr",
        alpha=1.0,
        tau=1e-7,
        max_edge_quantile=0.99,
        cap_inf=True,
        curv_summary="median",
        eta_floor=0.05,
        v_raw_mode="entropy_x_anti_dom",
        xi_coupling="eta_inverse",
        xi_q=0.05,
        xi_quantile=0.99,
        xi_eps_margin=0.02,
        phi_mode="pre_excess",
        delta=args.delta,
        gamma=args.gamma,
        pre_q=args.pre_q,
        phi_method="quantile",
        phi_q=0.99,
        phi_persist=3,
        phi_floor=1e-6,
        calm_policy="scan",
        calm_search_to=args.calm_search_to,
        calm_length_days=args.calm_length_days,
        calm_step_days=14,
        calm_topn=5,
        calm_knee_weight=1.0,
        calm_knee_tail_frac=0.25,
        calm_knee_ratio=1.50,
        calm_knee_slope_ann=0.0,
        calm_knee_mix=0.5,
        eval_after_calm=True,
        oh_persist_sens=2,
        oh_persist_confirm=5,
        out=args.out,
    )

    run(cfg)

if __name__ == "__main__":
    main()
