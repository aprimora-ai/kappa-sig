#!/usr/bin/env python3
"""
run_dotcom2000.py — Kappa-FIN: Dot-com Bubble / NASDAQ Crash (2000-2002)
=========================================================================
Note: ETF coverage for this era is limited; uses available proxies.
Period: 1998-01-01 to 2003-12-31

Author: David Ohio <odavidohio@gmail.com>
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kappa_fin.engine import Config, run

# ETF proxies available for late 1990s
TICKERS = ["SPY", "QQQ", "DIA", "GLD", "TLT"]

cfg = Config(
    tickers=TICKERS,
    start="1998-01-01",
    end="2003-12-31",
    window=22, k=3,
    dep_method="spearman", shrink_lambda=0.05,
    calm_policy="scan",
    calm_search_to="1999-12-31",
    calm_length_days=504,
    calm_step_days=14,
    calm_topn=5,
    delta=0.08, gamma=0.97, pre_q=0.968,
    eval_after_calm=True,
    oh_persist_sens=2, oh_persist_confirm=5,
    out="./out_dotcom2000",
)

if __name__ == "__main__":
    run(cfg)
