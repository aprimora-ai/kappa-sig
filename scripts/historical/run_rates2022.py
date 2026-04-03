#!/usr/bin/env python3
"""
run_rates2022.py — Kappa-FIN: Fed Rate Hike Cycle / Bond Crisis (2022)
=======================================================================
Period: 2021-01-01 to 2023-12-31

Author: David Ohio <odavidohio@gmail.com>
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kappa_fin.engine import Config, run

TICKERS = ["SPY", "QQQ", "IWM", "XLF", "TLT", "IEF", "LQD", "HYG", "GLD", "USO"]

cfg = Config(
    tickers=TICKERS,
    start="2021-01-01",
    end="2023-12-31",
    window=22, k=5,
    dep_method="spearman", shrink_lambda=0.05,
    calm_policy="scan",
    calm_search_to="2021-12-31",
    calm_length_days=252,
    calm_step_days=14,
    calm_topn=5,
    delta=0.08, gamma=0.97, pre_q=0.968,
    eval_after_calm=True,
    oh_persist_sens=2, oh_persist_confirm=5,
    out="./out_rates2022",
)

if __name__ == "__main__":
    run(cfg)
