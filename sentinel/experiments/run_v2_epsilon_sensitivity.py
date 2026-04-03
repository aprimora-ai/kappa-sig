#!/usr/bin/env python3
"""
Kappa v2 — Epsilon_q Sensitivity Analysis (0.10 vs 0.20 vs 0.30)
Tests how RQA recurrence threshold affects Layer 2 geometry.
David Ohio | 2026
"""
import os, sys, json
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, r"C:\Users\ohiod\Projects\kappa-fin")
from kappa_fin.engine_v5_fixed import run_v2, V2Config

REPORTS_DIR = Path(r"C:\Users\ohiod\Projects\Sentinel\data\reports")
V2_OUTPUT_DIR = Path(r"C:\Users\ohiod\Projects\Sentinel\data\v2_analysis")

EPSILON_VALUES = [0.10, 0.20, 0.30]

def find_universes():
    universes = []
    for d in sorted(REPORTS_DIR.iterdir()):
        if d.is_dir() and d.name.startswith("sentinel_"):
            sc = d / "kappa_v4_state.csv"
            vc = d / "kappa_v4_viscosity.csv"
            if sc.exists() and vc.exists():
                universes.append({"name": d.name.replace("sentinel_",""),
                                  "state_csv": str(sc), "visc_csv": str(vc)})
    return universes

def run_all():
    universes = find_universes()
    print(f"Found {len(universes)} universes.")
    print(f"Testing epsilon_q = {EPSILON_VALUES}\n")

    # Store results per epsilon
    all_results = {}  # {eps: {universe: summary}}

    for eps_q in EPSILON_VALUES:
        print(f"\n{'#'*70}")
        print(f"  EPSILON_Q = {eps_q}")
        print(f"{'#'*70}")

        cfg = V2Config()
        cfg.rqa_epsilon_q = eps_q
        results = {}

        for u in universes:
            nm = u["name"]
            # Suppress per-universe output for cleaner comparison
            # Redirect stdout temporarily
            import io, contextlib
            f_out = io.StringIO()
            try:
                with contextlib.redirect_stdout(f_out):
                    df, s = run_v2(u["state_csv"], u["visc_csv"], cfg)
                results[nm] = s

                # Save to subdirectory
                od = V2_OUTPUT_DIR / f"eps_{eps_q:.2f}" / nm
                od.mkdir(parents=True, exist_ok=True)
                df.to_csv(od / "kappa_v2_state.csv")
                with open(od / "kappa_v2_summary.json", "w") as jf:
                    json.dump(s, jf, indent=2, default=str)

            except Exception as e:
                print(f"  ERROR {nm}: {e}")
                results[nm] = {"error": str(e)}

        all_results[eps_q] = results

        # Quick summary for this epsilon
        n_high = sum(1 for s in results.values()
                     if s.get("geo_reliability") == "HIGH")
        n_low = sum(1 for s in results.values()
                    if s.get("geo_reliability") == "LOW")
        print(f"  eps_q={eps_q}: {n_high} HIGH / {n_low} LOW")

    # ══════════════════════════════════════════════════════════════════════════
    #  COMPARATIVE REPORT
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*100}")
    print(f"  EPSILON_Q SENSITIVITY ANALYSIS — COMPARATIVE REPORT")
    print(f"{'='*100}\n")

    # Header
    print(f"  {'Universe':25s}  {'C/Phi*':>7s}  {'kF':>6s}  |", end="")
    for eps in EPSILON_VALUES:
        print(f"  eps={eps:.2f}: {'tA':>5s} {'P30':>5s} {'T½':>6s} {'geo':>4s} {'Al':>2s}  |", end="")
    print()
    print("  " + "-" * 97)

    # Get all universe names, sorted by kF from first epsilon run
    first_eps = EPSILON_VALUES[0]
    measured = {k: v for k, v in all_results[first_eps].items()
                if not v.get("phi_star_estimated", False) and "error" not in v}
    insuf = {k: v for k, v in all_results[first_eps].items()
             if v.get("phi_star_estimated", False)}

    sorted_names = sorted(measured.keys(),
                          key=lambda x: all_results[first_eps][x].get("kappa_F_now", 0),
                          reverse=True)

    for nm in sorted_names:
        s0 = all_results[first_eps][nm]
        cn = s0.get("C_norm_now", 0)
        kf = s0.get("kappa_F_now", 0)
        print(f"  {nm:25s}  {cn:7.3f}  {kf:6.3f}  |", end="")

        for eps in EPSILON_VALUES:
            s = all_results[eps].get(nm, {})
            if "error" in s:
                print(f"  {'ERR':>5s} {'':>5s} {'':>6s} {'':>4s} {'':>2s}  |", end="")
                continue
            ta = s.get("theta_A_now", 0)
            p3 = s.get("P_collapse_30d", 0)
            th = s.get("T_half_now", float("inf"))
            gr = s.get("geo_reliability", "?")
            al = s.get("alert_level", "?")
            al_s = {"NOMINAL":"NM","WATCH":"WT","WARNING":"WR","EMERGENCY":"EM",
                     "BASELINE_INSUFFICIENT":"BI"}.get(al, "??")
            gr_s = "H" if gr == "HIGH" else "L"
            th_s = f"{th:6.1f}" if th < 9999 else "   inf"
            print(f"  {ta:5.2f} {p3:5.3f} {th_s} {gr_s:>4s} {al_s:>2s}  |", end="")
        print()

    if insuf:
        print(f"\n  --- BASELINE INSUFFICIENT ---")
        for nm in insuf:
            print(f"  {nm:25s}  [insufficient]")

    # ══════════════════════════════════════════════════════════════════════════
    #  FOCUS: ANOMALOUS UNIVERSES
    # ══════════════════════════════════════════════════════════════════════════
    focus = ["x_us_systemic", "x_brazil_vuln", "asia_pacific",
             "europe", "x_energy_geopolitics", "financials", "us_sectors"]
    print(f"\n\n{'='*100}")
    print(f"  FOCUS UNIVERSES — Θ_A and RQA detail across epsilon_q")
    print(f"{'='*100}")

    for nm in focus:
        print(f"\n  --- {nm} ---")
        for eps in EPSILON_VALUES:
            s = all_results[eps].get(nm, {})
            if "error" in s: continue
            ta = s.get("theta_A_now", 0)
            p3 = s.get("P_collapse_30d", 0)
            th = s.get("T_half_now", float("inf"))
            gr = s.get("geo_reliability", "?")
            al = s.get("alert_level", "?")
            cs = s.get("calm_rqa_stats", {})
            det_s = cs.get("DET", {}).get("sigma", 0)
            lam_s = cs.get("LAM", {}).get("sigma", 0)
            tt_mu = cs.get("TT", {}).get("mu", 0)
            tt_s = cs.get("TT", {}).get("sigma", 0)
            print(f"    eps={eps:.2f}: tA={ta:5.2f}  P30={p3:.3f}  T½={th:7.1f}  geo={gr}"
                  f"  σ_DET={det_s:.4f}  σ_LAM={lam_s:.4f}  TT:μ={tt_mu:.2f}/σ={tt_s:.4f}  [{al}]")

    # Geo reliability summary
    print(f"\n\n  Geometry Reliability Summary:")
    for eps in EPSILON_VALUES:
        res = all_results[eps]
        n_h = sum(1 for s in res.values() if s.get("geo_reliability") == "HIGH")
        n_l = sum(1 for s in res.values() if s.get("geo_reliability") == "LOW")
        print(f"    eps={eps:.2f}: {n_h} HIGH / {n_l} LOW")

    # Save full comparison
    cp = V2_OUTPUT_DIR / "epsilon_sensitivity_report.json"
    with open(cp, "w") as f:
        json.dump({
            "epsilon_values": EPSILON_VALUES,
            "results": {str(e): r for e, r in all_results.items()}
        }, f, indent=2, default=str)
    print(f"\n  Full report: {cp}")
    print(f"\nDone.")

if __name__ == "__main__":
    run_all()
