#!/usr/bin/env python3
"""
Kappa v2 — Run on all Sentinel universes (v5.4 — z-score geometry)
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
V2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def find_universes():
    universes = []
    for d in sorted(REPORTS_DIR.iterdir()):
        if d.is_dir() and d.name.startswith("sentinel_"):
            sc = d / "kappa_v4_state.csv"
            vc = d / "kappa_v4_viscosity.csv"
            if sc.exists() and vc.exists():
                universes.append({"name": d.name.replace("sentinel_",""), "state_csv": str(sc), "visc_csv": str(vc)})
    return universes

def phi_star_status(s):
    if not s.get("phi_star_estimated", False): return "measured"
    if s.get("phi_star", 0) <= 1e-4: return "no_damage_history"
    return "estimated_fallback"

def detect_anomalies(real_summaries):
    anomalies = []
    for nm, s in real_summaries.items():
        cn = s.get("C_norm_now", 0); p3 = s.get("P_collapse_30d", 0)
        kf = s.get("kappa_F_now", 0); al = s.get("alert_level", "")
        th = s.get("T_half_now", float("inf"))
        if cn > 0.9 and p3 > 0.3:
            anomalies.append((nm, "HIGH_P30_INTACT_CAPACITY", f"C/Phi*={cn:.3f} P30={p3:.3f}"))
        if al == "WARNING" and kf < 0.1:
            anomalies.append((nm, "WARNING_WITHOUT_FRAGILITY", f"alert=WR kF={kf:.4f}"))
        if cn > 0.8 and th < 30:
            anomalies.append((nm, "SHORT_THALF_HIGH_CAPACITY", f"C/Phi*={cn:.3f} T1/2={th:.1f}"))
        if al == "NOMINAL" and p3 > 0.2:
            anomalies.append((nm, "NOMINAL_HIGH_P30", f"alert=NM P30={p3:.3f}"))
    return anomalies

def run_all():
    universes = find_universes()
    print(f"Found {len(universes)} universes.\n")
    cfg = V2Config(); all_s = {}; errors = []
    for u in universes:
        nm = u["name"]
        print(f"\n{'='*60}\n  {nm}\n{'='*60}")
        od = V2_OUTPUT_DIR / nm; od.mkdir(parents=True, exist_ok=True)
        try:
            df, s = run_v2(u["state_csv"], u["visc_csv"], cfg)
            s["phi_star_status"] = phi_star_status(s)
            df.to_csv(od / "kappa_v2_state.csv")
            with open(od / "kappa_v2_summary.json", "w") as f: json.dump(s, f, indent=2, default=str)
            all_s[nm] = s
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}"); traceback.print_exc()
            errors.append({"universe": nm, "error": str(e)})

    print(f"\n\n{'='*70}")
    print(f"  KAPPA v2 CROSS-UNIVERSE REPORT (v5.4 — z-score geometry)")
    print(f"{'='*70}\n")

    real = {k:v for k,v in all_s.items() if v.get("phi_star_status") == "measured"}
    insuf = {k:v for k,v in all_s.items() if v.get("phi_star_status") != "measured"}

    su = sorted(real.items(), key=lambda x: x[1].get("kappa_F_now",0), reverse=True)
    for nm, s in su:
        al=s.get("alert_level","?"); cn=s.get("C_norm_now",0); kf=s.get("kappa_F_now",0)
        p3=s.get("P_collapse_30d",0); th=s.get("T_half_now",float("inf"))
        nf=s.get("n_false_recovery_days",0); gr=s.get("geo_reliability","?")
        ta=s.get("theta_A_now",0)
        ic={"NOMINAL":"NM","WATCH":"WT","WARNING":"WR","EMERGENCY":"EM"}.get(al,"??")
        geo_tag = f" tA={ta:.2f}" if ta > 0.01 else ""
        gr_tag = f" geo={gr}" if gr != "HIGH" else ""
        print(f"  {ic} {nm:35s}  C/Phi*={cn:8.3f}  kF={kf:6.3f}  "
              f"P30={p3:5.3f}  T1/2={th:8.1f}  FR={nf:3d}  [{al}]{geo_tag}{gr_tag}")

    if insuf:
        print(f"\n  --- BASELINE NOT MEASURED ---")
        for nm, s in insuf.items():
            nf=s.get("n_false_recovery_days",0); st=s.get("phi_star_status","?")
            print(f"  ?? {nm:35s}  [{st}]  FR={nf}")

    anomalies = detect_anomalies(real)
    if anomalies:
        print(f"\n  --- CANDIDATE ANOMALIES ({len(anomalies)}) ---")
        for nm, at, desc in anomalies:
            print(f"  >>  {nm:35s}  {at:30s}  {desc}")
    else:
        print(f"\n  --- No anomalies detected ---")

    # Geo reliability summary
    n_high = sum(1 for s in real.values() if s.get("geo_reliability") == "HIGH")
    n_low = sum(1 for s in real.values() if s.get("geo_reliability") == "LOW")
    print(f"\n  Geometry reliability: {n_high} HIGH / {n_low} LOW out of {len(real)} measured")

    cp = V2_OUTPUT_DIR / "v2_cross_universe_report.json"
    with open(cp, "w") as f:
        json.dump({
            "config": {k:v for k,v in cfg.__dict__.items()},
            "engine_version": "v5.4",
            "universes": all_s,
            "anomalies": [{"universe":a[0],"type":a[1],"detail":a[2]} for a in anomalies],
            "errors": errors,
            "n_measured": len(real), "n_insufficient": len(insuf),
            "n_geo_high": n_high, "n_geo_low": n_low,
        }, f, indent=2, default=str)

    print(f"\n  Report: {cp}")
    if errors: print(f"  {len(errors)} errors")
    print(f"\nDone. {len(real)} measured, {len(insuf)} insufficient, "
          f"{n_high} geo-HIGH, {n_low} geo-LOW, {len(anomalies)} anomalies.")

if __name__ == "__main__":
    run_all()
