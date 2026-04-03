#!/usr/bin/env python3
r"""
Kappa v2 -- Impact Sensitivity Module (Section 8.12)
=====================================================
Computes the Impact Sensitivity Index I_S(t) for all Sentinel universes.

The Damage Barrier Model (Proposition 8.1):
  Oh_crit(t) = (Oh_pre + delta) * g(Theta_A(t))
  I_S(t) = Oh(t) / Oh_crit(t)

When I_S < 1: system absorbing shocks within barrier (safe)
When I_S >= 1: barrier breached, damage expected

Operational proxy (uncalibrated, alpha not yet estimated):
  I_S_proxy(t) = Theta_A(t) * Oh(t) / (Oh_pre + delta)

Run after run_v2_monitor.py:
  cd C:\Users\ohiod\Projects\Sentinel
  python compute_impact_sensitivity.py

David Ohio | odavidohio@gmail.com | Independent Researcher
March 2026
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

V2_DIR = Path(r"C:\Users\ohiod\Projects\Sentinel\data\v2_analysis")
REPORTS_DIR = Path(r"C:\Users\ohiod\Projects\Sentinel\data\reports")
OUT_DIR = Path(r"C:\Users\ohiod\Projects\Sentinel\data\v2_monitoring")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Default delta from engine config
DELTA = 0.08

# I_S interpretation thresholds (Section 8.12.6, provisional)
IS_LOW = 0.2
IS_ELEVATED = 0.5
IS_HIGH = 1.0


def compute_impact_sensitivity(uid):
    """Compute I_S time series and current value for a universe."""
    v2_csv = V2_DIR / uid / "kappa_v2_state.csv"
    v2_json = V2_DIR / uid / "kappa_v2_summary.json"
    visc_csv = REPORTS_DIR / f"sentinel_{uid}" / "kappa_v4_viscosity.csv"

    if not v2_csv.exists() or not visc_csv.exists():
        return None

    df = pd.read_csv(v2_csv, index_col="date", parse_dates=True)
    visc = pd.read_csv(visc_csv, index_col=0)

    # Get Oh_pre from viscosity
    oh_pre = float(visc.loc["Oh_pre", "value"]) if "Oh_pre" in visc.index else 0.98

    # Get Theta_A and Oh
    theta_a = df["theta_A"].values if "theta_A" in df.columns else np.zeros(len(df))
    oh = df["Oh"].values if "Oh" in df.columns else np.zeros(len(df))

    # Compute I_S proxy: Theta_A * Oh / (Oh_pre + delta)
    barrier = oh_pre + DELTA
    i_s = theta_a * oh / max(barrier, 1e-12)

    # Current values
    i_s_now = float(i_s[-1])
    theta_now = float(theta_a[-1])
    oh_now = float(oh[-1])

    # Classification
    if i_s_now >= IS_HIGH:
        level = "BARRIER_BREACHED"
    elif i_s_now >= IS_ELEVATED:
        level = "HIGH_VULNERABILITY"
    elif i_s_now >= IS_LOW:
        level = "ELEVATED"
    else:
        level = "LOW"

    # Barrier gap: how far is Oh from the effective barrier?
    # Using simplified proxy: barrier_effective = barrier / max(theta_a, 0.01)
    # This is the Oh value that would make I_S = 1
    if theta_now > 0.01:
        oh_to_breach = barrier / theta_now
        gap = oh_to_breach - oh_now
        gap_pct = gap / max(oh_to_breach, 1e-12) * 100
    else:
        oh_to_breach = float("inf")
        gap = float("inf")
        gap_pct = 100.0

    # Historical stats
    i_s_max = float(np.max(i_s))
    i_s_mean_last30 = float(np.mean(i_s[-30:])) if len(i_s) >= 30 else float(np.mean(i_s))

    # Load v2 summary for additional context
    phi_star_estimated = False
    if v2_json.exists():
        with open(v2_json, encoding="utf-8") as f:
            summary = json.load(f)
        phi_star_estimated = summary.get("phi_star_estimated", False)

    return {
        "universe": uid,
        "I_S_now": round(i_s_now, 4),
        "I_S_level": level,
        "theta_A_now": round(theta_now, 4),
        "Oh_now": round(oh_now, 4),
        "Oh_pre": round(oh_pre, 4),
        "barrier": round(barrier, 4),
        "Oh_to_breach": round(oh_to_breach, 4) if oh_to_breach < 1e6 else None,
        "barrier_gap": round(gap, 4) if gap < 1e6 else None,
        "barrier_gap_pct": round(gap_pct, 1) if gap_pct < 200 else None,
        "I_S_max_historical": round(i_s_max, 4),
        "I_S_mean_30d": round(i_s_mean_last30, 4),
        "phi_star_estimated": phi_star_estimated,
        "date": str(df.index[-1].date()),
    }


def main():
    print("=" * 80)
    print("  KAPPA v2 -- IMPACT SENSITIVITY INDEX (Section 8.12)")
    print("  Damage Barrier Model: I_S = Theta_A * Oh / (Oh_pre + delta)")
    print("  David Ohio | Independent Researcher | March 2026")
    print("=" * 80)

    results = {}
    universes = sorted([d.name for d in V2_DIR.iterdir() if d.is_dir()])

    print(f"\n  {'Universe':25s}  {'I_S':>6s}  {'Level':>20s}  {'Theta_A':>8s}  {'Oh':>6s}  {'Gap':>8s}")
    print(f"  {'-'*85}")

    for uid in universes:
        r = compute_impact_sensitivity(uid)
        if r is None:
            continue

        results[uid] = r

        gap_str = f"{r['barrier_gap_pct']:.0f}%" if r["barrier_gap_pct"] is not None else "inf"
        marker = " <<<" if r["I_S_level"] in ("HIGH_VULNERABILITY", "BARRIER_BREACHED") else ""

        # Skip BASELINE_INSUFFICIENT
        if r["phi_star_estimated"]:
            print(f"  {uid:25s}  {'---':>6s}  {'BASELINE_INSUFF':>20s}  {r['theta_A_now']:8.3f}  {r['Oh_now']:6.3f}  {'---':>8s}")
            continue

        print(f"  {uid:25s}  {r['I_S_now']:6.3f}  {r['I_S_level']:>20s}  {r['theta_A_now']:8.3f}  {r['Oh_now']:6.3f}  {gap_str:>8s}{marker}")

    # Save
    out_file = OUT_DIR / "impact_sensitivity.json"
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": "barrier_proxy",
        "note": "I_S = Theta_A * Oh / (Oh_pre + delta). Uncalibrated proxy. Section 8.12.",
        "thresholds": {"low": IS_LOW, "elevated": IS_ELEVATED, "high": IS_HIGH},
        "universes": results,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    # Summary
    real = {k: v for k, v in results.items() if not v.get("phi_star_estimated")}
    n_breach = sum(1 for v in real.values() if v["I_S_level"] == "BARRIER_BREACHED")
    n_high = sum(1 for v in real.values() if v["I_S_level"] == "HIGH_VULNERABILITY")
    n_elev = sum(1 for v in real.values() if v["I_S_level"] == "ELEVATED")
    n_low = sum(1 for v in real.values() if v["I_S_level"] == "LOW")

    print(f"\n  Summary: {n_breach} BREACHED  {n_high} HIGH  {n_elev} ELEVATED  {n_low} LOW")
    print(f"  Saved: {out_file}")
    print("\nDone.")


if __name__ == "__main__":
    main()
