#!/usr/bin/env python3
"""
Kappa v2 — Historical Lead-Time Analysis
Measures ΔT_{G→S} and ΔT_{G→C} across completed and ongoing crises.

Tests Claim E: does geometric activation (t_G) precede structural activation (t_S)?

David Ohio | odavidohio@gmail.com | Independent Researcher
March 2026
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path

BASE = Path(r"C:\Users\ohiod\Projects\Sentinel\data\v2_analysis")

# All universes to analyze (EMERGENCY + WARNING + ANOMALOUS)
UNIVERSES = {
    # EMERGENCY (completed crisis cycle)
    "commodities": "commodities",
    "energy": "energy",
    "europe": "europe",
    "x_energy_geopolitics": "x_energy_geopolitics",
    "x_europe_vuln": "x_europe_vuln",
    # WARNING (partial damage)
    "financials": "financials",
    "us_sectors": "us_sectors",
    # ANOMALOUS (crystallized, no damage currently)
    "x_us_systemic": "x_us_systemic",
    "x_brazil_vuln": "x_brazil_vuln",
    "asia_pacific": "asia_pacific",
}

# Thresholds
THETA_A_THRESH = 0.5       # geometric activation threshold
THETA_A_PERSIST = 5        # consecutive days above threshold
C_NORM_THRESH_S = 0.90     # structural activation (mild)
C_NORM_THRESH_W = 0.50     # structural activation (severe)


def find_t_G(ta, dates, thresh=THETA_A_THRESH, persist=THETA_A_PERSIST):
    """Find first sustained geometric activation."""
    n = len(ta)
    for i in range(n - persist):
        if all(ta[i:i+persist] > thresh):
            return i, dates[i]
    # Fallback: try lower threshold
    for i in range(n - persist):
        if all(ta[i:i+persist] > 0.1):
            return i, dates[i]
    return None, None


def find_first_crossing(series, dates, threshold, direction="below"):
    """Find first time series crosses threshold."""
    for i in range(len(series)):
        if direction == "below" and series[i] < threshold:
            return i, dates[i]
        elif direction == "above" and series[i] > threshold:
            return i, dates[i]
    return None, None


def analyze_universe(uid, folder):
    """Run full lead-time analysis on one universe."""
    csv_path = BASE / folder / "kappa_v2_state.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    ta = df["theta_A"].values
    cn = df["C_norm"].values
    dates = df.index
    n = len(df)

    result = {
        "universe": uid,
        "n_steps": n,
        "period_start": str(dates[0].date()),
        "period_end": str(dates[-1].date()),
        "final_C_norm": float(cn[-1]),
        "final_theta_A": float(ta[-1]),
        "final_alert": df["alert_level"].iloc[-1],
    }

    print(f"\n{'─'*80}")
    print(f"  {uid}")
    print(f"  Period: {dates[0].date()} to {dates[-1].date()} ({n} steps)")
    print(f"  Final: C/Φ*={cn[-1]:.4f}  Θ_A={ta[-1]:.3f}  [{df['alert_level'].iloc[-1]}]")

    # t_G: geometric activation
    t_G_step, t_G_date = find_t_G(ta, dates)
    if t_G_date is not None:
        thresh_used = THETA_A_THRESH if ta[t_G_step] > THETA_A_THRESH else 0.1
        print(f"  t_G (Θ_A > {thresh_used} for {THETA_A_PERSIST}d): {t_G_date.date()} (step {t_G_step})")
        result["t_G"] = str(t_G_date.date())
        result["t_G_step"] = t_G_step
    else:
        print(f"  t_G: NOT FOUND")
        result["t_G"] = None

    # t_S: structural activation (C/Φ* < 0.90)
    t_S_step, t_S_date = find_first_crossing(cn, dates, C_NORM_THRESH_S, "below")
    if t_S_date is not None:
        print(f"  t_S (C/Φ* < {C_NORM_THRESH_S}): {t_S_date.date()} (step {t_S_step})")
        result["t_S"] = str(t_S_date.date())
        result["t_S_step"] = t_S_step
    else:
        print(f"  t_S: NOT REACHED")
        result["t_S"] = None

    # t_W: severe structural (C/Φ* < 0.50)
    t_W_step, t_W_date = find_first_crossing(cn, dates, C_NORM_THRESH_W, "below")
    if t_W_date is not None:
        print(f"  t_W (C/Φ* < {C_NORM_THRESH_W}): {t_W_date.date()} (step {t_W_step})")
        result["t_W"] = str(t_W_date.date())

    # t_C: post-irreversibility (C/Φ* < 0)
    t_C_step, t_C_date = find_first_crossing(cn, dates, 0, "below")
    if t_C_date is not None:
        print(f"  t_C (C/Φ* < 0): {t_C_date.date()} (step {t_C_step})")
        result["t_C"] = str(t_C_date.date())
        result["t_C_step"] = t_C_step
    else:
        result["t_C"] = None

    # ── Lead times ──
    if t_G_date is not None and t_S_date is not None:
        dT_GS = t_S_step - t_G_step
        days_GS = (t_S_date - t_G_date).days
        result["dT_GS_steps"] = dT_GS
        result["dT_GS_days"] = days_GS
        print(f"\n  ΔT_{{G→S}} = {dT_GS} steps ({days_GS} calendar days)")
        if dT_GS > 0:
            print(f"  >>> GEOMETRY PRECEDED STRUCTURE by {days_GS} days <<<")
        elif dT_GS < 0:
            print(f"  >>> STRUCTURE preceded geometry by {-days_GS} days")
            # Check if left-censored (damage before RQA warm-up)
            if t_S_step < 60:
                print(f"  *** LEFT-CENSORED: t_S at step {t_S_step} < RQA warm-up (60). "
                      f"Geometric detection was impossible by construction. ***")
                result["left_censored"] = True
        else:
            print(f"  >>> Simultaneous activation")
    else:
        result["dT_GS_steps"] = None
        result["dT_GS_days"] = None

    if t_G_date is not None and t_C_date is not None:
        dT_GC = t_C_step - t_G_step
        days_GC = (t_C_date - t_G_date).days
        result["dT_GC_steps"] = dT_GC
        result["dT_GC_days"] = days_GC
        print(f"  ΔT_{{G→C}} = {dT_GC} steps ({days_GC} calendar days) [to irreversibility]")
        if dT_GC > 0:
            print(f"  >>> GEOMETRY PRECEDED COLLAPSE by {days_GC} days <<<")

    # Special cases
    if t_G_date is None and t_S_date is not None:
        print(f"\n  NOTE: Structural damage WITHOUT geometric precursor")
        result["verdict"] = "NO_GEO_PRECURSOR"
    elif t_G_date is not None and t_S_date is None:
        crystallized_days = (dates[-1] - t_G_date).days
        print(f"\n  NOTE: Geometric activation WITHOUT structural damage ({crystallized_days}d and counting)")
        result["verdict"] = "GEO_ONLY_PENDING"
    elif result.get("dT_GS_days") is not None:
        if result["dT_GS_days"] > 0:
            result["verdict"] = "GEO_PRECEDED"
        elif result["dT_GS_days"] < 0:
            result["verdict"] = "STRUCT_PRECEDED"
            if result.get("left_censored"):
                result["verdict"] = "LEFT_CENSORED"
        else:
            result["verdict"] = "SIMULTANEOUS"

    # Timeline
    print(f"\n  Timeline:")
    events = []
    if t_G_date: events.append((t_G_step, "t_G", t_G_date))
    if t_S_date: events.append((t_S_step, "t_S", t_S_date))
    if t_W_date: events.append((t_W_step, "t_W", t_W_date))
    if t_C_date: events.append((t_C_step, "t_C", t_C_date))
    events.sort(key=lambda x: x[0])
    for step, label, date in events:
        bar_pos = int(50 * step / n)
        bar = "." * bar_pos + "|" + "." * (50 - bar_pos)
        cens = " [<warm-up]" if step < 60 and label in ("t_S", "t_W", "t_C") else ""
        print(f"    {label} [{bar}] {date.date()}{cens}")

    return result


def main():
    print("=" * 100)
    print("  KAPPA v2 — HISTORICAL LEAD-TIME ANALYSIS")
    print("  Measuring ΔT_{G→S} and ΔT_{G→C} across completed and ongoing crises")
    print("  David Ohio | Independent Researcher | March 2026")
    print("=" * 100)

    results = []
    for uid, folder in UNIVERSES.items():
        r = analyze_universe(uid, folder)
        if r:
            results.append(r)

    # ── SUMMARY TABLE ──
    print(f"\n\n{'='*100}")
    print(f"  SUMMARY: HISTORICAL LEAD TIMES")
    print(f"{'='*100}")
    print(f"\n  {'Universe':25s}  {'t_G':>12s}  {'t_S':>12s}  {'t_C':>12s}  "
          f"{'ΔT_G→S':>10s}  {'ΔT_G→C':>10s}  {'Verdict':>20s}")
    print(f"  {'─'*105}")

    for r in results:
        tg = r.get("t_G") or "—"
        ts = r.get("t_S") or "—"
        tc = r.get("t_C") or "—"
        dgs = f"{r['dT_GS_days']}d" if r.get("dT_GS_days") is not None else "—"
        dgc = f"{r['dT_GC_days']}d" if r.get("dT_GC_days") is not None else "—"
        verdict = r.get("verdict", "INSUFFICIENT")
        print(f"  {r['universe']:25s}  {tg:>12s}  {ts:>12s}  {tc:>12s}  "
              f"{dgs:>10s}  {dgc:>10s}  {verdict:>20s}")

    # ── VERDICTS ──
    geo_preceded = [r for r in results if r.get("verdict") == "GEO_PRECEDED"]
    struct_preceded = [r for r in results if r.get("verdict") == "STRUCT_PRECEDED"]
    left_censored = [r for r in results if r.get("verdict") == "LEFT_CENSORED"]
    no_geo = [r for r in results if r.get("verdict") == "NO_GEO_PRECURSOR"]
    geo_only = [r for r in results if r.get("verdict") == "GEO_ONLY_PENDING"]

    print(f"\n  VERDICTS:")
    print(f"    Geometry preceded structure:      {len(geo_preceded)}")
    print(f"    Structure preceded geometry:      {len(struct_preceded)}")
    print(f"    Left-censored (t_S < warm-up):   {len(left_censored)}")
    print(f"    No geometric precursor:           {len(no_geo)}")
    print(f"    Geometric only (pending):         {len(geo_only)}")

    # ── VALID CASES ONLY ──
    valid = [r for r in results
             if r.get("dT_GS_days") is not None and not r.get("left_censored")
             and r.get("t_S_step", 999) >= 60]
    print(f"\n  METHODOLOGICALLY VALID CASES (t_S ≥ step 60, no left-censoring):")
    print(f"    Total valid: {len(valid)}")
    for r in valid:
        v = r.get("verdict", "?")
        d = r.get("dT_GS_days", "?")
        print(f"      {r['universe']:25s}  ΔT_G→S = {d}d  [{v}]")

    if geo_preceded:
        lead_times = [r["dT_GS_days"] for r in geo_preceded]
        print(f"\n  LEAD TIMES (geometry preceded):")
        print(f"    Min:    {min(lead_times)} days")
        print(f"    Max:    {max(lead_times)} days")
        print(f"    Mean:   {np.mean(lead_times):.0f} days")
        print(f"    Median: {np.median(lead_times):.0f} days")

    # ── SAVE RESULTS ──
    out_path = BASE / "historical_lead_time_analysis.json"
    with open(out_path, "w") as f:
        json.dump({
            "thresholds": {
                "theta_A_thresh": THETA_A_THRESH,
                "theta_A_persist": THETA_A_PERSIST,
                "C_norm_thresh_S": C_NORM_THRESH_S,
                "C_norm_thresh_W": C_NORM_THRESH_W,
                "rqa_warmup_steps": 60,
            },
            "results": results,
            "summary": {
                "geo_preceded": len(geo_preceded),
                "struct_preceded": len(struct_preceded),
                "left_censored": len(left_censored),
                "no_geo_precursor": len(no_geo),
                "geo_only_pending": len(geo_only),
                "n_valid": len(valid),
            }
        }, f, indent=2)
    print(f"\n  Results saved: {out_path}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
