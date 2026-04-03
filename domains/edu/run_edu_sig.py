#!/usr/bin/env python3
"""
Kappa-SIG EDU: Obsessive Coherence in Educational Dropout (Revised)
=====================================================================
Uses the ACTUAL Kappa framework (KatashiAnalyzer + topological_metrics)
instead of proxy columns. Consistent with FIN/LLM/NEURO methodology:

  9 activity channels = correlation network
  Spearman correlation (rolling window) -> eigenstructure -> Kappa state

Data: OULAD (Open University Learning Analytics Dataset)
  Course: AAA 2014J, 4 cohorts (Pass, Fail, Distinction, Withdrawn)
  43 weeks, 9 activity channels, ~32K students

David Ohio | odavidohio@gmail.com | Independent Researcher
April 2026
"""
import sys, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from scipy.stats import entropy as sp_entropy

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ACTIVITY_COLS = ["clicks_dataplus", "clicks_forumng", "clicks_glossary",
                 "clicks_homepage", "clicks_oucollaborate", "clicks_oucontent",
                 "clicks_resource", "clicks_subpage", "clicks_url"]

COHORTS = ["pass", "fail", "distinction", "withdrawn"]
WINDOW = 10   # weeks
STEP = 1
GAMMA = 0.97  # Phi memory decay
K_SCALE = 4   # Xi scale factor


# ══════════════════════════════════════════════════════════
# KAPPA STATE FROM ACTIVITY CHANNEL CORRELATIONS
# Same methodology as FIN (engine v5.8c) and LLM (SIG inter-head)
# ══════════════════════════════════════════════════════════

def compute_kappa_from_window(window_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute Kappa state from a rolling window of activity channel data.
    Uses Spearman correlation between 9 activity channels.
    """
    activity = window_df[ACTIVITY_COLS].values  # (window, 9)
    n_ch = activity.shape[1]
    
    # Check for degenerate windows (channels with zero variance)
    variances = np.var(activity, axis=0)
    active = variances > 1e-10
    n_active = int(active.sum())
    
    if n_active < 3:
        return {"Oh": 0.0, "eta": 0.0, "mean_corr": 0.0, "DEF": 0.0,
                "Xi": 1.0, "degenerate": True, "n_active": n_active}

    # Spearman correlation between activity channels
    from scipy.stats import spearmanr
    rho, _ = spearmanr(activity[:, active])
    if rho.ndim == 0:
        rho = np.array([[1.0]])
    rho = np.nan_to_num(rho, nan=0.0)
    np.fill_diagonal(rho, 1.0)
    
    # Shrinkage (same as topological_metrics.py)
    n = rho.shape[0]
    C = 0.9 * rho + 0.1 * np.eye(n)
    
    # Eigenvalues
    eigvals = np.sort(np.abs(np.linalg.eigvalsh(C)))[::-1]
    eigvals = np.maximum(eigvals, 1e-12)
    total = eigvals.sum()
    
    # Oh: spectral concentration (lambda_1 / mean) — Ohio Number
    Oh = float(eigvals[0] / (total / n)) if n > 0 else 0.0
    
    # eta: Frobenius-based rigidity (same as topological_metrics.compute_eta)
    fro = np.linalg.norm(C, ord='fro')
    eta = float(1.0 + np.log1p(fro))

    # mean_corr: mean absolute off-diagonal correlation
    mask = ~np.eye(n, dtype=bool)
    mean_corr = float(np.mean(np.abs(C[mask])))
    
    # DEF: eigenvalue dominance gap (lambda_1 - lambda_2) / lambda_1
    DEF = float((eigvals[0] - eigvals[1]) / (eigvals[0] + 1e-10)) if n > 1 else 0.0
    
    # Xi: effective diversity (effective rank / n)
    eig_norm = eigvals / total
    eig_pos = eig_norm[eig_norm > 1e-12]
    eff_rank = np.exp(sp_entropy(eig_pos)) if len(eig_pos) > 0 else 1.0
    Xi = float(eff_rank / n)
    
    # Spectral entropy (from topological_metrics.compute_spectral_entropy)
    H_spec = float(-(eig_pos * np.log(eig_pos)).sum())
    
    return {"Oh": Oh, "eta": eta, "mean_corr": mean_corr, "DEF": DEF,
            "Xi": Xi, "H_spec": H_spec, "degenerate": False, "n_active": n_active}


def analyze_cohort(name: str, df: pd.DataFrame) -> Dict:
    """Analyze one cohort using rolling-window Kappa states."""
    n = len(df)
    states = []
    phi = 0.0
    
    for i in range(0, n - WINDOW + 1, STEP):
        window = df.iloc[i:i + WINDOW]
        s = compute_kappa_from_window(window)
        
        # Phi: exponential memory of Oh excursions
        oh_excess = max(0, s["Oh"] - 1.0) * 0.08
        phi = GAMMA * phi + oh_excess
        s["phi"] = phi
        s["week"] = i + WINDOW
        states.append(s)
    
    if not states:
        return {"cohort": name, "error": "no_states"}
    
    sdf = pd.DataFrame(states)
    n_steps = len(sdf)
    valid = sdf[~sdf["degenerate"]]

    if len(valid) < 3:
        return {"cohort": name, "error": "insufficient_valid_states",
                "n_total": n_steps, "n_degenerate": n_steps - len(valid)}
    
    STATE_COLS = ["Oh", "phi", "eta", "mean_corr", "DEF", "Xi"]
    
    # Mean states
    mean_state = {c: round(float(valid[c].mean()), 6) for c in STATE_COLS}
    
    # Late-phase (last 25%)
    late_start = int(len(valid) * 0.75)
    late = valid.iloc[late_start:]
    late_state = {c: round(float(late[c].mean()), 6) for c in STATE_COLS}
    
    # Oh drift (first quarter vs last quarter)
    first_q = valid.iloc[:len(valid)//4]
    last_q = valid.iloc[-len(valid)//4:]
    oh_drift = float(last_q["Oh"].mean() - first_q["Oh"].mean())
    
    # Obsessive Coherence Index
    # Coherence indicators: Oh, eta, DEF (all higher = more rigid)
    # Disorder indicators: Xi, (1-mean_corr) (higher = more flexible)
    coh = (mean_state["Oh"]/9 + mean_state["DEF"]) / 2.0  # Normalize Oh by n_channels
    dis = (mean_state["Xi"] + (1.0 - mean_state["mean_corr"])) / 2.0
    oc_score = coh - dis

    # Late-phase OC
    late_coh = (late_state["Oh"]/9 + late_state["DEF"]) / 2.0
    late_dis = (late_state["Xi"] + (1.0 - late_state["mean_corr"])) / 2.0
    late_oc = late_coh - late_dis
    
    return {
        "cohort": name, "n_weeks": n, "n_steps": n_steps,
        "n_degenerate": int((sdf["degenerate"]).sum()),
        "mean_state": mean_state, "late_state": late_state,
        "oc_score": round(oc_score, 6),
        "late_oc_score": round(late_oc, 6),
        "oh_drift": round(oh_drift, 6),
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("  KAPPA-SIG EDU: Obsessive Coherence (Revised - Proper Kappa)")
    print("  David Ohio | Independent Researcher | April 2026")
    print("  Method: Spearman correlation between 9 activity channels")
    print(f"  Window: {WINDOW} weeks, Step: {STEP}")
    print("=" * 70)
    
    results = {}
    for cohort_name in COHORTS:
        df = pd.read_csv(DATA_DIR / f"data_AAA_2014J_{cohort_name}.csv", parse_dates=["date"])
        r = analyze_cohort(cohort_name, df)
        results[cohort_name] = r
        
        if "error" in r:
            print(f"\n  [{cohort_name.upper()}] ERROR: {r['error']}")
            continue
        
        ms = r["mean_state"]
        print(f"\n  [{cohort_name.upper()}] ({r['n_weeks']} weeks, {r['n_steps']} steps, "
              f"{r['n_degenerate']} degenerate)")
        print(f"    Kappa State (correlation-based):")
        print(f"    {'':12s} {'Oh':>8s} {'phi':>8s} {'eta':>8s} {'m_corr':>8s} {'DEF':>8s} {'Xi':>8s}")
        print(f"    {'mean':12s} {ms['Oh']:8.4f} {ms['phi']:8.4f} {ms['eta']:8.4f} "
              f"{ms['mean_corr']:8.4f} {ms['DEF']:8.4f} {ms['Xi']:8.4f}")
        ls = r["late_state"]
        print(f"    {'late':12s} {ls['Oh']:8.4f} {ls['phi']:8.4f} {ls['eta']:8.4f} "
              f"{ls['mean_corr']:8.4f} {ls['DEF']:8.4f} {ls['Xi']:8.4f}")
        print(f"    OC={r['oc_score']:+.4f}  late_OC={r['late_oc_score']:+.4f}  "
              f"Oh_drift={r['oh_drift']:+.4f}")

    # ── Hypothesis test ──
    print(f"\n  {'='*70}")
    print("  OBSESSIVE COHERENCE HYPOTHESIS")
    print(f"  {'='*70}")
    
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    if len(valid_results) < 3:
        print("  INSUFFICIENT valid cohorts for hypothesis test")
    else:
        print(f"\n  {'Cohort':<15s} {'OC':>8s} {'Late OC':>10s} {'Oh Drift':>10s}")
        print(f"  {'-'*45}")
        for name in ["withdrawn", "fail", "pass", "distinction"]:
            r = valid_results.get(name)
            if r:
                print(f"  {name:<15s} {r['oc_score']:+8.4f} {r['late_oc_score']:+10.4f} "
                      f"{r['oh_drift']:+10.4f}")
        
        w = valid_results.get("withdrawn", {}).get("oh_drift", 0)
        f = valid_results.get("fail", {}).get("oh_drift", 0)
        p = valid_results.get("pass", {}).get("oh_drift", 0)
        
        print(f"\n  Oh drift ordering: W({w:+.4f}) vs F({f:+.4f}) vs P({p:+.4f})")
        if w > f and w > p:
            print(f"  Withdrawn has highest Oh drift: YES")
        else:
            print(f"  Withdrawn has highest Oh drift: NO")

    # ── Save ──
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "domain": "EDU",
        "dataset": "OULAD (AAA 2014J)",
        "method": "Spearman correlation between 9 activity channels, "
                  "rolling window, eigenstructure -> Kappa state. "
                  "Consistent with FIN (asset correlations) and LLM (head correlations).",
        "parameters": {"window": WINDOW, "step": STEP, "gamma": GAMMA, "n_channels": 9},
        "cohorts": {k: v for k, v in valid_results.items()},
    }
    out_file = OUT_DIR / "edu_obsessive_coherence.json"
    with open(out_file, "w") as fp:
        json.dump(output, fp, indent=2, default=str)
    
    elapsed = time.time() - t0
    print(f"\n  Results: {out_file}")
    print(f"  Time: {elapsed:.1f}s")
    print("=" * 70)

if __name__ == "__main__":
    main()
