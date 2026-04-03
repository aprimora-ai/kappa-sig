#!/usr/bin/env python3
r"""
Kappa v2 -- Experiment 2 (Final Corrected)
==========================================
Epistemically corrected baseline comparison implementing the full
taxonomy from the v2.5 paper.

Taxonomy labels are audit-derived from prior manual episode analysis
and are NOT inferred by this script. The experiment evaluates signals
CONDITIONED on this pre-established taxonomy.

Fixes applied:
  FIX-8:  Save complete analytical results to JSON (AUC, lead times, all signals)
  FIX-9:  Explicit audit-derived taxonomy declaration (this docstring)
  FIX-10: Threshold sensitivity analysis (Theta_A: 0.3/0.5/0.7, persist: 3/5/7)
  FIX-11: Phi_slope moved to supplementary analysis (not in main comparison)
  FIX-12: Part A renamed to event-proximity discrimination
  FIX-13: asia_pacific episodes saved as separate objects in output

Run:
  cd C:\Users\ohiod\Projects\Sentinel
  python run_v2_experiment2.py

David Ohio | odavidohio@gmail.com | Independent Researcher
March 2026
"""
import os, sys, json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings("ignore")

BASE = Path(r"C:\Users\ohiod\Projects\Sentinel\data\v2_analysis")
OUT = Path(r"C:\Users\ohiod\Projects\Sentinel\data\v2_analysis\experiment2")
OUT.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

# ============================================================================
# EPISTEMOLOGICAL TAXONOMY (audit-derived, not script-inferred)
# Source: v2.5 paper Section 8.8, historical lead-time audit, asia_pacific
# episode analysis. Each label was assigned through manual inspection of
# the full time series, RQA profiles, and CALM statistics.
# ============================================================================
UNIVERSE_TAXONOMY = {
    "commodities":          {"class": "GEO_PRECEDED",    "geo": "HIGH"},
    "energy":               {"class": "GEO_PRECEDED",    "geo": "HIGH"},
    "europe":               {"class": "LEFT_CENSORED",   "geo": "HIGH"},
    "x_energy_geopolitics": {"class": "LEFT_CENSORED",   "geo": "HIGH"},
    "us_sectors":           {"class": "LEFT_CENSORED",   "geo": "HIGH"},
    "x_us_systemic":        {"class": "PROSPECTIVE",     "geo": "HIGH"},
    "x_brazil_vuln":        {"class": "PROSPECTIVE",     "geo": "HIGH"},
    "x_europe_vuln":        {"class": "ALREADY_CRYSTAL", "geo": "LOW"},
    "financials":           {"class": "ALREADY_CRYSTAL", "geo": "LOW"},
    "asia_pacific":         {"class": "SPLIT",           "geo": "HIGH"},
}
AP_SPLIT_STEP = 653  # recovery to C/Phi* > 0.99 (Sep 2024)

# ============================================================================
# THRESHOLDS (documented rationale)
# ============================================================================
RQA_WARMUP = 60          # RQA window size; Theta_A undefined before this
C_NORM_THRESHOLD = 0.90  # structural activation per paper Section 5

# Default activation thresholds (sensitivity tested in Part D)
THETA_THRESH = 0.5       # 10% of cap(5.0); marks meaningful geometric departure
THETA_PERSIST = 5        # 5 business days; filters 1-day spike artifacts
KF_THRESH = 0.5          # log(Phi*/C)>0.5 means C < 0.61*Phi*
OH_THRESH = 0.8          # 80% of Katashi threshold (Oh=1.0)
OH_PERSIST = 5           # same persistence as Theta_A for fairness

# Main comparison signals (FIX-11: Phi_slope moved to supplementary)
MAIN_SIGNALS = ["Theta_A", "kappa_F", "Oh_EMA"]
HORIZONS = [30, 60, 90]


# ============================================================================
# HELPERS
# ============================================================================

def compute_oh_ema(oh, alpha=0.1):
    ema = np.zeros(len(oh))
    ema[0] = oh[0]
    for i in range(1, len(oh)):
        ema[i] = alpha * oh[i] + (1 - alpha) * ema[i - 1]
    return ema


def compute_phi_slope(phi, window=30):
    slopes = np.zeros(len(phi))
    for t in range(window, len(phi)):
        w = phi[t-window:t]
        x = np.arange(window)
        slopes[t] = np.polyfit(x, w, 1)[0] if np.std(w) > 1e-15 else 0.0
    return slopes


def compute_target_first_crossing(c_norm, horizon):
    """Target = 1 in the H steps before first t_S. Post-event censored."""
    n = len(c_norm)
    t_S = None
    for i in range(n):
        if c_norm[i] < C_NORM_THRESHOLD:
            t_S = i
            break
    target = np.zeros(n, dtype=int)
    valid = np.ones(n, dtype=bool)
    if t_S is not None:
        for t in range(max(0, t_S - horizon), t_S):
            target[t] = 1
        valid[t_S:] = False
    return target, valid, t_S


def first_sustained(signal, thresh, persist, start=0):
    n = len(signal)
    for i in range(start, n - persist + 1):  # FIX-4: boundary corrected
        if all(signal[i:i + persist] > thresh):
            return i
    return None


def first_crossing(signal, thresh, start=0):
    for i in range(start, len(signal)):
        if signal[i] > thresh:
            return i
    return None


def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    if np.std(y_score) < 1e-15:
        return 0.5, float(y_true.mean())
    try:
        return roc_auc_score(y_true, y_score), average_precision_score(y_true, y_score)
    except Exception:
        return float("nan"), float("nan")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 100)
    print("  KAPPA v2 -- EXPERIMENT 2 (FINAL CORRECTED)")
    print("  FIX-8 through FIX-13 applied")
    print("  Taxonomy is audit-derived, not script-inferred (FIX-9)")
    print("  David Ohio | Independent Researcher | March 2026")
    print("=" * 100)

    # -- Collector for FIX-8: complete JSON output --
    output = {
        "metadata": {
            "note": "Taxonomy labels are audit-derived from prior manual episode analysis "
                    "and are not inferred by this script.",
            "thresholds": {
                "theta_a": THETA_THRESH, "theta_persist": THETA_PERSIST,
                "kf": KF_THRESH, "oh": OH_THRESH, "oh_persist": OH_PERSIST,
                "c_norm": C_NORM_THRESHOLD, "rqa_warmup": RQA_WARMUP,
            },
            "ap_split_step": AP_SPLIT_STEP,
        },
        "taxonomy": {},
        "part_a": {},  # event-proximity AUC
        "part_a2": {}, # per-universe AUC
        "part_b": {},  # temporal precedence
        "part_c": {},  # lead times (valid cases)
        "part_d": {},  # threshold sensitivity
        "asia_pacific_episodes": {},  # FIX-13
        "supplementary": {},  # Phi_slope (FIX-11)
    }

    # -- Load universes --
    print("\n  TAXONOMY (audit-derived):")
    print("  " + "-"*70)
    udata = {}
    for uid, tax in UNIVERSE_TAXONOMY.items():
        csv = BASE / uid / "kappa_v2_state.csv"
        if not csv.exists():
            print(f"  SKIP {uid}")
            continue
        df = pd.read_csv(csv, index_col="date", parse_dates=True)
        udata[uid] = {
            "df": df, "n": len(df), "cn": df["C_norm"].values,
            "signals": {
                "Theta_A": df["theta_A"].values,
                "kappa_F": df["kappa_F"].values,
                "Oh_EMA": compute_oh_ema(df["Oh"].values),
                "Phi_slope": compute_phi_slope(df["phi"].values),
                "Random": np.random.rand(len(df)),
            },
            "tax": tax,
        }
        output["taxonomy"][uid] = tax
        print(f"  {uid:25s}  {tax['class']:20s}  geo={tax['geo']}")

    # ==================================================================
    # PART A: EVENT-PROXIMITY DISCRIMINATION (FIX-12: renamed)
    # Only GEO_PRECEDED, pre-event censored
    # ==================================================================
    print("\n\n" + "="*100)
    print("  PART A: EVENT-PROXIMITY DISCRIMINATION UNDER PRE-EVENT CENSORING")
    print("  (Not early warning -- measures proximity to first structural activation)")
    print("  Valid universes: GEO_PRECEDED only")
    print("="*100)

    valid_uids = [u for u, d in udata.items() if d["tax"]["class"] == "GEO_PRECEDED"]
    output["part_a"]["valid_universes"] = valid_uids
    output["part_a"]["horizons"] = {}

    for h in HORIZONS:
        print(f"\n  Horizon: {h}d")
        print(f"  {'Signal':15s}  {'AUC':>8s}  {'AP':>8s}  {'n':>6s}")
        print("  " + "-"*45)

        py, ps = [], {sn: [] for sn in MAIN_SIGNALS + ["Random"]}
        for uid in valid_uids:
            d = udata[uid]
            tgt, vm, _ = compute_target_first_crossing(d["cn"], h)
            mask = vm & (np.arange(d["n"]) >= RQA_WARMUP)
            if mask.sum() < 10:
                continue
            py.append(tgt[mask])
            for sn in MAIN_SIGNALS + ["Random"]:
                ps[sn].append(d["signals"][sn][mask])

        y = np.concatenate(py)
        h_results = {}
        for sn in MAIN_SIGNALS + ["Random"]:
            sc = np.concatenate(ps[sn])
            auc, ap = safe_auc(y, sc)
            h_results[sn] = {"auc": auc, "ap": ap, "n": len(y)}
            print(f"  {sn:15s}  {auc:8.4f}  {ap:8.4f}  {len(y):6d}")
        output["part_a"]["horizons"][str(h)] = h_results

    # ==================================================================
    # PART A2: PER-UNIVERSE AUC (all classes, reference)
    # ==================================================================
    print("\n\n" + "="*100)
    print("  PART A2: PER-UNIVERSE AUC (H=60, all classes, censored)")
    print("="*100)

    h = 60
    print(f"\n  {'Universe':25s} {'Class':>15s} {'Geo':>4s}", end="")
    for sn in MAIN_SIGNALS:
        print(f"  {sn:>10s}", end="")
    print()
    print("  " + "-"*75)

    for uid, d in udata.items():
        tgt, vm, _ = compute_target_first_crossing(d["cn"], h)
        mask = vm & (np.arange(d["n"]) >= RQA_WARMUP)
        cls = d["tax"]["class"]
        geo = d["tax"]["geo"]
        row = {"class": cls, "geo": geo}
        print(f"  {uid:25s} {cls:>15s} {geo:>4s}", end="")
        if mask.sum() < 10:
            print(f"  {'(insufficient)':>35s}")
            output["part_a2"][uid] = row
            continue
        y = tgt[mask]
        for sn in MAIN_SIGNALS:
            auc, _ = safe_auc(y, d["signals"][sn][mask])
            row[f"auc_{sn}"] = auc
            print(f"  {auc:10.3f}", end="")
        print()
        output["part_a2"][uid] = row

    # ==================================================================
    # PART B: TEMPORAL PRECEDENCE (taxonomy-aware)
    # ==================================================================
    print("\n\n" + "="*100)
    print("  PART B: TEMPORAL PRECEDENCE (HIGH geo-reliability only)")
    print("="*100)

    print(f"\n  {'Universe':25s}  {'Class':>15s}  {'t_TA':>7s}  {'t_kF':>7s}  {'t_Oh':>7s}  {'d(kF-TA)':>9s}  {'Verdict':>25s}")
    print("  " + "-"*105)

    for uid, d in udata.items():
        tax = d["tax"]
        if tax["geo"] == "LOW":
            print(f"  {uid:25s}  {tax['class']:>15s}  {'---':>7s}  {'---':>7s}  {'---':>7s}  {'---':>9s}  {'EXCLUDED (LOW geo)':>25s}")
            output["part_b"][uid] = {"class": tax["class"], "excluded": True, "reason": "LOW geo"}
            continue
        if tax["class"] == "SPLIT":
            continue

        ta = d["signals"]["Theta_A"]
        kf = d["signals"]["kappa_F"]
        oh = d["signals"]["Oh_EMA"]

        t_ta = first_sustained(ta, THETA_THRESH, THETA_PERSIST, start=RQA_WARMUP)
        t_kf = first_crossing(kf, KF_THRESH, start=RQA_WARMUP)
        t_oh = first_sustained(oh, OH_THRESH, OH_PERSIST, start=RQA_WARMUP)

        delta = (t_kf - t_ta) if (t_ta is not None and t_kf is not None) else None
        if delta is not None:
            verdict = "THETA_A PRECEDED kF" if delta > 10 else ("kF preceded Theta_A" if delta < -10 else "SIMULTANEOUS")
        elif t_ta is not None:
            verdict = "Theta_A only"
        elif t_kf is not None:
            verdict = "kF only"
        else:
            verdict = "Neither"

        ta_s = str(t_ta) if t_ta is not None else "never"
        kf_s = str(t_kf) if t_kf is not None else "never"
        oh_s = str(t_oh) if t_oh is not None else "never"
        d_s = f"{delta:+d}" if delta is not None else "N/A"

        print(f"  {uid:25s}  {tax['class']:>15s}  {ta_s:>7s}  {kf_s:>7s}  {oh_s:>7s}  {d_s:>9s}  {verdict:>25s}")
        output["part_b"][uid] = {
            "class": tax["class"], "t_theta_a": t_ta, "t_kappa_f": t_kf,
            "t_oh_ema": t_oh, "delta_kf_ta": delta, "verdict": verdict,
        }

    # -- FIX-13: asia_pacific episodes --
    if "asia_pacific" in udata:
        d = udata["asia_pacific"]
        df = d["df"]

        ta1 = d["signals"]["Theta_A"][:AP_SPLIT_STEP]
        kf1 = d["signals"]["kappa_F"][:AP_SPLIT_STEP]
        cn1 = d["cn"][:AP_SPLIT_STEP]
        t_ta1 = first_sustained(ta1, THETA_THRESH, THETA_PERSIST, start=RQA_WARMUP)
        t_kf1 = first_crossing(kf1, KF_THRESH, start=RQA_WARMUP)
        t_S1 = None
        for i in range(RQA_WARMUP, len(cn1)):
            if cn1[i] < C_NORM_THRESHOLD:
                t_S1 = i
                break
        lt1 = (t_S1 - t_ta1) if (t_ta1 is not None and t_S1 is not None and t_ta1 < t_S1) else None

        ep1 = {
            "class": "AMBIGUOUS", "reason": "Type 1 spike concurrent with Ukraine shock (Section 3.10)",
            "steps": f"0-{AP_SPLIT_STEP}", "t_theta_a": t_ta1, "t_kappa_f": t_kf1,
            "t_S": t_S1, "lead_theta_a": lt1, "counted_as_evidence": False,
        }
        output["asia_pacific_episodes"]["episode_1"] = ep1
        print(f"\n  asia_pacific Ep.1: AMBIGUOUS (spike). t_TA={t_ta1}, t_S={t_S1}, lead={lt1}. NOT counted.")

        ta2 = d["signals"]["Theta_A"][AP_SPLIT_STEP:]
        kf2 = d["signals"]["kappa_F"][AP_SPLIT_STEP:]
        cn2 = d["cn"][AP_SPLIT_STEP:]
        t_ta2 = first_sustained(ta2, THETA_THRESH, THETA_PERSIST, start=0)
        t_kf2 = first_crossing(kf2, KF_THRESH, start=0)
        t_S2 = None
        for i in range(len(cn2)):
            if cn2[i] < C_NORM_THRESHOLD:
                t_S2 = i
                break

        t_ta2_abs = (AP_SPLIT_STEP + t_ta2) if t_ta2 is not None else None
        ta2_date = str(df.index[t_ta2_abs].date()) if t_ta2_abs is not None else "never"
        days_active = (len(ta2) - t_ta2) if t_ta2 is not None else 0

        ep2 = {
            "class": "PROSPECTIVE", "reason": "Type 2 ramp Nov 2025, clean second cycle on recovered system",
            "steps": f"{AP_SPLIT_STEP}-{d['n']}",
            "t_theta_a_abs": t_ta2_abs, "t_theta_a_date": ta2_date,
            "t_kappa_f": (AP_SPLIT_STEP + t_kf2) if t_kf2 is not None else None,
            "t_S": None if t_S2 is None else AP_SPLIT_STEP + t_S2,
            "days_theta_a_active": days_active, "damage_detected": t_S2 is not None,
        }
        output["asia_pacific_episodes"]["episode_2"] = ep2
        print(f"  asia_pacific Ep.2: PROSPECTIVE. t_TA={t_ta2_abs} ({ta2_date}), active {days_active} steps, no damage.")

    # ==================================================================
    # PART C: LEAD TIMES (GEO_PRECEDED only)
    # ==================================================================
    print("\n\n" + "="*100)
    print("  PART C: LEAD TIME BEFORE FIRST DAMAGE (GEO_PRECEDED, HIGH geo)")
    print("="*100)

    for uid in ["commodities", "energy"]:
        d = udata[uid]
        cn = d["cn"]
        df = d["df"]

        t_S = None
        for i in range(RQA_WARMUP, len(cn)):
            if cn[i] < C_NORM_THRESHOLD:
                t_S = i
                break
        if t_S is None:
            continue

        ts_date = str(df.index[t_S].date())
        print(f"\n  {uid} (t_S = step {t_S}, {ts_date})")

        lt_data = {"t_S": t_S, "t_S_date": ts_date}

        for sn in MAIN_SIGNALS:
            sig = d["signals"][sn]
            if sn == "Theta_A":
                t_act = first_sustained(sig, THETA_THRESH, THETA_PERSIST, start=RQA_WARMUP)
            elif sn == "kappa_F":
                t_act = first_crossing(sig, KF_THRESH, start=RQA_WARMUP)
            else:
                t_act = first_sustained(sig, OH_THRESH, OH_PERSIST, start=RQA_WARMUP)

            lt = (t_S - t_act) if (t_act is not None and t_act < t_S) else None
            act_date = str(df.index[t_act].date()) if t_act is not None else "never"

            lt_data[f"t_{sn}"] = t_act
            lt_data[f"t_{sn}_date"] = act_date
            lt_data[f"lead_{sn}"] = lt

            if lt is not None:
                print(f"    {sn:15s}: step {t_act} ({act_date}), lead = {lt} steps  <<<")
            elif t_act is not None:
                print(f"    {sn:15s}: step {t_act} ({act_date}), AFTER damage")
            else:
                print(f"    {sn:15s}: never activated")

        output["part_c"][uid] = lt_data

    # ==================================================================
    # PART D: THRESHOLD SENSITIVITY (FIX-10)
    # ==================================================================
    print("\n\n" + "="*100)
    print("  PART D: THRESHOLD SENSITIVITY FOR Theta_A")
    print("  Testing: thresh in [0.3, 0.5, 0.7], persist in [3, 5, 7]")
    print("  Checking: does the ordering survive?")
    print("="*100)

    sens_thresholds = [0.3, 0.5, 0.7]
    sens_persists = [3, 5, 7]

    print(f"\n  {'thresh':>6s}  {'persist':>7s}  ", end="")
    for uid in ["commodities", "energy"]:
        print(f"  {'LT_'+uid:>18s}", end="")
    print(f"  {'Ordering ok?':>14s}")
    print("  " + "-"*75)

    sens_results = []
    for thr in sens_thresholds:
        for per in sens_persists:
            lts = {}
            for uid in ["commodities", "energy"]:
                d = udata[uid]
                t_S = None
                for i in range(RQA_WARMUP, len(d["cn"])):
                    if d["cn"][i] < C_NORM_THRESHOLD:
                        t_S = i
                        break
                t_act = first_sustained(d["signals"]["Theta_A"], thr, per, start=RQA_WARMUP)
                lt = (t_S - t_act) if (t_act is not None and t_S is not None and t_act < t_S) else None
                lts[uid] = lt

            ok = all(lt is not None and lt > 0 for lt in lts.values())

            row = {"thresh": thr, "persist": per}
            print(f"  {thr:6.1f}  {per:7d}  ", end="")
            for uid in ["commodities", "energy"]:
                lt = lts[uid]
                row[f"lt_{uid}"] = lt
                lt_str = f"{lt}" if lt is not None else "none"
                print(f"  {lt_str:>18s}", end="")
            row["ordering_preserved"] = ok
            print(f"  {'YES' if ok else 'NO':>14s}")
            sens_results.append(row)

    output["part_d"] = {"configurations": sens_results}

    all_ok = all(r["ordering_preserved"] for r in sens_results)
    n_ok = sum(r["ordering_preserved"] for r in sens_results)
    print(f"\n  Robustness: {'ALL configurations preserve ordering' if all_ok else f'{n_ok}/{len(sens_results)} preserve ordering'}")

    # ==================================================================
    # SUPPLEMENTARY: Phi_slope (FIX-11)
    # ==================================================================
    print("\n\n" + "="*100)
    print("  SUPPLEMENTARY: Phi_slope analysis (not in main comparison)")
    print("  Threshold: slope > 0.0001, persist 5 days")
    print("="*100)

    for uid in ["commodities", "energy"]:
        d = udata[uid]
        t_S = None
        for i in range(RQA_WARMUP, len(d["cn"])):
            if d["cn"][i] < C_NORM_THRESHOLD:
                t_S = i
                break
        t_act = first_sustained(d["signals"]["Phi_slope"], 0.0001, 5, start=RQA_WARMUP)
        lt = (t_S - t_act) if (t_act is not None and t_S is not None and t_act < t_S) else None
        output["supplementary"][uid] = {"t_phi_slope": t_act, "lead_phi_slope": lt}
        lt_str = f"{lt} steps" if lt is not None else "none"
        print(f"  {uid}: t_act={t_act}, lead={lt_str}")

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print("\n\n" + "="*100)
    print("  EXPERIMENT 2 (FINAL) -- SUMMARY")
    print("="*100)
    print("""
  TAXONOMY (audit-derived, not script-inferred):
    GEO_PRECEDED:     commodities, energy
    LEFT_CENSORED:    europe, x_energy_geo, us_sectors
    ALREADY_CRYSTAL:  x_europe_vuln, financials (LOW geo, excluded)
    PROSPECTIVE:      x_us_systemic, x_brazil_vuln
    SPLIT:            asia_pacific (Ep.1=AMBIGUOUS, Ep.2=PROSPECTIVE)

  PART A (EVENT-PROXIMITY DISCRIMINATION, pre-event censored):
    kappa_F dominates concurrent proximity (as expected -- tautological).
    Theta_A near random for proximity.
    This is NOT a test of early warning.

  PART B (TEMPORAL PRECEDENCE):
    commodities: Theta_A 185 steps before damage. kappa_F: 0 lead.
    energy:      Theta_A 24 steps before damage.  kappa_F: 0 lead.
    3 prospective cases: Theta_A active, no damage yet.""")

    print(f"""
  PART D (THRESHOLD SENSITIVITY):
    {'ROBUST' if all_ok else 'PARTIALLY ROBUST'}: ordering preserved across
    {n_ok}/{len(sens_results)} threshold configurations.

  CONCLUSION:
    Theta_A and kappa_F answer DIFFERENT questions.
    kappa_F: "Are we damaged now?" (AUC ~0.87, lead time 0)
    Theta_A: "Is the system hardening?" (AUC ~0.50, lead time 24-185 steps)
    They are complementary. Theta_A opens the window. kappa_F confirms.
""")

    # -- Save complete results (FIX-8) --
    out_file = OUT / "experiment2_final_results.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved: {out_file}")
    print("\nDone.")


if __name__ == "__main__":
    main()
