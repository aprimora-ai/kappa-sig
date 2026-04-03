#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Kappa-SIG: FIX-12 — CALM Audit & Alternative Baselines
=======================================================
Audits the CALM normalization effect on geometric detection by:

1. ANATOMY: For each universe, side-by-side comparison of Theta_A (CALM-dependent)
   vs SCR/lambda_1 (CALM-free) vs C_norm trajectory
   
2. VARIANTS: Test alternative Theta_A computation strategies:
   - V0: Current (z-score vs CALM) — baseline
   - V1: z-score vs global corpus percentiles (no per-universe CALM)
   - V2: z-score vs rolling window (adaptive baseline)
   - V3: Hybrid — SCR-weighted Theta_A (instant prior + temporal RQA)

3. RECALIBRATION: Re-run Experiment 3 logic with best variant

David Ohio | odavidohio@gmail.com | Independent Researcher
April 2026
"""
import sys, json, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

SENTINEL_DIR = Path(r"C:\Users\ohiod\Projects\Sentinel")
V2_DIR = SENTINEL_DIR / "data" / "v2_analysis"
SIG_DIR = SENTINEL_DIR / "data" / "sig"
OUT_DIR = SIG_DIR / "fix12"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Emblematic universes for detailed audit
AUDIT_UNIVERSES = [
    "commodities",           # GEO_PRECEDED, lead=269d
    "energy",                # GEO_PRECEDED, lead=35d
    "europe",                # LEFT_CENSORED, recovered by SIG
    "x_us_systemic",         # Crystallized 26 months, no damage
    "x_brazil_vuln",         # LEFT_CENSORED, NOT recovered
    "us_sectors",            # LEFT_CENSORED, recovered by SIG
    "ai_ecosystem",          # Healthy, good CALM
    "financials",            # ALREADY_CRYSTALLIZED
]


# ══════════════════════════════════════════════════════════════════
# PART 1: ANATOMY — Where does CALM flatten the signal?
# ══════════════════════════════════════════════════════════════════

def load_v2_state(uid):
    """Load v2 state CSV."""
    p = V2_DIR / uid / "kappa_v2_state.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, index_col="date", parse_dates=True)

def load_sig(uid):
    """Load SIG extracted data."""
    p = SIG_DIR / f"{uid}_sig.npz"
    if not p.exists():
        return None
    return np.load(p, allow_pickle=True)

def run_anatomy():
    """Side-by-side: Theta_A vs SCR vs C_norm for audit universes."""
    print("\n" + "=" * 70)
    print("  PART 1: ANATOMY — Where CALM flattens the signal")
    print("=" * 70)


    results = {}
    for uid in AUDIT_UNIVERSES:
        v2 = load_v2_state(uid)
        sig = load_sig(uid)
        if v2 is None or sig is None:
            print(f"\n  [{uid}] SKIP — missing data")
            continue

        n = min(len(v2), len(sig["scr"]))
        theta_a = v2["theta_A"].values[:n]
        c_norm = v2["C_norm"].values[:n]
        scr = sig["scr"][:n]
        lam1 = sig["lam1"][:n] if "lam1" in sig else scr * sig["n_assets"]
        d_mp = sig["d_mp_kl"][:n] if "d_mp_kl" in sig else np.zeros(n)

        # Find key dates
        t_S = None
        for i in range(n):
            if c_norm[i] < 0.90:
                t_S = i
                break

        t_G = None
        for i in range(n):
            if theta_a[i] > 0.5:
                t_G = i
                break

        # Find where SCR is high but Theta_A is low (CALM suppression zone)
        scr_high = scr > np.percentile(scr, 75)
        theta_low = theta_a < 0.1
        suppressed = scr_high & theta_low
        n_suppressed = int(np.sum(suppressed))
        frac_suppressed = n_suppressed / max(n, 1)


        # Of suppressed steps, how many precede damage within 90 steps?
        n_suppressed_before_damage = 0
        if t_S is not None:
            for i in range(n):
                if suppressed[i] and i < t_S and (t_S - i) <= 90:
                    n_suppressed_before_damage += 1

        # Classification
        if frac_suppressed > 0.20:
            calm_verdict = "SEVERE_SUPPRESSION"
        elif frac_suppressed > 0.05:
            calm_verdict = "MODERATE_SUPPRESSION"
        else:
            calm_verdict = "MINIMAL"

        results[uid] = {
            "n_steps": n,
            "t_S": t_S, "t_G": t_G,
            "theta_a_mean": float(np.mean(theta_a)),
            "scr_mean": float(np.mean(scr)),
            "n_suppressed": n_suppressed,
            "frac_suppressed": frac_suppressed,
            "n_suppressed_pre_damage_90": n_suppressed_before_damage,
            "calm_verdict": calm_verdict,
        }

        print(f"\n  [{uid}]")
        print(f"    t_S={t_S}  t_G={t_G}  steps={n}")
        print(f"    Theta_A: mean={np.mean(theta_a):.3f}  max={np.max(theta_a):.3f}")
        print(f"    SCR:     mean={np.mean(scr):.3f}  max={np.max(scr):.3f}")
        print(f"    Suppressed (SCR>p75 & Theta_A<0.1): {n_suppressed} steps ({frac_suppressed:.1%})")
        if n_suppressed_before_damage > 0:
            print(f"    >>> {n_suppressed_before_damage} suppressed steps within 90d of damage!")
        print(f"    Verdict: {calm_verdict}")

    return results


# ══════════════════════════════════════════════════════════════════
# PART 2: VARIANTS — Alternative Theta_A computations
# ══════════════════════════════════════════════════════════════════

def compute_rqa_simple(phi_series, window=60, epsilon_q=0.10, m=3):
    """Minimal RQA: DET, LAM, TT from delay embedding of phi."""
    from scipy.spatial.distance import pdist, squareform
    n = len(phi_series)
    if n < window + 10:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)

    # Estimate tau
    x = phi_series - np.mean(phi_series)
    tau = 1
    for t in range(1, min(20, n // 2)):
        if len(x[:-t]) < 2:
            continue
        c = np.corrcoef(x[:-t], x[t:])[0, 1]
        if c < 0:
            tau = t
            break

    DET_arr = np.full(n, np.nan)
    LAM_arr = np.full(n, np.nan)
    TT_arr = np.full(n, np.nan)


    for t in range(window, n):
        seg = phi_series[t - window:t]
        # Delay embedding
        max_idx = len(seg) - (m - 1) * tau
        if max_idx < 5:
            continue
        embedded = np.array([seg[i:i + (m - 1) * tau + 1:tau] for i in range(max_idx)])
        # Distance matrix
        dists = squareform(pdist(embedded))
        eps = np.quantile(dists[np.triu_indices(len(dists), 1)], epsilon_q)
        R = (dists <= max(eps, 1e-12)).astype(int)
        np.fill_diagonal(R, 0)
        total = R.sum()
        if total < 2:
            DET_arr[t] = 0.0; LAM_arr[t] = 0.0; TT_arr[t] = 1.0
            continue
        # DET: fraction in diagonal lines >= 2
        det_count = 0
        nn = len(R)
        for k in range(1, nn):
            diag = np.diag(R, k)
            run = 0
            for v in diag:
                if v:
                    run += 1
                else:
                    if run >= 2:
                        det_count += run
                    run = 0
            if run >= 2:
                det_count += run
        DET_arr[t] = det_count / max(total / 2, 1)


        # LAM: fraction in vertical lines >= 2
        lam_count = 0
        vert_lengths = []
        for col in range(nn):
            run = 0
            for row in range(nn):
                if R[row, col]:
                    run += 1
                else:
                    if run >= 2:
                        lam_count += run
                        vert_lengths.append(run)
                    run = 0
            if run >= 2:
                lam_count += run
                vert_lengths.append(run)
        LAM_arr[t] = lam_count / max(total, 1)
        TT_arr[t] = float(np.mean(vert_lengths)) if vert_lengths else 1.0

    return DET_arr, LAM_arr, TT_arr


def compute_theta_variants(uid, v2_df, sig_data):
    """Compute 4 variants of Theta_A for comparison."""
    n = min(len(v2_df), len(sig_data["scr"]))
    phi = v2_df["phi"].values[:n] if "phi" in v2_df.columns else np.zeros(n)

    # RQA (recompute from phi — simplified version)
    DET, LAM, TT = compute_rqa_simple(phi, window=60, epsilon_q=0.10)

    # V0: Current Theta_A (from v2 state — CALM z-score)
    V0 = v2_df["theta_A"].values[:n]

    # Identify CALM region (first 40%)
    calm_end = int(n * 0.4)
    calm_mask = np.zeros(n, dtype=bool)
    calm_mask[:calm_end] = True


    # --- V1: Global corpus percentiles (no per-universe CALM) ---
    # Use fixed percentiles from ALL universes combined
    # For now, use hardcoded quantiles derived from corpus
    GLOBAL_DET_MEAN, GLOBAL_DET_STD = 0.85, 0.10
    GLOBAL_LAM_MEAN, GLOBAL_LAM_STD = 0.90, 0.08
    GLOBAL_TT_MEAN, GLOBAL_TT_STD = 5.0, 3.0
    sigma_floor = 0.01
    w_D, w_L, w_T = 0.40, 0.35, 0.25

    V1 = np.zeros(n)
    for t in range(n):
        if np.isnan(DET[t]):
            continue
        z_det = max((DET[t] - GLOBAL_DET_MEAN) / max(GLOBAL_DET_STD, sigma_floor), 0)
        z_lam = max((LAM[t] - GLOBAL_LAM_MEAN) / max(GLOBAL_LAM_STD, sigma_floor), 0)
        z_tt = max((TT[t] - GLOBAL_TT_MEAN) / max(GLOBAL_TT_STD, sigma_floor), 0)
        V1[t] = min(w_D * z_det + w_L * z_lam + w_T * z_tt, 5.0)


    # --- V2: Rolling window baseline (adaptive, no fixed CALM) ---
    ROLL_W = 120  # 120-step trailing window as adaptive baseline
    V2 = np.zeros(n)
    for t in range(ROLL_W, n):
        if np.isnan(DET[t]):
            continue
        window_det = DET[t - ROLL_W:t]
        window_lam = LAM[t - ROLL_W:t]
        window_tt = TT[t - ROLL_W:t]
        valid_d = window_det[np.isfinite(window_det)]
        valid_l = window_lam[np.isfinite(window_lam)]
        valid_t = window_tt[np.isfinite(window_tt)]
        if len(valid_d) < 10:
            continue
        z_det = max((DET[t] - np.mean(valid_d)) / max(np.std(valid_d), sigma_floor), 0)
        z_lam = max((LAM[t] - np.mean(valid_l)) / max(np.std(valid_l), sigma_floor), 0)
        z_tt = max((TT[t] - np.mean(valid_t)) / max(np.std(valid_t), sigma_floor), 0)
        V2[t] = min(w_D * z_det + w_L * z_lam + w_T * z_tt, 5.0)


    # --- V3: Hybrid — SCR-weighted Theta_A ---
    # Theta_A_hybrid = V0 * (1 + alpha * SCR_z)
    # Where SCR_z = (SCR - SCR_median) / SCR_std
    # This boosts Theta_A when instantaneous spectral concentration is high
    scr = sig_data["scr"][:n].astype(float)
    scr_med = np.median(scr[:calm_end]) if calm_end > 10 else np.median(scr)
    scr_std = max(np.std(scr[:calm_end]), 0.01) if calm_end > 10 else max(np.std(scr), 0.01)
    alpha_hybrid = 0.5  # boost factor

    V3 = np.zeros(n)
    for t in range(n):
        scr_z = max((scr[t] - scr_med) / scr_std, 0)
        # If V0 is suppressed (near 0) but SCR is high, inject signal
        if V0[t] < 0.1 and scr_z > 1.0:
            V3[t] = min(scr_z * 0.5, 5.0)  # SCR-only fallback
        else:
            V3[t] = min(V0[t] * (1.0 + alpha_hybrid * scr_z), 5.0)

    return {
        "V0_calm_zscore": V0,
        "V1_global_percentile": V1,
        "V2_rolling_baseline": V2,
        "V3_hybrid_scr": V3,
    }


def run_variants():
    """Test all Theta_A variants across audit universes."""
    print("\n" + "=" * 70)
    print("  PART 2: VARIANT COMPARISON")
    print("  V0=CALM z-score | V1=Global | V2=Rolling | V3=Hybrid SCR")
    print("=" * 70)

    all_results = {}
    # Collect global RQA stats first (for V1 calibration)
    # For now using hardcoded estimates, to be refined

    for uid in AUDIT_UNIVERSES:
        v2 = load_v2_state(uid)
        sig = load_sig(uid)
        if v2 is None or sig is None:
            continue

        print(f"\n  [{uid}]")
        variants = compute_theta_variants(uid, v2, sig)
        n = len(variants["V0_calm_zscore"])


        # Target: CALM-free forward H=90
        labels = sig["labels"][:n]
        y_cf = np.zeros(n, dtype=int)
        for t in range(n):
            future = labels[t + 1:t + 1 + 90]
            if any(l == "DAMAGED" for l in future):
                y_cf[t] = 1
        # Also CALM-dependent target
        y_cd = np.array([1 if l in ("CRYSTALLIZING", "CRYSTALLIZED") else 0
                         for l in labels])

        uid_results = {}
        print(f"    {'Variant':<25s} {'AUC(CALM-dep)':>14s} {'AUC(CALM-free)':>14s}")
        print(f"    {'-'*55}")

        for vname, theta in variants.items():
            valid = np.isfinite(theta) & (theta >= 0)
            for target_name, y in [("calm_dep", y_cd), ("calm_free", y_cf)]:
                x = theta[valid]
                yt = y[valid]
                if len(set(yt)) < 2 or len(yt) < 50:
                    uid_results[f"{vname}_{target_name}"] = None
                    continue
                try:
                    auc = roc_auc_score(yt, x)
                    if auc < 0.5:
                        auc = 1 - auc
                    uid_results[f"{vname}_{target_name}"] = float(auc)
                except:
                    uid_results[f"{vname}_{target_name}"] = None


        for vname in variants:
            auc_cd = uid_results.get(f"{vname}_calm_dep")
            auc_cf = uid_results.get(f"{vname}_calm_free")
            cd_str = f"{auc_cd:.4f}" if auc_cd else "---"
            cf_str = f"{auc_cf:.4f}" if auc_cf else "---"
            print(f"    {vname:<25s} {cd_str:>14s} {cf_str:>14s}")

        all_results[uid] = uid_results

    # Cross-universe summary
    print(f"\n{'=' * 70}")
    print(f"  CROSS-UNIVERSE SUMMARY (CALM-free AUC)")
    print(f"{'=' * 70}")
    variant_names = ["V0_calm_zscore", "V1_global_percentile",
                     "V2_rolling_baseline", "V3_hybrid_scr"]
    print(f"  {'Universe':<22s}", end="")
    for vn in variant_names:
        print(f"  {vn.split('_',1)[0]:>6s}", end="")
    print()


    means = {vn: [] for vn in variant_names}
    for uid in AUDIT_UNIVERSES:
        if uid not in all_results:
            continue
        print(f"  {uid:<22s}", end="")
        for vn in variant_names:
            val = all_results[uid].get(f"{vn}_calm_free")
            if val:
                print(f"  {val:6.3f}", end="")
                means[vn].append(val)
            else:
                print(f"  {'---':>6s}", end="")
        print()

    print(f"\n  {'MEAN':<22s}", end="")
    best_vn = None
    best_mean = 0
    for vn in variant_names:
        if means[vn]:
            m = np.mean(means[vn])
            print(f"  {m:6.3f}", end="")
            if m > best_mean:
                best_mean = m
                best_vn = vn
        else:
            print(f"  {'---':>6s}", end="")
    print()

    if best_vn:
        v0_mean = np.mean(means["V0_calm_zscore"]) if means["V0_calm_zscore"] else 0
        gain = best_mean - v0_mean
        print(f"\n  Best variant: {best_vn} (mean AUC={best_mean:.4f})")
        print(f"  Gain over V0 (CALM z-score): {gain:+.4f}")
        if gain > 0.02:
            print(f"  >>> FIX-12 JUSTIFIED: alternative baseline improves detection")
        elif gain > 0.005:
            print(f"  >>> MARGINAL: small improvement, may not justify complexity")
        else:
            print(f"  >>> NO IMPROVEMENT: CALM z-score is adequate")

    return all_results


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  KAPPA-SIG FIX-12: CALM Audit & Alternative Baselines")
    print("  David Ohio | Independent Researcher | April 2026")
    print("=" * 70)
    print("\n  Principle: CALM remains foundational for structural baselining.")
    print("  Question: Is CALM-based normalization appropriate for ALL layers?")
    print("  Hypothesis: Geometric normalization may need layer-conditional treatment.")

    t0 = time.time()

    # Part 1: Anatomy
    anatomy = run_anatomy()

    # Part 2: Variants
    variants = run_variants()

    total = time.time() - t0


    # Final verdict
    print(f"\n{'=' * 70}")
    print(f"  FIX-12 VERDICT")
    print(f"{'=' * 70}")

    # Count suppression cases
    severe = sum(1 for r in anatomy.values() if r["calm_verdict"] == "SEVERE_SUPPRESSION")
    moderate = sum(1 for r in anatomy.values() if r["calm_verdict"] == "MODERATE_SUPPRESSION")
    print(f"\n  CALM suppression: {severe} severe, {moderate} moderate out of {len(anatomy)}")

    print(f"\n  Three principles for revised CALM usage:")
    print(f"    1. CALM remains foundational for structural baselining (Layer 1)")
    print(f"    2. Geometric observability is NOT guaranteed by structural calmness")
    print(f"    3. CALM-based normalization is layer-conditional, not universal")

    print(f"\n  Total time: {total:.0f}s")

    # Save
    out = {"anatomy": anatomy, "variants": {
        uid: {k: float(v) if v is not None else None for k, v in vr.items()}
        for uid, vr in variants.items()
    }}
    out_path = OUT_DIR / "fix12_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Results: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
