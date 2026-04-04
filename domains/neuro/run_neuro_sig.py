#!/usr/bin/env python3
"""
Kappa-SIG NEURO: Obsessive Coherence in EEG Seizure Prediction
Inter-channel correlation eigenstructure, mapped to SIG 5-observable framework.
Data: CHB-MIT Scalp EEG Database (21 epileptic patients) + EEGMMIDB (healthy controls)
Reads pre-processed katashi_state.csv files from legacy pipeline.
David Ohio | odavidohio@gmail.com | Independent Researcher | April 2026
"""
import json, time
import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Default data paths (D:\TopoCML legacy)
CHBMIT_DIR = Path(r"D:\TopoCML\scripts\kappa_eegs\out_chbmit_v2")
EEGMMIDB_DIR = Path(r"D:\TopoCML\scripts\kappa_eegs\out_eegmmidb")
LEGACY_DIR = Path(__file__).parent.parent.parent / "legacy" / "kappa_neuro"

N_CHANNELS = 23  # CHB-MIT standard montage (23 bipolar channels)
N_CHANNELS_EEGMMIDB = 64  # EEGMMIDB uses 64-channel BCI2000 montage


def map_legacy_to_sig(row, n_channels):
    """Map legacy katashi_state columns to SIG 5-observable framework.

    Legacy columns: Oh, eta(legacy), Xi(legacy=exp(H)), entropy(=H), dominance(=DEF), phi
    SIG columns: Oh, eta_sig(=1-H/Hmax), DEF, Xi_sig(=exp(H)/n), mean_corr(=proxy from Oh)

    Note: mean_corr is not directly available in legacy data.
    We estimate it as: mean_corr_proxy = (Oh - 1) / (n - 1), clamped to [0, 1].
    This is exact for equicorrelation matrices and approximate otherwise.
    """
    Oh = row["Oh"]
    entropy_H = row.get("entropy", np.nan)
    dominance = row.get("dominance", np.nan)
    Xi_raw = row.get("Xi", np.nan)

    # SIG eta: spectral rigidity = 1 - H/H_max
    H_max = np.log2(n_channels) if n_channels > 1 else 1.0
    if np.isfinite(entropy_H) and H_max > 0:
        eta_sig = 1.0 - entropy_H / H_max
    else:
        eta_sig = np.nan

    # SIG DEF: eigenvalue dominance gap (direct from legacy "dominance")
    DEF = dominance if np.isfinite(dominance) else np.nan

    # SIG Xi: normalized effective diversity = exp(H) / n
    if np.isfinite(Xi_raw) and n_channels > 0:
        Xi_sig = Xi_raw / n_channels
    else:
        Xi_sig = np.nan

    # mean_corr proxy: for equicorrelation matrix, Oh = 1 + (n-1)*rho_bar
    # => rho_bar = (Oh - 1) / (n - 1)
    if n_channels > 1:
        mc_proxy = np.clip((Oh - 1.0) / (n_channels - 1.0), 0.0, 1.0)
    else:
        mc_proxy = 0.0

    return {
        "Oh": Oh, "eta": eta_sig, "DEF": DEF,
        "Xi": Xi_sig, "mean_corr": mc_proxy, "phi": row.get("phi", 0.0)
    }


def load_patient_katashi(patient_dir, n_channels):
    """Load katashi_state.csv and map to SIG observables."""
    kstate = patient_dir / "kappa_run" / "katashi_state.csv"
    if not kstate.exists():
        return None
    df = pd.read_csv(kstate)
    if len(df) < 10:
        return None
    sig_rows = [map_legacy_to_sig(row, n_channels) for _, row in df.iterrows()]
    return pd.DataFrame(sig_rows)


def compute_group_stats(sig_df):
    """Compute mean SIG observables for a group/condition."""
    metrics = ["Oh", "eta", "DEF", "Xi", "mean_corr", "phi"]
    return {m: float(sig_df[m].mean()) for m in metrics if m in sig_df.columns}


def direction_test(stressed, nominal):
    """Test 5 predicted directional changes (SIG protocol)."""
    dirs = {
        "Oh": stressed["Oh"] > nominal["Oh"],
        "eta": stressed["eta"] > nominal["eta"],
        "DEF": stressed["DEF"] > nominal["DEF"],
        "Xi": stressed["Xi"] < nominal["Xi"],
        "mean_corr": stressed["mean_corr"] > nominal["mean_corr"],
    }
    return dirs


def oc_score(s):
    """Obsessive Coherence Index."""
    return (s["Oh"] / 23 + s["DEF"]) / 2 - (s["Xi"] + (1 - s["mean_corr"])) / 2


def main():
    t0 = time.time()
    print("=" * 70)
    print("  KAPPA-SIG NEURO: Obsessive Coherence in EEG")
    print("  David Ohio | Independent Researcher | April 2026")
    print("  SIG 5-observable framework on legacy katashi_state.csv")
    print("=" * 70)

    # ================================================================
    # 1. LOAD EPILEPTIC PATIENTS (CHB-MIT)
    # ================================================================
    epileptic_all = []
    chb_patients = sorted(CHBMIT_DIR.glob("chb*")) if CHBMIT_DIR.exists() else []
    print(f"\n  CHB-MIT patients found: {len(chb_patients)}")

    for pdir in chb_patients:
        sig = load_patient_katashi(pdir, N_CHANNELS)
        if sig is not None:
            stats = compute_group_stats(sig)
            stats["patient"] = pdir.name
            stats["n_windows"] = len(sig)
            epileptic_all.append(stats)
            print(f"    {pdir.name}: {len(sig)} windows, Oh={stats['Oh']:.4f}, "
                  f"eta={stats['eta']:.4f}, DEF={stats['DEF']:.4f}")

    # ================================================================
    # 2. LOAD HEALTHY SUBJECTS (EEGMMIDB, sample)
    # ================================================================
    healthy_all = []
    eeg_subjects = sorted(EEGMMIDB_DIR.glob("S*")) if EEGMMIDB_DIR.exists() else []
    sample_n = min(30, len(eeg_subjects))  # sample for speed
    print(f"\n  EEGMMIDB subjects found: {len(eeg_subjects)} (sampling {sample_n})")


    for sdir in eeg_subjects[:sample_n]:
        sig = load_patient_katashi(sdir, N_CHANNELS_EEGMMIDB)
        if sig is not None:
            stats = compute_group_stats(sig)
            stats["subject"] = sdir.name
            stats["n_windows"] = len(sig)
            healthy_all.append(stats)

    print(f"    Loaded: {len(healthy_all)} healthy subjects")

    # ================================================================
    # 3. ICTAL ANALYSIS (pre-ictal vs inter-ictal from legacy)
    # ================================================================
    ictal_path = LEGACY_DIR / "ictal_analysis.csv"
    ictal_results = {}
    if ictal_path.exists():
        ictal_df = pd.read_csv(ictal_path)
        print(f"\n  Ictal analysis: {len(ictal_df)} rows")
        for pid in ictal_df["patient_id"].unique():
            pdata = ictal_df[ictal_df["patient_id"] == pid]
            inter = pdata[pdata["state"] == "inter-ictal"].iloc[0] if len(pdata[pdata["state"] == "inter-ictal"]) > 0 else None
            pre = pdata[pdata["state"] == "pre-ictal"].iloc[0] if len(pdata[pdata["state"] == "pre-ictal"]) > 0 else None
            ictal = pdata[pdata["state"] == "ictal"].iloc[0] if len(pdata[pdata["state"] == "ictal"]) > 0 else None
            if inter is not None and pre is not None:
                ictal_results[pid] = {
                    "inter_Oh": float(inter["Oh_mean"]),
                    "pre_Oh": float(pre["Oh_mean"]),
                    "ictal_Oh": float(ictal["Oh_mean"]) if ictal is not None else None,
                    "inter_phi": float(inter["phi_mean"]),
                    "pre_phi": float(pre["phi_mean"]),
                    "pre_gt_inter": float(pre["Oh_mean"]) > float(inter["Oh_mean"]),
                }
                st = "+" if ictal_results[pid]["pre_gt_inter"] else "-"
                print(f"    {pid}: inter Oh={inter['Oh_mean']:.4f} -> pre Oh={pre['Oh_mean']:.4f} [{st}]")
        n_pre_gt = sum(1 for v in ictal_results.values() if v["pre_gt_inter"])
        print(f"    Pre-ictal Oh > Inter-ictal: {n_pre_gt}/{len(ictal_results)}")
    else:
        print("\n  Ictal analysis: not available (legacy file missing)")


    # ================================================================
    # 4. GROUP COMPARISON: Epileptic vs Healthy (SIG Direction Test)
    # ================================================================
    print("\n  " + "=" * 66)
    print("  OBSESSIVE COHERENCE: Epileptic vs Healthy")
    print("  " + "=" * 66)

    if epileptic_all and healthy_all:
        epi_df = pd.DataFrame(epileptic_all)
        hlt_df = pd.DataFrame(healthy_all)

        metrics = ["Oh", "eta", "DEF", "Xi", "mean_corr", "phi"]
        epi_mean = {m: float(epi_df[m].mean()) for m in metrics}
        hlt_mean = {m: float(hlt_df[m].mean()) for m in metrics}

        print(f"\n    {'Metric':<14s} {'Epileptic':>12s} {'Healthy':>12s} {'Delta':>10s}   Dir")
        print(f"    {'-'*58}")
        for m in metrics:
            d = epi_mean[m] - hlt_mean[m]
            print(f"    {m:<14s} {epi_mean[m]:12.6f} {hlt_mean[m]:12.6f} {d:+10.6f}")

        dirs = direction_test(epi_mean, hlt_mean)
        n_coh = sum(dirs.values())
        status = "CONFIRMED" if n_coh >= 4 else "PARTIAL"
        print(f"\n    Coherent directions: {n_coh}/5 -- {status}")

        oc_epi = oc_score(epi_mean)
        oc_hlt = oc_score(hlt_mean)
        print(f"    OC: epileptic={oc_epi:+.4f}  healthy={oc_hlt:+.4f}  delta={oc_epi-oc_hlt:+.4f}")
    else:
        epi_mean, hlt_mean, dirs, n_coh, status = {}, {}, {}, 0, "NO_DATA"
        oc_epi, oc_hlt = 0.0, 0.0
        if not epileptic_all:
            print("    [!] No epileptic data loaded (D: drive not available?)")
        if not healthy_all:
            print("    [!] No healthy data loaded (D: drive not available?)")


    # ================================================================
    # 5. INTRA-PATIENT ANALYSIS (eliminates channel-count confound)
    # ================================================================
    # For each epileptic patient, split time series into high-Oh (top 25%)
    # vs low-Oh (bottom 25%) windows. Same patient, same montage, same n.
    print(f"\n  " + "=" * 66)
    print("  INTRA-PATIENT DIRECTION TEST (same montage, no confound)")
    print("  " + "=" * 66)

    intra_results = []
    for pdir in chb_patients:
        sig = load_patient_katashi(pdir, N_CHANNELS)
        if sig is None or len(sig) < 40:
            continue
        q25 = sig["Oh"].quantile(0.25)
        q75 = sig["Oh"].quantile(0.75)
        low = sig[sig["Oh"] <= q25]
        high = sig[sig["Oh"] >= q75]
        if len(low) < 10 or len(high) < 10:
            continue
        low_m = compute_group_stats(low)
        high_m = compute_group_stats(high)
        dirs_p = direction_test(high_m, low_m)
        nc = sum(dirs_p.values())
        oc_h = oc_score(high_m)
        oc_l = oc_score(low_m)
        intra_results.append({
            "patient": pdir.name, "n_coherent": nc, "confirmed": nc >= 4,
            "oc_high": round(oc_h, 6), "oc_low": round(oc_l, 6),
            "oc_delta": round(oc_h - oc_l, 6),
            "high_Oh": round(high_m["Oh"], 4), "low_Oh": round(low_m["Oh"], 4),
        })
        st = "OK" if nc >= 4 else f"{nc}/5"
        print(f"    {pdir.name}: {nc}/5 {'CONFIRMED' if nc>=4 else 'PARTIAL'} "
              f"  Oh: {low_m['Oh']:.4f} -> {high_m['Oh']:.4f}  "
              f"OC delta={oc_h-oc_l:+.4f}")

    n_intra_ok = sum(1 for r in intra_results if r["confirmed"])
    print(f"\n    Intra-patient confirmed: {n_intra_ok}/{len(intra_results)}")

    # ================================================================
    # 6. CONSOLIDATED SUMMARY ANALYSIS (legacy fallback)
    # ================================================================
    consol_path = LEGACY_DIR / "consolidated_summary.csv"
    consol_results = {}
    if consol_path.exists():
        cdf = pd.read_csv(consol_path)
        epi_c = cdf[cdf["group"] == "epileptic"]
        hlt_c = cdf[cdf["group"] == "healthy"]
        print(f"\n  Consolidated summary: {len(epi_c)} epileptic, {len(hlt_c)} healthy")
        for col in ["Oh_mean", "Oh_max", "Oh_p95", "phi_mean", "phi_max",
                     "pct_Oh_gt_threshold", "max_consecutive_Oh_gt_threshold"]:
            if col in cdf.columns:
                e_val = float(epi_c[col].mean())
                h_val = float(hlt_c[col].mean())
                consol_results[col] = {"epileptic": round(e_val, 6),
                                       "healthy": round(h_val, 6),
                                       "delta": round(e_val - h_val, 6)}
                d_sign = "+" if e_val > h_val else "-"
                print(f"    {col:<38s}: epi={e_val:.4f}  hlt={h_val:.4f}  [{d_sign}]")

    # ================================================================
    # 7. SAVE RESULTS
    # ================================================================
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "domain": "NEURO",
        "framework": "SIG v5.8c (mapped from legacy katashi_state)",
        "datasets": {
            "epileptic": {"source": "CHB-MIT Scalp EEG", "n_patients": len(epileptic_all),
                          "n_channels": N_CHANNELS},
            "healthy": {"source": "EEGMMIDB", "n_subjects": len(healthy_all),
                        "n_channels": N_CHANNELS_EEGMMIDB},
        },

        "sig_mapping": {
            "Oh": "direct",
            "eta": "1 - entropy / log2(n_channels)",
            "DEF": "direct from legacy dominance",
            "Xi": "legacy Xi / n_channels",
            "mean_corr": "proxy: (Oh - 1) / (n - 1), exact for equicorrelation",
        },
        "group_comparison": {
            "epileptic_mean": {k: round(v, 6) for k, v in epi_mean.items()} if epi_mean else None,
            "healthy_mean": {k: round(v, 6) for k, v in hlt_mean.items()} if hlt_mean else None,
            "direction_test": {k: bool(v) for k, v in dirs.items()} if dirs else None,
            "n_coherent": n_coh,
            "confirmed": n_coh >= 4,
            "oc_epileptic": round(oc_epi, 6),
            "oc_healthy": round(oc_hlt, 6),
            "oc_delta": round(oc_epi - oc_hlt, 6),
        },
        "ictal_analysis": ictal_results if ictal_results else None,
        "intra_patient_analysis": {
            "method": "Top 25% Oh vs Bottom 25% Oh within same patient (same montage)",
            "n_patients_tested": len(intra_results),
            "n_confirmed": n_intra_ok,
            "patients": intra_results,
        },
        "consolidated_summary": consol_results if consol_results else None,
        "per_patient_epileptic": epileptic_all[:5] if epileptic_all else None,  # sample
        "notes": [
            "GROUP COMPARISON (epileptic vs healthy) is CONFOUNDED by channel count (23 vs 64)",
            "INTRA-PATIENT ANALYSIS is the valid test: same patient, same montage, no confound",
            "ICTAL ANALYSIS (pre-ictal vs inter-ictal) is valid: within-patient comparison",
            "mean_corr is a proxy (equicorrelation assumption), not measured directly",
            "CHB-MIT uses 23-channel bipolar montage; EEGMMIDB uses 64-channel BCI2000",
            "This domain is DEFERRED to dedicated Kappa-NEURO paper for clinical validation",
        ],
    }

    out_file = OUT_DIR / "neuro_obsessive_coherence.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Results: {out_file}")
    print(f"  Time: {time.time() - t0:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
