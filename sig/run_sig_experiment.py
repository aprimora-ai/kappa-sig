#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Kappa-SIG Phase 2: Experiment 4a — Window Sweep + Wheeler Test
================================================================
For each SIG method (S1, S2, S3), measure detection performance
at varying window sizes W = {1, 3, 5, 10, 15, 20, 30, 45, 60}.

Uses pre-extracted data from extract_sig_data.py (Phase 1).

David Ohio | odavidohio@gmail.com | Independent Researcher
April 2026
"""
import sys, json, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from collections import Counter

warnings.filterwarnings("ignore")

SIG_DIR = Path(r"C:\Users\ohiod\Projects\Sentinel\data\sig")
OUT_DIR = SIG_DIR / "experiments"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS = [1, 3, 5, 10, 15, 20, 30, 45, 60]


# ══════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════

def load_sig_data(uid):
    """Load pre-extracted SIG data for a universe."""
    path = SIG_DIR / f"{uid}_sig.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {
        "dates": data["dates"],
        "labels": data["labels"],
        "scr": data["scr"],
        "sgr": data["sgr"],
        "d_mp_kl": data["d_mp_kl"] if "d_mp_kl" in data else np.zeros_like(data["scr"]),
        "mp_excess": data["mp_excess"],
        "lz_norm": data["lz_norm"],
        "cdr": data["cdr"] if "cdr" in data else np.zeros_like(data["scr"]),
        "pe1": data["pe1"],
        "tcs": data["tcs"] if "tcs" in data else np.zeros_like(data["scr"]),
        "n_h1": data["n_h1"],
        "eigenvalues": data["eigenvalues"],
        "n_assets": int(data["n_assets"]),
        # Baselines
        "mean_corr": data["mean_corr"] if "mean_corr" in data else np.zeros_like(data["scr"]),
        "lam1": data["lam1"] if "lam1" in data else np.zeros_like(data["scr"]),
        "rvol": data["rvol"] if "rvol" in data else np.zeros_like(data["scr"]),
    }

def make_binary_target(labels):
    """Convert labels to binary: 1 = CRYSTALLIZING or CRYSTALLIZED, 0 = NOMINAL.
    NOTE: These labels depend on Theta_A which uses CALM normalization."""
    return np.array([1 if l in ("CRYSTALLIZING", "CRYSTALLIZED") else 0 for l in labels])

def make_damaged_target(labels):
    """Binary: 1 = DAMAGED, 0 = everything else."""
    return np.array([1 if l == "DAMAGED" else 0 for l in labels])

def make_calm_free_target(labels, H=90):
    """CALM-free forward-looking target: 1 if DAMAGED occurs within next H steps.
    Uses only C_norm-derived labels — no Theta_A, no CALM normalization.
    This is the same logic as Experiment 3 target."""
    n = len(labels)
    y = np.zeros(n, dtype=int)
    for t in range(n):
        future = labels[t+1:t+1+H]
        if any(l == "DAMAGED" for l in future):
            y[t] = 1
    return y


# ══════════════════════════════════════════════════════════════════
# WINDOWED METRICS
# ══════════════════════════════════════════════════════════════════

def rolling_mean(x, W):
    """Rolling mean over W steps. First W-1 values are NaN."""
    out = np.full_like(x, np.nan, dtype=float)
    if W <= 1:
        return x.astype(float)
    cs = np.cumsum(x)
    out[W-1:] = (cs[W-1:] - np.concatenate([[0], cs[:-W]])) / W
    return out

def rolling_slope(x, W):
    """Rolling linear slope over W steps (trend indicator)."""
    out = np.full_like(x, np.nan, dtype=float)
    if W < 3:
        return out
    t_vec = np.arange(W, dtype=float)
    t_mean = t_vec.mean()
    t_var = np.sum((t_vec - t_mean)**2)
    for i in range(W-1, len(x)):
        window = x[i-W+1:i+1].astype(float)
        if np.any(np.isnan(window)):
            continue
        out[i] = np.sum((t_vec - t_mean) * (window - window.mean())) / max(t_var, 1e-12)
    return out


# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 4a: WINDOW SWEEP PER METHOD
# ══════════════════════════════════════════════════════════════════

def evaluate_method_at_window(metric_values, labels_binary, W):
    """Compute AUC for a single metric at window size W."""
    # Rolling mean of metric
    smoothed = rolling_mean(metric_values, W)
    # Mask valid (non-NaN) entries
    valid = np.isfinite(smoothed) & np.isfinite(labels_binary.astype(float))
    if valid.sum() < 20:
        return None
    y = labels_binary[valid]
    x = smoothed[valid]
    if len(set(y)) < 2:
        return None
    try:
        auc = roc_auc_score(y, x)
        # If AUC < 0.5, signal is inversely correlated (flip)
        if auc < 0.5:
            auc = 1 - auc
        return float(auc)
    except:
        return None

def find_first_detection(metric_values, threshold, labels, W):
    """Find first step where windowed metric exceeds threshold."""
    smoothed = rolling_mean(metric_values, W)
    for i in range(len(smoothed)):
        if np.isfinite(smoothed[i]) and smoothed[i] > threshold:
            return i
    return None

def find_t_S(labels):
    """Find first DAMAGED step."""
    for i, l in enumerate(labels):
        if l == "DAMAGED":
            return i
    return None


def run_experiment_4a():
    """Experiment 4a: Window sweep for all methods."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 4a: WINDOW SWEEP")
    print("  Testing detection at W = " + str(WINDOWS))
    print("=" * 70)

    # Load all universes
    universes = {}
    for f in sorted(SIG_DIR.glob("*_sig.npz")):
        uid = f.stem.replace("_sig", "")
        data = load_sig_data(uid)
        if data is not None:
            universes[uid] = data
    print(f"\n  Loaded {len(universes)} universes")

    # Define metrics to test
    # S1: SCR (higher = more crystallized), SGR, MP excess
    # S2: 1-LZ_norm (higher = more crystallized, since LZ_norm drops)
    # S3: PE1 inverted (lower entropy = more crystallized)
    metrics_def = {
        "S1_scr":       ("scr", False),
        "S1_sgr":       ("sgr", False),
        "S1_d_mp_kl":   ("d_mp_kl", False),
        "S1_mp_excess": ("mp_excess", False),
        "S2_lz_inv":    ("lz_norm", True),
        "S2_cdr":       ("cdr", False),       # CDR positive = crystallizing
        "S3_pe1_inv":   ("pe1", True),
        "S3_tcs":       ("tcs", False),       # TCS higher = further from CALM
        # Baselines (should NOT beat SIG methods)
        "BL_mean_corr": ("mean_corr", False),
        "BL_lam1":      ("lam1", False),
        "BL_rvol":      ("rvol", False),      # Higher vol might = more crystallized
    }


    results = {}
    for mname, (field, invert) in metrics_def.items():
        results[mname] = {}
        for W in WINDOWS:
            aucs = []
            for uid, data in universes.items():
                raw = data[field].astype(float)
                if invert:
                    raw = -raw  # Flip so higher = crystallized
                y = make_binary_target(data["labels"])
                auc = evaluate_method_at_window(raw, y, W)
                if auc is not None:
                    aucs.append(auc)
            if aucs:
                results[mname][W] = {
                    "auc_mean": float(np.mean(aucs)),
                    "auc_std": float(np.std(aucs)),
                    "n_universes": len(aucs),
                }

    # Print results table
    print(f"\n  {'Method':<22s}", end="")
    for W in WINDOWS:
        print(f"  W={W:>2d}", end="")
    print()
    print(f"  {'-'*22}", end="")
    for _ in WINDOWS:
        print(f"  {'----':>5s}", end="")
    print()


    best_overall = {"method": None, "W": None, "auc": 0}
    for mname in metrics_def:
        print(f"  {mname:<22s}", end="")
        for W in WINDOWS:
            r = results[mname].get(W, {})
            auc = r.get("auc_mean", 0)
            if auc > 0:
                marker = "*" if auc >= 0.70 else " "
                print(f"  {auc:.2f}{marker}", end="")
                if auc > best_overall["auc"]:
                    best_overall = {"method": mname, "W": W, "auc": auc}
            else:
                print(f"  {'---':>5s}", end="")
        print()

    print(f"\n  Best: {best_overall['method']} at W={best_overall['W']} "
          f"(AUC={best_overall['auc']:.3f})")

    # Find W_min for each method (first W where AUC > 0.70)
    print(f"\n  W_min (first W with AUC > 0.70):")
    for mname in metrics_def:
        w_min = None
        for W in WINDOWS:
            r = results[mname].get(W, {})
            if r.get("auc_mean", 0) >= 0.70:
                w_min = W
                break
        status = f"W={w_min}" if w_min else "NOT REACHED"
        print(f"    {mname:<22s} {status}")

    return results


# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 4d: WHEELER TEST (minimal window)
# ══════════════════════════════════════════════════════════════════

def run_experiment_4d():
    """Wheeler test: can crystallization be detected from 1-10 steps?"""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 4d: WHEELER TEST (Extreme Minimal Window)")
    print("  Testing the 'same electron' hypothesis")
    print("=" * 70)

    WHEELER_WINDOWS = [1, 2, 3, 5, 7, 10]
    N_PERM = 500  # Permutations for significance

    universes = {}
    for f in sorted(SIG_DIR.glob("*_sig.npz")):
        uid = f.stem.replace("_sig", "")
        data = load_sig_data(uid)
        if data is not None:
            universes[uid] = data
    print(f"\n  Loaded {len(universes)} universes")

    # Instantaneous metrics: S1 spectral + S2 + S3 topological + baselines
    metrics = {
        "S1_scr":       ("scr", False),
        "S1_sgr":       ("sgr", False),
        "S1_d_mp_kl":   ("d_mp_kl", False),
        "S1_mp_excess": ("mp_excess", False),
        "S2_cdr":       ("cdr", False),
        "S3_pe1_inv":   ("pe1", True),
        "S3_tcs":       ("tcs", False),
        "BL_mean_corr": ("mean_corr", False),
        "BL_lam1":      ("lam1", False),
    }


    rng = np.random.RandomState(42)
    wheeler_results = {}

    for mname, (field, invert) in metrics.items():
        wheeler_results[mname] = {}
        for W in WHEELER_WINDOWS:
            # Pool all universes
            all_x, all_y = [], []
            for uid, data in universes.items():
                raw = data[field].astype(float)
                if invert:
                    raw = -raw
                y = make_binary_target(data["labels"])
                smoothed = rolling_mean(raw, W)
                valid = np.isfinite(smoothed)
                all_x.extend(smoothed[valid].tolist())
                all_y.extend(y[valid].tolist())

            all_x = np.array(all_x)
            all_y = np.array(all_y)

            if len(set(all_y)) < 2 or len(all_y) < 50:
                wheeler_results[mname][W] = {"auc_real": 0, "p_value": 1.0}
                continue

            auc_real = roc_auc_score(all_y, all_x)
            if auc_real < 0.5:
                all_x = -all_x
                auc_real = 1 - auc_real


            # Permutation test
            auc_perm = []
            for _ in range(N_PERM):
                y_shuf = rng.permutation(all_y)
                try:
                    a = roc_auc_score(y_shuf, all_x)
                    auc_perm.append(max(a, 1-a))
                except:
                    pass
            auc_perm = np.array(auc_perm)
            p_value = float(np.mean(auc_perm >= auc_real)) if len(auc_perm) > 0 else 1.0

            wheeler_results[mname][W] = {
                "auc_real": float(auc_real),
                "auc_perm_mean": float(auc_perm.mean()) if len(auc_perm) > 0 else 0,
                "auc_perm_p99": float(np.percentile(auc_perm, 99)) if len(auc_perm) > 0 else 0,
                "p_value": p_value,
                "n_obs": len(all_y),
                "significant": p_value < 0.01,
            }

    # Print Wheeler results
    print(f"\n  {'Method':<18s}", end="")
    for W in WHEELER_WINDOWS:
        print(f"  {'W='+str(W):>8s}", end="")
    print()
    print(f"  {'-'*18}", end="")
    for _ in WHEELER_WINDOWS:
        print(f"  {'--------':>8s}", end="")
    print()


    w_critical = {}
    for mname in metrics:
        print(f"  {mname:<18s}", end="")
        found_critical = None
        for W in WHEELER_WINDOWS:
            r = wheeler_results[mname].get(W, {})
            auc = r.get("auc_real", 0)
            sig = r.get("significant", False)
            marker = "**" if sig else "  "
            if auc > 0:
                print(f"  {auc:.3f}{marker}", end="")
                if sig and found_critical is None:
                    found_critical = W
            else:
                print(f"  {'---':>8s}", end="")
        print()
        w_critical[mname] = found_critical

    print(f"\n  W_critical (smallest W with p < 0.01):")
    for mname, wc in w_critical.items():
        status = f"W = {wc}  *** WHEELER CONFIRMED ***" if wc else "NOT SIGNIFICANT"
        print(f"    {mname:<18s} {status}")

    wheeler_confirmed = any(wc is not None and wc <= 5 for wc in w_critical.values())
    if wheeler_confirmed:
        print(f"\n  >>> WHEELER HYPOTHESIS SUPPORTED: crystallization detectable from <= 5 steps")
    else:
        any_sig = any(wc is not None for wc in w_critical.values())
        if any_sig:
            best_wc = min(wc for wc in w_critical.values() if wc is not None)
            print(f"\n  >>> Partial: detection above chance at W={best_wc}, but > 5 steps")
        else:
            print(f"\n  >>> WHEELER HYPOTHESIS NOT SUPPORTED at current sample size")

    return wheeler_results, w_critical


# ══════════════════════════════════════════════════════════════════
# EXPERIMENT: LEFT_CENSORED RECOVERY
# ══════════════════════════════════════════════════════════════════

# These universes had t_S < 60 steps (invisible to RQA)
LEFT_CENSORED = {
    "europe":              {"t_S": 16},
    "x_energy_geopolitics": {"t_S": 46},
    "us_sectors":          {"t_S": 46},
    "x_us_systemic":       {"t_S": 21},
    "x_brazil_vuln":       {"t_S": 21},
}

def run_left_censored_recovery():
    """Test if SIG methods detect crystallization in LEFT_CENSORED cases."""
    print("\n" + "=" * 70)
    print("  LEFT_CENSORED RECOVERY TEST")
    print("  Can SIG detect crystallization where RQA could not (t_S < 60)?")
    print("=" * 70)

    metrics_to_test = {
        "S1_scr":   ("scr", False, 0.50),    # threshold to be explored
        "S1_sgr":   ("sgr", False, 0.60),
        "S3_pe1":   ("pe1", True,  -0.50),   # inverted: lower = crystallized
    }


    def first_crossing(metric, threshold, W=1, persistence=3):
        """Forward-only: first step where windowed metric exceeds threshold
        for 'persistence' consecutive steps."""
        sm = rolling_mean(metric, W)
        run = 0
        for i, val in enumerate(sm):
            if np.isfinite(val) and val > threshold:
                run += 1
                if run >= persistence:
                    return i - persistence + 1
            else:
                run = 0
        return None

    # Calibrate thresholds from non-LEFT_CENSORED universes (forward-safe)
    print(f"\n  Calibrating thresholds from non-censored universes...")
    all_cal = {}
    for uid, info in LEFT_CENSORED.items():
        all_cal[uid] = True  # mark censored
    cal_vals = {mname: [] for mname in metrics_to_test}
    for f in sorted(SIG_DIR.glob("*_sig.npz")):
        uid = f.stem.replace("_sig", "")
        if uid in LEFT_CENSORED:
            continue
        data = load_sig_data(uid)
        if data is None:
            continue
        labels = data["labels"]
        for mname, (field, invert, _) in metrics_to_test.items():
            raw = data[field].astype(float)
            if invert:
                raw = -raw
            # Use values during CRYSTALLIZING/CRYSTALLIZED periods as positive calibration
            for i, l in enumerate(labels):
                if l in ("CRYSTALLIZING", "CRYSTALLIZED") and np.isfinite(raw[i]):
                    cal_vals[mname].append(raw[i])
    # Threshold = 50th percentile of crystallized values (conservative)
    cal_thresholds = {}
    for mname, vals in cal_vals.items():
        if vals:
            cal_thresholds[mname] = float(np.percentile(vals, 50))
        else:
            cal_thresholds[mname] = 0.0
    for mname, thr in cal_thresholds.items():
        print(f"    {mname}: threshold = {thr:+.3f} (p50 of crystallized)")

    recoveries = {}
    for uid, info in LEFT_CENSORED.items():
        t_S = info["t_S"]
        data = load_sig_data(uid)
        if data is None:
            print(f"\n  [{uid}] No SIG data available")
            continue

        print(f"\n  [{uid}] t_S={t_S} (RQA blind before step 60)")
        recoveries[uid] = {}

        for mname, (field, invert, _) in metrics_to_test.items():
            raw = data[field].astype(float)
            if invert:
                raw = -raw
            thr = cal_thresholds.get(mname, 0.0)

            for W in [1, 5, 10]:
                t_det = first_crossing(raw, thr, W=W, persistence=3)
                recovered = (t_det is not None) and (t_det < t_S)
                lead = t_S - t_det if recovered else None

                recoveries[uid][f"{mname}_W{W}"] = {
                    "t_detection": t_det,
                    "t_S": t_S,
                    "recovered": recovered,
                    "lead_steps": lead,
                    "threshold": thr,
                }

                marker = f"RECOVERED (lead={lead})" if recovered else "missed"
                t_str = f"t_det={t_det}" if t_det is not None else "t_det=None"
                print(f"    {mname} W={W:>2d}: {t_str} [{marker}]")

    # Summary
    n_recovered = 0
    for uid, methods in recoveries.items():
        if any(v.get("recovered", False) for v in methods.values()):
            n_recovered += 1

    print(f"\n  RECOVERY SUMMARY: {n_recovered}/{len(LEFT_CENSORED)} LEFT_CENSORED cases recovered")
    print(f"  (forward-only crossing with persistence=3, calibrated thresholds)")
    return recoveries


# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 4c: ENSEMBLE
# ══════════════════════════════════════════════════════════════════

def run_experiment_4c():
    """Build and evaluate ensemble of SIG methods."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 4c: ENSEMBLE CONSTRUCTION")
    print("=" * 70)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import calibration_curve


    # Load all universes
    universes = {}
    for f in sorted(SIG_DIR.glob("*_sig.npz")):
        uid = f.stem.replace("_sig", "")
        data = load_sig_data(uid)
        if data is not None:
            universes[uid] = data
    print(f"\n  Loaded {len(universes)} universes")

    # Build feature matrix for best window (W=20)
    W = 20
    all_features = []
    all_targets = []
    all_uids = []

    feature_names = ["scr", "sgr", "d_mp_kl", "mp_excess", "lz_inv", "cdr", "pe1_inv", "tcs",
                      "bl_mean_corr", "bl_lam1", "bl_rvol"]


    for uid, data in universes.items():
        n = len(data["scr"])
        scr_w   = rolling_mean(data["scr"].astype(float), W)
        sgr_w   = rolling_mean(data["sgr"].astype(float), W)
        dmpkl_w = rolling_mean(data["d_mp_kl"].astype(float), W)
        mp_w    = rolling_mean(data["mp_excess"].astype(float), W)
        lz_w    = rolling_mean(-data["lz_norm"].astype(float), W)
        cdr_w   = rolling_mean(data["cdr"].astype(float), W)
        pe_w    = rolling_mean(-data["pe1"].astype(float), W)
        tcs_w   = rolling_mean(data["tcs"].astype(float), W)
        mc_w    = rolling_mean(data["mean_corr"].astype(float), W)
        l1_w    = rolling_mean(data["lam1"].astype(float), W)
        rv_w    = rolling_mean(data["rvol"].astype(float), W)
        y = make_binary_target(data["labels"])

        for t in range(n):
            row = [scr_w[t], sgr_w[t], dmpkl_w[t], mp_w[t], lz_w[t], cdr_w[t],
                   pe_w[t], tcs_w[t], mc_w[t], l1_w[t], rv_w[t]]
            if any(np.isnan(v) for v in row):
                continue
            all_features.append(row)
            all_targets.append(y[t])
            all_uids.append(uid)

    X = np.array(all_features)
    y = np.array(all_targets)
    uids_arr = np.array(all_uids)
    print(f"  Feature matrix: {X.shape}, positive={y.sum()}, prevalence={y.mean():.3f}")


    if len(set(y)) < 2:
        print("  SKIP: no positive examples")
        return {}

    from sklearn.pipeline import Pipeline

    # Full fit (scaler + classifier as pipeline)
    pipe_full = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, C=1.0))
    ])
    pipe_full.fit(X, y)
    y_prob = pipe_full.predict_proba(X)[:, 1]
    auc_full = roc_auc_score(y, y_prob)
    coef = pipe_full.named_steps["clf"].coef_[0]

    print(f"\n  FULL FIT (W={W}):")
    print(f"    AUC = {auc_full:.4f}")
    print(f"    Coefficients:")
    for name, c in zip(feature_names, coef):
        print(f"      {name:<12s} {c:+.4f}")


    # LOUO-CV
    unique_uids = sorted(set(uids_arr))
    cv_aucs = []
    for test_uid in unique_uids:
        mask_train = uids_arr != test_uid
        mask_test = uids_arr == test_uid
        X_tr, y_tr = X[mask_train], y[mask_train]
        X_te, y_te = X[mask_test], y[mask_test]
        if len(set(y_tr)) < 2 or len(set(y_te)) < 2:
            continue
        pipe_cv = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, C=1.0))
        ])
        pipe_cv.fit(X_tr, y_tr)
        yp = pipe_cv.predict_proba(X_te)[:, 1]
        try:
            cv_aucs.append(roc_auc_score(y_te, yp))
        except:
            pass

    if cv_aucs:
        print(f"\n  LOUO-CV ({len(cv_aucs)} folds):")
        print(f"    AUC = {np.mean(cv_aucs):.4f} +/- {np.std(cv_aucs):.4f}")


    # Ablation: remove each feature
    print(f"\n  ABLATION:")
    print(f"    {'Removed':<15s} {'AUC':>8s} {'dAUC':>8s}")
    print(f"    {'-'*35}")
    ablation = {}
    for i, name in enumerate(feature_names):
        X_abl = np.delete(X, i, axis=1)
        pipe_abl = Pipeline([("scaler", StandardScaler()),
                             ("clf", LogisticRegression(max_iter=5000, C=1.0))])
        pipe_abl.fit(X_abl, y)
        auc_abl = roc_auc_score(y, pipe_abl.predict_proba(X_abl)[:, 1])
        d = auc_full - auc_abl
        ablation[name] = {"auc": auc_abl, "d_auc": d}
        crit = "CRITICAL" if d > 0.02 else "minor" if d > 0.005 else "redundant"
        print(f"    {name:<15s} {auc_abl:8.4f} {d:+8.4f}  [{crit}]")

    # Permutation test
    rng = np.random.RandomState(42)
    perm_aucs = []
    for _ in range(500):
        y_shuf = rng.permutation(y)
        if len(set(y_shuf)) < 2:
            continue
        pipe_p = Pipeline([("scaler", StandardScaler()),
                           ("clf", LogisticRegression(max_iter=5000, C=1.0))])
        pipe_p.fit(X, y_shuf)
        try:
            perm_aucs.append(roc_auc_score(y_shuf, pipe_p.predict_proba(X)[:, 1]))
        except:
            pass


    perm_aucs = np.array(perm_aucs)
    p_val = float(np.mean(perm_aucs >= auc_full))
    print(f"\n  PERMUTATION TEST (500 shuffles):")
    print(f"    Real AUC:     {auc_full:.4f}")
    print(f"    Shuffled max: {perm_aucs.max():.4f}" if len(perm_aucs) > 0 else "")
    print(f"    p-value:      {p_val:.4f}")
    print(f"    Verdict:      {'PASS' if p_val < 0.01 else 'FAIL'}")

    return {
        "auc_full": auc_full,
        "cv_auc_mean": float(np.mean(cv_aucs)) if cv_aucs else None,
        "cv_auc_std": float(np.std(cv_aucs)) if cv_aucs else None,
        "ablation": ablation,
        "perm_p_value": p_val,
        "coefficients": dict(zip(feature_names, coef.tolist())),
        "W": W,
    }


# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 4b: CROSS-UNIVERSE TRANSFERABILITY
# ══════════════════════════════════════════════════════════════════

def run_experiment_4b():
    """Test if crystallization signatures transfer between universes (LOUO per method)."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 4b: CROSS-UNIVERSE TRANSFERABILITY")
    print("  'Same electron' hypothesis — do signatures generalize?")
    print("=" * 70)

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    universes = {}
    for f in sorted(SIG_DIR.glob("*_sig.npz")):
        uid = f.stem.replace("_sig", "")
        data = load_sig_data(uid)
        if data is not None:
            universes[uid] = data
    print(f"\n  Loaded {len(universes)} universes")

    # Test each method individually via LOUO
    W = 20
    methods = {
        "S1_scr": ("scr", False),
        "S1_d_mp_kl": ("d_mp_kl", False),
        "S2_lz_inv": ("lz_norm", True),
        "S2_cdr": ("cdr", False),
        "S3_pe1_inv": ("pe1", True),
        "S3_tcs": ("tcs", False),
        "BL_mean_corr": ("mean_corr", False),
        "BL_lam1": ("lam1", False),
    }

    results_4b = {}
    for mname, (field, invert) in methods.items():
        # Build per-universe features
        per_uid = {}
        for uid, data in universes.items():
            raw = data[field].astype(float)
            if invert:
                raw = -raw
            smoothed = rolling_mean(raw, W)
            y = make_binary_target(data["labels"])
            valid = np.isfinite(smoothed)
            per_uid[uid] = {"x": smoothed[valid], "y": y[valid]}

        # LOUO: train on all except one, test on held-out
        louo_aucs = []
        for test_uid in per_uid:
            x_te = per_uid[test_uid]["x"]
            y_te = per_uid[test_uid]["y"]
            if len(set(y_te)) < 2 or len(y_te) < 20:
                continue
            x_tr = np.concatenate([per_uid[u]["x"] for u in per_uid if u != test_uid])
            y_tr = np.concatenate([per_uid[u]["y"] for u in per_uid if u != test_uid])
            if len(set(y_tr)) < 2:
                continue
            pipe = Pipeline([("scaler", StandardScaler()),
                             ("clf", LogisticRegression(max_iter=5000, C=1.0))])
            pipe.fit(x_tr.reshape(-1, 1), y_tr)
            yp = pipe.predict_proba(x_te.reshape(-1, 1))[:, 1]
            try:
                auc = roc_auc_score(y_te, yp)
                if auc < 0.5:
                    auc = 1 - auc
                louo_aucs.append(auc)
            except:
                pass

        if louo_aucs:
            mean_auc = float(np.mean(louo_aucs))
            std_auc = float(np.std(louo_aucs))
            degradation = 0  # No full-fit baseline for single features
            results_4b[mname] = {
                "louo_auc_mean": mean_auc,
                "louo_auc_std": std_auc,
                "n_folds": len(louo_aucs),
                "transfers": mean_auc > 0.55,  # Above chance
            }

    # Print results
    print(f"\n  {'Method':<18s} {'LOUO AUC':>10s} {'±std':>8s} {'Folds':>6s} {'Transfers?':>12s}")
    print(f"  {'-'*58}")
    for mname, r in sorted(results_4b.items(), key=lambda x: -x[1]["louo_auc_mean"]):
        tf = "YES" if r["transfers"] else "no"
        print(f"  {mname:<18s} {r['louo_auc_mean']:10.4f} {r['louo_auc_std']:8.4f}"
              f" {r['n_folds']:6d} {tf:>12s}")

    # Summary: do SIG methods beat baselines in transferability?
    sig_aucs = [r["louo_auc_mean"] for m, r in results_4b.items() if not m.startswith("BL_")]
    bl_aucs = [r["louo_auc_mean"] for m, r in results_4b.items() if m.startswith("BL_")]
    if sig_aucs and bl_aucs:
        best_sig = max(sig_aucs)
        best_bl = max(bl_aucs)
        print(f"\n  Best SIG LOUO: {best_sig:.4f}  |  Best Baseline LOUO: {best_bl:.4f}")
        if best_sig > best_bl:
            print(f"  >>> SIG methods beat baselines by {best_sig - best_bl:.4f}")
        else:
            print(f"  >>> WARNING: Baselines beat SIG methods by {best_bl - best_sig:.4f}")

    return results_4b


# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 4e: CALM-FREE COMPARISON
# ══════════════════════════════════════════════════════════════════

def run_experiment_4e():
    """Compare SIG detection with CALM-dependent vs CALM-free labels.

    CALM-dependent: target = CRYSTALLIZING/CRYSTALLIZED (from Theta_A, uses CALM z-score)
    CALM-free: target = will C_norm < 0.90 within 90 steps? (from Phi/Phi*, no CALM geometry)

    If CALM attenuates signal, CALM-free AUCs should be HIGHER than CALM-dependent.
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT 4e: CALM-FREE LABEL COMPARISON")
    print("  Does CALM normalization attenuate crystallization signal?")
    print("=" * 70)

    universes = {}
    for f in sorted(SIG_DIR.glob("*_sig.npz")):
        uid = f.stem.replace("_sig", "")
        data = load_sig_data(uid)
        if data is not None:
            universes[uid] = data
    print(f"\n  Loaded {len(universes)} universes")

    WINDOWS_TEST = [1, 5, 10, 20, 60]

    metrics_test = {
        "S1_scr":       ("scr", False),
        "S1_d_mp_kl":   ("d_mp_kl", False),
        "S2_lz_inv":    ("lz_norm", True),
        "S3_pe1_inv":   ("pe1", True),
        "BL_lam1":      ("lam1", False),
    }

    results = {"calm_dependent": {}, "calm_free": {}}

    for label_mode in ["calm_dependent", "calm_free"]:
        for mname, (field, invert) in metrics_test.items():
            results[label_mode][mname] = {}
            for W in WINDOWS_TEST:
                aucs = []
                for uid, data in universes.items():
                    raw = data[field].astype(float)
                    if invert:
                        raw = -raw
                    if label_mode == "calm_dependent":
                        y = make_binary_target(data["labels"])
                    else:
                        y = make_calm_free_target(data["labels"], H=90)
                    auc = evaluate_method_at_window(raw, y, W)
                    if auc is not None:
                        aucs.append(auc)
                if aucs:
                    results[label_mode][mname][W] = {
                        "auc_mean": float(np.mean(aucs)),
                        "auc_std": float(np.std(aucs)),
                        "n": len(aucs),
                    }

    # Print comparison table
    for label_mode in ["calm_dependent", "calm_free"]:
        mode_label = "CALM-DEPENDENT (Theta_A)" if label_mode == "calm_dependent" \
                     else "CALM-FREE (C_norm forward H=90)"
        print(f"\n  --- {mode_label} ---")
        print(f"  {'Method':<18s}", end="")
        for W in WINDOWS_TEST:
            print(f"  W={W:>2d}", end="")
        print()
        for mname in metrics_test:
            print(f"  {mname:<18s}", end="")
            for W in WINDOWS_TEST:
                r = results[label_mode][mname].get(W, {})
                auc = r.get("auc_mean", 0)
                if auc > 0:
                    print(f"  {auc:.3f}", end="")
                else:
                    print(f"  {'---':>5s}", end="")
            print()

    # Delta analysis: CALM-free minus CALM-dependent
    print(f"\n  --- DELTA (CALM-free - CALM-dependent) ---")
    print(f"  Positive = CALM was attenuating signal")
    print(f"  {'Method':<18s}", end="")
    for W in WINDOWS_TEST:
        print(f"  W={W:>2d}", end="")
    print()
    deltas = []
    for mname in metrics_test:
        print(f"  {mname:<18s}", end="")
        for W in WINDOWS_TEST:
            cd = results["calm_dependent"][mname].get(W, {}).get("auc_mean", 0)
            cf = results["calm_free"][mname].get(W, {}).get("auc_mean", 0)
            if cd > 0 and cf > 0:
                d = cf - cd
                deltas.append(d)
                marker = "+" if d > 0.01 else "-" if d < -0.01 else "~"
                print(f" {d:+.3f}{marker}", end="")
            else:
                print(f"  {'---':>5s}", end="")
        print()

    if deltas:
        mean_delta = np.mean(deltas)
        pos_frac = np.mean([d > 0.005 for d in deltas])
        print(f"\n  Mean delta: {mean_delta:+.4f}")
        print(f"  Fraction positive (CALM attenuates): {pos_frac:.1%}")
        if mean_delta > 0.01:
            print(f"  >>> CALM IS ATTENUATING: CALM-free targets yield better AUCs")
        elif mean_delta < -0.01:
            print(f"  >>> CALM IS HELPING: CALM-dependent targets yield better AUCs")
        else:
            print(f"  >>> CALM effect is negligible (delta ~ 0)")

    return results


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  KAPPA-SIG Phase 2: Structural Information Geometry Experiments")
    print("  David Ohio | Independent Researcher | April 2026")
    print("=" * 70)

    # Check data exists
    sig_files = list(SIG_DIR.glob("*_sig.npz"))
    if not sig_files:
        print("\n  ERROR: No SIG data found. Run extract_sig_data.py first.")
        return

    print(f"\n  Found {len(sig_files)} universe data files")

    all_results = {}


    # Experiment 4a: Window sweep
    t0 = time.time()
    results_4a = run_experiment_4a()
    all_results["experiment_4a"] = results_4a

    # Experiment 4d: Wheeler test
    wheeler_results, w_critical = run_experiment_4d()
    all_results["experiment_4d"] = {
        "results": {m: {str(k): v for k, v in wr.items()}
                    for m, wr in wheeler_results.items()},
        "w_critical": {m: v for m, v in w_critical.items()},
    }

    # LEFT_CENSORED recovery
    recovery = run_left_censored_recovery()
    all_results["left_censored_recovery"] = {
        uid: {k: {kk: (float(vv) if isinstance(vv, (np.floating, float)) else vv)
                   for kk, vv in v.items()}
              for k, v in methods.items()}
        for uid, methods in recovery.items()
    }

    # Experiment 4c: Ensemble
    ensemble = run_experiment_4c()
    all_results["experiment_4c"] = ensemble

    # Experiment 4b: Cross-universe transferability
    transfer = run_experiment_4b()
    all_results["experiment_4b"] = transfer

    # Experiment 4e: CALM-free comparison
    calm_comparison = run_experiment_4e()
    all_results["experiment_4e"] = calm_comparison


    total_time = time.time() - t0

    # Final summary
    print("\n" + "=" * 70)
    print("  KAPPA-SIG: FINAL SUMMARY")
    print("=" * 70)

    # Best individual method
    best_w_min = None
    best_method = None
    for mname, wdict in results_4a.items():
        for W in sorted(wdict.keys()):
            if wdict[W].get("auc_mean", 0) >= 0.70:
                if best_w_min is None or W < best_w_min:
                    best_w_min = W
                    best_method = mname
                break

    if best_method:
        print(f"\n  Best individual method: {best_method} at W={best_w_min}")
    else:
        print(f"\n  No individual method reached AUC >= 0.70")


    # Wheeler verdict
    wheeler_any = any(v is not None for v in w_critical.values())
    if wheeler_any:
        best_wc = min(v for v in w_critical.values() if v is not None)
        print(f"  Wheeler test: significant at W={best_wc}")
    else:
        print(f"  Wheeler test: not significant at any window")

    # Ensemble verdict
    if ensemble.get("auc_full"):
        print(f"  Ensemble AUC: {ensemble['auc_full']:.4f} (full) "
              f"{ensemble.get('cv_auc_mean', 'N/A')} (CV)")

    # Recovery count (fix: use "recovered" not "detected")
    n_rec = sum(1 for uid, methods in recovery.items()
                if any(v.get("recovered", False) for v in methods.values()))
    print(f"  LEFT_CENSORED recovered: {n_rec}/5")

    # CALM comparison verdict
    if calm_comparison:
        cd_aucs = []
        cf_aucs = []
        for mname in calm_comparison.get("calm_dependent", {}):
            for W, r in calm_comparison["calm_dependent"].get(mname, {}).items():
                cd_aucs.append(r.get("auc_mean", 0))
            for W, r in calm_comparison["calm_free"].get(mname, {}).items():
                cf_aucs.append(r.get("auc_mean", 0))
        if cd_aucs and cf_aucs:
            delta = np.mean(cf_aucs) - np.mean(cd_aucs)
            verdict = "ATTENUATING" if delta > 0.01 else "HELPING" if delta < -0.01 else "NEUTRAL"
            print(f"  CALM effect: {verdict} (delta={delta:+.4f})")
    print(f"\n  Total time: {total_time:.0f}s")

    # Save all results
    out_path = OUT_DIR / "sig_experiment_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
