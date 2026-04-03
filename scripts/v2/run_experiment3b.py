#!/usr/bin/env python3
r"""
Kappa v2 -- Experiment 3b: Ablation, Shuffle Test, Random Features
===================================================================
Three robustness tests for the calibrated hazard model (H=90):

1. ABLATION: Remove one covariate at a time, measure AUC drop
   - If removing Theta_A causes the biggest drop -> confirms dominance
   - If removing kappa_F causes no drop -> it's truly redundant

2. SHUFFLE TEST (Permutation): Shuffle target labels 1000x, refit
   - If real AUC >> shuffled distribution -> model is not fitting noise
   - Computes empirical p-value

3. RANDOM FEATURES: Add 4 random noise columns, refit
   - If random features get beta ~0 and real features keep their values
   - -> confirms signal is real, not overfitting to structure

David Ohio | odavidohio@gmail.com | Independent Researcher
March 2026
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
import warnings
warnings.filterwarnings("ignore")

V2_DIR = Path(r"C:\Users\ohiod\Projects\Sentinel\data\v2_analysis")
OUT_DIR = Path(r"C:\Users\ohiod\Projects\Sentinel\data\v2_analysis\experiment3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE = {"latam", "x_commodity_chain"}
H = 90
C_NORM_THRESH = 0.90
COVARIATES = ["kappa_F", "rho_bar_pos", "eta_norm", "theta_A"]


def load_universe(uid):
    csv_path = V2_DIR / uid / "kappa_v2_state.csv"
    summ_path = V2_DIR / uid / "kappa_v2_summary.json"
    if not csv_path.exists():
        return None
    if summ_path.exists():
        with open(summ_path) as f:
            summ = json.load(f)
        if summ.get("phi_star_estimated", False):
            return None
    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    df["rho_bar_pos"] = np.maximum(df["rho_bar"].values, 0.0)
    cn = df["C_norm"].values
    t_S = None
    for i in range(len(cn)):
        if cn[i] < C_NORM_THRESH:
            t_S = i
            break
    return df, t_S, uid


def build_dataset():
    X_all, y_all, uids_all = [], [], []
    universes = sorted([d.name for d in V2_DIR.iterdir()
                        if d.is_dir() and d.name not in EXCLUDE
                        and (d / "kappa_v2_state.csv").exists()])
    for uid in universes:
        result = load_universe(uid)
        if result is None:
            continue
        df, t_S, _ = result
        n = len(df)
        kf = df["kappa_F"].values
        rho_pos = np.maximum(df["rho_bar"].values, 0.0)
        eta_n = df["eta_norm"].values if "eta_norm" in df.columns else np.ones(n)
        theta = df["theta_A"].values if "theta_A" in df.columns else np.zeros(n)
        cn = df["C_norm"].values

        for t in range(60, n):
            if t_S is not None and t > t_S:
                continue
            future = cn[t+1:t+1+H]
            if len(future) == 0:
                continue
            y = 1 if np.any(future < C_NORM_THRESH) else 0
            X_row = [kf[t], rho_pos[t], eta_n[t], theta[t]]
            if any(np.isnan(x) for x in X_row):
                continue
            X_all.append(X_row)
            y_all.append(y)
            uids_all.append(uid)

    return np.array(X_all), np.array(y_all), np.array(uids_all)


def run_ablation(X, y):
    """Remove one covariate at a time, measure AUC drop."""
    print("\n" + "=" * 60)
    print("  TEST 1: ABLATION (remove one covariate at a time)")
    print("=" * 60)

    # Full model
    m_full = LogisticRegression(max_iter=5000, solver="lbfgs", C=1.0)
    m_full.fit(X, y)
    auc_full = roc_auc_score(y, m_full.predict_proba(X)[:, 1])
    brier_full = brier_score_loss(y, m_full.predict_proba(X)[:, 1])
    print(f"\n  Full model (all 4 covariates):")
    print(f"    AUC = {auc_full:.4f}  Brier = {brier_full:.4f}")

    results = {"full": {"AUC": auc_full, "Brier": brier_full}, "ablated": {}}

    print(f"\n  {'Removed':15s}  {'AUC':>8s}  {'dAUC':>8s}  {'Brier':>8s}  {'Interpretation':>20s}")
    print(f"  {'-' * 70}")

    for i, name in enumerate(COVARIATES):
        # Remove column i
        X_abl = np.delete(X, i, axis=1)
        m_abl = LogisticRegression(max_iter=5000, solver="lbfgs", C=1.0)
        m_abl.fit(X_abl, y)
        auc_abl = roc_auc_score(y, m_abl.predict_proba(X_abl)[:, 1])
        brier_abl = brier_score_loss(y, m_abl.predict_proba(X_abl)[:, 1])
        d_auc = auc_full - auc_abl

        if d_auc > 0.05:
            interp = "CRITICAL"
        elif d_auc > 0.01:
            interp = "important"
        elif d_auc > 0.001:
            interp = "minor"
        else:
            interp = "redundant"

        print(f"  {name:15s}  {auc_abl:8.4f}  {d_auc:+8.4f}  {brier_abl:8.4f}  {interp:>20s}")
        results["ablated"][name] = {
            "AUC": auc_abl, "dAUC": d_auc, "Brier": brier_abl,
            "interpretation": interp
        }

    # Also test: Theta_A alone (no other covariates)
    X_theta_only = X[:, 3:4]
    m_theta = LogisticRegression(max_iter=5000, solver="lbfgs", C=1.0)
    m_theta.fit(X_theta_only, y)
    auc_theta = roc_auc_score(y, m_theta.predict_proba(X_theta_only)[:, 1])
    print(f"\n  Theta_A ALONE:   AUC = {auc_theta:.4f}  (vs full {auc_full:.4f})")
    results["theta_alone"] = {"AUC": auc_theta}

    # kappa_F alone
    X_kf_only = X[:, 0:1]
    m_kf = LogisticRegression(max_iter=5000, solver="lbfgs", C=1.0)
    m_kf.fit(X_kf_only, y)
    auc_kf = roc_auc_score(y, m_kf.predict_proba(X_kf_only)[:, 1])
    print(f"  kappa_F ALONE:   AUC = {auc_kf:.4f}  (vs full {auc_full:.4f})")
    results["kf_alone"] = {"AUC": auc_kf}

    return results


def run_shuffle_test(X, y, n_perm=1000):
    """Permutation test: shuffle labels, compare AUC distribution."""
    print("\n" + "=" * 60)
    print(f"  TEST 2: SHUFFLE TEST ({n_perm} permutations)")
    print("=" * 60)

    # Real AUC
    m_real = LogisticRegression(max_iter=5000, solver="lbfgs", C=1.0)
    m_real.fit(X, y)
    auc_real = roc_auc_score(y, m_real.predict_proba(X)[:, 1])

    # Shuffled AUCs
    rng = np.random.RandomState(42)
    auc_shuffled = []
    for i in range(n_perm):
        y_shuf = rng.permutation(y)
        if len(set(y_shuf)) < 2:
            continue
        m_shuf = LogisticRegression(max_iter=5000, solver="lbfgs", C=1.0)
        m_shuf.fit(X, y_shuf)
        try:
            auc_s = roc_auc_score(y_shuf, m_shuf.predict_proba(X)[:, 1])
            auc_shuffled.append(auc_s)
        except:
            pass

    auc_shuffled = np.array(auc_shuffled)
    p_value = float(np.mean(auc_shuffled >= auc_real))

    print(f"\n  Real AUC:       {auc_real:.4f}")
    print(f"  Shuffled mean:  {auc_shuffled.mean():.4f} +/- {auc_shuffled.std():.4f}")
    print(f"  Shuffled max:   {auc_shuffled.max():.4f}")
    print(f"  Shuffled p95:   {np.percentile(auc_shuffled, 95):.4f}")
    print(f"  Shuffled p99:   {np.percentile(auc_shuffled, 99):.4f}")
    print(f"\n  p-value:        {p_value:.4f}")
    print(f"  Verdict:        {'PASS (p < 0.01)' if p_value < 0.01 else 'FAIL' if p_value >= 0.05 else 'MARGINAL'}")

    return {
        "AUC_real": auc_real,
        "AUC_shuffled_mean": float(auc_shuffled.mean()),
        "AUC_shuffled_std": float(auc_shuffled.std()),
        "AUC_shuffled_max": float(auc_shuffled.max()),
        "AUC_shuffled_p95": float(np.percentile(auc_shuffled, 95)),
        "AUC_shuffled_p99": float(np.percentile(auc_shuffled, 99)),
        "p_value": p_value,
        "n_permutations": n_perm,
        "verdict": "PASS" if p_value < 0.01 else "FAIL" if p_value >= 0.05 else "MARGINAL",
    }


def run_random_features(X, y, n_random=4, n_trials=100):
    """Add random noise features, check if real betas survive."""
    print("\n" + "=" * 60)
    print(f"  TEST 3: RANDOM FEATURES ({n_random} noise columns, {n_trials} trials)")
    print("=" * 60)

    # Real model betas
    m_real = LogisticRegression(max_iter=5000, solver="lbfgs", C=1.0)
    m_real.fit(X, y)
    real_betas = dict(zip(COVARIATES, m_real.coef_[0]))

    rng = np.random.RandomState(42)
    random_betas = {f"random_{i}": [] for i in range(n_random)}
    real_betas_with_noise = {name: [] for name in COVARIATES}

    for trial in range(n_trials):
        # Add random columns
        X_aug = np.column_stack([X] + [rng.randn(len(X)) for _ in range(n_random)])

        m_aug = LogisticRegression(max_iter=5000, solver="lbfgs", C=1.0)
        m_aug.fit(X_aug, y)

        for i, name in enumerate(COVARIATES):
            real_betas_with_noise[name].append(m_aug.coef_[0][i])
        for j in range(n_random):
            random_betas[f"random_{j}"].append(m_aug.coef_[0][len(COVARIATES) + j])

    print(f"\n  {'Feature':15s}  {'Original':>10s}  {'With noise':>12s}  {'Random beta':>12s}  {'Verdict':>10s}")
    print(f"  {'-' * 65}")

    results = {"real_features": {}, "random_features": {}}

    for name in COVARIATES:
        orig = real_betas[name]
        noisy = np.mean(real_betas_with_noise[name])
        noisy_std = np.std(real_betas_with_noise[name])
        drift = abs(noisy - orig) / max(abs(orig), 1e-12)
        verdict = "STABLE" if drift < 0.20 else "DRIFTED"
        print(f"  {name:15s}  {orig:+10.4f}  {noisy:+10.4f}+-{noisy_std:.3f}  {'---':>12s}  {verdict:>10s}")
        results["real_features"][name] = {
            "original_beta": orig,
            "mean_with_noise": float(noisy),
            "std_with_noise": float(noisy_std),
            "drift_pct": float(drift * 100),
            "stable": drift < 0.20,
        }

    for j in range(n_random):
        name = f"random_{j}"
        vals = np.array(random_betas[name])
        mean_r = np.mean(vals)
        std_r = np.std(vals)
        near_zero = abs(mean_r) < 0.5
        verdict = "NOISE (good)" if near_zero else "ABSORBED SIGNAL"
        print(f"  {name:15s}  {'---':>10s}  {'---':>12s}  {mean_r:+10.4f}+-{std_r:.3f}  {verdict:>10s}")
        results["random_features"][name] = {
            "mean_beta": float(mean_r),
            "std_beta": float(std_r),
            "near_zero": near_zero,
        }

    all_real_stable = all(r["stable"] for r in results["real_features"].values())
    all_random_zero = all(r["near_zero"] for r in results["random_features"].values())
    overall = "PASS" if all_real_stable and all_random_zero else "FAIL"
    print(f"\n  Overall: real features stable={all_real_stable}, random~zero={all_random_zero} -> {overall}")
    results["verdict"] = overall

    return results


def main():
    print("=" * 80)
    print("  KAPPA v2 -- EXPERIMENT 3b: ABLATION + SHUFFLE + RANDOM FEATURES")
    print("  Robustness validation for calibrated H=90 hazard model")
    print("  David Ohio | Independent Researcher | March 2026")
    print("=" * 80)

    print("\n[1] Building dataset (H=90, pre-event censored)...")
    X, y, uids = build_dataset()
    print(f"    N={len(y)}  positive={y.sum()}  prevalence={y.mean():.4f}")
    print(f"    Universes: {len(set(uids))}")

    # Test 1: Ablation
    ablation_results = run_ablation(X, y)

    # Test 2: Shuffle
    shuffle_results = run_shuffle_test(X, y, n_perm=1000)

    # Test 3: Random features
    random_results = run_random_features(X, y, n_random=4, n_trials=100)

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    # Find most critical ablation
    max_drop_name = max(ablation_results["ablated"],
                        key=lambda k: ablation_results["ablated"][k]["dAUC"])
    max_drop = ablation_results["ablated"][max_drop_name]["dAUC"]

    print(f"\n  Ablation:  Most critical = {max_drop_name} (dAUC={max_drop:+.4f})")
    print(f"  Shuffle:   p={shuffle_results['p_value']:.4f} -> {shuffle_results['verdict']}")
    print(f"  Random:    {random_results['verdict']}")

    all_pass = (shuffle_results["verdict"] == "PASS" and
                random_results["verdict"] == "PASS")
    print(f"\n  OVERALL: {'ALL TESTS PASS' if all_pass else 'SOME TESTS FAILED'}")

    # Save
    out_file = OUT_DIR / "experiment3b_robustness.json"
    combined = {
        "ablation": ablation_results,
        "shuffle_test": shuffle_results,
        "random_features": random_results,
        "overall_pass": all_pass,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"\n  Saved: {out_file}")
    print("\nDone.")


if __name__ == "__main__":
    main()
