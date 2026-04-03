#!/usr/bin/env python3
r"""
Kappa v2 -- Experiment 3: Beta Calibration via Historical Series
================================================================
Calibrates Layer 3 hazard model coefficients using pooled daily
observations from all Sentinel universes.

Target: y(t) = 1 if C_norm crosses below 0.90 for the FIRST TIME
        within the next H days.

Method:
  1. Pool all universe-days (excluding BASELINE_INSUFFICIENT)
  2. Apply pre-event censoring: exclude all days AFTER first t_S
  3. Fit logistic regression: logit(y) = b0 + bF*kF + br*rho+ + be*eta + bA*thetaA
  4. Leave-one-universe-out cross-validation
  5. Bootstrap (1000x) for confidence intervals
  6. Check: no beta changes sign, ECE < 0.15

David Ohio | odavidohio@gmail.com | Independent Researcher
March 2026
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, brier_score_loss,
                             log_loss, confusion_matrix)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings("ignore")

V2_DIR = Path(r"C:\Users\ohiod\Projects\Sentinel\data\v2_analysis")
OUT_DIR = Path(r"C:\Users\ohiod\Projects\Sentinel\data\v2_analysis\experiment3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Universes to EXCLUDE (baseline insufficient or insufficient data)
EXCLUDE = {"latam", "x_commodity_chain"}

# Horizons to calibrate
HORIZONS = [30, 60, 90]

# C_norm threshold for "structural activation"
C_NORM_THRESH = 0.90

# Covariates
COVARIATES = ["kappa_F", "rho_bar_pos", "eta_norm", "theta_A"]


def load_universe(uid):
    """Load v2 state CSV for a universe, compute derived features."""
    csv_path = V2_DIR / uid / "kappa_v2_state.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)

    # Check if baseline insufficient
    summ_path = V2_DIR / uid / "kappa_v2_summary.json"
    if summ_path.exists():
        with open(summ_path) as f:
            summ = json.load(f)
        if summ.get("phi_star_estimated", False):
            return None

    # Derived: positive rho_bar
    df["rho_bar_pos"] = np.maximum(df["rho_bar"].values, 0.0)

    # Find t_S: first time C_norm < threshold
    t_S = None
    cn = df["C_norm"].values
    for i in range(len(cn)):
        if cn[i] < C_NORM_THRESH:
            t_S = i
            break

    return df, t_S, uid


def build_dataset(horizons):
    """Build pooled dataset for all universes and horizons."""
    datasets = {}
    for H in horizons:
        datasets[H] = {"X": [], "y": [], "uid": [], "date": []}

    universes = sorted([d.name for d in V2_DIR.iterdir()
                        if d.is_dir() and d.name not in EXCLUDE
                        and (d / "kappa_v2_state.csv").exists()])

    stats = {}
    for uid in universes:
        result = load_universe(uid)
        if result is None:
            continue
        df, t_S, _ = result
        n = len(df)

        # Covariates
        kf = df["kappa_F"].values
        rho_pos = np.maximum(df["rho_bar"].values, 0.0)
        eta_n = df["eta_norm"].values if "eta_norm" in df.columns else np.ones(n)
        theta = df["theta_A"].values if "theta_A" in df.columns else np.zeros(n)
        cn = df["C_norm"].values

        stats[uid] = {"n": n, "t_S": t_S, "has_damage": t_S is not None}

        for H in horizons:
            for t in range(60, n):  # skip warm-up
                # Pre-event censoring: exclude days AFTER t_S
                if t_S is not None and t > t_S:
                    continue

                # Target: will C_norm cross below threshold within H steps?
                future = cn[t+1:t+1+H]
                if len(future) == 0:
                    continue
                y = 1 if np.any(future < C_NORM_THRESH) else 0

                # But if t_S is known and t_S is within [t+1, t+H], y=1
                # If t_S is None or t_S > t+H, y=0
                # This is already captured by checking future C_norm

                X_row = [kf[t], rho_pos[t], eta_n[t], theta[t]]

                # Skip rows with NaN
                if any(np.isnan(x) for x in X_row):
                    continue

                datasets[H]["X"].append(X_row)
                datasets[H]["y"].append(y)
                datasets[H]["uid"].append(uid)
                datasets[H]["date"].append(str(df.index[t].date()))

    for H in horizons:
        datasets[H]["X"] = np.array(datasets[H]["X"])
        datasets[H]["y"] = np.array(datasets[H]["y"])
        datasets[H]["uid"] = np.array(datasets[H]["uid"])

    return datasets, stats


def fit_and_evaluate(X, y, uids, H):
    """Fit logistic regression with leave-one-universe-out CV."""
    unique_uids = sorted(set(uids))

    # --- Full fit ---
    model = LogisticRegression(max_iter=5000, solver="lbfgs", C=1.0)
    model.fit(X, y)
    coefs = dict(zip(["beta_0"] + COVARIATES,
                      [model.intercept_[0]] + list(model.coef_[0])))
    y_prob = model.predict_proba(X)[:, 1]

    full_auc = roc_auc_score(y, y_prob) if len(set(y)) > 1 else 0.0
    full_brier = brier_score_loss(y, y_prob)
    full_logloss = log_loss(y, y_prob)

    # ECE (Expected Calibration Error)
    prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=10, strategy="uniform")
    ece = float(np.mean(np.abs(prob_true - prob_pred)))

    # --- Leave-one-universe-out CV ---
    cv_aucs = []
    cv_briers = []
    cv_coefs = {name: [] for name in ["beta_0"] + COVARIATES}

    for test_uid in unique_uids:
        mask_train = uids != test_uid
        mask_test = uids == test_uid

        X_train, y_train = X[mask_train], y[mask_train]
        X_test, y_test = X[mask_test], y[mask_test]

        if len(set(y_train)) < 2 or len(set(y_test)) < 2:
            continue
        if len(y_test) < 10:
            continue

        m = LogisticRegression(max_iter=5000, solver="lbfgs", C=1.0)
        m.fit(X_train, y_train)
        yp = m.predict_proba(X_test)[:, 1]

        try:
            cv_aucs.append(roc_auc_score(y_test, yp))
        except:
            pass
        cv_briers.append(brier_score_loss(y_test, yp))

        cv_coefs["beta_0"].append(m.intercept_[0])
        for i, name in enumerate(COVARIATES):
            cv_coefs[name].append(m.coef_[0][i])

    # --- Bootstrap (1000x) for coefficient CIs ---
    n_boot = 1000
    boot_coefs = {name: [] for name in ["beta_0"] + COVARIATES}
    rng = np.random.RandomState(42)

    for _ in range(n_boot):
        idx = rng.choice(len(X), size=len(X), replace=True)
        Xb, yb = X[idx], y[idx]
        if len(set(yb)) < 2:
            continue
        mb = LogisticRegression(max_iter=5000, solver="lbfgs", C=1.0)
        mb.fit(Xb, yb)
        boot_coefs["beta_0"].append(mb.intercept_[0])
        for i, name in enumerate(COVARIATES):
            boot_coefs[name].append(mb.coef_[0][i])

    # Coefficient summary
    coef_summary = {}
    sign_stable = True
    for name in ["beta_0"] + COVARIATES:
        vals = np.array(boot_coefs[name])
        ci_lo = float(np.percentile(vals, 2.5))
        ci_hi = float(np.percentile(vals, 97.5))
        median = float(np.median(vals))
        # Sign stability: does the 95% CI cross zero?
        crosses_zero = (ci_lo < 0 and ci_hi > 0)
        if name != "beta_0" and crosses_zero:
            sign_stable = False

        coef_summary[name] = {
            "value": coefs[name],
            "median_boot": median,
            "ci_2.5": ci_lo,
            "ci_97.5": ci_hi,
            "sign_stable": not crosses_zero,
        }

    # CV coefficient stability
    cv_sign_stable = True
    for name in ["beta_0"] + COVARIATES:
        if name == "beta_0":
            continue
        vals = cv_coefs[name]
        if len(vals) >= 3:
            if min(vals) < 0 and max(vals) > 0:
                cv_sign_stable = False

    return {
        "horizon": H,
        "n_obs": len(y),
        "n_positive": int(y.sum()),
        "prevalence": float(y.mean()),
        "n_universes": len(unique_uids),
        "coefficients": coef_summary,
        "full_fit": {
            "AUC": full_auc,
            "Brier": full_brier,
            "LogLoss": full_logloss,
            "ECE": ece,
        },
        "cv_louo": {
            "n_folds": len(cv_aucs),
            "AUC_mean": float(np.mean(cv_aucs)) if cv_aucs else None,
            "AUC_std": float(np.std(cv_aucs)) if cv_aucs else None,
            "Brier_mean": float(np.mean(cv_briers)) if cv_briers else None,
        },
        "sign_stability": {
            "bootstrap_all_stable": sign_stable,
            "cv_all_stable": cv_sign_stable,
        },
        "calibration_pass": sign_stable and ece < 0.15,
    }


def main():
    print("=" * 80)
    print("  KAPPA v2 -- EXPERIMENT 3: BETA CALIBRATION")
    print("  Logistic regression on pooled daily observations")
    print("  Leave-one-universe-out CV + Bootstrap(1000x)")
    print("  David Ohio | Independent Researcher | March 2026")
    print("=" * 80)

    print("\n[1] Building dataset...")
    datasets, stats = build_dataset(HORIZONS)

    print(f"\n  Universe inventory:")
    for uid, s in sorted(stats.items()):
        dmg = f"t_S={s['t_S']}" if s['has_damage'] else "no damage"
        print(f"    {uid:25s}  n={s['n']:4d}  {dmg}")

    results = {}
    for H in HORIZONS:
        X = datasets[H]["X"]
        y = datasets[H]["y"]
        uids = datasets[H]["uid"]
        print(f"\n{'='*60}")
        print(f"  HORIZON H={H} days")
        print(f"  N={len(y)}  positive={y.sum()}  prevalence={y.mean():.4f}")
        print(f"{'='*60}")

        if len(set(y)) < 2:
            print("  SKIP: no positive examples")
            continue

        r = fit_and_evaluate(X, y, uids, H)
        results[f"H{H}"] = r

        # Print results
        print(f"\n  COEFFICIENTS (calibrated):")
        for name, cs in r["coefficients"].items():
            sign_flag = "" if cs["sign_stable"] else " *** UNSTABLE ***"
            print(f"    {name:12s} = {cs['value']:+8.4f}  "
                  f"CI95=[{cs['ci_2.5']:+.4f}, {cs['ci_97.5']:+.4f}]{sign_flag}")

        print(f"\n  FULL FIT:")
        ff = r["full_fit"]
        print(f"    AUC={ff['AUC']:.4f}  Brier={ff['Brier']:.4f}  "
              f"LogLoss={ff['LogLoss']:.4f}  ECE={ff['ECE']:.4f}")

        cv = r["cv_louo"]
        if cv["AUC_mean"] is not None:
            print(f"\n  LEAVE-ONE-UNIVERSE-OUT CV ({cv['n_folds']} folds):")
            print(f"    AUC={cv['AUC_mean']:.4f} +/- {cv['AUC_std']:.4f}  "
                  f"Brier={cv['Brier_mean']:.4f}")

        ss = r["sign_stability"]
        print(f"\n  SIGN STABILITY: bootstrap={'PASS' if ss['bootstrap_all_stable'] else 'FAIL'}  "
              f"cv={'PASS' if ss['cv_all_stable'] else 'FAIL'}")
        print(f"  CALIBRATION: {'PASS' if r['calibration_pass'] else 'FAIL'}  "
              f"(ECE={ff['ECE']:.4f} < 0.15?  signs stable?)")

    # Save
    out_file = OUT_DIR / "experiment3_calibration.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {out_file}")

    # Print deployment-ready coefficients
    print(f"\n{'='*60}")
    print(f"  DEPLOYMENT-READY COEFFICIENTS")
    print(f"{'='*60}")
    for hkey, r in results.items():
        if not r.get("calibration_pass"):
            print(f"\n  {hkey}: CALIBRATION FAILED -- do not deploy")
            continue
        print(f"\n  {hkey}:")
        for name, cs in r["coefficients"].items():
            print(f"    {name} = {cs['value']:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
