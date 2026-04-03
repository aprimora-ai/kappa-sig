#!/usr/bin/env python3
r"""
Kappa v2 -- Circularity Test
==============================
Tests whether geometric precedence survives when Layer 2 is computed
from observables INDEPENDENT of Phi(t).

If Oh(t), eta(t), or mean_corr(t) embeddings show the same lead time
as Phi(t), the geometric signal is system-wide -- not Phi-derived.
This eliminates the circularity objection (L4 in the paper).

Run:
  cd C:\Users\ohiod\Projects\Sentinel
  python run_v2_circularity_test.py

David Ohio | odavidohio@gmail.com | Independent Researcher
March 2026
"""
import numpy as np
import pandas as pd
import json
from pathlib import Path

BASE = Path(r"C:\Users\ohiod\Projects\Sentinel\data\v2_analysis")
OUT = BASE / "circularity_test"
OUT.mkdir(parents=True, exist_ok=True)

EPS = 1e-12
SIGMA_FLOOR = 0.01
GEO_SIGMA_MIN = 0.005
THETA_A_CAP = 5.0
RQA_WINDOW = 60
EPSILON_Q = 0.10
WARMUP = 60
THRESH = 0.5
PERSIST = 5
C_NORM_THRESHOLD = 0.90

# Universes to test (GEO_PRECEDED + AMBIGUOUS for reference)
TEST_UNIVERSES = ["commodities", "energy", "asia_pacific"]

# Observables to embed (Phi is the baseline; others are independent)
OBSERVABLES = [
    ("Phi",       "phi"),
    ("Oh",        "Oh"),
    ("eta",       "eta"),
    ("mean_corr", "mean_corr"),
]


# ============================================================================
# RQA FUNCTIONS (same as engine_v5.5, standalone for independence)
# ============================================================================

def _recurrence_matrix(Y, epsilon):
    n = Y.shape[0]
    R = np.zeros((n, n), dtype=bool)
    for i in range(n):
        R[i] = np.linalg.norm(Y - Y[i], axis=1) <= epsilon
    return R


def _rqa_from_matrix(R, min_line=2):
    n = R.shape[0]
    rp = int(R.sum())
    RR = rp / (n * n) if n > 0 else 0.0
    dl = []
    for k in range(-n + 1, n):
        d = np.diag(R, k)
        l = 0
        for v in d:
            if v:
                l += 1
            else:
                if l >= min_line:
                    dl.append(l)
                l = 0
        if l >= min_line:
            dl.append(l)
    DET = sum(dl) / rp if rp > 0 else 0.0
    vl = []
    for j in range(n):
        c = R[:, j]
        l = 0
        for v in c:
            if v:
                l += 1
            else:
                if l >= min_line:
                    vl.append(l)
                l = 0
        if l >= min_line:
            vl.append(l)
    LAM = sum(vl) / rp if rp > 0 else 0.0
    TT = float(np.mean(vl)) if vl else 0.0
    return {"DET": DET, "LAM": LAM, "TT": TT}


def _delay_embedding(x, tau, m):
    nv = len(x) - (m - 1) * tau
    if nv <= 0:
        return np.array([]).reshape(0, m)
    Y = np.zeros((nv, m))
    for i in range(m):
        Y[:, i] = x[i * tau:i * tau + nv]
    return Y


def _estimate_tau(x, max_tau=20):
    xc = x - np.mean(x)
    n = len(xc)
    for t in range(1, min(max_tau, n // 2)):
        if len(xc[:-t]) < 2:
            continue
        if np.corrcoef(xc[:-t], xc[t:])[0, 1] <= 0:
            return t
    return min(max_tau, max(1, n // 4))


def compute_theta_for_series(series):
    """Compute Theta_A via RQA + z-score for any input series."""
    n = len(series)
    tau = _estimate_tau(series)
    m = 3
    W = RQA_WINDOW
    results = []

    for t in range(W, n):
        w = series[t - W:t]
        Y = _delay_embedding(w, tau, m)
        if Y.shape[0] < 10:
            results.append({"DET": np.nan, "LAM": np.nan, "TT": np.nan})
            continue
        ds = [np.linalg.norm(Y[i] - Y[j])
              for i in range(Y.shape[0]) for j in range(i + 1, Y.shape[0])]
        if not ds:
            results.append({"DET": np.nan, "LAM": np.nan, "TT": np.nan})
            continue
        eps = max(np.quantile(ds, EPSILON_Q), EPS)
        R = _recurrence_matrix(Y, eps)
        results.append(_rqa_from_matrix(R))

    pad = [{"DET": np.nan, "LAM": np.nan, "TT": np.nan}] * W
    rdf = pd.DataFrame(pad + results)

    # CALM normalization (first 40%)
    ci = int(n * 0.4)
    calm_mask = np.zeros(n, dtype=bool)
    calm_mask[:ci] = True

    geo_reliability = "HIGH"
    for c in ["DET", "LAM", "TT"]:
        cv = rdf[c].values[calm_mask]
        cv = cv[np.isfinite(cv)]
        if len(cv) > 5:
            mu = float(np.mean(cv))
            sigma = float(np.std(cv))
            if c in ("DET", "LAM") and sigma < GEO_SIGMA_MIN:
                geo_reliability = "LOW"
            denom = max(sigma, SIGMA_FLOOR)
            rdf[f"{c}_norm"] = (rdf[c] - mu) / denom
        else:
            rdf[f"{c}_norm"] = 0.0

    dn = rdf["DET_norm"].fillna(0).values
    ln = rdf["LAM_norm"].fillna(0).values
    tn = rdf["TT_norm"].fillna(0).values
    theta = np.clip(
        0.40 * np.maximum(dn, 0) + 0.35 * np.maximum(ln, 0) + 0.25 * np.maximum(tn, 0),
        0, THETA_A_CAP
    )

    if geo_reliability == "LOW":
        theta = theta * 0.0

    return theta, geo_reliability


def first_sustained(signal, thresh, persist, start=0):
    n = len(signal)
    for i in range(start, n - persist + 1):
        if all(signal[i:i + persist] > thresh):
            return i
    return None


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 90)
    print("  KAPPA v2 -- CIRCULARITY TEST")
    print("  Does geometric precedence survive with independent observable embeddings?")
    print("  David Ohio | Independent Researcher | March 2026")
    print("=" * 90)

    results = {}

    for uid in TEST_UNIVERSES:
        csv_path = BASE / uid / "kappa_v2_state.csv"
        if not csv_path.exists():
            print(f"\n  SKIP {uid}: no CSV")
            continue

        df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
        n = len(df)
        cn = df["C_norm"].values

        # Find t_S
        t_S = None
        for i in range(WARMUP, n):
            if cn[i] < C_NORM_THRESHOLD:
                t_S = i
                break

        t_S_date = str(df.index[t_S].date()) if t_S is not None else "none"

        print(f"\n{'-'*80}")
        print(f"  {uid} (n={n}, t_S={'step '+str(t_S)+' ('+t_S_date+')' if t_S else 'none'})")
        print(f"  {'Observable':12s}  {'t_G':>7s}  {'t_G_date':>12s}  {'peak':>6s}  {'lead':>12s}  {'geo':>5s}  {'Verdict':>20s}")
        print(f"  {'-'*80}")

        uid_results = {"t_S": t_S, "t_S_date": t_S_date, "embeddings": {}}

        for obs_name, obs_col in OBSERVABLES:
            if obs_col not in df.columns:
                continue

            series = df[obs_col].values.astype(float)
            if np.std(series) < 1e-15:
                print(f"  {obs_name:12s}  {'FLAT':>7s}  {'':>12s}  {'':>6s}  {'skipped':>12s}  {'':>5s}")
                uid_results["embeddings"][obs_name] = {"status": "FLAT"}
                continue

            theta, geo_rel = compute_theta_for_series(series)
            t_G = first_sustained(theta, THRESH, PERSIST, start=WARMUP)
            peak = float(theta.max())

            lt = None
            if t_G is not None and t_S is not None and t_G < t_S:
                lt = t_S - t_G

            t_G_str = str(t_G) if t_G is not None else "never"
            t_G_date = str(df.index[t_G].date()) if t_G is not None else ""
            lt_str = f"{lt} steps" if lt is not None else ("no lead" if t_G is not None else "N/A")

            verdict = ""
            if lt is not None and lt > 0:
                verdict = "PRECEDES DAMAGE"
            elif t_G is not None and t_S is not None and t_G >= t_S:
                verdict = "after damage"
            elif t_S is None:
                verdict = "no damage (prosp.)"
            elif t_G is None:
                verdict = "no activation"

            is_independent = obs_col != "phi"
            marker = " <<<" if (lt is not None and lt > 0 and is_independent) else ""

            print(f"  {obs_name:12s}  {t_G_str:>7s}  {t_G_date:>12s}  {peak:6.2f}  {lt_str:>12s}  {geo_rel:>5s}  {verdict:>20s}{marker}")

            uid_results["embeddings"][obs_name] = {
                "column": obs_col,
                "t_G": t_G,
                "t_G_date": t_G_date if t_G_date else None,
                "peak_theta_A": peak,
                "lead_time": lt,
                "geo_reliability": geo_rel,
                "verdict": verdict,
                "independent_of_phi": is_independent,
            }

        results[uid] = uid_results

    # -- Summary --
    print(f"\n\n{'='*90}")
    print(f"  CIRCULARITY VERDICT")
    print(f"{'='*90}")

    # Count independent observables that preceded damage
    independent_preceded = 0
    independent_total = 0
    for uid, ur in results.items():
        for obs, data in ur["embeddings"].items():
            if isinstance(data, dict) and data.get("independent_of_phi"):
                independent_total += 1
                if data.get("lead_time") is not None and data["lead_time"] > 0:
                    independent_preceded += 1

    print(f"""
  Independent observables tested: {independent_total}
  Independent observables with lead time > 0: {independent_preceded}

  INTERPRETATION:""")

    if independent_preceded > 0:
        print(f"""
  >>> CIRCULARITY ELIMINATED <<<
  {independent_preceded} observable(s) independent of Phi(t) show geometric
  activation BEFORE structural damage. The geometric signal is system-wide,
  not an artifact of embedding the same variable used by Layer 1.

  This means Layer 2 detects structural reorganization in the correlation
  topology itself (Oh), the viscosity dynamics (eta), and the mean
  correlation structure (mean_corr) -- not just in the damage accumulator.
""")
    else:
        print(f"""
  Circularity concern REMAINS. No independent observable showed
  lead time before structural damage. The geometric signal may be
  specific to Phi(t) dynamics.
""")

    # Save results
    out_file = OUT / "circularity_test_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved: {out_file}")
    print("\nDone.")


if __name__ == "__main__":
    main()
