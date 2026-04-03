#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Kappa-SIG: LSCC — Latent Structural Crystallization Coordinate
================================================================
Tests for the latent variable hypothesis:
  T1: Regime separation in latent space (PCA/UMAP on z)
  T2: Partial correlation with manual observables
  T3: Incremental value above best analytic signals
  T4: Cross-universe stability of dominant latent direction
  T5: Temporal dynamics of LSCC before damage
  T6: Performance on problematic (CALM-saturated) universes

David Ohio | odavidohio@gmail.com | Independent Researcher
April 2026
"""
import sys, json, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SENTINEL_DIR = Path(r"C:\Users\ohiod\Projects\Sentinel")
V2_DIR = SENTINEL_DIR / "data" / "v2_analysis"
SIG_DIR = SENTINEL_DIR / "data" / "sig"
S4S5_DIR = SIG_DIR / "s4_s5"
OUT_DIR = SIG_DIR / "lscc"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STATE_COLS = ["Oh", "phi", "eta", "mean_corr", "DEF", "Xi"]
N_STATE = len(STATE_COLS)
CALM_SATURATED = {"brazil_sectors", "commodities", "financials",
                  "x_europe_vuln", "x_global_contagion", "latam", "x_commodity_chain"}


# ══════════════════════════════════════════════════════════════════
# AUTOENCODER (same architecture as S5)
# ══════════════════════════════════════════════════════════════════

class StructuralAutoencoder(nn.Module):
    def __init__(self, input_dim, window, latent_dim=8):
        super().__init__()
        flat_dim = input_dim * window
        self.window = window; self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, latent_dim))
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, flat_dim))
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid())
    def forward(self, x):
        flat = x.reshape(x.size(0), -1)
        z = self.encoder(flat)
        recon = self.decoder(z).reshape(x.size(0), self.window, self.input_dim)
        prob = self.classifier(z).squeeze(-1)
        return recon, z, prob


# ══════════════════════════════════════════════════════════════════
# DATA LOADING + LATENT EXTRACTION
# ══════════════════════════════════════════════════════════════════

def load_model_and_scaler():
    """Load trained S5 autoencoder."""
    ckpt = torch.load(S4S5_DIR / "autoencoder.pt", map_location=DEVICE, weights_only=False)
    model = StructuralAutoencoder(N_STATE, ckpt["window"], ckpt["latent_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    scaler = StandardScaler()
    scaler.mean_ = ckpt["scaler_mean"]
    scaler.scale_ = ckpt["scaler_scale"]
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)
    return model, scaler, int(ckpt["window"]), int(ckpt["latent_dim"])


def load_universe(uid):
    """Load state + SIG data for one universe."""
    state_path = V2_DIR / uid / "kappa_v2_state.csv"
    sig_path = SIG_DIR / f"{uid}_sig.npz"
    if not state_path.exists():
        return None
    df = pd.read_csv(state_path, index_col="date", parse_dates=True)
    missing = [c for c in STATE_COLS if c not in df.columns]
    if missing:
        return None
    states = df[STATE_COLS].values.astype(np.float32)
    states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
    c_norm = df["C_norm"].values if "C_norm" in df.columns else np.ones(len(df))
    theta_a = df["theta_A"].values if "theta_A" in df.columns else np.zeros(len(df))


    # CALM-free target
    y_cf = np.zeros(len(df), dtype=np.float32)
    for t in range(len(df)):
        future = c_norm[t+1:t+91]
        if len(future) > 0 and np.any(future < 0.90):
            y_cf[t] = 1.0
    # Regime labels for T1
    labels = []
    for t in range(len(df)):
        if c_norm[t] < 0.90:
            labels.append("DAMAGED")
        elif y_cf[t] > 0:
            labels.append("PRE_DAMAGE")
        else:
            labels.append("NOMINAL")
    # SIG features (if available)
    sig = {}
    if sig_path.exists():
        sd = np.load(sig_path, allow_pickle=True)
        for k in ["scr", "lam1", "d_mp_kl", "lz_norm", "pe1", "tcs", "mean_corr"]:
            if k in sd:
                sig[k] = sd[k]
    return {"states": states, "y_cf": y_cf, "c_norm": c_norm,
            "theta_a": theta_a, "labels": np.array(labels),
            "sig": sig, "n": len(df)}


def extract_latent(model, scaler, states, window):
    """Extract latent z vectors for all valid windows."""
    normed = scaler.transform(states).astype(np.float32)
    n = len(normed)
    z_all = []
    with torch.no_grad():
        for t in range(window, n):
            x = torch.tensor(normed[t-window:t], dtype=torch.float32)
            x = x.unsqueeze(0).to(DEVICE)
            _, z, _ = model(x)
            z_all.append(z.cpu().numpy().squeeze())
    return np.array(z_all)  # shape: (n - window, latent_dim)


# ══════════════════════════════════════════════════════════════════
# TEST 1: REGIME SEPARATION IN LATENT SPACE
# ══════════════════════════════════════════════════════════════════

def run_test_1(all_z, all_labels, all_uids):
    """PCA on z, measure separation between NOMINAL / PRE_DAMAGE / DAMAGED."""
    print("\n" + "=" * 70)
    print("  TEST 1: REGIME SEPARATION IN LATENT SPACE")
    print("=" * 70)
    from sklearn.decomposition import PCA

    Z = np.vstack(all_z)
    L = np.concatenate(all_labels)
    pca = PCA(n_components=min(8, Z.shape[1]))
    Z_pca = pca.fit_transform(Z)

    print(f"\n  PCA explained variance: {pca.explained_variance_ratio_[:4].round(3)}")
    print(f"  PC1: {pca.explained_variance_ratio_[0]:.1%}  PC2: {pca.explained_variance_ratio_[1]:.1%}")


    # Centroid distances per regime
    regimes = ["NOMINAL", "PRE_DAMAGE", "DAMAGED"]
    centroids = {}
    for r in regimes:
        mask = L == r
        if mask.sum() > 0:
            centroids[r] = Z_pca[mask].mean(axis=0)
            print(f"  {r}: n={mask.sum()}, PC1_mean={centroids[r][0]:.3f}, PC2_mean={centroids[r][1]:.3f}")

    # Separation metric: distance between NOMINAL and PRE_DAMAGE centroids
    if "NOMINAL" in centroids and "PRE_DAMAGE" in centroids:
        sep_np = np.linalg.norm(centroids["NOMINAL"][:2] - centroids["PRE_DAMAGE"][:2])
        print(f"\n  NOMINAL → PRE_DAMAGE centroid distance (PC1-2): {sep_np:.4f}")
    else:
        sep_np = 0.0

    if "PRE_DAMAGE" in centroids and "DAMAGED" in centroids:
        sep_pd = np.linalg.norm(centroids["PRE_DAMAGE"][:2] - centroids["DAMAGED"][:2])
        print(f"  PRE_DAMAGE → DAMAGED centroid distance (PC1-2): {sep_pd:.4f}")
    else:
        sep_pd = 0.0


    # Silhouette score
    from sklearn.metrics import silhouette_score
    valid = L != ""
    if len(set(L[valid])) >= 2 and sum(valid) > 100:
        sil = silhouette_score(Z_pca[valid][:5000], L[valid][:5000])  # subsample for speed
        print(f"  Silhouette score: {sil:.4f}")
    else:
        sil = 0.0

    # PC1 as LSCC candidate: AUC for PRE_DAMAGE detection
    y_bin = (L == "PRE_DAMAGE").astype(int)
    if len(set(y_bin)) == 2:
        auc_pc1 = roc_auc_score(y_bin, Z_pca[:, 0])
        if auc_pc1 < 0.5: auc_pc1 = 1 - auc_pc1
        print(f"  PC1 as LSCC → AUC for PRE_DAMAGE: {auc_pc1:.4f}")
    else:
        auc_pc1 = None

    verdict = "PASS" if sep_np > 0.5 and sil > 0.05 else "WEAK" if sep_np > 0.2 else "FAIL"
    print(f"\n  Verdict: {verdict}")
    return {"sep_np": sep_np, "sep_pd": sep_pd, "silhouette": sil,
            "auc_pc1": auc_pc1, "pca_var": pca.explained_variance_ratio_.tolist(),
            "verdict": verdict}


# ══════════════════════════════════════════════════════════════════
# ALIGNMENT HELPER
# ══════════════════════════════════════════════════════════════════

def build_aligned_latent_observables(all_z, all_sig, obs_names):
    """Align z vectors and SIG features per-universe, then concatenate."""
    aligned = {name: [] for name in obs_names}
    aligned_z = []
    for z_u, sig_u in zip(all_z, all_sig):
        n_z = len(z_u)
        if n_z == 0:
            continue
        aligned_z.append(z_u)
        for name in obs_names:
            vals = sig_u.get(name, None)
            if vals is None:
                aligned[name].append(np.full(n_z, np.nan))
                continue
            vals = np.asarray(vals, dtype=float).reshape(-1)
            if len(vals) >= n_z:
                aligned[name].append(vals[:n_z])
            else:
                pad = np.full(n_z - len(vals), np.nan)
                aligned[name].append(np.concatenate([vals, pad]))
    Z = np.vstack(aligned_z)
    obs = {name: np.concatenate(parts) for name, parts in aligned.items()}
    return Z, obs


# ══════════════════════════════════════════════════════════════════
# TEST 2: PARTIAL CORRELATION WITH MANUAL OBSERVABLES
# ══════════════════════════════════════════════════════════════════

def run_test_2(all_z, all_sig, all_labels):
    """Does LSCC correlate partially with SCR, LZ, TCS but not collapse into any?"""
    print("\n" + "=" * 70)
    print("  TEST 2: PARTIAL CORRELATION WITH OBSERVABLES")
    print("=" * 70)
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression

    obs_names = ["scr", "lam1", "d_mp_kl", "lz_norm", "pe1", "tcs"]
    Z, obs_vectors = build_aligned_latent_observables(all_z, all_sig, obs_names)

    pca = PCA(n_components=min(8, Z.shape[1]))
    Z_pca = pca.fit_transform(Z)
    lscc = Z_pca[:, 0]

    print(f"\n  Aligned: {len(lscc)} vectors, {len(obs_vectors)} observables")
    for name, vals in obs_vectors.items():
        n_valid = np.sum(np.isfinite(vals))
        print(f"    {name}: {n_valid} valid / {len(vals)} total")

    print(f"\n  LSCC (PC1) correlations with observables:")
    print(f"  {'Observable':<15s} {'Pearson r':>10s} {'|r|':>8s}")
    print(f"  {'-'*35}")
    correlations = {}
    usable_cols = []
    usable_names = []
    for name in obs_names:
        vals = obs_vectors[name]
        valid = np.isfinite(vals) & np.isfinite(lscc)
        if valid.sum() < 100:
            continue
        r = float(np.corrcoef(lscc[valid], vals[valid])[0, 1])
        correlations[name] = r
        print(f"  {name:<15s} {r:+10.4f} {abs(r):8.4f}")
        usable_cols.append(vals)
        usable_names.append(name)

    r2 = None
    if usable_cols:
        X_obs = np.column_stack(usable_cols)
        valid = np.all(np.isfinite(X_obs), axis=1) & np.isfinite(lscc)
        if valid.sum() > 200:
            reg = LinearRegression().fit(X_obs[valid], lscc[valid])
            r2 = float(reg.score(X_obs[valid], lscc[valid]))
            print(f"\n  Linear R² (observables → LSCC): {r2:.4f}")
            print(f"  Residual variance: {1 - r2:.4f}")
            if r2 > 0.90:
                print("  >>> WARNING: LSCC nearly collapses into linear combo")
            elif r2 > 0.50:
                print("  >>> GOOD: LSCC partially explained but carries extra info")
            else:
                print("  >>> STRONG: LSCC captures substantial non-linear structure")

    max_corr = max(abs(r) for r in correlations.values()) if correlations else 0
    if max_corr == 0 and not correlations:
        verdict = "FAILED_ALIGNMENT"
    elif max_corr < 0.80:
        verdict = "COMPOSITE"
    elif max_corr > 0.95:
        verdict = "COLLAPSED"
    else:
        verdict = "PARTIAL"

    print(f"\n  Max |r| with any single observable: {max_corr:.4f}")
    print(f"  Verdict: {verdict}")
    return {"correlations": correlations, "r2_linear": r2,
            "max_corr": float(max_corr), "verdict": verdict}


# ══════════════════════════════════════════════════════════════════
# TEST 3: INCREMENTAL VALUE ABOVE ANALYTIC SIGNALS
# ══════════════════════════════════════════════════════════════════

def run_test_3(all_z, all_sig, all_y, all_uids_flat):
    """Does adding LSCC improve AUC above SCR/TCS/LZ alone? (fold-local PCA)"""
    print("\n" + "=" * 70)
    print("  TEST 3: INCREMENTAL VALUE (LOUO, fold-local PCA)")
    print("=" * 70)
    from sklearn.decomposition import PCA

    obs_names = ["scr", "lam1", "d_mp_kl", "lz_norm", "tcs"]

    # Build per-row table with uid, y, z, and aligned observables
    rows = []
    for z_u, sig_u, y_u, uid_u in zip(all_z, all_sig, all_y, all_uids_flat):
        n = min(len(z_u), len(np.asarray(uid_u)), len(np.asarray(y_u)))
        z_u = z_u[:n]
        y_arr = np.asarray(y_u, dtype=float).reshape(-1)[:n]
        uid_arr = np.asarray(uid_u)[:n]
        obs_u = {}
        for name in obs_names:
            vals = sig_u.get(name, None)
            if vals is None:
                obs_u[name] = np.full(n, np.nan)
            else:
                vals = np.asarray(vals, dtype=float).reshape(-1)
                obs_u[name] = vals[:n] if len(vals) >= n else np.concatenate([vals, np.full(n - len(vals), np.nan)])
        for i in range(n):
            row = {"uid": uid_arr[i], "y": y_arr[i], "z": z_u[i]}
            for name in obs_names:
                row[name] = obs_u[name][i]
            rows.append(row)

    uids = sorted({r["uid"] for r in rows})
    print(f"  Total rows: {len(rows)}, universes: {len(uids)}")

    pca_holder = {"pca": None}
    results = {}

    def eval_model(build_X_fn, model_name):
        aucs = []
        for test_uid in uids:
            train = [r for r in rows if r["uid"] != test_uid]
            test = [r for r in rows if r["uid"] == test_uid]
            if len(train) < 100 or len(test) < 20:
                continue
            X_tr, y_tr = build_X_fn(train, fit=True)
            X_te, y_te = build_X_fn(test, fit=False)
            v_tr = np.all(np.isfinite(X_tr), axis=1) & np.isfinite(y_tr)
            v_te = np.all(np.isfinite(X_te), axis=1) & np.isfinite(y_te)
            X_tr, y_tr = X_tr[v_tr], y_tr[v_tr]
            X_te, y_te = X_te[v_te], y_te[v_te]
            if len(set(y_tr)) < 2 or len(set(y_te)) < 2:
                continue
            pipe = Pipeline([("s", StandardScaler()),
                             ("c", LogisticRegression(max_iter=5000, C=1.0))])
            pipe.fit(X_tr, y_tr)
            yp = pipe.predict_proba(X_te)[:, 1]
            try:
                auc = roc_auc_score(y_te, yp)
                if auc < 0.5: auc = 1 - auc
                aucs.append(auc)
            except: pass
        if aucs:
            results[model_name] = {"auc": float(np.mean(aucs)),
                                   "std": float(np.std(aucs)), "n": len(aucs)}

    def build_lscc(subset, fit):
        Z = np.vstack([r["z"] for r in subset])
        y = np.array([r["y"] for r in subset])
        if fit:
            pca = PCA(n_components=min(8, Z.shape[1]))
            Zp = pca.fit_transform(Z)
            pca_holder["pca"] = pca
        else:
            Zp = pca_holder["pca"].transform(Z)
        return Zp[:, [0]], y

    def build_obs(name):
        def fn(subset, fit):
            X = np.array([[r[name]] for r in subset], dtype=float)
            y = np.array([r["y"] for r in subset])
            return X, y
        return fn

    def build_obs_plus_lscc(name):
        def fn(subset, fit):
            Z = np.vstack([r["z"] for r in subset])
            y = np.array([r["y"] for r in subset])
            if fit:
                pca = PCA(n_components=min(8, Z.shape[1]))
                Zp = pca.fit_transform(Z)
                pca_holder["pca"] = pca
            else:
                Zp = pca_holder["pca"].transform(Z)
            obs = np.array([[r[name]] for r in subset], dtype=float)
            return np.hstack([obs, Zp[:, [0]]]), y
        return fn

    def build_all_analytic(subset, fit):
        X = np.array([[r[n] for n in obs_names] for r in subset], dtype=float)
        y = np.array([r["y"] for r in subset])
        return X, y

    def build_all_plus_lscc(subset, fit):
        Z = np.vstack([r["z"] for r in subset])
        y = np.array([r["y"] for r in subset])
        if fit:
            pca = PCA(n_components=min(8, Z.shape[1]))
            Zp = pca.fit_transform(Z)
            pca_holder["pca"] = pca
        else:
            Zp = pca_holder["pca"].transform(Z)
        obs = np.array([[r[n] for n in obs_names] for r in subset], dtype=float)
        return np.hstack([obs, Zp[:, [0]]]), y

    eval_model(build_lscc, "LSCC_only")
    for name in obs_names:
        eval_model(build_obs(name), f"{name}_only")
        eval_model(build_obs_plus_lscc(name), f"{name}+LSCC")
    eval_model(build_all_analytic, "all_analytic")
    eval_model(build_all_plus_lscc, "all_analytic+LSCC")

    print(f"\n  {'Model':<25s} {'LOUO AUC':>10s} {'±std':>8s} {'Folds':>6s}")
    print(f"  {'-'*52}")
    for k, v in sorted(results.items(), key=lambda kv: -kv[1]["auc"]):
        print(f"  {k:<25s} {v['auc']:10.4f} {v['std']:8.4f} {v['n']:6d}")

    if "all_analytic" in results and "all_analytic+LSCC" in results:
        delta = results["all_analytic+LSCC"]["auc"] - results["all_analytic"]["auc"]
        print(f"\n  Incremental value of LSCC above all analytics: {delta:+.4f}")
        if delta > 0.02:
            print("  >>> LSCC adds substantial value")
        elif delta > 0.005:
            print("  >>> LSCC adds marginal value")
        else:
            print("  >>> LSCC adds no value above analytics")

    return results


# ══════════════════════════════════════════════════════════════════
# TEST 4: CROSS-UNIVERSE STABILITY OF DOMINANT DIRECTION
# ══════════════════════════════════════════════════════════════════

def run_test_4(per_universe_z):
    """Does the same latent direction matter across universes?"""
    print("\n" + "=" * 70)
    print("  TEST 4: CROSS-UNIVERSE STABILITY")
    print("=" * 70)
    from sklearn.decomposition import PCA

    # Per-universe PCA: extract PC1 direction
    pc1_directions = {}
    for uid, z_data in per_universe_z.items():
        z = z_data["z"]
        if len(z) < 50:
            continue
        pca = PCA(n_components=min(4, z.shape[1]))
        pca.fit(z)
        pc1_directions[uid] = pca.components_[0]  # first PC direction


    # Cosine similarity matrix between PC1 directions
    uids = sorted(pc1_directions.keys())
    n = len(uids)
    cos_sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d1, d2 = pc1_directions[uids[i]], pc1_directions[uids[j]]
            cos_sim[i, j] = abs(np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-12))

    # Average off-diagonal similarity
    mask = ~np.eye(n, dtype=bool)
    mean_cos = float(cos_sim[mask].mean())
    min_cos = float(cos_sim[mask].min())
    print(f"\n  {n} universes with PC1 directions computed")
    print(f"  Mean |cos(PC1_i, PC1_j)|: {mean_cos:.4f}")
    print(f"  Min  |cos(PC1_i, PC1_j)|: {min_cos:.4f}")

    if mean_cos > 0.70:
        verdict = "COORDINATE"
        print("  >>> STRONG: same direction dominates across universes → LSCC is a coordinate")
    elif mean_cos > 0.40:
        verdict = "PARTIAL"
        print("  >>> PARTIAL: some alignment → LSCC is universe-dependent fingerprint")
    else:
        verdict = "FINGERPRINT"
        print("  >>> WEAK: directions diverge → better called fingerprint than coordinate")

    return {"mean_cosine": mean_cos, "min_cosine": min_cos, "n_universes": n, "verdict": verdict}


# ══════════════════════════════════════════════════════════════════
# TEST 5: TEMPORAL DYNAMICS OF LSCC BEFORE DAMAGE
# ══════════════════════════════════════════════════════════════════

def run_test_5(per_universe_z):
    """Does LSCC drift systematically before C_norm drops?"""
    print("\n" + "=" * 70)
    print("  TEST 5: TEMPORAL DYNAMICS BEFORE DAMAGE")
    print("=" * 70)
    from sklearn.decomposition import PCA

    # Global PCA for consistent LSCC direction
    all_z = np.vstack([d["z"] for d in per_universe_z.values()])
    pca = PCA(n_components=min(8, all_z.shape[1]))
    pca.fit(all_z)


    results = {}
    for uid, d in per_universe_z.items():
        z = d["z"]; c_norm = d["c_norm"]; y_cf = d["y_cf"]
        n = len(z)
        if n < 100:
            continue
        # Project onto global PC1 = LSCC
        lscc = pca.transform(z)[:, 0]
        # Find t_S (first C_norm < 0.90)
        t_S = None
        for t in range(n):
            if c_norm[t] < 0.90:
                t_S = t; break
        if t_S is None or t_S < 30:
            continue

        # LSCC in 3 windows: far before, near before, during damage
        w_far = lscc[max(0, t_S-90):max(1, t_S-30)]
        w_near = lscc[max(0, t_S-30):t_S]
        w_during = lscc[t_S:min(n, t_S+30)]
        w_nominal = lscc[:max(1, t_S-90)]

        far_mean = float(np.mean(w_far)) if len(w_far) > 0 else np.nan
        near_mean = float(np.mean(w_near)) if len(w_near) > 0 else np.nan
        during_mean = float(np.mean(w_during)) if len(w_during) > 0 else np.nan
        nom_mean = float(np.mean(w_nominal)) if len(w_nominal) > 0 else np.nan

        # Drift = monotonic increase toward damage?
        drift = near_mean - nom_mean if np.isfinite(near_mean) and np.isfinite(nom_mean) else 0
        accel = near_mean - far_mean if np.isfinite(near_mean) and np.isfinite(far_mean) else 0


        results[uid] = {
            "t_S": t_S, "nom_mean": nom_mean, "far_mean": far_mean,
            "near_mean": near_mean, "during_mean": during_mean,
            "drift": drift, "acceleration": accel,
        }
        drifts_sign = "+" if drift > 0 else "-"
        print(f"  [{uid}] t_S={t_S}  nom={nom_mean:+.3f}  near={near_mean:+.3f}  "
              f"drift={drift:+.3f}{drifts_sign}  accel={accel:+.3f}")

    # Summary
    if results:
        drifts = [r["drift"] for r in results.values() if np.isfinite(r["drift"])]
        pos_frac = np.mean([d > 0 for d in drifts]) if drifts else 0
        mean_drift = np.mean(drifts) if drifts else 0
        print(f"\n  Universes with t_S: {len(results)}")
        print(f"  Mean drift (near - nominal): {mean_drift:+.4f}")
        print(f"  Fraction with positive drift: {pos_frac:.1%}")

        if pos_frac > 0.70:
            print("  >>> LSCC drifts systematically before damage")
            verdict = "DRIFT_CONFIRMED"
        elif pos_frac > 0.50:
            print("  >>> LSCC drift is mixed")
            verdict = "PARTIAL"
        else:
            print("  >>> No systematic drift detected")
            verdict = "NO_DRIFT"
    else:
        verdict = "INSUFFICIENT_DATA"
        print("  No universes with valid t_S")

    return {"per_universe": results, "verdict": verdict}


# ══════════════════════════════════════════════════════════════════
# TEST 6: PERFORMANCE ON CALM-SATURATED UNIVERSES
# ══════════════════════════════════════════════════════════════════

def run_test_6(per_universe_z):
    """Does S5/LSCC help where Layer 2 is blind?"""
    print("\n" + "=" * 70)
    print("  TEST 6: CALM-SATURATED UNIVERSE PERFORMANCE")
    print("=" * 70)
    from sklearn.decomposition import PCA

    # Global PCA
    all_z = np.vstack([d["z"] for d in per_universe_z.values()])
    pca = PCA(n_components=min(8, all_z.shape[1]))
    pca.fit(all_z)


    sat_aucs = []; high_aucs = []
    print(f"\n  {'Universe':<25s} {'CALM':>12s} {'AUC(LSCC)':>10s}")
    print(f"  {'-'*50}")

    for uid, d in per_universe_z.items():
        z = d["z"]; y_cf = d["y_cf"]
        if len(z) < 50 or len(set(y_cf)) < 2:
            continue
        lscc = pca.transform(z)[:, 0]
        try:
            auc = roc_auc_score(y_cf, lscc)
            if auc < 0.5: auc = 1 - auc
        except:
            continue

        is_sat = uid in CALM_SATURATED
        status = "SATURATED" if is_sat else "HIGH"
        if is_sat: sat_aucs.append(auc)
        else: high_aucs.append(auc)
        print(f"  {uid:<25s} {status:>12s} {auc:10.4f}")


    if sat_aucs and high_aucs:
        mean_sat = np.mean(sat_aucs); mean_high = np.mean(high_aucs)
        delta = mean_sat - mean_high
        print(f"\n  CALM_SATURATED (Layer2 blind): mean AUC = {mean_sat:.4f} (n={len(sat_aucs)})")
        print(f"  HIGH (Layer2 working):         mean AUC = {mean_high:.4f} (n={len(high_aucs)})")
        print(f"  Delta: {delta:+.4f}")
        if mean_sat > 0.60:
            print("  >>> LSCC provides signal even where Layer 2 is blind")
            verdict = "LSCC_COVERS_BLIND_SPOT"
        elif mean_sat > 0.55:
            print("  >>> LSCC provides weak signal in saturated universes")
            verdict = "PARTIAL_COVERAGE"
        else:
            print("  >>> LSCC also struggles in saturated universes")
            verdict = "NO_COVERAGE"
    else:
        verdict = "INSUFFICIENT"
        mean_sat = mean_high = delta = None

    return {"sat_aucs": sat_aucs, "high_aucs": high_aucs,
            "mean_sat": mean_sat, "mean_high": mean_high, "verdict": verdict}


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  KAPPA-SIG: LSCC — Latent Structural Crystallization Coordinate")
    print("  David Ohio | Independent Researcher | April 2026")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    t0 = time.time()

    # Load model
    print("\n  Loading S5 autoencoder...")
    model, scaler, window, latent_dim = load_model_and_scaler()
    print(f"  Window={window}, latent_dim={latent_dim}")


    # Load all universes and extract latent vectors
    print("  Loading universes and extracting latent vectors...")
    per_universe_z = {}
    all_z = []; all_labels = []; all_y = []
    all_sig = []; all_uids_flat = []

    for d in sorted(V2_DIR.iterdir()):
        if not d.is_dir():
            continue
        uid = d.name
        data = load_universe(uid)
        if data is None:
            continue
        z = extract_latent(model, scaler, data["states"], window)
        n_z = len(z)
        if n_z < 20:
            continue


        # Align labels/targets with z (offset by window)
        labels_aligned = data["labels"][window:][:n_z]
        y_aligned = data["y_cf"][window:][:n_z]
        c_norm_aligned = data["c_norm"][window:][:n_z]

        # SIG features aligned — take from end to match z offset
        sig_aligned = {}
        for k, v in data["sig"].items():
            v = np.asarray(v, dtype=float).reshape(-1)
            # SIG .npz has its own window offset; align to n_z from the tail
            if len(v) >= n_z:
                sig_aligned[k] = v[-n_z:]
            elif len(v) > 0:
                # Pad front with NaN if SIG is shorter
                pad = np.full(n_z - len(v), np.nan)
                sig_aligned[k] = np.concatenate([pad, v])

        per_universe_z[uid] = {
            "z": z, "labels": labels_aligned, "y_cf": y_aligned,
            "c_norm": c_norm_aligned, "sig": sig_aligned,
        }
        all_z.append(z)
        all_labels.append(labels_aligned)
        all_y.append(y_aligned)
        all_sig.append(sig_aligned)
        all_uids_flat.append(np.full(n_z, uid))
        print(f"    {uid}: {n_z} steps, z.shape={z.shape}")

    print(f"  Total: {len(per_universe_z)} universes, "
          f"{sum(len(d['z']) for d in per_universe_z.values())} latent vectors")


    results = {}

    # T1: Regime separation
    results["T1_regime_separation"] = run_test_1(all_z, all_labels, all_uids_flat)

    # T2: Partial correlation
    results["T2_partial_correlation"] = run_test_2(all_z, all_sig, all_labels)

    # T3: Incremental value
    results["T3_incremental_value"] = run_test_3(all_z, all_sig, all_y, all_uids_flat)

    # T4: Cross-universe stability
    results["T4_cross_stability"] = run_test_4(per_universe_z)

    # T5: Temporal dynamics
    results["T5_temporal_dynamics"] = run_test_5(per_universe_z)

    # T6: CALM-saturated performance
    results["T6_calm_saturated"] = run_test_6(per_universe_z)


    # ── FINAL SUMMARY ──
    total_time = time.time() - t0
    print("\n" + "=" * 70)
    print("  LSCC FINAL VERDICT")
    print("=" * 70)

    verdicts = {k: v.get("verdict", "?") for k, v in results.items()
                if isinstance(v, dict) and "verdict" in v}
    for test, v in verdicts.items():
        status = "PASS" if v in ("PASS", "COMPOSITE", "COORDINATE",
                                  "DRIFT_CONFIRMED", "LSCC_COVERS_BLIND_SPOT") else \
                 "WEAK" if v in ("WEAK", "PARTIAL", "PARTIAL_COVERAGE") else "FAIL"
        print(f"  {test}: {v} [{status}]")

    n_pass = sum(1 for v in verdicts.values()
                 if v in ("PASS", "COMPOSITE", "COORDINATE",
                          "DRIFT_CONFIRMED", "LSCC_COVERS_BLIND_SPOT"))
    n_total = len(verdicts)


    if n_pass >= 4:
        overall = "STRONG EVIDENCE"
        print(f"\n  >>> LSCC has strong empirical support ({n_pass}/{n_total} tests pass)")
        print("  >>> The latent structural crystallization coordinate is a candidate")
        print("  >>> composite observable for pre-damage vulnerability detection.")
    elif n_pass >= 2:
        overall = "MODERATE EVIDENCE"
        print(f"\n  >>> LSCC has moderate support ({n_pass}/{n_total} tests pass)")
        print("  >>> Worth investigating further but not yet definitive.")
    else:
        overall = "WEAK EVIDENCE"
        print(f"\n  >>> LSCC evidence is weak ({n_pass}/{n_total} tests pass)")

    print(f"\n  Total time: {total_time:.0f}s ({total_time/60:.1f}m)")

    # Save
    out_path = OUT_DIR / "lscc_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"verdicts": verdicts, "overall": overall,
                   "results": {k: {kk: (float(vv) if isinstance(vv, (float, np.floating)) else vv)
                                   for kk, vv in v.items() if not isinstance(vv, (np.ndarray, list, dict))}
                               for k, v in results.items() if isinstance(v, dict)}},
                  f, indent=2, default=str)
    print(f"  Results: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
