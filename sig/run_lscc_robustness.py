#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Kappa-SIG: LSCC Robustness Tests
==================================
Before publication, stress-test the LSCC finding:
  R1: Seed stability (5 seeds, retrain from scratch)
  R2: Latent dimension sensitivity (4, 8, 16, 32)
  R3: Encoder-only (no classifier head during training)
  R4: Temporal split (train pre-2024, test 2024+)
  R5: Artifact check (shuffled labels, random z baseline)

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
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SENTINEL_DIR = Path(r"C:\Users\ohiod\Projects\Sentinel")
V2_DIR = SENTINEL_DIR / "data" / "v2_analysis"
SIG_DIR = SENTINEL_DIR / "data" / "sig"
OUT_DIR = SIG_DIR / "lscc_robustness"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STATE_COLS = ["Oh", "phi", "eta", "mean_corr", "DEF", "Xi"]
N_STATE = len(STATE_COLS)


# ══════════════════════════════════════════════════════════════════
# AUTOENCODER + DATA
# ══════════════════════════════════════════════════════════════════

class StructuralAutoencoder(nn.Module):
    def __init__(self, input_dim, window, latent_dim=8, use_classifier=True):
        super().__init__()
        flat_dim = input_dim * window
        self.window = window; self.input_dim = input_dim
        self.use_classifier = use_classifier
        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, latent_dim))
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, flat_dim))
        if use_classifier:
            self.classifier = nn.Sequential(
                nn.Linear(latent_dim, 16), nn.ReLU(),
                nn.Linear(16, 1), nn.Sigmoid())
    def forward(self, x):
        flat = x.reshape(x.size(0), -1)
        z = self.encoder(flat)
        recon = self.decoder(z).reshape(x.size(0), self.window, self.input_dim)
        if self.use_classifier:
            prob = self.classifier(z).squeeze(-1)
        else:
            prob = torch.zeros(x.size(0), device=x.device)
        return recon, z, prob


class WindowDataset(Dataset):
    def __init__(self, states_list, labels_list, window=20):
        self.samples = []; self.labels = []
        for states, y in zip(states_list, labels_list):
            for t in range(window, len(states)):
                self.samples.append(states[t-window:t])
                self.labels.append(y[t])
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        return (torch.tensor(self.samples[idx], dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.float32))

def load_all_data(window=20):
    """Load all universes, return per-universe data."""
    universes = {}
    for d in sorted(V2_DIR.iterdir()):
        if not d.is_dir(): continue
        uid = d.name
        f = d / "kappa_v2_state.csv"
        if not f.exists(): continue
        df = pd.read_csv(f, index_col="date", parse_dates=True)
        if any(c not in df.columns for c in STATE_COLS): continue
        states = np.nan_to_num(df[STATE_COLS].values.astype(np.float32))
        c_norm = df["C_norm"].values if "C_norm" in df.columns else np.ones(len(df))

        # CALM-free target
        y_cf = np.zeros(len(df), dtype=np.float32)
        for t in range(len(df)):
            future = c_norm[t+1:t+91]
            if len(future) > 0 and np.any(future < 0.90):
                y_cf[t] = 1.0
        dates = df.index
        universes[uid] = {"states": states, "y_cf": y_cf,
                          "dates": dates, "n": len(df)}
    return universes

def train_and_extract(universes, seed, latent_dim=8, window=20,
                      epochs=60, use_classifier=True):
    """Train autoencoder with given seed, extract z for all universes."""
    torch.manual_seed(seed); np.random.seed(seed)
    all_states = [u["states"] for u in universes.values()]
    all_flat = np.concatenate(all_states)
    scaler = StandardScaler(); scaler.fit(all_flat)
    normed = [scaler.transform(s).astype(np.float32) for s in all_states]
    all_y = [u["y_cf"] for u in universes.values()]
    dataset = WindowDataset(normed, all_y, window)
    loader = DataLoader(dataset, batch_size=256, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    model = StructuralAutoencoder(N_STATE, window, latent_dim, use_classifier).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    recon_fn = nn.MSELoss(); class_fn = nn.BCELoss()
    alpha = 0.5 if use_classifier else 0.0

    for epoch in range(epochs):
        for x_b, y_b in loader:
            x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            recon, z, prob = model(x_b)
            loss = recon_fn(recon, x_b)
            if use_classifier:
                loss = loss + alpha * class_fn(prob, y_b)
            loss.backward(); optimizer.step()

    # Extract z for all universes
    model.eval()
    per_uid = {}
    uids = list(universes.keys())
    for i, uid in enumerate(uids):
        states_n = scaler.transform(universes[uid]["states"]).astype(np.float32)
        n = len(states_n)
        z_list = []
        with torch.no_grad():
            for t in range(window, n):
                x = torch.tensor(states_n[t-window:t]).unsqueeze(0).to(DEVICE)
                _, z, _ = model(x)
                z_list.append(z.cpu().numpy().squeeze())

        z_arr = np.array(z_list)
        y_aligned = universes[uid]["y_cf"][window:][:len(z_arr)]
        per_uid[uid] = {"z": z_arr, "y": y_aligned}
    return per_uid, model

def louo_lscc(per_uid):
    """LOUO with fold-local PCA on z -> PC1 -> logistic regression."""
    uids = sorted(per_uid.keys())
    aucs = []
    for test_uid in uids:
        # Build train/test
        z_train = np.vstack([per_uid[u]["z"] for u in uids if u != test_uid])
        y_train = np.concatenate([per_uid[u]["y"] for u in uids if u != test_uid])
        z_test = per_uid[test_uid]["z"]
        y_test = per_uid[test_uid]["y"]
        if len(set(y_train)) < 2 or len(set(y_test)) < 2 or len(y_test) < 20:
            continue
        # Fold-local PCA
        pca = PCA(n_components=min(8, z_train.shape[1]))
        X_tr = pca.fit_transform(z_train)[:, [0]]
        X_te = pca.transform(z_test)[:, [0]]
        pipe = Pipeline([("s", StandardScaler()),
                         ("c", LogisticRegression(max_iter=5000))])
        pipe.fit(X_tr, y_train)
        yp = pipe.predict_proba(X_te)[:, 1]
        try:
            auc = roc_auc_score(y_test, yp)
            if auc < 0.5: auc = 1 - auc
            aucs.append(auc)
        except: pass
    return float(np.mean(aucs)) if aucs else None, aucs


# ══════════════════════════════════════════════════════════════════
# R1: SEED STABILITY
# ══════════════════════════════════════════════════════════════════

def run_r1(universes):
    """Retrain with 5 different seeds. LSCC should be stable."""
    print("\n" + "=" * 70)
    print("  R1: SEED STABILITY (5 seeds)")
    print("=" * 70)
    seeds = [42, 123, 7, 2026, 999]
    results = []
    for seed in seeds:
        print(f"  Seed {seed}...", end=" ", flush=True)
        per_uid, _ = train_and_extract(universes, seed=seed)
        auc, fold_aucs = louo_lscc(per_uid)
        results.append({"seed": seed, "auc": auc, "std": float(np.std(fold_aucs))})
        print(f"LOUO AUC = {auc:.4f} ± {np.std(fold_aucs):.4f}")
    aucs = [r["auc"] for r in results if r["auc"]]
    mean_a = np.mean(aucs); std_a = np.std(aucs)
    cv = std_a / mean_a if mean_a > 0 else 999
    print(f"\n  Mean AUC across seeds: {mean_a:.4f} ± {std_a:.4f} (CV={cv:.3f})")
    verdict = "STABLE" if cv < 0.05 else "UNSTABLE" if cv > 0.15 else "MODERATE"
    print(f"  Verdict: {verdict}")
    return {"seeds": results, "mean": mean_a, "std": std_a, "cv": cv, "verdict": verdict}


# ══════════════════════════════════════════════════════════════════
# R2: LATENT DIMENSION SENSITIVITY
# ══════════════════════════════════════════════════════════════════

def run_r2(universes):
    """Test latent_dim = 4, 8, 16, 32."""
    print("\n" + "=" * 70)
    print("  R2: LATENT DIMENSION SENSITIVITY")
    print("=" * 70)
    dims = [4, 8, 16, 32]
    results = []
    for dim in dims:
        print(f"  latent_dim={dim}...", end=" ", flush=True)
        per_uid, _ = train_and_extract(universes, seed=42, latent_dim=dim)
        auc, _ = louo_lscc(per_uid)
        results.append({"dim": dim, "auc": auc})
        print(f"LOUO AUC = {auc:.4f}")
    aucs = [r["auc"] for r in results if r["auc"]]
    best = max(results, key=lambda r: r["auc"] or 0)
    spread = max(aucs) - min(aucs) if aucs else 0
    print(f"\n  Best: dim={best['dim']} (AUC={best['auc']:.4f}), spread={spread:.4f}")
    verdict = "ROBUST" if spread < 0.05 else "SENSITIVE" if spread > 0.15 else "MODERATE"
    print(f"  Verdict: {verdict}")
    return {"dims": results, "spread": spread, "verdict": verdict}


# ══════════════════════════════════════════════════════════════════
# R3: ENCODER-ONLY (no classifier head)
# ══════════════════════════════════════════════════════════════════

def run_r3(universes):
    """Train AE with reconstruction loss only — no classifier."""
    print("\n" + "=" * 70)
    print("  R3: ENCODER-ONLY (no classifier head)")
    print("=" * 70)
    print("  Training with classifier...", end=" ", flush=True)
    per_uid_cls, _ = train_and_extract(universes, seed=42, use_classifier=True)
    auc_cls, _ = louo_lscc(per_uid_cls)
    print(f"AUC = {auc_cls:.4f}")
    print("  Training WITHOUT classifier...", end=" ", flush=True)
    per_uid_enc, _ = train_and_extract(universes, seed=42, use_classifier=False)
    auc_enc, _ = louo_lscc(per_uid_enc)
    print(f"AUC = {auc_enc:.4f}")
    delta = auc_enc - auc_cls if auc_enc and auc_cls else 0
    print(f"\n  With classifier:    {auc_cls:.4f}")
    print(f"  Without classifier: {auc_enc:.4f}")
    print(f"  Delta: {delta:+.4f}")

    if auc_enc and auc_enc > 0.70:
        verdict = "STRUCTURE_REAL"
        print("  >>> LSCC exists even without classifier — structure is in the data")
    elif auc_enc and auc_enc > 0.55:
        verdict = "PARTIAL"
        print("  >>> Encoder-only captures some signal, classifier boosts it")
    else:
        verdict = "CLASSIFIER_DEPENDENT"
        print("  >>> LSCC depends on classifier head — may be supervised artifact")
    return {"auc_with_cls": auc_cls, "auc_no_cls": auc_enc,
            "delta": delta, "verdict": verdict}

# ══════════════════════════════════════════════════════════════════
# R4: TEMPORAL SPLIT (train pre-2024, test 2024+)
# ══════════════════════════════════════════════════════════════════

def run_r4(universes, window=20):
    """Train on pre-2024 data, test on 2024+. No universe leakage."""
    print("\n" + "=" * 70)
    print("  R4: TEMPORAL SPLIT (train < 2024, test >= 2024)")
    print("=" * 70)
    cutoff = pd.Timestamp("2024-01-01")

    torch.manual_seed(42); np.random.seed(42)
    # Split each universe temporally
    train_states = []; train_y = []; test_data = {}
    all_flat_train = []
    for uid, u in universes.items():
        dates = u["dates"]
        mask_tr = dates < cutoff
        n_tr = mask_tr.sum()
        if n_tr < window + 30:
            continue
        all_flat_train.append(u["states"][mask_tr])
        train_states.append(u["states"][mask_tr])
        train_y.append(u["y_cf"][mask_tr])
        mask_te = dates >= cutoff
        if mask_te.sum() > window + 20:
            test_data[uid] = {
                "states": u["states"][mask_te],
                "y_cf": u["y_cf"][mask_te]}
    if not all_flat_train or not test_data:
        print("  SKIP: insufficient temporal split")
        return {"verdict": "INSUFFICIENT"}
    scaler = StandardScaler()
    scaler.fit(np.concatenate(all_flat_train))

    normed_tr = [scaler.transform(s).astype(np.float32) for s in train_states]
    dataset = WindowDataset(normed_tr, train_y, window)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    model = StructuralAutoencoder(N_STATE, window, 8, True).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    recon_fn = nn.MSELoss(); class_fn = nn.BCELoss()
    for epoch in range(60):
        for x_b, y_b in loader:
            x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            recon, z, prob = model(x_b)
            loss = recon_fn(recon, x_b) + 0.5 * class_fn(prob, y_b)
            loss.backward(); optimizer.step()
    # Extract z for train (for PCA fit)
    model.eval()
    z_train_all = []
    for states in normed_tr:
        with torch.no_grad():
            for t in range(window, len(states)):
                x = torch.tensor(states[t-window:t]).unsqueeze(0).to(DEVICE)
                _, z, _ = model(x)
                z_train_all.append(z.cpu().numpy().squeeze())
    z_train_all = np.array(z_train_all)
    pca = PCA(n_components=8); pca.fit(z_train_all)

    # Fit classifier on train PC1
    X_tr_pca = pca.transform(z_train_all)[:, [0]]
    y_tr_flat = np.concatenate([y[window:] for y in train_y])[:len(X_tr_pca)]
    pipe = Pipeline([("s", StandardScaler()),
                     ("c", LogisticRegression(max_iter=5000))])
    pipe.fit(X_tr_pca, y_tr_flat)
    # Evaluate on test universes
    test_aucs = []
    for uid, td in test_data.items():
        normed_te = scaler.transform(td["states"]).astype(np.float32)
        z_te = []
        with torch.no_grad():
            for t in range(window, len(normed_te)):
                x = torch.tensor(normed_te[t-window:t]).unsqueeze(0).to(DEVICE)
                _, z, _ = model(x)
                z_te.append(z.cpu().numpy().squeeze())
        if len(z_te) < 20: continue
        z_te = np.array(z_te)
        X_te_pca = pca.transform(z_te)[:, [0]]
        y_te = td["y_cf"][window:][:len(z_te)]
        if len(set(y_te)) < 2: continue
        yp = pipe.predict_proba(X_te_pca)[:, 1]
        auc = roc_auc_score(y_te, yp)
        if auc < 0.5: auc = 1 - auc
        test_aucs.append(auc)
        print(f"  [{uid}] AUC = {auc:.4f}")

    if test_aucs:
        mean_auc = float(np.mean(test_aucs))
        print(f"\n  Temporal split AUC: {mean_auc:.4f} ± {np.std(test_aucs):.4f} ({len(test_aucs)} universes)")
        verdict = "PASS" if mean_auc > 0.70 else "WEAK" if mean_auc > 0.55 else "FAIL"
    else:
        mean_auc = None; verdict = "INSUFFICIENT"
    print(f"  Verdict: {verdict}")
    return {"mean_auc": mean_auc, "n_test": len(test_aucs), "verdict": verdict}

# ══════════════════════════════════════════════════════════════════
# R5: ARTIFACT CHECK (shuffled labels + random z)
# ══════════════════════════════════════════════════════════════════

def run_r5(universes):
    """Two null hypothesis tests."""
    print("\n" + "=" * 70)
    print("  R5: ARTIFACT CHECK")
    print("=" * 70)
    # 5a: Train normally, evaluate with SHUFFLED labels
    print("  5a: Real model, shuffled labels...")
    per_uid, _ = train_and_extract(universes, seed=42)
    rng = np.random.RandomState(42)
    per_uid_shuf = {}
    for uid, d in per_uid.items():
        per_uid_shuf[uid] = {"z": d["z"], "y": rng.permutation(d["y"])}
    auc_shuf, _ = louo_lscc(per_uid_shuf)
    print(f"  Shuffled labels LOUO AUC: {auc_shuf:.4f}" if auc_shuf else "  N/A")

    # 5b: Random z vectors (same shape), real labels
    print("  5b: Random z, real labels...")
    per_uid_rand = {}
    for uid, d in per_uid.items():
        per_uid_rand[uid] = {"z": rng.randn(*d["z"].shape).astype(np.float32),
                             "y": d["y"]}
    auc_rand, _ = louo_lscc(per_uid_rand)
    print(f"  Random z LOUO AUC: {auc_rand:.4f}" if auc_rand else "  N/A")

    # 5c: Real model, real labels (reference)
    auc_real, _ = louo_lscc(per_uid)
    print(f"\n  Reference (real): {auc_real:.4f}")
    print(f"  Shuffled labels:  {auc_shuf:.4f}")
    print(f"  Random z:         {auc_rand:.4f}")
    gap_shuf = (auc_real - auc_shuf) if auc_real and auc_shuf else 0
    gap_rand = (auc_real - auc_rand) if auc_real and auc_rand else 0
    print(f"  Gap (real - shuffled): {gap_shuf:+.4f}")
    print(f"  Gap (real - random z): {gap_rand:+.4f}")
    if gap_shuf > 0.15 and gap_rand > 0.15:
        verdict = "NO_ARTIFACT"
        print("  >>> PASS: large gap between real and null → signal is genuine")
    elif gap_shuf > 0.05:
        verdict = "MARGINAL"
        print("  >>> MARGINAL: some gap but not definitive")
    else:
        verdict = "ARTIFACT_RISK"
        print("  >>> WARNING: gap too small — possible artifact")
    return {"auc_real": auc_real, "auc_shuffled": auc_shuf,
            "auc_random_z": auc_rand, "verdict": verdict}


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  KAPPA-SIG: LSCC Robustness Tests")
    print("  David Ohio | Independent Researcher | April 2026")
    print(f"  Device: {DEVICE}")
    print("=" * 70)
    t0 = time.time()
    universes = load_all_data()
    print(f"  Loaded {len(universes)} universes")
    results = {}
    results["R1"] = run_r1(universes)
    results["R2"] = run_r2(universes)
    results["R3"] = run_r3(universes)
    results["R4"] = run_r4(universes)
    results["R5"] = run_r5(universes)

    total = time.time() - t0
    print("\n" + "=" * 70)
    print("  LSCC ROBUSTNESS VERDICT")
    print("=" * 70)
    verdicts = {k: v.get("verdict", "?") for k, v in results.items()}
    n_pass = 0
    for test, v in verdicts.items():
        ok = v in ("STABLE", "ROBUST", "STRUCTURE_REAL", "PASS", "NO_ARTIFACT")
        weak = v in ("MODERATE", "PARTIAL", "WEAK", "MARGINAL")
        status = "PASS" if ok else "WEAK" if weak else "FAIL"
        if ok: n_pass += 1
        print(f"  {test}: {v} [{status}]")
    print(f"\n  {n_pass}/5 tests pass cleanly")
    if n_pass >= 4:
        print("  >>> LSCC is ROBUST — ready for publication")
    elif n_pass >= 3:
        print("  >>> LSCC is moderately robust — publishable with caveats")
    elif n_pass >= 2:
        print("  >>> LSCC needs more investigation")
    else:
        print("  >>> LSCC finding is fragile")
    print(f"\n  Total time: {total:.0f}s ({total/60:.1f}m)")
    out = OUT_DIR / "lscc_robustness.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results: {out}")
    print("=" * 70)

if __name__ == "__main__":
    main()
