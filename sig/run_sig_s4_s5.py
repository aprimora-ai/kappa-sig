#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Kappa-SIG: S4 (Neural ODE) + S5 (Autoencoder) — Learned Methods
=================================================================
S4: Train Neural ODE on state trajectories, upsample short windows,
    compute RQA on virtual trajectory. Tests if crystallized dynamics
    can be reconstructed from minimal data.

S5: Train autoencoder on windowed state trajectories, classify regime
    from latent fingerprint. Tests if structural regimes can be
    compressed into a transferable representation.

Both use GPU (RTX 4060 Ti) when available.

David Ohio | odavidohio@gmail.com | Independent Researcher
April 2026
"""
import sys, json, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SENTINEL_DIR = Path(r"C:\Users\ohiod\Projects\Sentinel")
V2_DIR = SENTINEL_DIR / "data" / "v2_analysis"
SIG_DIR = SENTINEL_DIR / "data" / "sig"
OUT_DIR = SIG_DIR / "s4_s5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE = {"latam", "x_commodity_chain"}
# State vector: [Oh, phi, eta, mean_corr, DEF, Xi]
STATE_COLS = ["Oh", "phi", "eta", "mean_corr", "DEF", "Xi"]
N_STATE = len(STATE_COLS)

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# ══════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════

def load_all_trajectories():
    """Load state trajectories + labels from all v2 universes."""
    universes = {}
    for d in sorted(V2_DIR.iterdir()):
        if not d.is_dir() or d.name in EXCLUDE:
            continue
        state_path = d / "kappa_v2_state.csv"
        if not state_path.exists():
            continue
        uid = d.name
        df = pd.read_csv(state_path, index_col="date", parse_dates=True)

        # Check columns exist
        missing = [c for c in STATE_COLS if c not in df.columns]
        if missing:
            continue
        states = df[STATE_COLS].values.astype(np.float32)
        states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)

        # Labels: CALM-free forward target (will C_norm < 0.90 within 90 steps?)
        c_norm = df["C_norm"].values if "C_norm" in df.columns else np.ones(len(df))
        y = np.zeros(len(df), dtype=np.float32)
        for t in range(len(df)):
            future = c_norm[t+1:t+91]
            if len(future) > 0 and np.any(future < 0.90):
                y[t] = 1.0

        # Also CALM-dependent labels
        theta_a = df["theta_A"].values if "theta_A" in df.columns else np.zeros(len(df))
        y_calm = np.zeros(len(df), dtype=np.float32)
        for t in range(len(df)):
            if theta_a[t] >= 2.0:
                y_calm[t] = 1.0
            elif theta_a[t] > 0.1:
                y_calm[t] = 1.0

        universes[uid] = {
            "states": states, "y_calmfree": y,
            "y_calm": y_calm, "n": len(df),
        }
    return universes


# ══════════════════════════════════════════════════════════════════
# S4: NEURAL ODE — Temporal Super-Resolution
# ══════════════════════════════════════════════════════════════════

class ODEFunc(nn.Module):
    """Neural network f_theta: R^6 -> R^6 for dS/dt = f(S)."""
    def __init__(self, n_state=N_STATE, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_state, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_state),
        )
    def forward(self, t, y):
        return self.net(y)


class TrajectoryDataset(Dataset):
    """Pairs of (S(t), S(t+1)) for ODE training."""
    def __init__(self, all_states):
        self.pairs = []
        for states in all_states:
            for t in range(len(states) - 1):
                self.pairs.append((states[t], states[t+1]))
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        s0, s1 = self.pairs[idx]
        return torch.tensor(s0), torch.tensor(s1)

def train_neural_ode(universes, epochs=50, lr=1e-3, batch_size=256):
    """Train Neural ODE on all universe trajectories."""
    from torchdiffeq import odeint
    print(f"\n  Training Neural ODE on {DEVICE}...")

    # Normalize states
    all_states = [u["states"] for u in universes.values()]
    all_flat = np.concatenate(all_states, axis=0)
    scaler = StandardScaler()
    scaler.fit(all_flat)
    normed = [scaler.transform(s) for s in all_states]

    dataset = TrajectoryDataset(normed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"  Training pairs: {len(dataset)}")

    model = ODEFunc(N_STATE, hidden=64).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    t_span = torch.tensor([0.0, 1.0]).to(DEVICE)

    for epoch in range(epochs):
        total_loss = 0; n_batch = 0
        for s0, s1 in loader:
            s0, s1 = s0.to(DEVICE), s1.to(DEVICE)
            optimizer.zero_grad()
            pred = odeint(model, s0, t_span, method="euler")[1]
            loss = nn.MSELoss()(pred, s1)
            loss.backward()
            optimizer.step()
            total_loss += loss.item(); n_batch += 1
        if (epoch + 1) % 10 == 0:
            avg = total_loss / max(n_batch, 1)
            print(f"    Epoch {epoch+1}/{epochs}: loss={avg:.6f}")

    return model, scaler


def upsample_trajectory(model, scaler, states_short, factor=10):
    """Upsample a short trajectory using the trained ODE model."""
    from torchdiffeq import odeint
    normed = scaler.transform(states_short)
    virtual = [normed[0]]
    t_span = torch.tensor([0.0, 1.0 / factor]).to(DEVICE)

    with torch.no_grad():
        for t in range(len(normed) - 1):
            s = torch.tensor(normed[t], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            for sub in range(factor):
                s_next = odeint(model, s, t_span, method="euler")[1]
                virtual.append(s_next.cpu().numpy().squeeze())
                s = s_next

    virtual = np.array(virtual)
    return scaler.inverse_transform(virtual)


def compute_rqa_simple(series, window=60, epsilon_q=0.10):
    """Simplified RQA: returns DET for a scalar series."""
    n = len(series)
    if n < window:
        return 0.0
    # Use last 'window' points
    seg = series[-window:]
    tau = 1; m = 3
    max_idx = len(seg) - (m - 1) * tau
    if max_idx < 5:
        return 0.0
    embedded = np.array([seg[i:i + (m - 1) * tau + 1:tau] for i in range(max_idx)])
    from scipy.spatial.distance import pdist, squareform
    dists = squareform(pdist(embedded))
    ds_flat = dists[np.triu_indices(len(dists), 1)]
    if len(ds_flat) == 0:
        return 0.0
    eps = max(np.quantile(ds_flat, epsilon_q), 1e-12)
    R = (dists <= eps).astype(int)
    np.fill_diagonal(R, 0)
    total = R.sum()
    if total < 2:
        return 0.0
    # DET: fraction in diagonal lines >= 2
    det_count = 0
    nn = len(R)
    for k in range(1, nn):
        diag = np.diag(R, k)
        run = 0
        for v in diag:
            if v: run += 1
            else:
                if run >= 2: det_count += run
                run = 0
        if run >= 2: det_count += run
    return det_count / max(total / 2, 1)


def evaluate_s4(model, scaler, universes, W_short=20, factor=10):
    """Evaluate S4: upsample short windows, compute RQA on virtual."""
    print(f"\n  Evaluating S4 (W={W_short}, factor={factor}x)...")
    all_det_virtual = []
    all_y = []
    all_usi = []
    n_eval = 0

    for uid, data in universes.items():
        states = data["states"]
        y = data["y_calmfree"]
        n = len(states)

        for t in range(W_short + 60, n, 20):  # Sample every 20 steps
            short = states[t - W_short:t]
            try:
                virtual = upsample_trajectory(model, scaler, short, factor)
                # RQA on phi column of virtual trajectory
                phi_virtual = virtual[:, 1]  # phi is column 1
                det = compute_rqa_simple(phi_virtual, window=60)
                all_det_virtual.append(det)
                all_y.append(y[t])
                n_eval += 1
            except Exception:
                continue

        if n_eval > 0 and n_eval % 200 == 0:
            print(f"    Processed {n_eval} windows...")

    all_det = np.array(all_det_virtual)
    all_y = np.array(all_y)
    print(f"  Evaluated {n_eval} windows")

    if len(set(all_y)) < 2 or n_eval < 20:
        print("  SKIP: insufficient data for evaluation")
        return {"auc": None}

    try:
        auc = roc_auc_score(all_y, all_det)
        if auc < 0.5: auc = 1 - auc
    except:
        auc = None

    result = {
        "auc_calmfree": float(auc) if auc else None,
        "n_eval": n_eval,
        "det_mean": float(np.mean(all_det)),
        "det_std": float(np.std(all_det)),
        "W_short": W_short,
        "factor": factor,
    }
    print(f"  S4 Result: AUC={auc:.4f}" if auc else "  S4 Result: AUC=N/A")
    return result


# ══════════════════════════════════════════════════════════════════
# S5: AUTOENCODER — Structural Fingerprint
# ══════════════════════════════════════════════════════════════════

class StructuralAutoencoder(nn.Module):
    """Encoder-decoder with regime classifier on latent space."""
    def __init__(self, input_dim, window, latent_dim=8):
        super().__init__()
        flat_dim = input_dim * window
        self.window = window
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, flat_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, window, input_dim) -> flatten
        flat = x.reshape(x.size(0), -1)
        z = self.encoder(flat)
        recon = self.decoder(z).reshape(x.size(0), self.window, self.input_dim)
        regime_prob = self.classifier(z).squeeze(-1)
        return recon, z, regime_prob


class WindowDataset(Dataset):
    """Windowed state trajectories with labels."""
    def __init__(self, all_states, all_labels, window=20):
        self.samples = []
        self.labels = []
        for states, y in zip(all_states, all_labels):
            n = len(states)
            for t in range(window, n):
                self.samples.append(states[t-window:t])
                self.labels.append(y[t])
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        return (torch.tensor(self.samples[idx], dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.float32))


def train_autoencoder(universes, window=20, latent_dim=8, epochs=60,
                      lr=1e-3, batch_size=256, label_mode="calmfree"):
    """Train S5 autoencoder with joint reconstruction + classification loss."""
    print(f"\n  Training Autoencoder on {DEVICE} (W={window}, latent={latent_dim})...")

    # Normalize
    all_states = [u["states"] for u in universes.values()]
    all_flat = np.concatenate(all_states, axis=0)
    scaler = StandardScaler()
    scaler.fit(all_flat)
    normed = [scaler.transform(s).astype(np.float32) for s in all_states]

    label_key = "y_calmfree" if label_mode == "calmfree" else "y_calm"
    all_labels = [u[label_key] for u in universes.values()]

    dataset = WindowDataset(normed, all_labels, window)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"  Training samples: {len(dataset)}")


    model = StructuralAutoencoder(N_STATE, window, latent_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    recon_loss_fn = nn.MSELoss()
    class_loss_fn = nn.BCELoss()
    alpha_class = 0.5  # weight for classification vs reconstruction

    for epoch in range(epochs):
        total_recon = 0; total_class = 0; n_batch = 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            optimizer.zero_grad()
            recon, z, prob = model(x_batch)
            l_recon = recon_loss_fn(recon, x_batch)
            l_class = class_loss_fn(prob, y_batch)
            loss = l_recon + alpha_class * l_class
            loss.backward()
            optimizer.step()
            total_recon += l_recon.item()
            total_class += l_class.item()
            n_batch += 1


        if (epoch + 1) % 10 == 0:
            avg_r = total_recon / max(n_batch, 1)
            avg_c = total_class / max(n_batch, 1)
            print(f"    Epoch {epoch+1}/{epochs}: recon={avg_r:.6f} class={avg_c:.4f}")

    return model, scaler


def evaluate_s5(universes, window=20, latent_dim=8, epochs=60,
                batch_size=256, label_mode="calmfree"):
    """Evaluate S5 with LOUO cross-validation."""
    print(f"\n  Evaluating S5 (LOUO, W={window}, latent={latent_dim})...")
    label_key = "y_calmfree" if label_mode == "calmfree" else "y_calm"

    uids = sorted(universes.keys())
    louo_aucs = []

    for test_uid in uids:
        # Train on all except test
        train_states = []
        train_labels = []
        for uid in uids:
            if uid == test_uid:
                continue
            train_states.append(universes[uid]["states"])
            train_labels.append(universes[uid][label_key])


        # Normalize using training data only
        all_flat = np.concatenate(train_states, axis=0)
        scaler = StandardScaler()
        scaler.fit(all_flat)
        normed_train = [scaler.transform(s).astype(np.float32)
                        for s in train_states]

        dataset = WindowDataset(normed_train, train_labels, window)
        if len(dataset) < 100:
            continue
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Train
        model = StructuralAutoencoder(N_STATE, window, latent_dim).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        recon_fn = nn.MSELoss()
        class_fn = nn.BCELoss()


        for epoch in range(epochs):
            for x_b, y_b in loader:
                x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                optimizer.zero_grad()
                recon, z, prob = model(x_b)
                loss = recon_fn(recon, x_b) + 0.5 * class_fn(prob, y_b)
                loss.backward()
                optimizer.step()

        # Test on held-out universe
        test_states = universes[test_uid]["states"]
        test_y = universes[test_uid][label_key]
        normed_test = scaler.transform(test_states).astype(np.float32)


        # Build test windows
        test_x = []
        test_yt = []
        for t in range(window, len(normed_test)):
            test_x.append(normed_test[t-window:t])
            test_yt.append(test_y[t])
        if len(test_x) < 20 or len(set(test_yt)) < 2:
            continue

        test_x = torch.tensor(np.array(test_x), dtype=torch.float32).to(DEVICE)
        test_yt = np.array(test_yt)

        model.eval()
        with torch.no_grad():
            _, _, probs = model(test_x)
            probs = probs.cpu().numpy()

        try:
            auc = roc_auc_score(test_yt, probs)
            if auc < 0.5: auc = 1 - auc
            louo_aucs.append(auc)
            print(f"    [{test_uid}] AUC={auc:.4f}")
        except:
            pass


    result = {
        "louo_auc_mean": float(np.mean(louo_aucs)) if louo_aucs else None,
        "louo_auc_std": float(np.std(louo_aucs)) if louo_aucs else None,
        "n_folds": len(louo_aucs),
        "window": window,
        "latent_dim": latent_dim,
        "label_mode": label_mode,
    }
    if louo_aucs:
        print(f"\n  S5 LOUO: AUC={result['louo_auc_mean']:.4f} "
              f"+/- {result['louo_auc_std']:.4f} ({len(louo_aucs)} folds)")
    else:
        print("  S5: No valid folds")
    return result


def evaluate_s5_fullfit(model, scaler, universes, window=20,
                        label_mode="calmfree"):
    """Evaluate S5 full-fit AUC (in-sample, for comparison with S1-S3)."""
    print(f"\n  S5 full-fit evaluation...")
    label_key = "y_calmfree" if label_mode == "calmfree" else "y_calm"
    all_probs = []
    all_y = []

    model.eval()
    for uid, data in universes.items():
        normed = scaler.transform(data["states"]).astype(np.float32)
        y = data[label_key]
        for t in range(window, len(normed)):
            x = torch.tensor(normed[t-window:t], dtype=torch.float32)
            x = x.unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                _, z, prob = model(x)
            all_probs.append(prob.cpu().item())
            all_y.append(y[t])


    all_probs = np.array(all_probs)
    all_y = np.array(all_y)
    if len(set(all_y)) < 2:
        return {"auc_fullfit": None}
    try:
        auc = roc_auc_score(all_y, all_probs)
        if auc < 0.5: auc = 1 - auc
    except:
        auc = None
    print(f"  S5 full-fit: AUC={auc:.4f}" if auc else "  S5 full-fit: N/A")
    return {"auc_fullfit": float(auc) if auc else None, "n": len(all_y)}


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  KAPPA-SIG: S4 (Neural ODE) + S5 (Autoencoder)")
    print("  David Ohio | Independent Researcher | April 2026")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    t0 = time.time()
    results = {}

    # Load data
    print("\n  Loading trajectories...")
    universes = load_all_trajectories()
    print(f"  Loaded {len(universes)} universes")
    total_steps = sum(u["n"] for u in universes.values())
    print(f"  Total steps: {total_steps}")


    # ── S4: Neural ODE ──
    print("\n" + "=" * 70)
    print("  S4: NEURAL ODE — Temporal Super-Resolution")
    print("=" * 70)

    ode_model, ode_scaler = train_neural_ode(universes, epochs=50, lr=1e-3)
    s4_result = evaluate_s4(ode_model, ode_scaler, universes, W_short=20)
    results["s4_neural_ode"] = s4_result

    # Save ODE model
    ode_path = OUT_DIR / "neural_ode.pt"
    torch.save({"model_state": ode_model.state_dict(),
                "scaler_mean": ode_scaler.mean_,
                "scaler_scale": ode_scaler.scale_}, ode_path)
    print(f"  Model saved: {ode_path}")


    # ── S5: Autoencoder ──
    print("\n" + "=" * 70)
    print("  S5: AUTOENCODER — Structural Fingerprint")
    print("=" * 70)

    # Full-fit first (for comparison with S1-S3)
    ae_model, ae_scaler = train_autoencoder(
        universes, window=20, latent_dim=8, epochs=60,
        label_mode="calmfree")
    s5_fullfit = evaluate_s5_fullfit(
        ae_model, ae_scaler, universes, window=20,
        label_mode="calmfree")
    results["s5_autoencoder_fullfit"] = s5_fullfit

    # Save AE model
    ae_path = OUT_DIR / "autoencoder.pt"
    torch.save({"model_state": ae_model.state_dict(),
                "scaler_mean": ae_scaler.mean_,
                "scaler_scale": ae_scaler.scale_,
                "window": 20, "latent_dim": 8}, ae_path)
    print(f"  Model saved: {ae_path}")


    # LOUO evaluation (the real test of transferability)
    s5_louo_cf = evaluate_s5(universes, window=20, latent_dim=8,
                             epochs=40, label_mode="calmfree")
    results["s5_autoencoder_louo_calmfree"] = s5_louo_cf

    s5_louo_cd = evaluate_s5(universes, window=20, latent_dim=8,
                             epochs=40, label_mode="calm")
    results["s5_autoencoder_louo_calm"] = s5_louo_cd


    # ── SUMMARY ──
    total_time = time.time() - t0
    print("\n" + "=" * 70)
    print("  S4/S5 FINAL SUMMARY")
    print("=" * 70)

    print(f"\n  S4 Neural ODE:")
    s4a = s4_result.get("auc_calmfree")
    print(f"    Virtual RQA AUC (CALM-free): {s4a:.4f}" if s4a else
          "    Virtual RQA AUC: N/A")

    print(f"\n  S5 Autoencoder:")
    s5f = s5_fullfit.get("auc_fullfit")
    print(f"    Full-fit AUC (CALM-free): {s5f:.4f}" if s5f else
          "    Full-fit AUC: N/A")
    s5l = s5_louo_cf.get("louo_auc_mean")
    print(f"    LOUO AUC (CALM-free): {s5l:.4f}" if s5l else
          "    LOUO AUC: N/A")
    s5lcd = s5_louo_cd.get("louo_auc_mean")
    print(f"    LOUO AUC (CALM-dep): {s5lcd:.4f}" if s5lcd else
          "    LOUO AUC (CALM-dep): N/A")


    # Comparison reference (from previous experiments)
    print(f"\n  Reference (S1-S3 from run_sig_experiment.py):")
    print(f"    Best S1 individual: D_MP_KL AUC=0.655 (W=60, CALM-dep)")
    print(f"    Best S1 CALM-free:  SCR AUC=0.739 (W=60)")
    print(f"    Best LOUO:          TCS=0.627, LZ=0.626, D_MP_KL=0.622")
    print(f"    Ensemble (11 feat): AUC=0.647 (full), 0.528 (LOUO)")

    # Verdict
    if s5l and s5l > 0.627:
        print(f"\n  >>> S5 BEATS best S1-S3 LOUO ({s5l:.4f} > 0.627)")
    elif s5l and s5l > 0.55:
        print(f"\n  >>> S5 above chance but below S1-S3 ({s5l:.4f})")
    else:
        print(f"\n  >>> S5 not significant or failed")

    print(f"\n  Total time: {total_time:.0f}s ({total_time/60:.1f}m)")


    # Save results
    out_path = OUT_DIR / "s4_s5_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results: {out_path}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
