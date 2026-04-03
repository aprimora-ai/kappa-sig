#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kappa Method v2 — engine_v5.py
================================
Predictive Structural Dynamics via Capacity, Attractor Geometry, and Hazard.

Operates as an additional analysis layer on top of engine_v4 output.
Consumes kappa_v4_state.csv + kappa_v4_viscosity.csv and produces v2 quantities.

Three layers:
  Layer 1 (Core — definitional):    C(t), ρ(t), ρ̄(t), T̂_exhaust, κ_F, False Recovery
  Layer 2 (Geometric — theoretical): RQA (DET, LAM, TT), Θ_A(t)  [conditional on data]
  Layer 3 (Prognostic — inferential): h(t), P_collapse, T½, CSD, PCS  [requires calibration]

Author: David Ohio | odavidohio@gmail.com | Independent Researcher
Date: March 2026
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

EPS = 1e-12


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class V2Config:
    # ── Layer 1: Core ─────────────────────────────────────────────────────────
    gamma: float = 0.97              # inherited from v1
    ema_alpha: float = 0.1           # smoothing for ρ̄ (operational convention)
    theta_fr: float = 0.30           # False Recovery: C/Φ* threshold
    t_fr: int = 90                   # False Recovery: lookback window (steps)

    # ── Layer 2: Geometric Extension ──────────────────────────────────────────
    n1_min: int = 60                 # minimum steps for RQA
    n2_min: int = 180                # minimum steps for D₂, λ₁
    rqa_window: int = 60             # rolling window for RQA
    rqa_epsilon_q: float = 0.10      # recurrence threshold as quantile of distances
    rqa_min_line: int = 2            # minimum diagonal/vertical line length

    # ── Layer 3: Prognostic Module ────────────────────────────────────────────
    # β coefficients — PLACEHOLDER: to be calibrated from historical corpus
    # These are initial estimates; §7.3 validation required before deployment
    beta_0: float = -6.0             # baseline log-odds (low background risk)
    beta_F: float = 1.0              # log-capacity ratio sensitivity
    beta_rho: float = 0.5            # depletion rate sensitivity
    beta_eta: float = 0.3            # rigidity sensitivity
    beta_A: float = 0.5              # attractor transition sensitivity

    # CSD parameters
    csd_window: int = 40             # rolling window for CSD indicators

    # PCS thresholds
    kf_threshold: float = 2.0        # κ_F threshold for PCS
    theta_a_threshold: float = 1.0   # Θ_A threshold for PCS
    csd_threshold: float = 2.0       # CSD threshold for PCS


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — STRUCTURAL CAPACITY (CORE)
# Epistemological status: Definitional (C, ρ, κ_F) / Convention-bound (ρ̄, T̂, FR)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_capacity(phi: np.ndarray, phi_star: float) -> np.ndarray:
    """
    C(t) = Φ* − Φ(t)   [strictly definitional]

    C(t) ∈ (−∞, Φ*]:
      C > 0: remaining capacity
      C = 0: exhaustion
      C < 0: post-irreversibility regime
    """
    return phi_star - phi


def compute_capacity_dynamics(C: np.ndarray, phi: np.ndarray,
                               Oh: np.ndarray, Oh_pre: float,
                               delta: float, gamma: float
                               ) -> Dict[str, np.ndarray]:
    """
    Capacity dynamics [strictly definitional]:

      C_t = γ·C_{t-1} + (1−γ)·Φ* − D_t
      ρ_t = D_t − (1−γ)·Φ_{t-1}
      ΔC_t = (1−γ)·Φ_{t-1} − D_t

    Returns dict with ρ (depletion rate) and D (damage drive).
    """
    D = np.maximum(Oh - Oh_pre - delta, 0.0)
    n = len(phi)
    rho = np.zeros(n)
    for i in range(1, n):
        rho[i] = D[i] - (1.0 - gamma) * phi[i - 1]
    rho[0] = D[0]  # no prior state
    return {"rho": rho, "D": D}


def compute_rho_bar(rho: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """
    ρ̄(t) = EMA of ρ(t)   [operational convention-bound, α=0.1 fixed]
    """
    rho_bar = np.zeros_like(rho)
    rho_bar[0] = rho[0]
    for i in range(1, len(rho)):
        rho_bar[i] = alpha * rho[i] + (1.0 - alpha) * rho_bar[i - 1]
    return rho_bar


def compute_t_exhaust(C: np.ndarray, rho_bar: np.ndarray) -> np.ndarray:
    """
    T̂_exhaust(t) = C₊(t) / ρ̄(t)   [convention-bound]

    Only defined when ρ̄(t) > 0 (active depletion).
    Returns ∞ when system is recovering.
    """
    C_plus = np.maximum(C, EPS)
    t_exh = np.full_like(C, np.inf)
    active = rho_bar > EPS
    t_exh[active] = C_plus[active] / rho_bar[active]
    return t_exh


def compute_kappa_f(C: np.ndarray, phi_star: float) -> np.ndarray:
    """
    κ_F(t) = ln(Φ* / C₊(t))   [strictly definitional]

    Minimal fragility index:
      κ_F = 0 when C = Φ* (pristine)
      κ_F → ∞ as C → 0⁺ (approaching exhaustion)
    """
    C_plus = np.maximum(C, EPS)
    return np.log(phi_star / C_plus)


def detect_false_recovery(Oh: np.ndarray, C: np.ndarray, phi_star: float,
                           regimes: np.ndarray, cfg: V2Config
                           ) -> np.ndarray:
    """
    False Recovery detector [operational convention-bound]:

    Fires when:
      1. Oh(t) < 1.0  (instantaneous pressure below Katashi threshold)
      2. C(t)/Φ* < θ_FR  (structural capacity critically low)
      3. System was in Katashi within preceding T_FR steps

    Returns boolean array.
    """
    n = len(Oh)
    fr = np.zeros(n, dtype=bool)

    for i in range(n):
        # Condition 1: Oh below Katashi threshold
        if Oh[i] >= 1.0:
            continue

        # Condition 2: capacity below fraction
        if C[i] / phi_star >= cfg.theta_fr:
            continue

        # Condition 3: was in Katashi recently
        lookback_start = max(0, i - cfg.t_fr)
        was_katashi = False
        for j in range(lookback_start, i):
            if regimes[j] == "Katashi":
                was_katashi = True
                break

        if was_katashi:
            fr[i] = True

    return fr


def run_layer1(state_df: pd.DataFrame, phi_star: float, Oh_pre: float,
               cfg: V2Config) -> pd.DataFrame:
    """
    Run complete Layer 1 analysis. Returns DataFrame with all Core quantities.
    """
    Oh = state_df["Oh"].values
    phi = state_df["phi"].values
    eta = state_df["eta"].values
    regimes = state_df["regime"].values

    # Strictly definitional
    C = compute_capacity(phi, phi_star)
    dynamics = compute_capacity_dynamics(C, phi, Oh, Oh_pre, cfg.gamma * 0.08, cfg.gamma)
    rho = dynamics["rho"]
    kf = compute_kappa_f(C, phi_star)

    # Convention-bound
    rho_bar = compute_rho_bar(rho, cfg.ema_alpha)
    t_exhaust = compute_t_exhaust(C, rho_bar)
    false_rec = detect_false_recovery(Oh, C, phi_star, regimes, cfg)

    # Build output
    v2 = pd.DataFrame(index=state_df.index)
    v2["C"] = C
    v2["C_norm"] = C / phi_star  # C/Φ* ∈ (−∞, 1]
    v2["rho"] = rho
    v2["rho_bar"] = rho_bar
    v2["kappa_F"] = kf
    v2["T_exhaust"] = t_exhaust
    v2["false_recovery"] = false_rec
    v2["D"] = dynamics["D"]

    return v2


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — ATTRACTOR GEOMETRY (GEOMETRIC EXTENSION)
# Epistemological status: Theoretical (proposition) / Implementation-dependent (estimators)
# ═══════════════════════════════════════════════════════════════════════════════

def _recurrence_matrix(Y: np.ndarray, epsilon: float) -> np.ndarray:
    """Compute recurrence matrix from embedded trajectory."""
    n = Y.shape[0]
    R = np.zeros((n, n), dtype=bool)
    for i in range(n):
        dists = np.linalg.norm(Y - Y[i], axis=1)
        R[i] = dists <= epsilon
    return R


def _rqa_from_matrix(R: np.ndarray, min_line: int = 2) -> Dict[str, float]:
    """
    Extract RQA measures from recurrence matrix.

    Returns: RR, DET, LAM, TT
    """
    n = R.shape[0]
    total_points = n * n
    recurrence_points = int(R.sum())
    RR = recurrence_points / total_points if total_points > 0 else 0.0

    # Diagonal lines (DET)
    diag_lengths = []
    for k in range(-n + 1, n):
        diag = np.diag(R, k)
        length = 0
        for val in diag:
            if val:
                length += 1
            else:
                if length >= min_line:
                    diag_lengths.append(length)
                length = 0
        if length >= min_line:
            diag_lengths.append(length)

    diag_points = sum(diag_lengths)
    DET = diag_points / recurrence_points if recurrence_points > 0 else 0.0

    # Vertical lines (LAM, TT)
    vert_lengths = []
    for j in range(n):
        col = R[:, j]
        length = 0
        for val in col:
            if val:
                length += 1
            else:
                if length >= min_line:
                    vert_lengths.append(length)
                length = 0
        if length >= min_line:
            vert_lengths.append(length)

    vert_points = sum(vert_lengths)
    LAM = vert_points / recurrence_points if recurrence_points > 0 else 0.0
    TT = float(np.mean(vert_lengths)) if vert_lengths else 0.0

    return {"RR": RR, "DET": DET, "LAM": LAM, "TT": TT}


def _delay_embedding(x: np.ndarray, tau: int, m: int) -> np.ndarray:
    """
    Construct delay embedding of scalar time series.

    Y(t) = [x(t), x(t-τ), x(t-2τ), ..., x(t-(m-1)τ)]
    """
    n = len(x)
    n_vectors = n - (m - 1) * tau
    if n_vectors <= 0:
        return np.array([]).reshape(0, m)
    Y = np.zeros((n_vectors, m))
    for i in range(m):
        Y[:, i] = x[i * tau: i * tau + n_vectors]
    return Y


def _estimate_tau(x: np.ndarray, max_tau: int = 20) -> int:
    """
    Estimate embedding delay via first minimum of automutual information.
    Simplified: uses autocorrelation first zero-crossing as proxy.
    """
    x_centered = x - np.mean(x)
    n = len(x_centered)
    for tau in range(1, min(max_tau, n // 2)):
        acf = np.corrcoef(x_centered[:-tau], x_centered[tau:])[0, 1]
        if acf <= 0:
            return tau
    return min(max_tau, n // 4)


def compute_rqa_rolling(phi: np.ndarray, cfg: V2Config,
                         calm_mask: np.ndarray = None
                         ) -> Optional[pd.DataFrame]:
    """
    Compute RQA measures on rolling windows of Φ(t) embedding.

    Returns DataFrame with DET, LAM, TT, and CALM-normalized versions.
    Returns None if insufficient data.
    """
    n = len(phi)
    if n < cfg.n1_min:
        return None

    # Estimate embedding parameters
    tau = _estimate_tau(phi)
    m = 3  # conservative default; FNN would refine this

    # Compute RQA on rolling windows
    W = cfg.rqa_window
    results = []

    for t in range(W, n):
        window = phi[t - W:t]
        Y = _delay_embedding(window, tau, m)
        if Y.shape[0] < 10:
            results.append({"DET": np.nan, "LAM": np.nan, "TT": np.nan})
            continue

        # Compute pairwise distances for epsilon
        dists = []
        for i in range(Y.shape[0]):
            for j in range(i + 1, Y.shape[0]):
                dists.append(np.linalg.norm(Y[i] - Y[j]))
        if not dists:
            results.append({"DET": np.nan, "LAM": np.nan, "TT": np.nan})
            continue

        epsilon = np.quantile(dists, cfg.rqa_epsilon_q)
        R = _recurrence_matrix(Y, epsilon)
        rqa = _rqa_from_matrix(R, cfg.rqa_min_line)
        results.append(rqa)

    # Pad beginning with NaN
    padding = [{"DET": np.nan, "LAM": np.nan, "TT": np.nan}] * W
    rqa_df = pd.DataFrame(padding + results)

    # CALM normalization
    if calm_mask is not None and calm_mask.sum() >= W:
        for col in ["DET", "LAM", "TT"]:
            calm_vals = rqa_df[col].values[calm_mask]
            calm_vals = calm_vals[np.isfinite(calm_vals)]
            if len(calm_vals) > 5:
                calm_mean = np.mean(calm_vals)
                calm_std = np.std(calm_vals)
                if col in ("DET", "LAM"):
                    # Normalized excess: (val - CALM) / (1 - CALM)
                    denom = max(1.0 - calm_mean, EPS)
                    rqa_df[f"{col}_norm"] = (rqa_df[col] - calm_mean) / denom
                else:
                    # Z-score for TT
                    rqa_df[f"{col}_norm"] = (rqa_df[col] - calm_mean) / max(calm_std, EPS)
            else:
                rqa_df[f"{col}_norm"] = 0.0
    else:
        for col in ["DET", "LAM", "TT"]:
            rqa_df[f"{col}_norm"] = 0.0

    return rqa_df


def compute_theta_a(rqa_df: Optional[pd.DataFrame],
                     w_D: float = 0.40, w_L: float = 0.35, w_T: float = 0.25
                     ) -> np.ndarray:
    """
    Θ_A(t) — Attractor Transition Score.

    RQA-primary form:
      Θ_A = w_D·DET_norm + w_L·LAM_norm + w_T·TT_norm

    Returns zeros if Layer 2 is not applicable.
    """
    if rqa_df is None:
        return np.array([0.0])

    det_n = rqa_df.get("DET_norm", pd.Series(0.0, index=rqa_df.index)).fillna(0).values
    lam_n = rqa_df.get("LAM_norm", pd.Series(0.0, index=rqa_df.index)).fillna(0).values
    tt_n = rqa_df.get("TT_norm", pd.Series(0.0, index=rqa_df.index)).fillna(0).values

    theta = w_D * np.maximum(det_n, 0) + w_L * np.maximum(lam_n, 0) + w_T * np.maximum(tt_n, 0)
    return theta


def run_layer2(state_df: pd.DataFrame, cfg: V2Config,
               calm_mask: np.ndarray = None) -> pd.DataFrame:
    """
    Run Layer 2 analysis. Conditional on data sufficiency.
    """
    phi = state_df["phi"].values
    n = len(phi)

    v2_geo = pd.DataFrame(index=state_df.index)

    if n < cfg.n1_min:
        # Insufficient data — Layer 2 not applied
        v2_geo["theta_A"] = 0.0
        v2_geo["layer2_status"] = "NOT_APPLIED"
        return v2_geo

    rqa_df = compute_rqa_rolling(phi, cfg, calm_mask)

    if rqa_df is not None and len(rqa_df) == n:
        v2_geo["DET"] = rqa_df["DET"].values
        v2_geo["LAM"] = rqa_df["LAM"].values
        v2_geo["TT"] = rqa_df["TT"].values
        v2_geo["theta_A"] = compute_theta_a(rqa_df)
        v2_geo["layer2_status"] = np.where(n >= cfg.n2_min, "FULL", "RQA_ONLY")
    else:
        v2_geo["theta_A"] = 0.0
        v2_geo["layer2_status"] = "FAILED"

    return v2_geo


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — STRUCTURAL HAZARD FUNCTION (PROGNOSTIC MODULE)
# Epistemological status: Inferential — requires calibration
# ═══════════════════════════════════════════════════════════════════════════════

def compute_hazard_discrete(kf: np.ndarray, rho_bar_pos: np.ndarray,
                             eta_norm: np.ndarray, theta_a: np.ndarray,
                             cfg: V2Config) -> np.ndarray:
    """
    Discrete hazard via logit link [inferential]:

      logit(h_t) = β₀ + β_F·F(t) + β_ρ·ρ̄⁺(t) + β_η·η̃(t) + β_A·Θ_A(t)
      h_t = sigmoid(logit)

    Nested models:
      A (Core-only):       β₀ + β_F·F + β_ρ·ρ̄⁺
      B (Core+Rigidity):   A + β_η·η̃
      C (Full):            B + β_A·Θ_A
    """
    logit_h = (cfg.beta_0
               + cfg.beta_F * kf
               + cfg.beta_rho * rho_bar_pos
               + cfg.beta_eta * eta_norm
               + cfg.beta_A * theta_a)

    # Sigmoid with numerical stability
    logit_h = np.clip(logit_h, -20.0, 20.0)
    h = 1.0 / (1.0 + np.exp(-logit_h))
    return h


def compute_collapse_prob(h: np.ndarray, delta: int = 30) -> np.ndarray:
    """
    P_collapse(t, Δ) = 1 − ∏_{j=1}^{Δ} (1 − h_{t+j})   [discrete form]

    Approximation under locally constant hazard:
      P ≈ 1 − (1 − h_t)^Δ
    """
    return 1.0 - np.power(np.maximum(1.0 - h, EPS), delta)


def compute_half_life(h: np.ndarray) -> np.ndarray:
    """
    T_{1/2}(t) ≈ ln(2) / h(t)   for small h.

    Exact discrete: T½ = ln(0.5) / ln(1 − h_t)
    """
    h_safe = np.maximum(h, EPS)
    # Exact discrete form
    t_half = np.log(0.5) / np.log(np.maximum(1.0 - h_safe, EPS))
    return np.maximum(t_half, 0.0)


def compute_csd(C: np.ndarray, window: int = 40,
                calm_mask: np.ndarray = None) -> np.ndarray:
    """
    Critical Slowing Down composite [implementation-dependent]:

      CSD(t) = (α₁(t) − α₁_CALM) / σ_{α₁,CALM} + (σ(t) − σ_CALM) / σ_{σ,CALM}

    Where α₁ = AR(1) coefficient, σ = std of C(t) in rolling window.
    """
    n = len(C)
    ar1 = np.full(n, np.nan)
    sigma = np.full(n, np.nan)

    for t in range(window, n):
        w = C[t - window:t]
        if np.std(w) < EPS:
            ar1[t] = 0.0
            sigma[t] = 0.0
            continue
        # AR(1) = correlation of C(t) with C(t-1)
        ar1[t] = np.corrcoef(w[:-1], w[1:])[0, 1] if len(w) > 2 else 0.0
        sigma[t] = np.std(w)

    # CALM normalization
    if calm_mask is not None:
        calm_ar1 = ar1[calm_mask & np.isfinite(ar1)]
        calm_sigma = sigma[calm_mask & np.isfinite(sigma)]
        if len(calm_ar1) > 5 and len(calm_sigma) > 5:
            ar1_calm_mean = np.mean(calm_ar1)
            ar1_calm_std = max(np.std(calm_ar1), EPS)
            sigma_calm_mean = np.mean(calm_sigma)
            sigma_calm_std = max(np.std(calm_sigma), EPS)

            ar1_z = (ar1 - ar1_calm_mean) / ar1_calm_std
            sigma_z = (sigma - sigma_calm_mean) / sigma_calm_std

            csd = np.nan_to_num(ar1_z + sigma_z, nan=0.0)
            return csd

    return np.zeros(n)


def compute_pcs(kf: np.ndarray, theta_a: np.ndarray, csd: np.ndarray,
                cfg: V2Config) -> np.ndarray:
    """
    Prediction Confidence Score [operational heuristic]:

      PCS = min(κ_F/threshold, 1) × min(Θ_A/threshold, 1)

    When Θ_A unavailable (Layer 2 not applied), PCS = min(κ_F/threshold, 1).
    CSD used as independent confirmation signal, not multiplied in.
    """
    kf_term = np.minimum(kf / cfg.kf_threshold, 1.0)

    # Check if theta_a carries signal
    if np.max(np.abs(theta_a)) > EPS:
        theta_term = np.minimum(theta_a / cfg.theta_a_threshold, 1.0)
        pcs = kf_term * np.maximum(theta_term, 0.0)
    else:
        pcs = kf_term

    return np.clip(pcs, 0.0, 1.0)


def run_layer3(v2_core: pd.DataFrame, v2_geo: pd.DataFrame,
               state_df: pd.DataFrame, cfg: V2Config,
               calm_mask: np.ndarray = None) -> pd.DataFrame:
    """
    Run Layer 3 analysis.
    """
    kf = v2_core["kappa_F"].values
    rho_bar_pos = np.maximum(v2_core["rho_bar"].values, 0.0)
    eta = state_df["eta"].values

    # Normalize eta by CALM
    if calm_mask is not None and calm_mask.sum() > 5:
        eta_calm = np.mean(eta[calm_mask])
        eta_norm = eta / max(eta_calm, EPS)
    else:
        eta_norm = eta / max(np.median(eta), EPS)

    theta_a = v2_geo["theta_A"].values if "theta_A" in v2_geo.columns else np.zeros(len(kf))

    # Hazard
    h = compute_hazard_discrete(kf, rho_bar_pos, eta_norm, theta_a, cfg)

    # Derived quantities
    p30 = compute_collapse_prob(h, delta=30)
    p90 = compute_collapse_prob(h, delta=90)
    t_half = compute_half_life(h)

    # CSD
    C = v2_core["C"].values
    csd = compute_csd(C, cfg.csd_window, calm_mask)

    # PCS
    pcs = compute_pcs(kf, theta_a, csd, cfg)

    # Alert levels
    C_norm = v2_core["C_norm"].values
    alerts = np.full(len(h), "NOMINAL", dtype=object)
    alerts[C_norm < 0.50] = "WATCH"
    alerts[(pcs > 0.3) & (t_half < 90)] = "WARNING"
    alerts[(pcs > 0.7) & (t_half < 30)] = "EMERGENCY"

    v2_prog = pd.DataFrame(index=state_df.index)
    v2_prog["h"] = h
    v2_prog["P_collapse_30"] = p30
    v2_prog["P_collapse_90"] = p90
    v2_prog["T_half"] = t_half
    v2_prog["eta_norm"] = eta_norm
    v2_prog["CSD"] = csd
    v2_prog["PCS"] = pcs
    v2_prog["alert_level"] = alerts

    return v2_prog


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — FULL V2 PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_v2(state_csv: str, viscosity_csv: str,
           cfg: V2Config = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Run complete Kappa v2 analysis on engine_v4 output.

    Args:
        state_csv:     path to kappa_v4_state.csv
        viscosity_csv: path to kappa_v4_viscosity.csv
        cfg:           V2Config (uses defaults if None)

    Returns:
        (v2_df, summary)
        v2_df:   DataFrame with all v2 quantities joined to v4 state
        summary: dict with key metrics and status
    """
    if cfg is None:
        cfg = V2Config()

    # Load v4 data
    state_df = pd.read_csv(state_csv, index_col="date", parse_dates=True)
    visc = pd.read_csv(viscosity_csv, index_col=0)

    phi_star = float(visc.loc["phi_star", "value"])
    Oh_pre = float(state_df["Oh_pre"].iloc[0])
    gamma = cfg.gamma

    # Reconstruct CALM mask from regime data
    # Heuristic: early period with predominantly Nagare
    regimes = state_df["regime"].values
    n = len(state_df)
    # Use first 40% as CALM proxy (consistent with engine_v4's calm_search_to)
    calm_end_idx = int(n * 0.4)
    calm_mask = np.zeros(n, dtype=bool)
    calm_mask[:calm_end_idx] = True

    print(f"[v2] Loaded: {n} steps, Φ*={phi_star:.6f}, Oh_pre={Oh_pre:.4f}")

    # ── Layer 1 ──────────────────────────────────────────────────────────────
    print("[v2] Layer 1 — Structural Capacity (Core)...")
    v2_core = run_layer1(state_df, phi_star, Oh_pre, cfg)

    n_fr = int(v2_core["false_recovery"].sum())
    c_now = float(v2_core["C_norm"].iloc[-1])
    kf_now = float(v2_core["kappa_F"].iloc[-1])
    print(f"     C/Φ* now = {c_now:.4f}  |  κ_F now = {kf_now:.4f}  |  "
          f"False Recovery days = {n_fr}")

    # ── Layer 2 ──────────────────────────────────────────────────────────────
    print("[v2] Layer 2 — Attractor Geometry...")
    v2_geo = run_layer2(state_df, cfg, calm_mask)
    l2_status = v2_geo["layer2_status"].iloc[-1] if "layer2_status" in v2_geo.columns else "N/A"
    theta_now = float(v2_geo["theta_A"].iloc[-1]) if "theta_A" in v2_geo.columns else 0.0
    print(f"     Status: {l2_status}  |  Θ_A now = {theta_now:.4f}")

    # ── Layer 3 ──────────────────────────────────────────────────────────────
    print("[v2] Layer 3 — Prognostic Module...")
    v2_prog = run_layer3(v2_core, v2_geo, state_df, cfg, calm_mask)

    h_now = float(v2_prog["h"].iloc[-1])
    p30_now = float(v2_prog["P_collapse_30"].iloc[-1])
    t_half_now = float(v2_prog["T_half"].iloc[-1])
    alert_now = v2_prog["alert_level"].iloc[-1]
    print(f"     h(t) = {h_now:.6f}  |  P(30d) = {p30_now:.4f}  |  "
          f"T½ = {t_half_now:.1f}  |  Alert: {alert_now}")

    # ── Combine ──────────────────────────────────────────────────────────────
    v2_df = state_df.join(v2_core).join(v2_geo).join(v2_prog)

    summary = {
        "n_steps": n,
        "phi_star": phi_star,
        # Layer 1
        "C_now": float(v2_core["C"].iloc[-1]),
        "C_norm_now": c_now,
        "kappa_F_now": kf_now,
        "rho_bar_now": float(v2_core["rho_bar"].iloc[-1]),
        "T_exhaust_now": float(v2_core["T_exhaust"].iloc[-1]),
        "n_false_recovery_days": n_fr,
        # Layer 2
        "layer2_status": l2_status,
        "theta_A_now": theta_now,
        # Layer 3
        "h_now": h_now,
        "P_collapse_30d": p30_now,
        "P_collapse_90d": float(v2_prog["P_collapse_90"].iloc[-1]),
        "T_half_now": t_half_now,
        "CSD_now": float(v2_prog["CSD"].iloc[-1]),
        "PCS_now": float(v2_prog["PCS"].iloc[-1]),
        "alert_level": alert_now,
    }

    print(f"\n[v2] ✅ Complete. Alert level: {alert_now}")
    return v2_df, summary


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Kappa v2 — engine_v5")
    p.add_argument("--state", type=str, required=True, help="Path to kappa_v4_state.csv")
    p.add_argument("--visc", type=str, required=True, help="Path to kappa_v4_viscosity.csv")
    p.add_argument("--out", type=str, default="./v2_output", help="Output directory")
    args = p.parse_args()

    import os
    os.makedirs(args.out, exist_ok=True)

    v2_df, summary = run_v2(args.state, args.visc)

    # Save
    v2_df.to_csv(os.path.join(args.out, "kappa_v2_state.csv"))

    import json
    with open(os.path.join(args.out, "kappa_v2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[v2] Output saved to {args.out}/")
