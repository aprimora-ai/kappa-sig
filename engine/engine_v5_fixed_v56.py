#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Kappa Method v2 -- engine_v5.py  (v5.6 -- multi-observable geometric verification)
===========================================================================
v5.6 changes (2026-03-31):
  FIX-10: Multi-observable geometric spot check (Section 8.11 of paper)
          - Computes current Theta_A for Oh(t), eta(t), mean_corr(t) independently
          - Reports consensus: ALIGNED / PARTIAL / NONE
          - Resolves circularity concern (L4): if independent observables agree
            with Phi(t) geometry, the signal is system-wide, not Phi-derived

v5.5 changes (2026-03-30):
  FIX-8: Geometric activation classifier (Section 3.10 of paper)
         - Classifies Theta_A activation as Type 1 (spike) or Type 2 (ramp)
         - Only Type 2 (ramp) carries structural information
         - Spike = appears abruptly, dissipates in days (shock echo)
         - Ramp = builds gradually over weeks, persists months (reorganization)

  FIX-9: Geometric activation tracking (t_G, crystallization duration)
         - Detects first sustained Theta_A activation (t_G)
         - Tracks duration of geometric crystallization
         - Reports in summary JSON for prospective monitoring

  Prior fixes (v5.0-v5.4):
  FIX-1: phi_star=0 -> BASELINE_INSUFFICIENT
  FIX-2: Theta_A cap(5.0)
  FIX-3: kappa_F capped at 20.0
  FIX-4: Multi-source alert logic
  FIX-5: Geometry reliability gate (sigma-based)
  FIX-6: eta smoothed via EMA + cap at 5.0
  FIX-7: RQA normalization changed from ratio to z-score for ALL measures

Author: David Ohio | odavidohio@gmail.com | Independent Researcher
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

EPS = 1e-12
KAPPA_F_CAP = 20.0
THETA_A_CAP = 5.0
ETA_NORM_CAP = 5.0
PHI_STAR_MIN = 1e-4

# FIX-7: sigma floor for z-score normalization
SIGMA_FLOOR = 0.01

# FIX-7: geo reliability based on sigma
GEO_SIGMA_MIN = 0.005

# FIX-8: Geometric activation thresholds (Section 3.10)
THETA_G_THRESH = 0.5      # Theta_A threshold for geometric activation
THETA_G_PERSIST = 5        # Sustained days for activation
SPIKE_REVERT_WINDOW = 20   # Steps within which a spike reverts
RAMP_MIN_GROWTH = 15       # Minimum steps of growth for ramp classification
RAMP_MIN_PERSIST = 30      # Minimum steps of persistence for ramp classification


@dataclass
class V2Config:
    gamma: float = 0.97; ema_alpha: float = 0.1; theta_fr: float = 0.30; t_fr: int = 90
    n1_min: int = 60; n2_min: int = 180; rqa_window: int = 60
    rqa_epsilon_q: float = 0.10; rqa_min_line: int = 2
    beta_0: float = -6.0; beta_F: float = 1.0; beta_rho: float = 0.5
    beta_eta: float = 0.3; beta_A: float = 0.5; csd_window: int = 40
    kf_threshold: float = 2.0; theta_a_threshold: float = 1.0; csd_threshold: float = 2.0
    h_warn: float = 0.02; h_emerg: float = 0.10
    t_warn: int = 90; t_emerg: int = 30
    kf_warn: float = 1.5
    ema_alpha_eta: float = 0.1


# ==============================================================================
# LAYER 1 -- STRUCTURAL CAPACITY (CORE)
# ==============================================================================

def compute_capacity(phi, phi_star): return phi_star - phi

def compute_capacity_dynamics(C, phi, Oh, Oh_pre, delta, gamma):
    D = np.maximum(Oh - Oh_pre - delta, 0.0)
    rho = np.zeros(len(phi))
    for i in range(1, len(phi)): rho[i] = D[i] - (1.0 - gamma) * phi[i-1]
    rho[0] = D[0]
    return {"rho": rho, "D": D}

def compute_rho_bar(rho, alpha=0.1):
    rb = np.zeros_like(rho); rb[0] = rho[0]
    for i in range(1, len(rho)): rb[i] = alpha*rho[i] + (1-alpha)*rb[i-1]
    return rb

def compute_t_exhaust(C, rho_bar):
    Cp = np.maximum(C, EPS); t = np.full_like(C, np.inf)
    a = rho_bar > EPS; t[a] = Cp[a]/rho_bar[a]
    return t

def compute_kappa_f(C, phi_star):
    Cp = np.maximum(C, EPS)
    return np.clip(np.log(phi_star / Cp), 0.0, KAPPA_F_CAP)

def detect_false_recovery(Oh, C, phi_star, regimes, cfg):
    n = len(Oh); fr = np.zeros(n, dtype=bool)
    for i in range(n):
        if Oh[i] >= 1.0: continue
        if C[i] / phi_star >= cfg.theta_fr: continue
        for j in range(max(0, i-cfg.t_fr), i):
            if regimes[j] == "Katashi": fr[i] = True; break
    return fr

def run_layer1(state_df, phi_star, Oh_pre, cfg):
    Oh, phi, reg = state_df["Oh"].values, state_df["phi"].values, state_df["regime"].values
    C = compute_capacity(phi, phi_star)
    dyn = compute_capacity_dynamics(C, phi, Oh, Oh_pre, cfg.gamma*0.08, cfg.gamma)
    rho = dyn["rho"]; kf = compute_kappa_f(C, phi_star)
    rb = compute_rho_bar(rho, cfg.ema_alpha)
    te = compute_t_exhaust(C, rb)
    fr = detect_false_recovery(Oh, C, phi_star, reg, cfg)
    v2 = pd.DataFrame(index=state_df.index)
    v2["C"]=C; v2["C_norm"]=C/phi_star; v2["rho"]=rho; v2["rho_bar"]=rb
    v2["kappa_F"]=kf; v2["T_exhaust"]=te; v2["false_recovery"]=fr; v2["D"]=dyn["D"]
    return v2


# ==============================================================================
# LAYER 2 -- ATTRACTOR GEOMETRY
# ==============================================================================

def _recurrence_matrix(Y, epsilon):
    n=Y.shape[0]; R=np.zeros((n,n),dtype=bool)
    for i in range(n): R[i] = np.linalg.norm(Y-Y[i],axis=1) <= epsilon
    return R

def _rqa_from_matrix(R, min_line=2):
    n=R.shape[0]; rp=int(R.sum())
    RR=rp/(n*n) if n>0 else 0.0
    dl=[]
    for k in range(-n+1,n):
        d=np.diag(R,k); l=0
        for v in d:
            if v: l+=1
            else:
                if l>=min_line: dl.append(l)
                l=0
        if l>=min_line: dl.append(l)
    DET=sum(dl)/rp if rp>0 else 0.0
    vl=[]
    for j in range(n):
        c=R[:,j]; l=0
        for v in c:
            if v: l+=1
            else:
                if l>=min_line: vl.append(l)
                l=0
        if l>=min_line: vl.append(l)
    LAM=sum(vl)/rp if rp>0 else 0.0
    TT=float(np.mean(vl)) if vl else 0.0
    return {"RR":RR,"DET":DET,"LAM":LAM,"TT":TT}

def _delay_embedding(x, tau, m):
    nv=len(x)-(m-1)*tau
    if nv<=0: return np.array([]).reshape(0,m)
    Y=np.zeros((nv,m))
    for i in range(m): Y[:,i]=x[i*tau:i*tau+nv]
    return Y

def _estimate_tau(x, max_tau=20):
    xc=x-np.mean(x); n=len(xc)
    for t in range(1, min(max_tau, n//2)):
        if len(xc[:-t])<2: continue
        if np.corrcoef(xc[:-t],xc[t:])[0,1]<=0: return t
    return min(max_tau, max(1, n//4))

def compute_rqa_rolling(phi, cfg, calm_mask=None):
    n = len(phi)
    if n < cfg.n1_min:
        return None, "HIGH", {}
    tau = _estimate_tau(phi)
    m = 3; W = cfg.rqa_window; results = []
    for t in range(W, n):
        w = phi[t-W:t]
        Y = _delay_embedding(w, tau, m)
        if Y.shape[0] < 10:
            results.append({"DET": np.nan, "LAM": np.nan, "TT": np.nan}); continue
        ds = [np.linalg.norm(Y[i]-Y[j])
              for i in range(Y.shape[0]) for j in range(i+1, Y.shape[0])]
        if not ds:
            results.append({"DET": np.nan, "LAM": np.nan, "TT": np.nan}); continue
        eps = max(np.quantile(ds, cfg.rqa_epsilon_q), EPS)
        R = _recurrence_matrix(Y, eps)
        results.append(_rqa_from_matrix(R, cfg.rqa_min_line))
    pad = [{"DET": np.nan, "LAM": np.nan, "TT": np.nan}] * W
    rdf = pd.DataFrame(pad + results)
    geo_reliability = "HIGH"; calm_stats = {}
    if calm_mask is not None and calm_mask.sum() >= W:
        for c in ["DET", "LAM", "TT"]:
            cv = rdf[c].values[calm_mask]; cv = cv[np.isfinite(cv)]
            if len(cv) > 5:
                mu = float(np.mean(cv)); sigma = float(np.std(cv))
                calm_stats[c] = {"mu": mu, "sigma": sigma, "n": len(cv)}
                if c in ("DET", "LAM") and sigma < GEO_SIGMA_MIN: geo_reliability = "LOW"
                rdf[f"{c}_norm"] = (rdf[c] - mu) / max(sigma, SIGMA_FLOOR)
            else:
                rdf[f"{c}_norm"] = 0.0; calm_stats[c] = {"mu": 0, "sigma": 0, "n": 0}
    else:
        for c in ["DET", "LAM", "TT"]:
            rdf[f"{c}_norm"] = 0.0; calm_stats[c] = {"mu": 0, "sigma": 0, "n": 0}
    return rdf, geo_reliability, calm_stats

def compute_theta_a(rdf, w_D=0.40, w_L=0.35, w_T=0.25):
    if rdf is None: return np.array([0.0])
    dn = rdf.get("DET_norm", pd.Series(0.0, index=rdf.index)).fillna(0).values
    ln_ = rdf.get("LAM_norm", pd.Series(0.0, index=rdf.index)).fillna(0).values
    tn = rdf.get("TT_norm", pd.Series(0.0, index=rdf.index)).fillna(0).values
    th = w_D * np.maximum(dn, 0) + w_L * np.maximum(ln_, 0) + w_T * np.maximum(tn, 0)
    return np.clip(th, 0.0, THETA_A_CAP)


# ==============================================================================
# FIX-8/9: GEOMETRIC ACTIVATION CLASSIFIER & TRACKER (Section 3.10)
# ==============================================================================

def detect_geometric_activation(theta_a, dates=None):
    n = len(theta_a)
    result = {"t_G": None, "t_G_date": None, "activation_type": "NONE",
              "crystallization_duration": 0, "currently_active": False,
              "ramp_growth_steps": 0, "persistence_steps": 0}
    if n < THETA_G_PERSIST: return result
    t_G = None
    for i in range(n - THETA_G_PERSIST + 1):
        if all(theta_a[i:i+THETA_G_PERSIST] > THETA_G_THRESH): t_G = i; break
    if t_G is None: return result
    result["t_G"] = int(t_G)
    if dates is not None and t_G < len(dates): result["t_G_date"] = str(dates[t_G])
    tail_active = all(theta_a[max(0, n-THETA_G_PERSIST):] > THETA_G_THRESH)
    result["currently_active"] = bool(tail_active)
    if tail_active:
        result["crystallization_duration"] = int(n - t_G)
    else:
        for i in range(t_G + THETA_G_PERSIST, n):
            if theta_a[i] < 0.1: result["crystallization_duration"] = int(i - t_G); break
        else: result["crystallization_duration"] = int(n - t_G)
    rqa_warmup = 60; early_activation = (t_G <= rqa_warmup + 5)
    reverted = False
    for i in range(t_G + THETA_G_PERSIST, min(t_G + SPIKE_REVERT_WINDOW, n)):
        if theta_a[i] < 0.1: reverted = True; break
    growth_steps = 0
    if t_G > 0:
        for i in range(t_G - 1, max(0, t_G - 60) - 1, -1):
            if theta_a[i] < theta_a[i + 1] or theta_a[i] > 0.05: growth_steps += 1
            else: break
    result["ramp_growth_steps"] = int(growth_steps)
    persist = 0
    for i in range(t_G, n):
        if theta_a[i] > THETA_G_THRESH: persist += 1
        else: break
    result["persistence_steps"] = int(persist)
    if early_activation and reverted: result["activation_type"] = "SPIKE"
    elif reverted and growth_steps < RAMP_MIN_GROWTH: result["activation_type"] = "SPIKE"
    elif growth_steps >= RAMP_MIN_GROWTH and persist >= RAMP_MIN_PERSIST: result["activation_type"] = "RAMP"
    elif persist >= RAMP_MIN_PERSIST: result["activation_type"] = "RAMP"
    elif reverted: result["activation_type"] = "SPIKE"
    else: result["activation_type"] = "BUILDING"
    return result

def run_layer2(state_df, cfg, calm_mask=None):
    phi = state_df["phi"].values; n = len(phi)
    v2g = pd.DataFrame(index=state_df.index)
    if n < cfg.n1_min:
        v2g["theta_A"] = 0.0; v2g["layer2_status"] = "NOT_APPLIED"; v2g["geo_reliability"] = "N/A"
        return v2g, {}, {"activation_type": "NONE", "t_G": None}
    rdf, geo_rel, calm_stats = compute_rqa_rolling(phi, cfg, calm_mask)
    if rdf is not None and len(rdf) == n:
        v2g["DET"] = rdf["DET"].values; v2g["LAM"] = rdf["LAM"].values; v2g["TT"] = rdf["TT"].values
        theta = compute_theta_a(rdf)
        if geo_rel == "LOW": theta = theta * 0.0
        v2g["theta_A"] = theta
        v2g["layer2_status"] = np.where(n >= cfg.n2_min, "FULL", "RQA_ONLY")
        v2g["geo_reliability"] = geo_rel
        geo_activation = detect_geometric_activation(theta, state_df.index)
    else:
        v2g["theta_A"] = 0.0; v2g["layer2_status"] = "FAILED"; v2g["geo_reliability"] = "FAILED"
        calm_stats = {}; geo_activation = {"activation_type": "NONE", "t_G": None}
    return v2g, calm_stats, geo_activation


# ==============================================================================
# FIX-10: MULTI-OBSERVABLE GEOMETRIC SPOT CHECK (Section 8.11)
# ==============================================================================

INDEPENDENT_OBSERVABLES = [("Oh", "Oh"), ("eta", "eta"), ("mean_corr", "mean_corr")]

def _spot_check_rqa(series, rqa_window=60, epsilon_q=0.10, calm_frac=0.4):
    n = len(series)
    if n < rqa_window + 10: return 0.0, "N/A"
    tau = _estimate_tau(series); m = 3; ci = int(n * calm_frac); W = rqa_window
    calm_rqa = []
    for t in range(W, ci):
        w = series[t - W:t]; Y = _delay_embedding(w, tau, m)
        if Y.shape[0] < 10: continue
        ds = [np.linalg.norm(Y[i] - Y[j]) for i in range(Y.shape[0]) for j in range(i+1, Y.shape[0])]
        if not ds: continue
        eps = max(np.quantile(ds, epsilon_q), EPS)
        calm_rqa.append(_rqa_from_matrix(_recurrence_matrix(Y, eps)))
    if len(calm_rqa) < 5: return 0.0, "LOW"
    calm_stats = {}; geo_rel = "HIGH"
    for c in ["DET", "LAM", "TT"]:
        vals = [r[c] for r in calm_rqa if not np.isnan(r[c])]
        if len(vals) > 3:
            mu = np.mean(vals); sigma = np.std(vals); calm_stats[c] = (mu, sigma)
            if c in ("DET", "LAM") and sigma < GEO_SIGMA_MIN: geo_rel = "LOW"
        else: calm_stats[c] = (0, 0)
    if geo_rel == "LOW": return 0.0, "LOW"
    w = series[-W:]; Y = _delay_embedding(w, tau, m)
    if Y.shape[0] < 10: return 0.0, geo_rel
    ds = [np.linalg.norm(Y[i] - Y[j]) for i in range(Y.shape[0]) for j in range(i+1, Y.shape[0])]
    if not ds: return 0.0, geo_rel
    eps = max(np.quantile(ds, epsilon_q), EPS)
    rqa = _rqa_from_matrix(_recurrence_matrix(Y, eps))
    norms = []
    for c in ["DET", "LAM", "TT"]:
        mu, sigma = calm_stats[c]; z = max((rqa[c] - mu) / max(sigma, SIGMA_FLOOR), 0); norms.append(z)
    return float(np.clip(0.40*norms[0] + 0.35*norms[1] + 0.25*norms[2], 0, THETA_A_CAP)), geo_rel

def multi_observable_spot_check(state_df, cfg):
    results = {}
    for obs_name, obs_col in INDEPENDENT_OBSERVABLES:
        if obs_col not in state_df.columns:
            results[obs_name] = {"theta_A": 0.0, "geo": "N/A", "status": "missing"}; continue
        series = state_df[obs_col].values.astype(float)
        if np.std(series) < 1e-15:
            results[obs_name] = {"theta_A": 0.0, "geo": "N/A", "status": "flat"}; continue
        theta, geo = _spot_check_rqa(series, cfg.rqa_window, cfg.rqa_epsilon_q)
        results[obs_name] = {"theta_A": theta, "geo": geo, "activated": theta > THETA_G_THRESH, "status": "ok"}
    n_active = sum(1 for r in results.values() if r.get("activated", False))
    n_valid = sum(1 for r in results.values() if r.get("status") == "ok")
    results["_agreement"] = {"n_active": n_active, "n_valid": n_valid,
        "consensus": "ALIGNED" if n_active == n_valid and n_valid > 0
                     else ("PARTIAL" if n_active > 0 else "NONE")}
    return results


# ==============================================================================
# LAYER 3 -- STRUCTURAL HAZARD FUNCTION
# ==============================================================================

def smooth_eta(eta, alpha=0.1):
    eta_bar = np.zeros_like(eta); eta_bar[0] = eta[0]
    for i in range(1, len(eta)): eta_bar[i] = alpha * eta[i] + (1.0 - alpha) * eta_bar[i - 1]
    return eta_bar

def compute_hazard_discrete(kf, rbp, en, ta, cfg):
    lo = cfg.beta_0 + cfg.beta_F*kf + cfg.beta_rho*rbp + cfg.beta_eta*en + cfg.beta_A*ta
    return 1.0 / (1.0 + np.exp(-np.clip(lo, -20, 20)))

def compute_collapse_prob(h, delta=30):
    return 1.0 - np.power(np.maximum(1.0 - h, EPS), delta)

def compute_half_life(h):
    return np.clip(np.log(0.5) / np.log(np.maximum(1.0 - np.maximum(h, EPS), EPS)), 0, 1e6)

def compute_csd(C, window=40, calm_mask=None):
    n = len(C); ar1 = np.full(n, np.nan); sig = np.full(n, np.nan)
    for t in range(window, n):
        w = C[t-window:t]
        if np.std(w) < EPS: ar1[t] = 0.0; sig[t] = 0.0; continue
        ar1[t] = np.corrcoef(w[:-1], w[1:])[0, 1] if len(w) > 2 else 0.0
        sig[t] = np.std(w)
    if calm_mask is not None:
        ca = ar1[calm_mask & np.isfinite(ar1)]; cs = sig[calm_mask & np.isfinite(sig)]
        if len(ca) > 5 and len(cs) > 5:
            return np.nan_to_num((ar1 - np.mean(ca)) / max(np.std(ca), EPS)
                + (sig - np.mean(cs)) / max(np.std(cs), EPS), nan=0.0)
    return np.zeros(n)

def compute_pcs(kf, ta, csd, cfg):
    kt = np.minimum(np.maximum(kf, 0) / cfg.kf_threshold, 1.0)
    if np.max(np.abs(ta)) > EPS:
        tt = np.minimum(ta / cfg.theta_a_threshold, 1.0)
        return np.clip(kt * np.maximum(tt, 0), 0, 1)
    return np.clip(kt, 0, 1)

def compute_alerts(C_norm, kf, h, t_half, pcs, phi_star_estimated, cfg):
    n = len(C_norm); alerts = np.full(n, "NOMINAL", dtype=object)
    if phi_star_estimated: alerts[:] = "BASELINE_INSUFFICIENT"; return alerts
    alerts[(C_norm < 0.50) | (kf > cfg.kf_warn)] = "WATCH"
    alerts[((h >= cfg.h_warn) & (C_norm < 0.70)) | (t_half < cfg.t_warn)] = "WARNING"
    alerts[(C_norm < 0.0) | ((h >= cfg.h_emerg) & (C_norm < 0.50))] = "EMERGENCY"
    return alerts

def run_layer3(v2c, v2g, sdf, cfg, cm=None, phi_star_estimated=False):
    kf = v2c["kappa_F"].values; rbp = np.maximum(v2c["rho_bar"].values, 0)
    eta_raw = sdf["eta"].values; eta_smoothed = smooth_eta(eta_raw, cfg.ema_alpha_eta)
    if cm is not None and cm.sum() > 5: en = eta_smoothed / max(np.mean(eta_smoothed[cm]), EPS)
    else: en = eta_smoothed / max(np.median(eta_smoothed), EPS)
    en = np.clip(en, 0.0, ETA_NORM_CAP)
    ta = v2g["theta_A"].values if "theta_A" in v2g.columns else np.zeros(len(kf))
    h = compute_hazard_discrete(kf, rbp, en, ta, cfg)
    p30 = compute_collapse_prob(h, 30); p90 = compute_collapse_prob(h, 90); th = compute_half_life(h)
    csd = compute_csd(v2c["C"].values, cfg.csd_window, cm)
    pcs = compute_pcs(kf, ta, csd, cfg); cn = v2c["C_norm"].values
    al = compute_alerts(cn, kf, h, th, pcs, phi_star_estimated, cfg)
    r = pd.DataFrame(index=sdf.index)
    r["h"]=h; r["P_collapse_30"]=p30; r["P_collapse_90"]=p90; r["T_half"]=th
    r["eta_norm"]=en; r["CSD"]=csd; r["PCS"]=pcs; r["alert_level"]=al
    return r


# ==============================================================================
# MAIN
# ==============================================================================

def run_v2(state_csv, viscosity_csv, cfg=None):
    if cfg is None: cfg = V2Config()
    sdf = pd.read_csv(state_csv, index_col="date", parse_dates=True)
    visc = pd.read_csv(viscosity_csv, index_col=0)
    psr = float(visc.loc["phi_star", "value"]); phi = sdf["phi"].values
    phi_star_estimated = False
    if psr < PHI_STAR_MIN:
        phi_star_estimated = True; pm = float(np.max(phi))
        if pm > PHI_STAR_MIN: ps = pm * 2.0; print(f"[v2] WARNING: phi*=0. Using 2*max(phi)={ps:.6f} [BASELINE_INSUFFICIENT]")
        else: ps = PHI_STAR_MIN; print(f"[v2] WARNING: phi*=0, max(phi)~0. [BASELINE_INSUFFICIENT]")
    else: ps = psr
    Op = float(sdf["Oh_pre"].iloc[0]); n = len(sdf)
    ci = int(n * 0.4); cm = np.zeros(n, dtype=bool); cm[:ci] = True
    est_tag = " [BASELINE_INSUFFICIENT]" if phi_star_estimated else ""
    print(f"[v2] Loaded: {n} steps, phi*={ps:.6f}{est_tag}, Oh_pre={Op:.4f}")

    # Layer 1
    print("[v2] Layer 1 -- Structural Capacity...")
    v2c = run_layer1(sdf, ps, Op, cfg)
    nfr = int(v2c["false_recovery"].sum()); cn = float(v2c["C_norm"].iloc[-1]); kn = float(v2c["kappa_F"].iloc[-1])
    if phi_star_estimated: print(f"     [BASELINE_INSUFFICIENT] FR={nfr}")
    else: print(f"     C/phi*={cn:.4f}  kF={kn:.4f}  FR={nfr}")

    # Layer 2
    print("[v2] Layer 2 -- Attractor Geometry (z-score)...")
    v2g, calm_stats, geo_activation = run_layer2(sdf, cfg, cm)
    ls = v2g["layer2_status"].iloc[-1]
    gr = v2g["geo_reliability"].iloc[-1] if "geo_reliability" in v2g.columns else "N/A"
    tn = float(v2g["theta_A"].iloc[-1])
    cs_str = ""
    for m_name in ["DET", "LAM", "TT"]:
        if m_name in calm_stats:
            cs = calm_stats[m_name]; cs_str += f" {m_name}:mu={cs['mu']:.3f}/sigma={cs['sigma']:.4f}"
    ga = geo_activation; ga_str = ""
    if ga["t_G"] is not None:
        ga_str = f"  t_G={ga['t_G']}({ga.get('t_G_date','?')}) type={ga['activation_type']} dur={ga['crystallization_duration']}steps"
        if ga["currently_active"]: ga_str += " [ACTIVE]"
    print(f"     {ls}  thetaA={tn:.4f}  geo={gr}{cs_str}")
    if ga_str: print(f"     GEO_ACTIVATION:{ga_str}")

    # Layer 2b -- Multi-Observable Spot Check (FIX-10)
    print("[v2] Layer 2b -- Multi-Observable Spot Check...")
    obs_check = multi_observable_spot_check(sdf, cfg)
    agreement = obs_check.get("_agreement", {}); obs_strs = []
    for oname in ["Oh", "eta", "mean_corr"]:
        oc = obs_check.get(oname, {})
        if oc.get("status") == "ok":
            act_flag = "*" if oc.get("activated") else " "
            obs_strs.append(f"{oname}={oc['theta_A']:.2f}{act_flag}")
    print(f"     {' '.join(obs_strs)}  consensus={agreement.get('consensus', '?')}")

    # Layer 3
    print("[v2] Layer 3 -- Prognostic Module...")
    v2p = run_layer3(v2c, v2g, sdf, cfg, cm, phi_star_estimated)
    hn = float(v2p["h"].iloc[-1]); pn = float(v2p["P_collapse_30"].iloc[-1])
    thn = float(v2p["T_half"].iloc[-1]); an = v2p["alert_level"].iloc[-1]
    print(f"     h={hn:.6f}  P30={pn:.4f}  T1/2={thn:.1f}  [{an}]")

    # Combine
    v2df = sdf.join(v2c).join(v2g).join(v2p)
    s = {
        "n_steps": n, "phi_star": ps, "phi_star_estimated": phi_star_estimated,
        "phi_star_status": "estimated_fallback" if phi_star_estimated else "measured",
        "C_now": float(v2c["C"].iloc[-1]), "C_norm_now": cn, "kappa_F_now": kn,
        "rho_bar_now": float(v2c["rho_bar"].iloc[-1]),
        "T_exhaust_now": float(v2c["T_exhaust"].iloc[-1]),
        "n_false_recovery_days": nfr,
        "layer2_status": ls, "geo_reliability": gr, "theta_A_now": tn,
        "calm_rqa_stats": calm_stats,
        "geo_activation": {
            "t_G": ga["t_G"], "t_G_date": ga.get("t_G_date"),
            "activation_type": ga["activation_type"],
            "crystallization_duration": ga["crystallization_duration"],
            "currently_active": ga["currently_active"],
            "ramp_growth_steps": ga.get("ramp_growth_steps", 0),
            "persistence_steps": ga.get("persistence_steps", 0),
        },
        "h_now": hn, "P_collapse_30d": pn,
        "P_collapse_90d": float(v2p["P_collapse_90"].iloc[-1]),
        "T_half_now": thn,
        "CSD_now": float(v2p["CSD"].iloc[-1]),
        "PCS_now": float(v2p["PCS"].iloc[-1]),
        "alert_level": an,
        "multi_observable": {k: v for k, v in obs_check.items()},
    }
    print(f"\n[v2] Done. [{an}]")
    return v2df, s


if __name__ == "__main__":
    import argparse, os, json
    p = argparse.ArgumentParser()
    p.add_argument("--state", required=True)
    p.add_argument("--visc", required=True)
    p.add_argument("--out", default="./v2_output")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df, s = run_v2(a.state, a.visc)
    df.to_csv(os.path.join(a.out, "kappa_v2_state.csv"))
    with open(os.path.join(a.out, "kappa_v2_summary.json"), "w") as f:
        json.dump(s, f, indent=2, default=str)
