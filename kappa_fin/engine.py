#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Kappa-FIN v0.1 — Rayleigh + Prandtl (robust) + CALM scan + Dual Crossing
+ CALM penalty_knee_phi_tail (anti "phase-transition-inside-baseline")

Mudança central vs v4.5:
  - Mantém o calm_score atual (estabilidade global),
  - Adiciona um termo de penalidade que detecta "joelho"/aceleração no phi
    no último quartil do segmento candidato (phi calculado LOCALMENTE no segmento,
    sem depender do CALM global).

Intuição física:
  - Drift/entropia crescente é OK (vida real),
  - O que não pode é o candidato CALM conter uma transição de regime interna
    (subcrítico -> supercrítico) — isso "contamina" o baseline.

Outputs:
  - kappa_fin_state.csv
  - analysis_report.txt (indicadores + crossings)
  - observables.png
  - ohio_number.png
  - damage_phi.png
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from scipy import stats as scipy_stats

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import networkx as nx
except Exception:
    nx = None

try:
    import gudhi as gd
except Exception:
    gd = None

EPS = 1e-12


# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # Data
    tickers: List[str]
    start: str
    end: str
    interval: str = "1d"

    # Rolling
    window: int = 22
    k: int = 5

    # Dependência / estabilidade
    dep_method: str = "spearman"
    shrink_lambda: float = 0.05
    dist_mode: str = "corr"
    alpha: float = 1.0

    # PH stage
    tau: float = 1e-7
    max_edge_quantile: float = 0.99
    cap_inf: bool = True

    # Curvatura / rigidez
    weight_mode: str = "similarity"
    curv_summary: str = "median"
    eta_floor: float = 0.05

    # v_raw
    v_raw_mode: str = "entropy_x_anti_dom"

    # Xi / Ohio
    xi_coupling: str = "eta_inverse"
    xi_q: float = 0.05
    xi_method: str = "quantile"
    xi_quantile: float = 0.99
    xi_eps_margin: float = 0.02

    # Phi / dano
    phi_mode: str = "pre_excess"
    delta: float = 0.05
    gamma: float = 0.985
    pre_q: float = 0.95
    phi_method: str = "quantile"
    phi_q: float = 0.99
    phi_persist: int = 3
    phi_floor: float = 1e-6

    # CALM
    calm_policy: str = "scan"
    calm_start: Optional[str] = None
    calm_end: Optional[str] = None
    calm_length_days: int = 504
    calm_step_days: int = 14
    calm_topn: int = 5
    calm_search_to: Optional[str] = None
    calm_ensemble: bool = False

    # CALM v4.6: penalty_knee_phi_tail
    calm_knee_weight: float = 1.0          # peso global do penalty
    calm_knee_tail_frac: float = 0.25      # fração final do segmento usada como "tail"
    calm_knee_ratio: float = 1.50          # se dphi_tail > ratio * dphi_head => penaliza
    calm_knee_slope_ann: float = 0.0       # penaliza se slope(phi_tail) anualizado > isso
    calm_knee_mix: float = 0.5             # mistura (0..1): ratio vs slope (0.5 = balanceado)

    # EVAL - Dual crossing
    eval_after_calm: bool = True
    oh_persist_sens: int = 2
    oh_persist_confirm: int = 5

    # Output
    out: str = "./out_v46"


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def parse_date(s: str) -> pd.Timestamp:
    return pd.to_datetime(s, utc=False)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def rolling_windows_index(n: int, w: int) -> List[Tuple[int, int]]:
    return [(i - w, i) for i in range(w, n + 1)]

def shrink_corr(C: np.ndarray, lam: float) -> np.ndarray:
    n = C.shape[0]
    return (1.0 - lam) * C + lam * np.eye(n)

def corr_from_returns(R: np.ndarray, method: str) -> np.ndarray:
    if method == "spearman":
        ranks = np.apply_along_axis(lambda x: pd.Series(x).rank().to_numpy(), 0, R)
        C = np.corrcoef(ranks, rowvar=False)
    else:
        C = np.corrcoef(R, rowvar=False)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    C = np.clip(C, -1.0, 1.0)
    np.fill_diagonal(C, 1.0)
    return C

def dist_from_corr(C: np.ndarray, mode: str, alpha: float) -> np.ndarray:
    D = 1.0 - C
    D = np.clip(D, 0.0, 2.0)
    if abs(alpha - 1.0) > 1e-9:
        D = np.power(D, alpha)
    np.fill_diagonal(D, 0.0)
    return D

def clamp_k(k: int, n: int) -> int:
    return max(1, min(int(k), max(1, n - 1)))

def build_knn_graph(D: np.ndarray, k: int) -> "nx.Graph":
    if nx is None:
        raise RuntimeError("networkx não disponível. Instale: pip install networkx")
    n = D.shape[0]
    k_eff = clamp_k(k, n)

    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        idx = np.argsort(D[i, :])
        idx = [j for j in idx if j != i]
        for j in idx[:k_eff]:
            w = float(D[i, j])
            if not G.has_edge(i, j):
                G.add_edge(i, j, dist=w)
            else:
                G[i][j]["dist"] = min(G[i][j]["dist"], w)
    return G

def compute_forman_ricci_summary(G: "nx.Graph", summary: str = "median") -> float:
    if nx is None:
        raise RuntimeError("networkx não disponível.")

    deg = dict(G.degree())
    curvs = []
    for u, v in G.edges():
        common = len(set(G.neighbors(u)).intersection(set(G.neighbors(v))))
        fr = 4.0 - deg[u] - deg[v] + 3.0 * common
        curvs.append(fr)

    if len(curvs) == 0:
        return 0.0

    curvs = np.asarray(curvs, dtype=float)
    return float(np.mean(curvs)) if summary == "mean" else float(np.median(curvs))

def compute_ph_entropy_and_dominance(D: np.ndarray, max_edge_quantile: float, cap_inf: bool) -> Tuple[float, float, int]:
    if gd is None:
        return 0.0, 0.0, 0

    n = D.shape[0]
    vals = D[np.triu_indices(n, 1)]
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 0.0, 0.0, 0

    thr = float(np.quantile(vals, max_edge_quantile))
    thr = max(thr, EPS)

    rips = gd.RipsComplex(distance_matrix=D, max_edge_length=thr)
    st = rips.create_simplex_tree(max_dimension=2)
    st.compute_persistence()

    H1 = st.persistence_intervals_in_dimension(1)
    if H1 is None or len(H1) == 0:
        return 0.0, 0.0, 0

    births = H1[:, 0]
    deaths = H1[:, 1]
    if cap_inf:
        finite_deaths = deaths[np.isfinite(deaths)]
        cap = float(np.max(finite_deaths)) if len(finite_deaths) else float(thr)
        deaths = np.where(np.isfinite(deaths), deaths, cap)

    lifetimes = deaths - births
    lifetimes = lifetimes[np.isfinite(lifetimes)]
    lifetimes = lifetimes[lifetimes > EPS]
    if len(lifetimes) == 0:
        return 0.0, 0.0, 0

    s = float(np.sum(lifetimes))
    p = lifetimes / (s + EPS)
    entropy = float(-np.sum(p * np.log(p + EPS)))
    dominance = float(np.max(lifetimes) / (s + EPS))
    return entropy, dominance, len(lifetimes)

def compute_v_raw(entropy: float, dominance: float, mode: str) -> float:
    if mode == "entropy_x_anti_dom":
        return float(max(entropy, 0.0) * max(1.0 - dominance, 0.0))
    return float(max(entropy, 0.0) * max(1.0 - dominance, 0.0))

def persist_crossing(x: np.ndarray, threshold: float, persist: int, start_idx: int = 0) -> Optional[int]:
    x = np.asarray(x, dtype=float)
    n = len(x)
    persist = max(int(persist), 1)
    start_idx = max(int(start_idx), 0)
    if start_idx >= n:
        return None

    run = 0
    for i in range(start_idx, n):
        if x[i] > threshold:
            run += 1
            if run >= persist:
                return int(i - persist + 1)
        else:
            run = 0
    return None

def compute_phi_series(
    Oh: np.ndarray,
    calm_mask: np.ndarray,
    delta: float,
    gamma: float,
    pre_q: float,
    phi_floor: float
) -> Tuple[np.ndarray, float]:
    Oh = np.asarray(Oh, dtype=float)
    calm_mask = np.asarray(calm_mask, dtype=bool)

    if np.sum(calm_mask) >= 10:
        Oh_pre = float(np.quantile(Oh[calm_mask], pre_q))
    else:
        Oh_pre = float(np.quantile(Oh, pre_q))

    drive = np.maximum(Oh - (Oh_pre + delta), 0.0)
    phi = np.zeros_like(Oh, dtype=float)
    for i in range(len(Oh)):
        prev = phi[i - 1] if i > 0 else 0.0
        phi[i] = max(gamma * prev + drive[i], phi_floor)
    return phi, Oh_pre

def phi_critical(phi: np.ndarray, calm_mask: np.ndarray, method: str, q: float) -> float:
    phi = np.asarray(phi, dtype=float)
    calm_mask = np.asarray(calm_mask, dtype=bool)
    base = phi[calm_mask] if np.sum(calm_mask) >= 10 else phi
    if len(base) == 0:
        return float(np.quantile(phi, q)) if len(phi) else 0.0
    return float(np.quantile(base, q))

def xi_critical(Xi: np.ndarray, calm_mask: np.ndarray, method: str, quantile: float, eps_margin: float) -> float:
    Xi = np.asarray(Xi, dtype=float)
    calm_mask = np.asarray(calm_mask, dtype=bool)
    base = Xi[calm_mask] if np.sum(calm_mask) >= 10 else Xi
    if len(base) == 0:
        return 1.0
    xi_q = float(np.quantile(base, quantile))
    xi_max = float(np.max(base))
    xi_raw = max(xi_q, xi_max)
    return float(xi_raw * (1.0 + eps_margin))


# ═══════════════════════════════════════════════════════════════════════
# Core structural state
# ═══════════════════════════════════════════════════════════════════════

def compute_structural_state_window(Rw: np.ndarray, cfg: Config) -> Dict[str, float]:
    C = corr_from_returns(Rw, cfg.dep_method)
    C = shrink_corr(C, cfg.shrink_lambda)
    D = dist_from_corr(C, cfg.dist_mode, cfg.alpha)

    G = build_knn_graph(D, cfg.k)
    curv = compute_forman_ricci_summary(G, cfg.curv_summary)

    eta = 1.0 / (abs(curv) + float(cfg.eta_floor))

    entropy, dominance, h1_count = compute_ph_entropy_and_dominance(D, cfg.max_edge_quantile, cfg.cap_inf)
    v_raw = compute_v_raw(entropy, dominance, cfg.v_raw_mode)

    mean_corr = float(np.mean(C[np.triu_indices(C.shape[0], 1)])) if C.shape[0] > 1 else 0.0

    return {
        "mean_corr": mean_corr,
        "curv_summary": float(curv),
        "eta": float(eta),
        "entropy": float(entropy),
        "dominance": float(dominance),
        "h1_count": int(h1_count),
        "v_raw": float(v_raw),
    }

def compute_full_state(returns: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    R = returns.to_numpy(dtype=float)
    dates = returns.index

    rows = []
    idx = []
    for i0, i1 in rolling_windows_index(len(dates), cfg.window):
        Rw = R[i0:i1, :]
        d = dates[i1 - 1]
        rows.append(compute_structural_state_window(Rw, cfg))
        idx.append(d)

    df = pd.DataFrame(rows, index=pd.to_datetime(idx))
    df.index.name = "date"
    return df


# ═══════════════════════════════════════════════════════════════════════
# CALM scan
# ═══════════════════════════════════════════════════════════════════════

def calm_candidates(dates: pd.DatetimeIndex, search_to: pd.Timestamp, length_days: int, step_days: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    end = search_to
    if end not in dates:
        valid = dates[dates <= end]
        if len(valid) == 0:
            return []
        end = valid[-1]

    cands = []
    while True:
        start = end - pd.Timedelta(days=int(length_days))
        if start < dates[0]:
            # Window does not fully fit before dates[0].
            # Use the earliest available date instead of silently dropping.
            start = dates[0]
        s = dates[dates >= start][0]
        e = dates[dates <= end][-1]
        n_days = len(dates[(dates >= s) & (dates <= e)])
        # Only accept windows with at least half the requested length.
        if n_days >= int(length_days * 0.5 / 365 * 252):
            cands.append((s, e))
        # Advance backward
        end = end - pd.Timedelta(days=int(step_days))
        if end <= dates[0]:
            break

    cands.reverse()
    return cands


def _validate_calm_scan(top: list, cfg: "Config") -> None:
    """Raise a clear error if CALM was not properly identified."""
    if len(top) == 0 or top[0][2] == float("inf"):
        raise RuntimeError(
            "[CALM ERROR] No valid CALM window found.\n"
            f"  calm_search_to={cfg.calm_search_to}, calm_length_days={cfg.calm_length_days}\n"
            "  The data series is too short relative to the requested CALM window.\n"
            "  Solutions:\n"
            "    1. Reduce --calm_length_days (e.g. 252 instead of 504)\n"
            "    2. Move --calm_search_to later (more data before the search boundary)\n"
            "    3. Extend --start date to provide more pre-crisis history"
        )

def _local_oh_phi_for_segment(seg: pd.DataFrame, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula Oh e phi *localmente* dentro do segmento, SEM depender de CALM global.
    Objetivo: detectar joelho/entrada em aceleração dentro do próprio candidato.
    """
    eta = seg["eta"].to_numpy(dtype=float)
    eta = np.where(np.isfinite(eta), eta, 0.0)

    # eta_ref local (robusto)
    eta_ref = float(np.quantile(eta, cfg.xi_q)) if len(eta) else 1.0
    eta_ref = max(eta_ref, EPS)

    # Xi local
    if cfg.xi_coupling == "eta_inverse":
        Xi = eta_ref / (eta + EPS)
    else:
        Xi = (eta + EPS) / eta_ref
    Xi = np.where(np.isfinite(Xi), Xi, 0.0)
    Xi = np.clip(Xi, 0.0, 1e6)

    # Xi_c local (mesma regra do v4.5)
    # calm_mask = all True, porque é uma calibração interna ao segmento
    m = np.ones(len(Xi), dtype=bool)
    Xi_c = xi_critical(Xi, m, cfg.xi_method, cfg.xi_quantile, cfg.xi_eps_margin)
    Xi_c = max(Xi_c, EPS)

    Oh = Xi / Xi_c
    Oh = np.where(np.isfinite(Oh), Oh, 0.0)
    Oh = np.clip(Oh, 0.0, 1e6)

    # phi local (baseline interno do próprio segmento)
    phi, _ = compute_phi_series(Oh, m, cfg.delta, cfg.gamma, cfg.pre_q, cfg.phi_floor)
    return Oh, phi

def _penalty_knee_phi_tail(seg: pd.DataFrame, cfg: Config) -> float:
    """
    Penaliza candidatos CALM que contenham uma transição interna (joelho) em phi,
    especialmente no tail (último quartil).

    Estratégia:
      - Compara intensidade de crescimento de phi no tail vs head (razão de dphi médio).
      - Mede slope(phi) no tail (anualizado), penalizando aceleração sustentada.
      - Mistura ambas com cfg.calm_knee_mix.
    """
    if cfg.calm_knee_weight <= 0:
        return 0.0

    n = len(seg)
    if n < 40:
        return 0.0

    tail_frac = float(cfg.calm_knee_tail_frac)
    tail_frac = min(max(tail_frac, 0.10), 0.50)  # segurança
    tail_start = int((1.0 - tail_frac) * n)
    tail_start = min(max(tail_start, 5), n - 5)

    # phi local
    _, phi = _local_oh_phi_for_segment(seg, cfg)
    phi = np.asarray(phi, dtype=float)
    if not np.all(np.isfinite(phi)):
        phi = np.where(np.isfinite(phi), phi, np.nan)
        phi = pd.Series(phi).interpolate(limit_direction="both").to_numpy(dtype=float)

    # incrementos
    dphi = np.diff(phi)
    if len(dphi) < 10:
        return 0.0

    # head vs tail em dphi (robusto)
    dphi_head = dphi[:max(3, tail_start - 1)]
    dphi_tail = dphi[max(0, tail_start - 1):]

    head_mean = float(np.nanmedian(np.maximum(dphi_head, 0.0)))  # foca em acúmulo
    tail_mean = float(np.nanmedian(np.maximum(dphi_tail, 0.0)))

    ratio = tail_mean / (head_mean + EPS)

    # componente 1: razão tail/head
    ratio_excess = max(0.0, ratio - float(cfg.calm_knee_ratio))
    pen_ratio = ratio_excess  # linear e simples

    # componente 2: slope no tail (anualizado)
    phi_tail = phi[tail_start:]
    x = np.arange(len(phi_tail), dtype=float)
    if len(phi_tail) >= 8:
        slope, _, _, _, _ = scipy_stats.linregress(x, phi_tail)
        slope_ann = float(slope * 252.0)
    else:
        slope_ann = 0.0

    slope_excess = max(0.0, slope_ann - float(cfg.calm_knee_slope_ann))
    # normaliza suavemente para não explodir:
    pen_slope = slope_excess / (abs(float(cfg.calm_knee_slope_ann)) + 1.0)

    mix = float(cfg.calm_knee_mix)
    mix = min(max(mix, 0.0), 1.0)

    penalty = (1.0 - mix) * pen_ratio + mix * pen_slope
    return float(cfg.calm_knee_weight * penalty)

def calm_score(seg: pd.DataFrame, cfg: Config) -> float:
    """
    Score de CALM (v4.6):
      - base = exatamente o calm_score do v4.5
      - + penalty_knee_phi_tail (se habilitado)
    """
    eta = seg["eta"].to_numpy(dtype=float)
    mc = seg["mean_corr"].to_numpy(dtype=float)
    vr = seg["v_raw"].to_numpy(dtype=float)

    deta = np.diff(eta)
    dmc = np.diff(mc)

    base = (
        float(np.nanstd(eta)) +
        float(np.nanstd(mc)) +
        float(np.nanmean(np.abs(deta))) +
        float(np.nanmean(np.abs(dmc))) +
        0.25 * float(np.nanmean(vr))
    )

    if not np.isfinite(base):
        return float("inf")

    pen = _penalty_knee_phi_tail(seg, cfg)
    total = base + pen
    return float(total)

def choose_calm_scan(state_df: pd.DataFrame, cfg: Config) -> Tuple[pd.Timestamp, pd.Timestamp, List[Tuple[pd.Timestamp, pd.Timestamp, float]]]:
    dates = state_df.index
    search_to = parse_date(cfg.calm_search_to) if cfg.calm_search_to else dates[int(len(dates) * 0.4)]

    cands = calm_candidates(dates, search_to, cfg.calm_length_days, cfg.calm_step_days)

    scored = []
    for s, e in cands:
        seg = state_df.loc[(state_df.index >= s) & (state_df.index <= e)]
        if len(seg) < max(20, cfg.window):
            continue
        scored.append((s, e, calm_score(seg, cfg)))

    scored.sort(key=lambda x: x[2])
    top = scored[:max(int(cfg.calm_topn), 1)]
    _validate_calm_scan(top, cfg)
    chosen = top[0]
    return chosen[0], chosen[1], top

def build_calm_mask(index: pd.DatetimeIndex, calm_start: pd.Timestamp, calm_end: pd.Timestamp) -> np.ndarray:
    idx = pd.to_datetime(index)
    mask = (idx >= calm_start) & (idx <= calm_end)
    return np.asarray(mask, dtype=bool)


# ═══════════════════════════════════════════════════════════════════════
# Ohio + Phi
# ═══════════════════════════════════════════════════════════════════════

def compute_xi_and_oh(state_df: pd.DataFrame, calm_mask: np.ndarray, cfg: Config) -> Tuple[np.ndarray, float, float]:
    eta = state_df["eta"].to_numpy(dtype=float)
    calm_mask = np.asarray(calm_mask, dtype=bool)

    if np.sum(calm_mask) >= 10:
        eta_ref = float(np.quantile(eta[calm_mask], cfg.xi_q))
    else:
        eta_ref = float(np.quantile(eta, cfg.xi_q))
    eta_ref = max(eta_ref, EPS)

    if cfg.xi_coupling == "eta_inverse":
        Xi = eta_ref / (eta + EPS)
    else:
        Xi = (eta + EPS) / eta_ref

    Xi = np.where(np.isfinite(Xi), Xi, 0.0)
    Xi = np.clip(Xi, 0.0, 1e6)

    Xi_c = xi_critical(Xi, calm_mask, cfg.xi_method, cfg.xi_quantile, cfg.xi_eps_margin)
    Xi_c = max(Xi_c, EPS)

    Oh = Xi / Xi_c
    Oh = np.where(np.isfinite(Oh), Oh, 0.0)
    Oh = np.clip(Oh, 0.0, 1e6)

    return Oh, Xi_c, eta_ref


# ═══════════════════════════════════════════════════════════════════════
# Análise de Aquecimento
# ═══════════════════════════════════════════════════════════════════════

def compute_warming_indicators(df: pd.DataFrame, calm_mask: np.ndarray) -> Dict:
    """Computa indicadores de aquecimento independentes de threshold."""
    oh = df["Oh"].values
    oh_calm = oh[calm_mask]

    # Thresholds baseados no CALM
    oh_p90_calm = float(np.percentile(oh_calm, 90))
    oh_p95_calm = float(np.percentile(oh_calm, 95))

    # Análise por ano
    years = df.index.year.unique()
    yearly = {}

    for year in years:
        mask_year = df.index.year == year
        oh_year = oh[mask_year]

        if len(oh_year) < 20:
            continue

        # Tendência
        x = np.arange(len(oh_year))
        slope, _, r, p, _ = scipy_stats.linregress(x, oh_year)
        slope_annual = slope * 252

        # Frequência
        freq_p90 = 100 * np.mean(oh_year > oh_p90_calm)
        freq_p95 = 100 * np.mean(oh_year > oh_p95_calm)
        freq_1 = 100 * np.mean(oh_year > 1.0)

        # Autocorrelação
        acf1 = np.corrcoef(oh_year[:-1], oh_year[1:])[0, 1] if len(oh_year) > 1 else 0.0

        # Runs
        above = oh_year > oh_p90_calm
        runs = []
        run_len = 0
        for a in above:
            if a:
                run_len += 1
            else:
                if run_len > 0:
                    runs.append(run_len)
                run_len = 0
        if run_len > 0:
            runs.append(run_len)

        # Phi
        if "phi" in df.columns:
            phi_year = df.loc[mask_year, "phi"].values
            phi_max = float(np.max(phi_year))
        else:
            phi_max = 0.0

        yearly[year] = {
            "slope": slope_annual,
            "p_val": p,
            "trend": "↑" if slope > 0 and p < 0.05 else "↓" if slope < 0 and p < 0.05 else "—",
            "freq_p90": freq_p90,
            "freq_p95": freq_p95,
            "freq_1": freq_1,
            "acf1": acf1,
            "mean_run": np.mean(runs) if runs else 0,
            "max_run": max(runs) if runs else 0,
            "phi_max": phi_max,
        }

    return {
        "oh_p90_calm": oh_p90_calm,
        "oh_p95_calm": oh_p95_calm,
        "yearly": yearly,
    }

def find_all_crossings(df: pd.DataFrame, calm_end_idx: int, oh_p95_calm: float, phi_c: float, cfg: Config) -> Dict:
    """Encontra todos os crossings com diferentes thresholds e persistências."""
    oh = df["Oh"].values
    phi = df["phi"].values if "phi" in df.columns else np.zeros_like(oh)
    dates = df.index

    def first_cross(series, threshold, persist):
        idx = persist_crossing(series, threshold, persist, calm_end_idx)
        if idx is not None:
            return str(dates[idx].date())
        return "NONE"

    return {
        # Oh > 1.0 (absoluto)
        "oh_1_sens": first_cross(oh, 1.0, cfg.oh_persist_sens),
        "oh_1_confirm": first_cross(oh, 1.0, cfg.oh_persist_confirm),

        # Oh > P95(CALM) (relativo)
        "oh_p95_sens": first_cross(oh, oh_p95_calm, cfg.oh_persist_sens),
        "oh_p95_confirm": first_cross(oh, oh_p95_calm, cfg.oh_persist_confirm),

        # phi > phi_c
        "phi_sens": first_cross(phi, phi_c, cfg.oh_persist_sens),
        "phi_confirm": first_cross(phi, phi_c, cfg.phi_persist),
    }

def generate_analysis_report(
    cfg: Config,
    calm_start: pd.Timestamp,
    calm_end: pd.Timestamp,
    warming: Dict,
    crossings: Dict,
    eta_ref: float,
    Xi_c: float,
    Oh_pre: float,
    phi_c: float
) -> str:
    """Gera relatório completo de análise."""
    lines = []
    lines.append("=" * 70)
    lines.append("Kappa-FIN v0.1 — RELATÓRIO DE ANÁLISE")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Período analisado: {cfg.start} -> {cfg.end}")
    lines.append(f"CALM: {calm_start.date()} -> {calm_end.date()}")
    lines.append("")
    lines.append("PARÂMETROS DE CALIBRAÇÃO:")
    lines.append(f"  eta_ref      = {eta_ref:.6f}")
    lines.append(f"  Xi_c         = {Xi_c:.6f}")
    lines.append(f"  Oh_pre       = {Oh_pre:.6f}")
    lines.append(f"  phi_c        = {phi_c:.6f}")
    if phi_c < 1e-4:
        lines.append("  [NOTE] phi_c is at the floor: the CALM period had zero drive")
        lines.append("  accumulation, which is the expected behavior of a well-identified")
        lines.append("  baseline. The meaningful threshold is Oh_pre + delta above which")
        lines.append("  damage begins to accumulate. phi_c does not inflate the crossing.")
    lines.append(f"  Oh P90(CALM) = {warming['oh_p90_calm']:.4f}")
    lines.append(f"  Oh P95(CALM) = {warming['oh_p95_calm']:.4f}")
    lines.append("")
    lines.append("CALM penalty_knee_phi_tail (v4.6):")
    lines.append(f"  knee_weight   = {cfg.calm_knee_weight:.3f}")
    lines.append(f"  tail_frac     = {cfg.calm_knee_tail_frac:.2f}")
    lines.append(f"  ratio_thresh  = {cfg.calm_knee_ratio:.2f}")
    lines.append(f"  slope_ann_thr = {cfg.calm_knee_slope_ann:.3f}")
    lines.append(f"  mix           = {cfg.calm_knee_mix:.2f}")
    lines.append("")

    lines.append("=" * 70)
    lines.append("PARTE 1: INDICADORES DE AQUECIMENTO")
    lines.append("=" * 70)
    lines.append("")

    lines.append("1.1 TENDÊNCIA DO Oh (slope anualizado)")
    lines.append("-" * 50)
    for year, data in sorted(warming["yearly"].items()):
        sig = "***" if data["p_val"] < 0.001 else "**" if data["p_val"] < 0.01 else "*" if data["p_val"] < 0.05 else ""
        lines.append(f"  {year}: {data['trend']} {data['slope']:+.3f}/ano {sig}")
    lines.append("")

    lines.append("1.2 FREQUÊNCIA ACIMA DOS LIMIARES")
    lines.append("-" * 50)
    lines.append(f"  {'Ano':<6} {'%>P90':>8} {'%>P95':>8} {'%>1.0':>8}")
    for year, data in sorted(warming["yearly"].items()):
        lines.append(f"  {year:<6} {data['freq_p90']:>7.1f}% {data['freq_p95']:>7.1f}% {data['freq_1']:>7.1f}%")
    lines.append("")

    lines.append("1.3 AUTOCORRELAÇÃO E RUNS")
    lines.append("-" * 50)
    lines.append(f"  {'Ano':<6} {'ACF(1)':>8} {'Run médio':>10} {'Run máx':>8}")
    for year, data in sorted(warming["yearly"].items()):
        lines.append(f"  {year:<6} {data['acf1']:>8.3f} {data['mean_run']:>10.1f} {data['max_run']:>8.0f}")
    lines.append("")

    lines.append("1.4 DANO ACUMULADO (phi_max)")
    lines.append("-" * 50)
    for year, data in sorted(warming["yearly"].items()):
        lines.append(f"  {year}: {data['phi_max']:.3f}")
    lines.append("")

    lines.append("=" * 70)
    lines.append("PARTE 2: CROSSINGS (após CALM)")
    lines.append("=" * 70)
    lines.append("")

    lines.append(f"persist_sens={cfg.oh_persist_sens} | persist_confirm={cfg.oh_persist_confirm}")
    lines.append("")

    lines.append("2.1 CROSSING Oh > 1.0 (absoluto)")
    lines.append("-" * 50)
    lines.append(f"  Sensível (p={cfg.oh_persist_sens}):     {crossings['oh_1_sens']}")
    lines.append(f"  Confirmatório (p={cfg.oh_persist_confirm}): {crossings['oh_1_confirm']}")
    lines.append("")

    lines.append(f"2.2 CROSSING Oh > P95(CALM) = {warming['oh_p95_calm']:.4f}")
    lines.append("-" * 50)
    lines.append(f"  Sensível (p={cfg.oh_persist_sens}):     {crossings['oh_p95_sens']}")
    lines.append(f"  Confirmatório (p={cfg.oh_persist_confirm}): {crossings['oh_p95_confirm']}")
    lines.append("")

    lines.append(f"2.3 CROSSING phi > phi_c = {phi_c:.4f}")
    lines.append("-" * 50)
    lines.append(f"  Sensível (p={cfg.oh_persist_sens}):     {crossings['phi_sens']}")
    lines.append(f"  Confirmatório (p={cfg.phi_persist}): {crossings['phi_confirm']}")
    lines.append("")

    lines.append("=" * 70)
    lines.append("CONCLUSÃO")
    lines.append("=" * 70)
    lines.append("")

    yearly = warming["yearly"]
    years_sorted = sorted(yearly.keys())

    for i in range(len(years_sorted) - 1):
        y1, y2 = years_sorted[i], years_sorted[i + 1]
        if yearly[y1]["slope"] < 0 and yearly[y2]["slope"] > 0:
            lines.append(f"✓ INVERSÃO DE TENDÊNCIA: {y1} ({yearly[y1]['trend']}) → {y2} ({yearly[y2]['trend']})")
            break

    for i in range(len(years_sorted) - 1):
        y1, y2 = years_sorted[i], years_sorted[i + 1]
        if yearly[y2]["freq_p90"] > 1.5 * yearly[y1]["freq_p90"]:
            lines.append(f"✓ AUMENTO DE FREQUÊNCIA: {y1}={yearly[y1]['freq_p90']:.1f}% → {y2}={yearly[y2]['freq_p90']:.1f}%")
            break

    if crossings["oh_p95_confirm"] != "NONE":
        lines.append(f"✓ PRIMEIRO CROSSING CONFIRMATÓRIO (Oh>P95): {crossings['oh_p95_confirm']}")

    if crossings["phi_confirm"] != "NONE":
        lines.append(f"✓ CROSSING DE DANO (phi>phi_c): {crossings['phi_confirm']}")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════

def plot_observables(df: pd.DataFrame, outdir: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    ax1.plot(df.index, df["entropy"], label="entropy", alpha=0.8)
    ax1.set_title("Entropy (H1)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ax2.plot(df.index, df["dominance"], label="dominance", alpha=0.8)
    ax2.set_title("Dominance")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    ax3.plot(df.index, df["v_raw"], label="v_raw", alpha=0.8)
    ax3.set_title("V_raw (entropy × anti-dominance)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    ax4.plot(df.index, df["mean_corr"], label="mean_corr", alpha=0.8)
    ax4.set_title("Mean Correlation")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "observables.png"), dpi=150)
    plt.close(fig)

def plot_ohio(df: pd.DataFrame, outdir: str, oh_p95_calm: float, calm_end: pd.Timestamp) -> None:
    fig = plt.figure(figsize=(14, 6))

    plt.plot(df.index, df["Oh"], label="Oh(t)", linewidth=0.8, alpha=0.9)
    plt.axhline(1.0, linestyle="--", color="red", alpha=0.7, label="Oh = 1.0")
    plt.axhline(oh_p95_calm, linestyle="--", color="orange", alpha=0.7, label=f"Oh = P95(CALM) = {oh_p95_calm:.3f}")
    plt.axvline(calm_end, linestyle=":", color="green", alpha=0.5, label=f"CALM end: {calm_end.date()}")

    plt.title("Ohio Number (Oh) — Rayleigh Criterion")
    plt.xlabel("Date")
    plt.ylabel("Oh")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "ohio_number.png"), dpi=150)
    plt.close(fig)

def plot_phi(df: pd.DataFrame, outdir: str, phi_c: float, calm_end: pd.Timestamp) -> None:
    fig = plt.figure(figsize=(14, 6))

    plt.plot(df.index, df["phi"], label="phi(t)", linewidth=0.8, alpha=0.9)
    plt.axhline(phi_c, linestyle="--", color="red", alpha=0.7, label=f"phi_c = {phi_c:.3f}")
    plt.axvline(calm_end, linestyle=":", color="green", alpha=0.5, label=f"CALM end: {calm_end.date()}")

    plt.title("Damage Integral (phi) — Prandtl Criterion")
    plt.xlabel("Date")
    plt.ylabel("phi")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "damage_phi.png"), dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Data download
# ═══════════════════════════════════════════════════════════════════════

def download_prices(cfg: Config) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance não disponível. Instale: pip install yfinance")

    data = yf.download(
        tickers=cfg.tickers,
        start=cfg.start,
        end=cfg.end,
        interval=cfg.interval,
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if data is None or len(data) == 0:
        raise RuntimeError("yfinance retornou vazio.")

    if isinstance(data.columns, pd.MultiIndex):
        lvl1 = data.columns.get_level_values(1)
        field = "Adj Close" if "Adj Close" in set(lvl1) else "Close"
        close = data.xs(field, axis=1, level=1, drop_level=False)
        close.columns = close.columns.get_level_values(0)
    else:
        if "Adj Close" in data.columns:
            close = data[["Adj Close"]].rename(columns={"Adj Close": cfg.tickers[0]})
        elif "Close" in data.columns:
            close = data[["Close"]].rename(columns={"Close": cfg.tickers[0]})
        else:
            raise RuntimeError("Não achei coluna Close/Adj Close no retorno do yfinance.")

    close = close.dropna(how="all").ffill()

    missing_cols = [t for t in cfg.tickers if t not in close.columns]
    if missing_cols:
        print(f"[A][WARN] Tickers ausentes no download: {missing_cols}")

    before = list(close.columns)
    close = close.dropna(axis=1, how="any")
    removed = [c for c in before if c not in close.columns]
    if removed:
        print(f"[A][WARN] Tickers removidos por NaN após ffill: {removed}")

    close = close.dropna()
    return close

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    ret = np.log(prices).diff()
    ret = ret.replace([np.inf, -np.inf], np.nan).dropna()
    return ret


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def run(cfg: Config) -> None:
    ensure_dir(cfg.out)

    print("=" * 70)
    print("Kappa-FIN v0.1 — Rayleigh + Prandtl + CALM scan + Dual Crossing")
    print("=" * 70)
    print(f"Tickers: {cfg.tickers}")
    print(f"Period:  {cfg.start} -> {cfg.end} | interval={cfg.interval}")
    print(f"Window:  {cfg.window} | k={cfg.k} | dep={cfg.dep_method} shrink={cfg.shrink_lambda}")
    print(f"PH:      tau={cfg.tau:g} | max_edge_q={cfg.max_edge_quantile} | cap_inf={cfg.cap_inf}")
    print(f"Curv:    summary={cfg.curv_summary} | eta_floor={cfg.eta_floor}")
    print(f"Xi:      coupling={cfg.xi_coupling} | xi_q={cfg.xi_q} | xi_quantile={cfg.xi_quantile} margin={cfg.xi_eps_margin}")
    print(f"Phi:     mode={cfg.phi_mode} delta={cfg.delta} gamma={cfg.gamma} pre_q={cfg.pre_q} q={cfg.phi_q} persist={cfg.phi_persist}")
    print(f"CALM:    policy={cfg.calm_policy} | length={cfg.calm_length_days}d | step={cfg.calm_step_days}d | topn={cfg.calm_topn}")
    print(f"CALM v4.6 penalty_knee_phi_tail: weight={cfg.calm_knee_weight} tail_frac={cfg.calm_knee_tail_frac} ratio={cfg.calm_knee_ratio} slope_ann_thr={cfg.calm_knee_slope_ann} mix={cfg.calm_knee_mix}")
    print(f"EVAL:    after_calm={cfg.eval_after_calm} | oh_persist_sens={cfg.oh_persist_sens} | oh_persist_confirm={cfg.oh_persist_confirm}")
    print("-" * 70)

    # A) Download
    print("[A] Baixando dados...")
    prices = download_prices(cfg)
    print(f"[A] Tickers úteis: {prices.shape[1]}")
    returns = compute_returns(prices)
    print(f"[A] Returns shape: {returns.shape}")

    # B) Structural state
    print("[B] Computando estado estrutural...")
    state = compute_full_state(returns, cfg)
    print(f"[B] Structural state rows: {len(state)}")

    # C) CALM
    if cfg.calm_policy == "fixed":
        if not cfg.calm_start or not cfg.calm_end:
            raise ValueError("calm_policy=fixed requer --calm_start e --calm_end")
        calm_start = parse_date(cfg.calm_start)
        calm_end = parse_date(cfg.calm_end)
        top = [(calm_start, calm_end, float("nan"))]
        print(f"[CALM] fixed: {calm_start.date()} -> {calm_end.date()}")
    else:
        calm_start, calm_end, top = choose_calm_scan(state, cfg)
        print(f"[CALM] scan chosen: {calm_start.date()} -> {calm_end.date()} | score={top[0][2]:.4f}")
        print("[CALM] top candidates:")
        for i, (s, e, sc) in enumerate(top[:cfg.calm_topn], 1):
            print(f"  {i:02d}. {s.date()} -> {e.date()} | score={sc:.4f}")

    calm_mask = build_calm_mask(state.index, calm_start, calm_end)

    # D) Thresholds
    if cfg.calm_policy == "scan" and cfg.calm_ensemble:
        Xi_cs, phi_cs, eta_refs, Oh_pres = [], [], [], []
        for (s, e, _) in top[:cfg.calm_topn]:
            m = build_calm_mask(state.index, s, e)
            Oh_i, Xi_c_i, eta_ref_i = compute_xi_and_oh(state, m, cfg)
            phi_i, Oh_pre_i = compute_phi_series(Oh_i, m, cfg.delta, cfg.gamma, cfg.pre_q, cfg.phi_floor)
            phi_c_i = phi_critical(phi_i, m, cfg.phi_method, cfg.phi_q)
            Xi_cs.append(Xi_c_i)
            phi_cs.append(phi_c_i)
            eta_refs.append(eta_ref_i)
            Oh_pres.append(Oh_pre_i)

        Xi_c = float(np.median(Xi_cs))
        phi_c = float(np.median(phi_cs))
        eta_ref = float(np.median(eta_refs))
        Oh_pre = float(np.median(Oh_pres))

        Oh, _, _ = compute_xi_and_oh(state, calm_mask, cfg)
        drive = np.maximum(Oh - (Oh_pre + cfg.delta), 0.0)
        phi = np.zeros_like(Oh, dtype=float)
        for i in range(len(Oh)):
            prev = phi[i - 1] if i > 0 else 0.0
            phi[i] = max(cfg.gamma * prev + drive[i], cfg.phi_floor)

        print(f"[ENSEMBLE] Xi_c={Xi_c:.6f} | phi_c={phi_c:.6f} | eta_ref={eta_ref:.6f} | Oh_pre={Oh_pre:.6f}")
    else:
        Oh, Xi_c, eta_ref = compute_xi_and_oh(state, calm_mask, cfg)
        phi, Oh_pre = compute_phi_series(Oh, calm_mask, cfg.delta, cfg.gamma, cfg.pre_q, cfg.phi_floor)
        phi_c = phi_critical(phi, calm_mask, cfg.phi_method, cfg.phi_q)
        print(f"[THRESH] Xi_c={Xi_c:.6f} | phi_c={phi_c:.6f} | eta_ref={eta_ref:.6f} | Oh_pre={Oh_pre:.6f}")

    # E) Store in dataframe
    df = state.copy()
    df["Oh"] = Oh
    df["phi"] = phi
    df["phi_c"] = phi_c
    df["eta_ref"] = eta_ref
    df["Xi_c"] = Xi_c
    df["Oh_pre"] = Oh_pre

    # F) Análise de aquecimento
    print("[C] Analisando indicadores de aquecimento...")
    warming = compute_warming_indicators(df, calm_mask)
    print(f"[C] Oh P90(CALM)={warming['oh_p90_calm']:.4f} | Oh P95(CALM)={warming['oh_p95_calm']:.4f}")

    # G) Crossings
    if cfg.eval_after_calm:
        calm_idx = np.where(calm_mask)[0]
        eval_start_idx = int(calm_idx.max() + 1) if len(calm_idx) else 0
    else:
        eval_start_idx = 0

    crossings = find_all_crossings(df, eval_start_idx, warming["oh_p95_calm"], phi_c, cfg)

    print(f"[D] CROSSINGS (após {calm_end.date()}):")
    print(f"    Oh>1.0:      sens={crossings['oh_1_sens']} | confirm={crossings['oh_1_confirm']}")
    print(f"    Oh>P95:      sens={crossings['oh_p95_sens']} | confirm={crossings['oh_p95_confirm']}")
    print(f"    phi>phi_c:   sens={crossings['phi_sens']} | confirm={crossings['phi_confirm']}")

    # H) Gerar relatório
    report = generate_analysis_report(
        cfg, calm_start, calm_end, warming, crossings,
        eta_ref, Xi_c, Oh_pre, phi_c
    )

    # I) Salvar outputs
    out_csv = os.path.join(cfg.out, "kappa_fin_state.csv")
    df.to_csv(out_csv, index=True)
    print(f"[OK] Saved: {out_csv}")

    out_report = os.path.join(cfg.out, "analysis_report.txt")
    with open(out_report, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[OK] Saved: {out_report}")

    # J) Plots
    plot_observables(df, cfg.out)
    plot_ohio(df, cfg.out, warming["oh_p95_calm"], calm_end)
    plot_phi(df, cfg.out, phi_c, calm_end)
    print(f"[OK] Saved: observables.png, ohio_number.png, damage_phi.png")

    print("")
    print(report)
    print("✅ Done.")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Kappa-FIN v0.1")

    p.add_argument("--tickers", type=str, required=True)
    p.add_argument("--start", type=str, required=True)
    p.add_argument("--end", type=str, required=True)
    p.add_argument("--interval", type=str, default="1d")

    p.add_argument("--window", type=int, default=22)
    p.add_argument("--k", type=int, default=5)

    p.add_argument("--dep_method", type=str, default="spearman")
    p.add_argument("--shrink_lambda", type=float, default=0.05)
    p.add_argument("--dist_mode", type=str, default="corr")
    p.add_argument("--alpha", type=float, default=1.0)

    p.add_argument("--tau", type=float, default=1e-7)
    p.add_argument("--max_edge_quantile", type=float, default=0.99)
    p.add_argument("--cap_inf", action="store_true", default=True)

    p.add_argument("--weight_mode", type=str, default="similarity")
    p.add_argument("--curv_summary", type=str, default="median")
    p.add_argument("--eta_floor", type=float, default=0.05)

    p.add_argument("--v_raw_mode", type=str, default="entropy_x_anti_dom")

    p.add_argument("--xi_coupling", type=str, default="eta_inverse")
    p.add_argument("--xi_q", type=float, default=0.05)
    p.add_argument("--xi_method", type=str, default="quantile")
    p.add_argument("--xi_quantile", type=float, default=0.99)
    p.add_argument("--xi_eps_margin", type=float, default=0.02)

    p.add_argument("--phi_mode", type=str, default="pre_excess")
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--gamma", type=float, default=0.985)
    p.add_argument("--pre_q", type=float, default=0.95)
    p.add_argument("--phi_method", type=str, default="quantile")
    p.add_argument("--phi_q", type=float, default=0.99)
    p.add_argument("--phi_persist", type=int, default=3)
    p.add_argument("--phi_floor", type=float, default=1e-6)

    p.add_argument("--calm_policy", type=str, default="scan")
    p.add_argument("--calm_start", type=str, default=None)
    p.add_argument("--calm_end", type=str, default=None)
    p.add_argument("--calm_length_days", type=int, default=504)
    p.add_argument("--calm_step_days", type=int, default=14)
    p.add_argument("--calm_topn", type=int, default=5)
    p.add_argument("--calm_search_to", type=str, default=None)
    p.add_argument("--calm_ensemble", action="store_true", default=False)

    # v4.6 CALM knee penalty knobs
    p.add_argument("--calm_knee_weight", type=float, default=1.0)
    p.add_argument("--calm_knee_tail_frac", type=float, default=0.25)
    p.add_argument("--calm_knee_ratio", type=float, default=1.50)
    p.add_argument("--calm_knee_slope_ann", type=float, default=0.0)
    p.add_argument("--calm_knee_mix", type=float, default=0.5)

    p.add_argument("--eval_after_calm", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--oh_persist_sens", type=int, default=2)
    p.add_argument("--oh_persist_confirm", type=int, default=5)

    p.add_argument("--out", type=str, default="./out_v46")

    a = p.parse_args()
    tickers = [t.strip() for t in a.tickers.split(",") if t.strip()]

    return Config(
        tickers=tickers,
        start=a.start,
        end=a.end,
        interval=a.interval,
        window=a.window,
        k=a.k,
        dep_method=a.dep_method,
        shrink_lambda=a.shrink_lambda,
        dist_mode=a.dist_mode,
        alpha=a.alpha,
        tau=a.tau,
        max_edge_quantile=a.max_edge_quantile,
        cap_inf=bool(a.cap_inf),
        weight_mode=a.weight_mode,
        curv_summary=a.curv_summary,
        eta_floor=a.eta_floor,
        v_raw_mode=a.v_raw_mode,
        xi_coupling=a.xi_coupling,
        xi_q=a.xi_q,
        xi_method=a.xi_method,
        xi_quantile=a.xi_quantile,
        xi_eps_margin=a.xi_eps_margin,
        phi_mode=a.phi_mode,
        delta=a.delta,
        gamma=a.gamma,
        pre_q=a.pre_q,
        phi_method=a.phi_method,
        phi_q=a.phi_q,
        phi_persist=a.phi_persist,
        phi_floor=a.phi_floor,
        calm_policy=a.calm_policy,
        calm_start=a.calm_start,
        calm_end=a.calm_end,
        calm_length_days=a.calm_length_days,
        calm_step_days=a.calm_step_days,
        calm_topn=a.calm_topn,
        calm_search_to=a.calm_search_to,
        calm_ensemble=bool(a.calm_ensemble),
        calm_knee_weight=a.calm_knee_weight,
        calm_knee_tail_frac=a.calm_knee_tail_frac,
        calm_knee_ratio=a.calm_knee_ratio,
        calm_knee_slope_ann=a.calm_knee_slope_ann,
        calm_knee_mix=a.calm_knee_mix,
        eval_after_calm=bool(a.eval_after_calm),
        oh_persist_sens=a.oh_persist_sens,
        oh_persist_confirm=a.oh_persist_confirm,
        out=a.out,
    )

def main() -> None:
    cfg = parse_args()
    run(cfg)

if __name__ == "__main__":
    main()
