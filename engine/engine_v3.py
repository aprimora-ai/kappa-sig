#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kappa-FIN v3.0 — Implementação exata do Método Kappa (README §IV–V)
=====================================================================

Correções fundamentais em relação ao engine v4.6:

  1. DISTÂNCIA ANGULAR  d_ij = sqrt(2(1 − C_ij))
     O engine anterior usava D = 1 − C (linear). A distância angular é a
     métrica correta derivada da geometria de hiperesfera em ℝⁿ e satisfaz
     a desigualdade triangular — requisito para Rips complex.

  2. Ξ(t) — DIVERSIDADE ESTRUTURAL correta
     Antes: Xi = eta_ref / (eta + ε)  →  isso é o INVERSO de η, não diversidade.
     Agora: Ξ(t) = h1_count(t) normalizado pelo baseline CALM.
     h1_count = número de ciclos H1 = trajetórias topologicamente independentes.

  3. DEF(t) — DIVERGÊNCIA ESTADO-FASE (novo)
     DEF(t) = ‖x(t) − P(ẋ(t))‖
     Implementado como divergência angular entre o vetor de estado S(t)
     e a variação ΔS(t), ambos normalizados. Mede incoerência entre
     onde o sistema está e como está se movendo.

  4. Oh(t) — OHIO NUMBER — implementação exata do method.md §6
     Oh(t) = Ξ(t) / Ξ_c  onde Ξ(t) = η_ref / η(t)
     η_ref = quantile(η[CALM], q=0.05)   (≠ média)
     Ξ_c   = quantile(Ξ[CALM], 0.99) × 1.02
     Interpretando corretamente o Critério Rayleigh: Oh é acoplamento
     estrutural normalizado, não distância Euclidiana 3D.
     η tem variância não-zero mesmo com h1=0 → num. estável em qualquer cenário.

  5. S(t) = (Oh, Φ, η, Ξ, DEF) ∈ ℝ⁵ como vetor de primeira classe

  6. CLASSIFICAÇÃO DE REGIME: Nagare / Utsuroi / Katashi

  7. VISCOSIDADE ESTRUTURAL ν_s = τ_Katashi × η_Katashi / (Ξ_Katashi + ε)
     (Hipótese David Ohio, Março 2026)

  8. Φ* — limiar de irreversibilidade (Proposição B.15)
     Estimado como joelho da curva de Φ ordenada.

  9. EXPANSÃO DE DADOS:
     - Suporte a FNSPID parquet nativo
     - Presets multi-mercado (equities, bonds, commodities, FX, EM)
     - Suporte a séries temporais longas via pandas_datareader / FRED

Autor: David Ohio <odavidohio@gmail.com>
DOI:   10.5281/zenodo.18883821
"""

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    import networkx as nx
except ImportError:
    nx = None

try:
    import gudhi as gd
except ImportError:
    gd = None

EPS = 1e-12

# ═══════════════════════════════════════════════════════════════════════════════
# PRESETS DE DADOS MULTI-MERCADO
# ═══════════════════════════════════════════════════════════════════════════════

ASSET_PRESETS: Dict[str, List[str]] = {
    # Equities globais (alta cobertura yfinance)
    "us_core": [
        "SPY", "QQQ", "IWM", "XLF", "XLK", "XLE", "XLI", "XLV",
        "JPM", "BAC", "C", "GS", "MS", "AAPL", "MSFT",
    ],
    "us_sectors": [
        "XLF", "XLK", "XLE", "XLI", "XLV", "XLY", "XLP", "XLU", "XLB", "XLRE",
    ],
    "global_equity": [
        "SPY",   # US
        "EWJ",   # Japan
        "EWG",   # Germany
        "EWU",   # UK
        "EWC",   # Canada
        "EWA",   # Australia
        "EEM",   # Emerging Markets
        "FXI",   # China
        "EWZ",   # Brazil
        "EWY",   # Korea
        "EWW",   # Mexico
        "EWT",   # Taiwan
    ],
    "multi_asset": [
        "SPY",   # US equities
        "TLT",   # Long bonds
        "IEF",   # Medium bonds
        "GLD",   # Gold
        "USO",   # Oil
        "UUP",   # Dollar
        "HYG",   # High yield
        "EEM",   # EM equities
        "VNQ",   # Real estate
        "GSG",   # Commodities
    ],
    "crisis_core": [
        "SPY", "QQQ", "IWM", "TLT", "GLD",
        "JPM", "BAC", "C", "GS",
        "XLF", "XLE", "HYG", "EEM",
    ],
    "bonds_rates": [
        "TLT", "IEF", "SHY", "HYG", "LQD",
        "EMB", "MBB", "TIP",
    ],
    "commodities_fx": [
        "GLD", "SLV", "USO", "UNG", "DBA",
        "UUP", "FXE", "FXY", "FXB", "FXA",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConfigV3:
    # ── Dados ──────────────────────────────────────────────────────────────────
    tickers:        List[str]
    start:          str
    end:            str
    interval:       str   = "1d"
    preset:         Optional[str] = None          # nome de ASSET_PRESETS
    parquet_path:   Optional[str] = None          # caminho para FNSPID parquet

    # ── Janela rolante ─────────────────────────────────────────────────────────
    window:         int   = 22                    # dias úteis por janela
    k:              int   = 5                     # vizinhos kNN

    # ── Dependência ────────────────────────────────────────────────────────────
    dep_method:     str   = "spearman"
    shrink_lambda:  float = 0.05                  # Ledoit-Wolf shrinkage

    # ── Geometria — distância angular (README §V.1) ───────────────────────────
    # d_ij = sqrt(2*(1 - C_ij))  ← implementação exata do README
    # alpha: exponente extra (1.0 = distância angular pura)
    dist_alpha:     float = 1.0

    # ── PH / topologia ─────────────────────────────────────────────────────────
    max_edge_quantile: float = 0.99
    cap_inf:        bool  = True

    # ── η(t): rigidez dinâmica ─────────────────────────────────────────────────
    curv_summary:   str   = "median"
    eta_floor:      float = 0.05

    # ── Φ(t): memória estrutural (README §IV.3) ────────────────────────────────
    gamma:          float = 0.97                  # taxa de dissipação
    delta:          float = 0.08                  # margem acima de Oh_pre
    pre_q:          float = 0.968                 # quantil de Oh_pre sobre CALM
    phi_floor:      float = 1e-6
    phi_q:          float = 0.99                  # quantil para phi_c
    phi_persist:    int   = 3

    # ── CALM (Baseline) ────────────────────────────────────────────────────────
    calm_policy:        str   = "scan"
    calm_start:         Optional[str] = None
    calm_end:           Optional[str] = None
    calm_length_days:   int   = 504
    calm_step_days:     int   = 14
    calm_topn:          int   = 5
    calm_search_to:     Optional[str] = None

    # CALM anti-knee penalty (mantido do v4.6)
    calm_knee_weight:   float = 1.0
    calm_knee_tail_frac: float = 0.25
    calm_knee_ratio:    float = 1.50
    calm_knee_slope_ann: float = 0.0
    calm_knee_mix:      float = 0.5

    # ── Regime: limites para classificação ────────────────────────────────────
    # Oh < oh_nagare_c  → Nagare
    # oh_nagare_c ≤ Oh < oh_katashi_c → Utsuroi
    # Oh ≥ oh_katashi_c → Katashi
    oh_nagare_c:    float = 0.85   # q75 do CALM (ajustado por dados)
    oh_katashi_c:   float = 1.0    # cruzamento formal

    # ── Eval / crossings ──────────────────────────────────────────────────────
    eval_after_calm:    bool  = True
    oh_persist_sens:    int   = 2
    oh_persist_confirm: int   = 5

    # ── Output ─────────────────────────────────────────────────────────────────
    out:            str   = "./results/out_v3"


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS MATEMÁTICOS
# ═══════════════════════════════════════════════════════════════════════════════

def parse_date(s: str) -> pd.Timestamp:
    return pd.to_datetime(s, utc=False)

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def rolling_windows(n: int, w: int) -> List[Tuple[int, int]]:
    return [(i - w, i) for i in range(w, n + 1)]

def shrink_corr(C: np.ndarray, lam: float) -> np.ndarray:
    """Ledoit-Wolf shrinkage: C̃ = (1-λ)C + λI"""
    n = C.shape[0]
    return (1.0 - lam) * C + lam * np.eye(n)

def corr_spearman(R: np.ndarray) -> np.ndarray:
    """Correlação de Spearman robusta via ranks."""
    ranks = np.apply_along_axis(lambda x: pd.Series(x).rank().to_numpy(), 0, R)
    C = np.corrcoef(ranks, rowvar=False)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    C = np.clip(C, -1.0, 1.0)
    np.fill_diagonal(C, 1.0)
    return C

def angular_distance(C: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Distância angular geodésica — implementação exata do README §V.1:
        d_ij = sqrt(2*(1 − C_ij))

    Esta é a distância de corda na hipersfera unitária em ℝⁿ.
    É uma métrica válida (satisfaz desigualdade triangular) — requisito
    para construção correta do Rips complex.

    alpha: exponente extra (1.0 = distância angular pura)
    """
    D = np.sqrt(np.clip(2.0 * (1.0 - C), 0.0, None))
    if abs(alpha - 1.0) > 1e-9:
        D = np.power(D, alpha)
    np.fill_diagonal(D, 0.0)
    return D

def clamp_k(k: int, n: int) -> int:
    """
    Limita k para garantir esparsidade do grafo kNN.

    Com k >= N-1 (ex: k=5, N=5 → k_eff=4), o grafo torna-se K_N
    (grafo completo), onde toda janela tem a mesma estrutura e portanto
    FR(e) = constante → η = constante → Oh = constante.
    
    Regra: k_eff <= floor((N-1)/2) garante que nenhum nó conecta a
    mais da metade dos vizinhos, preservando variância estrutural.
    Mínimo absoluto = 2 (grafo conexo mínimo para PH H1).
    """
    sparse_max = max(2, (n - 1) // 2)
    return max(1, min(int(k), sparse_max))

def persist_crossing(x: np.ndarray, thr: float, persist: int,
                     start: int = 0) -> Optional[int]:
    x = np.asarray(x, dtype=float)
    run = 0
    for i in range(max(0, start), len(x)):
        if x[i] > thr:
            run += 1
            if run >= persist:
                return i - persist + 1
        else:
            run = 0
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# GRAFO kNN + CURVATURA DE FORMAN-RICCI
# ═══════════════════════════════════════════════════════════════════════════════

def build_knn_graph(D: np.ndarray, k: int):
    if nx is None:
        raise RuntimeError("networkx não instalado: pip install networkx")
    n = D.shape[0]
    k_eff = clamp_k(k, n)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        idx = np.argsort(D[i])
        for j in [x for x in idx if x != i][:k_eff]:
            w = float(D[i, j])
            if not G.has_edge(i, j):
                G.add_edge(i, j, dist=w)
            else:
                G[i][j]["dist"] = min(G[i][j]["dist"], w)
    return G

def forman_ricci(G) -> float:
    """Curvatura de Forman-Ricci: FR(e) = 4 − deg(u) − deg(v) + 3·|triângulos|"""
    if nx is None:
        return 0.0
    deg = dict(G.degree())
    curvs = []
    for u, v in G.edges():
        shared = len(set(G.neighbors(u)) & set(G.neighbors(v)))
        curvs.append(4.0 - deg[u] - deg[v] + 3.0 * shared)
    if not curvs:
        return 0.0
    arr = np.asarray(curvs)
    return float(np.median(arr))


# ═══════════════════════════════════════════════════════════════════════════════
# HOMOLOGIA PERSISTENTE H1 — README §V.2
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ph(D: np.ndarray, max_edge_q: float, cap_inf: bool
               ) -> Tuple[float, float, int]:
    """
    Computa H1 (ciclos) via Rips complex sobre a distância angular D.

    Retorna:
        entropy   — H(lifetimes): entropia de Shannon das persistências H1
        dominance — max_lifetime / Σlifetimes: dominância do ciclo mais longo
        h1_count  — número de ciclos H1 = Ξ(t) bruto

    v(t) = entropy × (1 − dominance) = H(1−D) do README §V.2
    """
    if gd is None:
        return 0.0, 0.0, 0

    n = D.shape[0]
    vals = D[np.triu_indices(n, 1)]
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 0.0, 0.0, 0

    thr = float(np.quantile(vals, max_edge_q))
    thr = max(thr, EPS)

    rips = gd.RipsComplex(distance_matrix=D, max_edge_length=thr)
    st   = rips.create_simplex_tree(max_dimension=2)
    st.compute_persistence()

    H1 = st.persistence_intervals_in_dimension(1)
    if H1 is None or len(H1) == 0:
        return 0.0, 0.0, 0

    births = H1[:, 0]
    deaths = H1[:, 1]
    if cap_inf:
        finite = deaths[np.isfinite(deaths)]
        cap = float(np.max(finite)) if len(finite) else float(thr)
        deaths = np.where(np.isfinite(deaths), deaths, cap)

    lifetimes = deaths - births
    lifetimes = lifetimes[np.isfinite(lifetimes) & (lifetimes > EPS)]
    if len(lifetimes) == 0:
        return 0.0, 0.0, 0

    s = float(np.sum(lifetimes))
    p = lifetimes / (s + EPS)
    entropy   = float(-np.sum(p * np.log(p + EPS)))
    dominance = float(np.max(lifetimes) / (s + EPS))
    return entropy, dominance, len(lifetimes)


# ═══════════════════════════════════════════════════════════════════════════════
# NÚCLEO: ESTADO ESTRUTURAL POR JANELA
# ═══════════════════════════════════════════════════════════════════════════════

def window_state(Rw: np.ndarray, cfg: ConfigV3) -> Dict[str, float]:
    """
    Computa o estado estrutural de uma janela de retornos Rw ∈ ℝ^{w×N}.

    Pipeline exato do README:
      1. C(t)  = Spearman + Ledoit-Wolf shrinkage
      2. D(t)  = sqrt(2*(1 − C(t)))  — distância angular (README §V.1)
      3. G(t)  = grafo kNN sobre D(t)
      4. η(t)  = 1 / (|Forman-Ricci(G)| + floor)   — rigidez dinâmica
      5. PH H1 → entropy, dominance, h1_count
      6. v(t)  = entropy × (1 − dominance)  = H(1-D)  — README §V.2
      7. Ξ_raw(t) = h1_count  (normalizado depois pelo CALM)
    """
    C       = corr_spearman(Rw)
    C       = shrink_corr(C, cfg.shrink_lambda)
    D       = angular_distance(C, cfg.dist_alpha)          # ← FIX fundamental v3
    G       = build_knn_graph(D, cfg.k)
    curv    = forman_ricci(G)
    eta     = 1.0 / (abs(curv) + cfg.eta_floor)
    entropy, dominance, h1_count = compute_ph(D, cfg.max_edge_quantile, cfg.cap_inf)
    v_raw   = float(max(entropy, 0.0) * max(1.0 - dominance, 0.0))
    mean_corr = float(np.mean(C[np.triu_indices(C.shape[0], 1)])) \
                if C.shape[0] > 1 else 0.0

    return {
        "mean_corr":   mean_corr,
        "curv":        float(curv),
        "eta":         float(eta),
        "entropy":     float(entropy),
        "dominance":   float(dominance),
        "h1_count":    int(h1_count),
        "v_raw":       float(v_raw),
    }

def compute_full_state(returns: pd.DataFrame, cfg: ConfigV3) -> pd.DataFrame:
    R, dates = returns.to_numpy(dtype=float), returns.index
    rows, idx = [], []
    for i0, i1 in rolling_windows(len(dates), cfg.window):
        rows.append(window_state(R[i0:i1], cfg))
        idx.append(dates[i1 - 1])
    df = pd.DataFrame(rows, index=pd.to_datetime(idx))
    df.index.name = "date"
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# CALM SCAN (mantido do v4.6 com anti-knee penalty)
# ═══════════════════════════════════════════════════════════════════════════════

def _calm_candidates(dates: pd.DatetimeIndex, search_to: pd.Timestamp,
                     length_days: int, step_days: int
                     ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    end = search_to if search_to in dates else dates[dates <= search_to][-1]
    cands = []
    while True:
        s_raw = end - pd.Timedelta(days=length_days)
        s = dates[dates >= max(s_raw, dates[0])][0]
        e = dates[dates <= end][-1]
        n = len(dates[(dates >= s) & (dates <= e)])
        if n >= int(length_days * 0.5 / 365 * 252):
            cands.append((s, e))
        end -= pd.Timedelta(days=step_days)
        if end <= dates[0]:
            break
    cands.reverse()
    return cands

def _local_phi(seg: pd.DataFrame, cfg: ConfigV3) -> np.ndarray:
    """Φ local para o segmento candidato (sem CALM global)."""
    v  = seg["v_raw"].to_numpy(dtype=float)
    et = seg["eta"].to_numpy(dtype=float)
    # proxy Oh local = v × eta normalizado internamente
    raw = v * et
    pre = float(np.quantile(raw, cfg.pre_q)) if len(raw) else 0.0
    drive = np.maximum(raw - (pre + cfg.delta), 0.0)
    phi = np.zeros(len(raw))
    for i in range(len(raw)):
        prev = phi[i-1] if i > 0 else 0.0
        phi[i] = max(cfg.gamma * prev + drive[i], cfg.phi_floor)
    return phi

def _calm_score(seg: pd.DataFrame, cfg: ConfigV3) -> float:
    eta = seg["eta"].to_numpy(dtype=float)
    mc  = seg["mean_corr"].to_numpy(dtype=float)
    vr  = seg["v_raw"].to_numpy(dtype=float)
    base = (float(np.nanstd(eta)) + float(np.nanstd(mc))
            + float(np.nanmean(np.abs(np.diff(eta))))
            + float(np.nanmean(np.abs(np.diff(mc))))
            + 0.25 * float(np.nanmean(vr)))
    if not np.isfinite(base):
        return float("inf")

    # Anti-knee penalty
    n, w = len(seg), cfg.calm_knee_weight
    if w > 0 and n >= 40:
        phi = _local_phi(seg, cfg)
        dp  = np.diff(phi)
        t0  = int((1 - cfg.calm_knee_tail_frac) * n)
        h_m = float(np.nanmedian(np.maximum(dp[:max(3, t0-1)], 0.0)))
        t_m = float(np.nanmedian(np.maximum(dp[max(0, t0-1):], 0.0)))
        ratio_ex = max(0.0, t_m / (h_m + EPS) - cfg.calm_knee_ratio)
        phi_tail = phi[t0:]
        slope_ann = 0.0
        if len(phi_tail) >= 8:
            x = np.arange(len(phi_tail), dtype=float)
            slope_ann = float(scipy_stats.linregress(x, phi_tail)[0] * 252)
        slope_ex = max(0.0, slope_ann - cfg.calm_knee_slope_ann) / \
                   (abs(cfg.calm_knee_slope_ann) + 1.0)
        mix = min(max(cfg.calm_knee_mix, 0.0), 1.0)
        base += w * ((1 - mix) * ratio_ex + mix * slope_ex)

    return float(base)

def choose_calm(state: pd.DataFrame, cfg: ConfigV3
                ) -> Tuple[pd.Timestamp, pd.Timestamp,
                           List[Tuple[pd.Timestamp, pd.Timestamp, float]]]:
    dates    = state.index
    to       = (parse_date(cfg.calm_search_to) if cfg.calm_search_to
                else dates[int(len(dates) * 0.4)])
    cands    = _calm_candidates(dates, to, cfg.calm_length_days, cfg.calm_step_days)
    scored   = []
    for s, e in cands:
        seg = state.loc[s:e]
        if len(seg) < max(20, cfg.window):
            continue
        scored.append((s, e, _calm_score(seg, cfg)))
    scored.sort(key=lambda x: x[2])
    top = scored[:max(cfg.calm_topn, 1)]
    if not top or top[0][2] == float("inf"):
        raise RuntimeError(
            "[CALM ERROR] Nenhuma janela CALM encontrada.\n"
            "  Aumente --start (mais histórico), reduza --calm_length_days\n"
            "  ou mova --calm_search_to para mais tarde."
        )
    # best = argmin para manter compatibilidade; top = lista completa para ensemble
    return top[0][0], top[0][1], top


def calm_ensemble_params(
    state: pd.DataFrame,
    top: List[Tuple[pd.Timestamp, pd.Timestamp, float]],
    cfg: "ConfigV3",
) -> Dict:
    """
    Ensemble CALM — portado de katashi_run.py (kappa_eegs).

    Em vez de usar apenas o CALM argmin (top[0]), agrega os top-N candidatos
    por mediana ponderada (1/score), produzindo parâmetros de calibração mais
    robustos:

        eta_ref  = median( quantile(η[CALM_i], 0.05)  for i in top )
        Xi_c     = median( quantile(Ξ_Oh[CALM_i], 0.99)*1.02 for i in top )
        Oh_pre   = median( quantile(Oh[CALM_i], pre_q)  for i in top )
        xi_c     = median( quantile(h1[CALM_i], 0.95)   for i in top )

    Retorna dict com todos os parâmetros calibrados + metadados por candidato.
    """
    eta  = state["eta"].to_numpy(dtype=float)
    h1   = state["h1_count"].to_numpy(dtype=float)

    eta_refs, Xi_cs, Oh_pres, xi_cs, scores = [], [], [], [], []

    for (cs_i, ce_i, score_i) in top:
        mask_i = np.asarray((state.index >= cs_i) & (state.index <= ce_i), dtype=bool)
        if mask_i.sum() < max(10, cfg.window):
            continue

        eta_c   = eta[mask_i]
        eta_r   = max(float(np.quantile(eta_c, 0.05)), EPS)
        Xi_Oh_i = eta_r / (eta + EPS)
        Xi_Oh_c = Xi_Oh_i[mask_i]
        Xi_c_i  = max(float(np.quantile(Xi_Oh_c, 0.99)) * 1.02, EPS)
        Oh_i    = Xi_Oh_i / Xi_c_i
        Oh_pre_i = float(np.quantile(Oh_i[mask_i], cfg.pre_q))
        h1_c    = h1[mask_i]
        xi_c_i  = max(float(np.quantile(h1_c, 0.95)) if len(h1_c) > 0 else 1.0, 1.0)

        eta_refs.append(eta_r)
        Xi_cs.append(Xi_c_i)
        Oh_pres.append(Oh_pre_i)
        xi_cs.append(xi_c_i)
        scores.append(score_i if np.isfinite(score_i) else 1e9)

    if not eta_refs:
        # fallback: só o best
        cs0, ce0, _ = top[0]
        mask0 = np.asarray((state.index >= cs0) & (state.index <= ce0), dtype=bool)
        eta_r0 = max(float(np.quantile(eta[mask0], 0.05)), EPS)
        Xi_Oh0 = eta_r0 / (eta + EPS)
        Xi_c0  = max(float(np.quantile(Xi_Oh0[mask0], 0.99)) * 1.02, EPS)
        Oh0    = Xi_Oh0 / Xi_c0
        return dict(
            eta_ref  = eta_r0,
            Xi_c     = Xi_c0,
            Oh_pre   = float(np.quantile(Oh0[mask0], cfg.pre_q)),
            xi_c     = max(float(np.quantile(h1[mask0], 0.95)) if mask0.sum() > 0 else 1.0, 1.0),
            n_calms  = 1,
            candidates = [{
                "start": str(top[0][0].date()), "end": str(top[0][1].date()),
                "score": float(top[0][2]), "eta_ref": eta_r0,
                "Xi_c": Xi_c0, "Oh_pre": float(np.quantile(Oh0[mask0], cfg.pre_q)),
            }],
        )

    # Agregação por mediana
    eta_ref = float(np.median(eta_refs))
    Xi_c    = float(np.median(Xi_cs))
    Oh_pre  = float(np.median(Oh_pres))
    xi_c    = float(np.median(xi_cs))

    candidates = [
        {"start": str(c[0].date()), "end": str(c[1].date()), "score": float(c[2]),
         "eta_ref": float(er), "Xi_c": float(xc),
         "Oh_pre": float(op)}
        for c, er, xc, op in zip(top, eta_refs, Xi_cs, Oh_pres)
    ]

    print(f"   [CALM-ensemble] N={len(eta_refs)} candidatos  "
          f"eta_ref={eta_ref:.6f}  Xi_c={Xi_c:.6f}  Oh_pre={Oh_pre:.4f}")

    return dict(
        eta_ref    = eta_ref,
        Xi_c       = Xi_c,
        Oh_pre     = Oh_pre,
        xi_c       = xi_c,
        n_calms    = len(eta_refs),
        candidates = candidates,
    )

def calm_mask(index: pd.DatetimeIndex, cs: pd.Timestamp, ce: pd.Timestamp
              ) -> np.ndarray:
    return np.asarray((index >= cs) & (index <= ce), dtype=bool)


# ═══════════════════════════════════════════════════════════════════════════════
# OBSERVÁVEIS FORMAIS — README §IV
# ═══════════════════════════════════════════════════════════════════════════════

def compute_Oh(
    state: pd.DataFrame,
    mask: np.ndarray,
    cfg: "ConfigV3",
    *,
    eta_ref: Optional[float] = None,
    Xi_c: Optional[float] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Oh(t) — Ohio Number (Critério Rayleigh) — method.md §6

        η_ref = quantile(η[CALM], q=0.05)       ← flexibilidade de referência
        Ξ(t)  = η_ref / η(t)                    ← índice de acoplamento estrutural
        Ξ_c   = quantile(Ξ[CALM], 0.99) × 1.02  ← limiar crítico calibrado
        Oh(t) = Ξ(t) / Ξ_c                      ← Ohio Number (adimensional)

    Se eta_ref e Xi_c forem fornecidos (vindos de calm_ensemble_params),
    usa esses valores diretamente — sem recalcular a partir da mask.
    Caso contrário, recalcula a partir da mask (comportamento legado).
    """
    eta = state["eta"].to_numpy(dtype=float)

    if eta_ref is None:
        eta_calm = eta[mask] if mask.sum() > 0 else eta
        eta_ref  = max(float(np.quantile(eta_calm, 0.05)), EPS)
    else:
        eta_ref  = max(eta_ref, EPS)

    Xi_Oh = eta_ref / (eta + EPS)

    if Xi_c is None:
        Xi_Oh_calm = Xi_Oh[mask] if mask.sum() > 0 else Xi_Oh
        Xi_c = max(float(np.quantile(Xi_Oh_calm, 0.99)) * 1.02, EPS)
    else:
        Xi_c = max(Xi_c, EPS)

    Oh = Xi_Oh / Xi_c

    Oh_calm_q95 = float(np.quantile(Oh[mask], 0.95)) if mask.sum() > 0 else float(Oh.max())

    meta = {
        "eta_ref":     eta_ref,
        "Xi_c":        Xi_c,
        "Oh_calm_q95": Oh_calm_q95,
    }
    return Oh, meta

def compute_Xi(state: pd.DataFrame, mask: np.ndarray, *, xi_c: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """
    Ξ(t) — Diversidade Estrutural (README §IV.4)
    "Richness of coexisting independent paths."

    Implementação v3: número de ciclos H1 (trajetórias topologicamente
    independentes) normalizado pelo baseline CALM.

    Ξ(t) = h1_count(t) / Ξ_calm_q95
    Ξ > 1 → mais trajetórias independentes que no CALM → mais dispersão possível
    Ξ → 0 → sistema empobrecido → entropia presa (alta viscosidade)
    """
    h1 = state["h1_count"].to_numpy(dtype=float)
    if xi_c is None:
        h1_calm = h1[mask] if mask.sum() > 0 else h1
        xi_c = float(np.quantile(h1_calm, 0.95)) if len(h1_calm) > 0 else 1.0
    xi_c = max(xi_c, 1.0)   # pelo menos 1 ciclo como referência
    Xi = h1 / xi_c
    return Xi, xi_c

def compute_Phi(
    Oh: np.ndarray,
    mask: np.ndarray,
    cfg: "ConfigV3",
    *,
    Oh_pre: Optional[float] = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Φ(t) — Memória Estrutural (README §IV.2, §IV.3)

    Φ_t = γ · Φ_{t-1} + max(0, Oh_t − Oh_pre)

    Se Oh_pre for fornecido (vindo de calm_ensemble_params), usa diretamente.
    Caso contrário, recalcula a partir da mask (comportamento legado).
    """
    if Oh_pre is None:
        Oh_calm = Oh[mask] if mask.sum() >= 10 else Oh
        Oh_pre  = float(np.quantile(Oh_calm, cfg.pre_q))
    drive   = np.maximum(Oh - (Oh_pre + cfg.delta), 0.0)
    phi     = np.zeros_like(Oh)
    for i in range(len(Oh)):
        prev   = phi[i-1] if i > 0 else 0.0
        phi[i] = max(cfg.gamma * prev + drive[i], cfg.phi_floor)
    phi_calm = phi[mask] if mask.sum() >= 10 else phi
    phi_c   = float(np.quantile(phi_calm, cfg.phi_q))
    return phi, Oh_pre, phi_c

def compute_DEF(Oh: np.ndarray, Xi: np.ndarray, eta: np.ndarray,
                v_raw: np.ndarray) -> np.ndarray:
    """
    DEF(t) — Divergência Estado-Fase (README §IV.5)

    DEF(t) = ‖x(t) − P(ẋ(t))‖

    Implementação v3: divergência angular entre o vetor de estado S(t)
    e a variação ΔS(t), ambos normalizados.

    DEF(t) = 1 − |cos(S̃(t), ΔS̃(t))|

    Interpretação:
    - DEF ≈ 0: sistema se move na direção do seu estado atual (coerente / viscoso)
    - DEF ≈ 1: sistema se move perpendicularmente ao seu estado (oscilante)
    - DEF > 1: sistema se move contra o seu estado (collapse)

    Alto DEF durante stress → sistema tentando escapar do Katashi → baixa viscosidade
    Baixo DEF durante stress → sistema coerentemente aprofundando o Katashi → alta viscosidade
    """
    # Matriz de estado S(t) ∈ ℝ^{n×4}
    S = np.column_stack([Oh, Xi, eta, v_raw]).astype(float)

    # Normaliza colunas para mesma escala
    col_std = np.where(S.std(axis=0) > EPS, S.std(axis=0), 1.0)
    S_norm  = S / col_std

    # Variação ΔS(t) = S(t) - S(t-1)  (derivada discreta)
    dS      = np.vstack([np.zeros((1, S_norm.shape[1])),
                         np.diff(S_norm, axis=0)])

    # Normas
    norm_S  = np.linalg.norm(S_norm, axis=1, keepdims=True) + EPS
    norm_dS = np.linalg.norm(dS, axis=1, keepdims=True) + EPS

    # Coseno entre posição e velocidade
    cos_sim = np.sum((S_norm / norm_S) * (dS / norm_dS), axis=1)

    # DEF = 1 - |cos|  ∈ [0, 1]  (0 = alinhado, 1 = perpendicular)
    DEF = 1.0 - np.abs(cos_sim)
    DEF = np.clip(DEF, 0.0, 1.0)
    return DEF


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFICAÇÃO DE REGIME — Nagare / Utsuroi / Katashi
# ═══════════════════════════════════════════════════════════════════════════════

def classify_regime(Oh: np.ndarray, Phi: np.ndarray, DEF: np.ndarray,
                    mask: np.ndarray, cfg: ConfigV3) -> np.ndarray:
    """
    Classifica cada timestep em um dos três regimes do Método Kappa:

      Nagare  (流)  — Regime saudável: fluxo livre, adaptabilidade intacta
      Utsuroi (移)  — Regime transiente: instabilidade crescente, decisões reversíveis
      Katashi (固)  — Regime crítico: qualquer intervenção pode causar colapso sistêmico

    Critérios baseados no README §VII:
      Nagare:  Oh < oh_nagare_c AND Phi < phi_c_calm
      Katashi: Oh >= oh_katashi_c OR Phi significativamente elevado
      Utsuroi: intermediário

    Retorna array de strings: "Nagare", "Utsuroi", "Katashi"
    """
    Oh_calm_q75 = float(np.quantile(Oh[mask], 0.75)) if mask.sum() > 0 else 0.75
    Oh_calm_q95 = float(np.quantile(Oh[mask], 0.95)) if mask.sum() > 0 else 1.0
    Phi_calm_q75= float(np.quantile(Phi[mask], 0.75)) if mask.sum() > 0 else 0.0
    Phi_calm_q95= float(np.quantile(Phi[mask], 0.95)) if mask.sum() > 0 else 0.0

    n = len(Oh)
    regimes = np.full(n, "Nagare", dtype=object)

    for i in range(n):
        in_katashi = (Oh[i] >= Oh_calm_q95) or (Phi[i] >= Phi_calm_q95 * 2.0)
        in_utsuroi = (Oh[i] >= Oh_calm_q75) or (Phi[i] >= Phi_calm_q75)
        if in_katashi:
            regimes[i] = "Katashi"
        elif in_utsuroi:
            regimes[i] = "Utsuroi"

    return regimes


# ═══════════════════════════════════════════════════════════════════════════════
# VISCOSIDADE ESTRUTURAL — Hipótese David Ohio (Mar 2026)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_viscosity(Phi: np.ndarray, Xi: np.ndarray, eta: np.ndarray,
                      regimes: np.ndarray
                      ) -> Dict[str, float]:
    """
    ν_s = τ_Katashi_max × η_Katashi / (Ξ_Katashi + ε)

    Mede por quanto tempo o sistema FICA PRESO no Katashi (τ), com quanta
    rigidez (η), e com quantas trajetórias disponíveis para escapar (Ξ).

    Alta ν_s → Φ acumula → Φ preditivo (Tipo A)
    ν_s ≈ 0  → Φ não acumula → correto negativo (Tipo C)
    """
    katashi_mask = regimes == "Katashi"

    # τ_Katashi: maior run consecutivo em Katashi
    runs, cur = [], 0
    for s in katashi_mask:
        if s: cur += 1
        else:
            if cur > 0: runs.append(cur); cur = 0
    if cur > 0: runs.append(cur)
    tau_max  = float(max(runs)) if runs else 0.0
    tau_mean = float(np.mean(runs)) if runs else 0.0

    # η e Ξ durante Katashi
    eta_K = float(eta[katashi_mask].mean()) if katashi_mask.sum() > 0 else 0.0
    xi_K  = float(Xi[katashi_mask].mean())  if katashi_mask.sum() > 0 else 0.0

    # ∫Φ dt  (soma discreta) — memória total acumulada
    phi_integral = float(Phi.sum())

    # ν_s é uma propriedade observável intrínseca do fluxo estrutural —
    # existe independentemente de Φ acumulado, assim como viscosidade
    # de um fluido existe independentemente de tensão aplicada.
    #
    # Problema de degenerescência: quando Ξ_Katashi → 0 (sem ciclos H1),
    # o denominador colapsa. A solução correta é calibrar o piso de Ξ
    # sobre o CALM — o estado mais fluído observado — não usar EPS arbitrário.
    # Piso: q5 de Ξ durante o período inteiro (mínimo observável real).
    xi_floor = max(float(np.quantile(Xi[Xi > 0], 0.05)) if (Xi > 0).sum() > 5 else EPS, EPS)
    nu_s = tau_max * eta_K / (xi_K + xi_floor)

    # Φ* estimado: joelho da curva Φ ordenada (limiar de irreversibilidade)
    phi_sorted = np.sort(Phi[Phi > Phi.mean()])
    phi_star = 0.0
    if len(phi_sorted) > 10:
        x = np.linspace(0, 1, len(phi_sorted))
        # Joelho: ponto de máxima curvatura
        d1  = np.gradient(phi_sorted, x)
        d2  = np.gradient(d1, x)
        curv_knee = np.abs(d2) / (1.0 + d1**2)**1.5
        knee_idx  = int(np.argmax(curv_knee))
        phi_star  = float(phi_sorted[knee_idx])

    return {
        "tau_Katashi_max":  tau_max,
        "tau_Katashi_mean": tau_mean,
        "eta_Katashi":      eta_K,
        "Xi_Katashi":       xi_K,
        "nu_s":             nu_s,
        "phi_integral":     phi_integral,
        "phi_max":          float(Phi.max()),
        "phi_star":         phi_star,
        "frac_Katashi":     float(katashi_mask.mean()),
        "frac_Nagare":      float((regimes == "Nagare").mean()),
        "frac_Utsuroi":     float((regimes == "Utsuroi").mean()),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CROSSINGS / DETECÇÃO DE EVENTOS
# ═══════════════════════════════════════════════════════════════════════════════

def find_crossings(Oh: np.ndarray, Phi: np.ndarray, regimes: np.ndarray,
                   dates: pd.DatetimeIndex, Oh_c95: float, phi_c: float,
                   eval_start: int, cfg: ConfigV3) -> Dict:
    def fc(arr, thr, persist):
        idx = persist_crossing(arr, thr, persist, eval_start)
        return str(dates[idx].date()) if idx is not None else "NONE"

    # Primeiro cruzamento Oh > limiar
    oh1  = fc(Oh, 1.0,    cfg.oh_persist_sens)
    oh1c = fc(Oh, 1.0,    cfg.oh_persist_confirm)
    ohq  = fc(Oh, Oh_c95, cfg.oh_persist_sens)
    ohqc = fc(Oh, Oh_c95, cfg.oh_persist_confirm)

    # Primeiro Φ > phi_c
    phi_s = fc(Phi, phi_c, cfg.oh_persist_sens)
    phi_c2 = fc(Phi, phi_c, cfg.phi_persist)

    # Primeira entrada em Katashi (persistente)
    katashi_int = (regimes == "Katashi").astype(float)
    k_idx = persist_crossing(katashi_int, 0.5, cfg.oh_persist_confirm, eval_start)
    kat_s = str(dates[k_idx].date()) if k_idx is not None else "NONE"

    return {
        "oh_1_sens":       oh1,  "oh_1_confirm":  oh1c,
        "oh_q95_sens":     ohq,  "oh_q95_confirm":ohqc,
        "phi_sens":        phi_s,"phi_confirm":   phi_c2,
        "katashi_confirm": kat_s,
        "Oh_c95":          Oh_c95,
        "phi_c":           phi_c,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS — DASHBOARD COMPLETO S(t)
# ═══════════════════════════════════════════════════════════════════════════════

REGIME_COLORS = {"Nagare": "#2e7d32", "Utsuroi": "#f57f17", "Katashi": "#c62828"}

def plot_dashboard(df: pd.DataFrame, calm_end: pd.Timestamp,
                   crossings: Dict, viscosity: Dict,
                   out: str, title: str = "") -> None:
    """
    Dashboard completo com os 5 observáveis formais S(t) + regimes.
    """
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor("#f5f6fa")
    gs = gridspec.GridSpec(4, 2, hspace=0.42, wspace=0.28,
                           left=0.06, right=0.97, top=0.93, bottom=0.05)

    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(8)]
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(alpha=0.2)

    dates = df.index

    def shade_regimes(ax):
        if "regime" not in df.columns:
            return
        reg = df["regime"].values
        i = 0
        while i < len(reg):
            j = i
            while j < len(reg) and reg[j] == reg[i]:
                j += 1
            c = REGIME_COLORS.get(reg[i], "#888888")
            ax.axvspan(dates[i], dates[j-1], alpha=0.08, color=c)
            i = j

    def vline(ax, lbl, col, style):
        ax.axvline(calm_end, color=col, lw=1.2, linestyle=style, alpha=0.7,
                   label=f"CALM end ({calm_end.date()})" if lbl else "")
        if crossings.get("phi_confirm", "NONE") != "NONE":
            cd = pd.Timestamp(crossings["phi_confirm"])
            ax.axvline(cd, color="#8b0000", lw=1.5, linestyle="--", alpha=0.85)

    # ── P1: Oh(t) — Pressão de Regime ────────────────────────────────────────
    ax = axes[0]
    shade_regimes(ax)
    ax.plot(dates, df["Oh"], color="#1a4a7a", lw=1.4, label="Oh(t) — Pressão")
    ax.axhline(crossings["Oh_c95"], color="#e65100", lw=1.0, ls="--", alpha=0.8,
               label=f"Oh_c95 = {crossings['Oh_c95']:.3f}")
    ax.axhline(1.0, color="#c62828", lw=0.8, ls=":", alpha=0.6, label="Oh = 1.0")
    vline(ax, True, "green", ":")
    ax.set_title("Oh(t) — Pressão de Regime\n"
                 "[Distância normalizada do baseline CALM]", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")

    # ── P2: Φ(t) — Memória Estrutural ────────────────────────────────────────
    ax = axes[1]
    shade_regimes(ax)
    ax.fill_between(dates, df["phi"], alpha=0.4, color="#1a4a7a")
    ax.plot(dates, df["phi"], color="#1a4a7a", lw=1.4, label="Φ(t) — Memória")
    ax.axhline(crossings["phi_c"], color="#c62828", lw=1.0, ls="--", alpha=0.8,
               label=f"Φ_c = {crossings['phi_c']:.4f}")
    if viscosity["phi_star"] > 0:
        ax.axhline(viscosity["phi_star"], color="#8b0000", lw=1.0, ls="-.", alpha=0.7,
                   label=f"Φ* = {viscosity['phi_star']:.4f} (irrevers.)")
    vline(ax, False, "green", ":")
    ax.set_title("Φ(t) — Memória Estrutural\n"
                 "[Dano acumulado: Φ_t = γΦ_{t-1} + max(0, Oh_t − Oh_pre)]",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")

    # ── P3: η(t) — Rigidez Dinâmica ──────────────────────────────────────────
    ax = axes[2]
    shade_regimes(ax)
    ax.plot(dates, df["eta"], color="#006064", lw=1.2, label="η(t) — Rigidez")
    vline(ax, False, "green", ":")
    ax.set_title("η(t) — Rigidez Dinâmica\n"
                 "[η = 1/(|Forman-Ricci| + ε)]", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    # ── P4: Ξ(t) — Diversidade Estrutural ────────────────────────────────────
    ax = axes[3]
    shade_regimes(ax)
    ax.plot(dates, df["Xi"], color="#4a148c", lw=1.2, label="Ξ(t) — Diversidade")
    ax.axhline(1.0, color="#888", lw=0.8, ls="--", alpha=0.6, label="Ξ = Ξ_CALM_q95")
    vline(ax, False, "green", ":")
    ax.set_title("Ξ(t) — Diversidade Estrutural\n"
                 "[Ciclos H1 normalizados: trajetórias independentes]",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    # ── P5: DEF(t) — Divergência Estado-Fase ─────────────────────────────────
    ax = axes[4]
    shade_regimes(ax)
    ax.plot(dates, df["DEF"], color="#bf360c", lw=1.0, alpha=0.8,
            label="DEF(t) — Divergência Estado-Fase")
    ax.fill_between(dates, df["DEF"], alpha=0.25, color="#bf360c")
    vline(ax, False, "green", ":")
    ax.set_title("DEF(t) — Divergência Estado-Fase\n"
                 "[Incoerência entre posição e dinâmica do sistema]",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    # ── P6: v(t) — Complexidade Topológica ───────────────────────────────────
    ax = axes[5]
    shade_regimes(ax)
    ax.plot(dates, df["v_raw"], color="#558b2f", lw=1.2, label="v(t) = H×(1−D)")
    vline(ax, False, "green", ":")
    ax.set_title("v(t) — Complexidade Topológica\n"
                 "[v = H(1−D): entropia × anti-dominância H1]",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    # ── P7: Viscosidade e Regime ──────────────────────────────────────────────
    ax = axes[6]
    if "regime" in df.columns:
        reg_map = {"Nagare": 0, "Utsuroi": 1, "Katashi": 2}
        reg_num = np.array([reg_map.get(r, 0) for r in df["regime"]])
        for r, c in REGIME_COLORS.items():
            mask_r = np.array(df["regime"]) == r
            if mask_r.any():
                ax.fill_between(dates, np.where(mask_r, reg_num, np.nan),
                                alpha=0.6, color=c, label=r, step="post")
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["Nagare", "Utsuroi", "Katashi"])
    nu = viscosity.get("nu_s", 0.0)
    ax.set_title(f"Classificação de Regime: Nagare / Utsuroi / Katashi\n"
                 f"ν_s = {nu:.2f}  |  ∫Φ = {viscosity['phi_integral']:.3f}  "
                 f"|  τ_Katashi_max = {viscosity['tau_Katashi_max']:.0f}d",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")

    # ── P8: Correlação média (contexto) ───────────────────────────────────────
    ax = axes[7]
    ax.plot(dates, df["mean_corr"], color="#37474f", lw=1.0, alpha=0.8,
            label="Correlação média (Spearman)")
    vline(ax, False, "green", ":")
    ax.set_title("Correlação Média (Spearman + Ledoit-Wolf)\n"
                 "[Contexto de sincronização do mercado]",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    suptitle = (f"Kappa-FIN v3.0 — {title}\n"
                "S(t) = (Oh, Φ, η, Ξ, DEF)  |  d_ij = √(2(1−C_ij))  "
                f"|  γ={df.attrs.get('gamma', '?')}  δ={df.attrs.get('delta', '?')}")
    fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=0.99)
    fig.text(0.5, 0.01,
             "David Ohio · odavidohio@gmail.com · Kappa-FIN v3 · DOI: 10.5281/zenodo.18883821",
             ha="center", fontsize=9, color="#666")

    # Legenda de regimes no rodapé
    for r, c in REGIME_COLORS.items():
        fig.patches.append(plt.Rectangle((0, 0), 0, 0, color=c, alpha=0.5, label=r))

    out_path = os.path.join(out, "kappa_v3_dashboard.png")
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Dashboard: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD E DADOS
# ═══════════════════════════════════════════════════════════════════════════════

def load_from_parquet(path: str, tickers: Optional[List[str]],
                      start: str, end: str) -> pd.DataFrame:
    """Carrega preços de FNSPID parquet."""
    print(f"   [DATA] Carregando FNSPID: {path}")
    prices = pd.read_parquet(path)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.loc[start:end]
    if tickers:
        avail = [t for t in tickers if t in prices.columns]
        if not avail:
            raise ValueError(f"Nenhum ticker encontrado no parquet: {tickers[:5]}...")
        prices = prices[avail]
    prices = prices.dropna(how="all").ffill().dropna(axis=1, how="any").dropna()
    print(f"   [DATA] Parquet: {prices.shape[1]} tickers, {len(prices)} dias")
    return prices

def download_prices(cfg: ConfigV3) -> pd.DataFrame:
    """Download via yfinance com limpeza robusta.

    Suporta MultiIndex em qualquer ordem (Price, Ticker) ou (Ticker, Price),
    conforme versão do yfinance. Usa filtro NaN permissivo para preservar
    o máximo de ativos possível, cortando o período ao máximo de cobertura
    comum.
    """
    if yf is None:
        raise RuntimeError("yfinance não instalado: pip install yfinance")

    tickers = cfg.tickers
    print(f"   [DATA] Download yfinance: {len(tickers)} tickers  {cfg.start}→{cfg.end}")

    raw = yf.download(
        tickers=tickers, start=cfg.start, end=cfg.end,
        interval=cfg.interval, group_by="ticker",
        auto_adjust=True, progress=False, threads=True,
    )
    if raw is None or len(raw) == 0:
        raise RuntimeError("yfinance retornou vazio.")

    # ── Extrai coluna Close de forma robusta, independente da versão yfinance ──
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = set(raw.columns.get_level_values(0))
        lvl1 = set(raw.columns.get_level_values(1))
        if "Close" in lvl0:
            # Ordem (Price, Ticker) → raw["Close"] dá DataFrame com tickers como colunas
            prices = raw["Close"]
        elif "Close" in lvl1:
            # Ordem (Ticker, Price) → xs no level=1
            prices = raw.xs("Close", axis=1, level=1)
        else:
            # Fallback: tentar pegar qualquer campo numérico coerente
            field = [f for f in raw.columns.get_level_values(0) if f not in
                     ("Open", "High", "Low", "Volume", "Dividends", "Stock Splits")]
            if field:
                prices = raw[field[0]]
                if isinstance(prices, pd.Series):
                    prices = prices.to_frame(name=tickers[0])
            else:
                raise RuntimeError("Não foi possível localizar coluna Close no MultiIndex.")
    else:
        col = "Close" if "Close" in raw.columns else raw.columns[0]
        if len(tickers) == 1:
            prices = raw[[col]].rename(columns={col: tickers[0]})
        else:
            prices = raw[[col]]

    # ── Limpeza permissiva: preserva o máximo de ativos E o período histórico ──
    prices.index = pd.to_datetime(prices.index)
    before = list(prices.columns)

    # 1. Remove tickers completamente sem dados
    prices = prices.dropna(axis=1, how="all")

    # 2. Forward-fill (máx 5 dias) para preencher feriados locais
    prices = prices.ffill(limit=5)

    # 3. Determina quando cada ticker começa efetivamente
    first_valid = prices.apply(lambda s: s.first_valid_index())

    # 4. Remove tickers que só começam depois de calm_search_to.
    #    Esses tickers não cobrem o período CALM e tornam common_start posterior
    #    à janela de calibração, causando IndexError em _calm_candidates.
    calm_cutoff = pd.Timestamp(cfg.calm_search_to) if cfg.calm_search_to else None
    if calm_cutoff is None:
        # sem calm_search_to explícito: usa 40% do período como proxy
        period_start = pd.Timestamp(cfg.start)
        period_end   = pd.Timestamp(cfg.end)
        calm_cutoff  = period_start + (period_end - period_start) * 0.4
    late_tickers = first_valid[first_valid > calm_cutoff].index.tolist()
    if late_tickers:
        prices = prices.drop(columns=late_tickers)
        print(f"   [WARN] Removidos por cobertura tardia (começam após calm_search_to={calm_cutoff.date()}): "
              f"{late_tickers}")
        first_valid = first_valid.drop(late_tickers)

    # 5. Corta o início para o primeiro dia em que TODOS os tickers restantes têm dados
    common_start = first_valid.max()
    prices = prices.loc[common_start:]

    # 6. Remove tickers com <50% de linhas restantes (cobertura muito esparça)
    min_rows = max(int(0.5 * len(prices)), cfg.window * 2)
    prices = prices.dropna(axis=1, thresh=min_rows)

    # 7. Remove dias residuais com qualquer NaN
    prices = prices.dropna(how="any")

    removed = [c for c in before if c not in prices.columns]
    if removed:
        print(f"   [WARN] Removidos (NaN ou cobertura insuficiente): {removed}")
    if len(prices.columns) < 4:
        raise RuntimeError(
            f"Apenas {len(prices.columns)} ticker(s) com dados suficientes "
            f"no período {cfg.start}→{cfg.end}. "
            f"O Método Kappa requer mínimo 4 ativos. "
            f"Adicione tickers ao cenário ou amplie o período."
        )
    print(f"   [DATA] Pronto: {prices.shape[1]} tickers, {len(prices)} dias  "
          f"(início efetivo: {prices.index[0].date()})")
    return prices

def get_prices(cfg: ConfigV3) -> pd.DataFrame:
    """Roteador: parquet > yfinance."""
    if cfg.parquet_path and os.path.exists(cfg.parquet_path):
        return load_from_parquet(cfg.parquet_path, cfg.tickers, cfg.start, cfg.end)
    return download_prices(cfg)

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    ret = np.log(prices).diff().replace([np.inf, -np.inf], np.nan).dropna()
    return ret


# ═══════════════════════════════════════════════════════════════════════════════
# RELATÓRIO
# ═══════════════════════════════════════════════════════════════════════════════

def build_report(cfg: ConfigV3, n_tickers: int, n_days: int,
                 calm_start: pd.Timestamp, calm_end: pd.Timestamp,
                 Oh_meta: Dict, phi_c: float, Oh_pre: float, xi_c: float,
                 crossings: Dict, viscosity: Dict) -> str:
    L = []
    L += ["=" * 72,
          "Kappa-FIN v3.0 — RELATÓRIO DE ANÁLISE COMPLETA",
          "Implementação exata do Método Kappa (README §IV–V)",
          "Autor: David Ohio <odavidohio@gmail.com>",
          "=" * 72, ""]

    L += [f"Período:    {cfg.start} → {cfg.end}",
          f"Tickers:    {n_tickers}  ({cfg.window}d janela, k={cfg.k})",
          f"Retornos:   {n_days} dias úteis",
          f"Distância:  d_ij = sqrt(2*(1 − C_ij))  ← distância angular v3",
          f"CALM:       {calm_start.date()} → {calm_end.date()}",
          ""]

    L += ["─" * 72,
          "PARÂMETROS DE CALIBRAÇÃO (S(t) = (Oh, Φ, η, Ξ, DEF))",
          "─" * 72]

    L += [f"  η_ref (CALM q5 de η)        = {Oh_meta['eta_ref']:.6f}",
          f"  Ξ_c  (CALM q99 de Ξ(η))    = {Oh_meta['Xi_c']:.6f}",
          f"  Oh_c95 (CALM q95)         = {crossings['Oh_c95']:.4f}",
          f"  Oh_pre (Φ baseline, q{cfg.pre_q:.3f}) = {Oh_pre:.4f}",
          f"  Φ_c  (CALM q99 de Φ)      = {phi_c:.6f}",
          f"  Φ*   (irreversibilidade)  = {viscosity['phi_star']:.6f}",
          f"  h1_Ξ_c (CALM q95 de h1)   = {xi_c:.4f}",
          f"  γ    (dissipação Φ)       = {cfg.gamma}",
          f"  δ    (margem Oh→Φ)        = {cfg.delta}",
          ""]

    L += ["─" * 72, "VISCOSIDADE ESTRUTURAL (ν_s)", "─" * 72]
    v = viscosity
    L += [f"  ν_s = τ_Katashi_max × η_Katashi / (Ξ_Katashi + ε)",
          f"      = {v['tau_Katashi_max']:.0f} × {v['eta_Katashi']:.4f} / "
          f"({v['Xi_Katashi']:.4f} + ε)",
          f"      = {v['nu_s']:.4f}",
          f"  ∫Φ dt              = {v['phi_integral']:.4f}",
          f"  τ_Katashi_max      = {v['tau_Katashi_max']:.0f} dias",
          f"  τ_Katashi_mean     = {v['tau_Katashi_mean']:.1f} dias",
          f"  Fração Nagare      = {v['frac_Nagare']*100:.1f}%",
          f"  Fração Utsuroi     = {v['frac_Utsuroi']*100:.1f}%",
          f"  Fração Katashi     = {v['frac_Katashi']*100:.1f}%",
          ""]

    L += ["─" * 72, "CROSSINGS (após CALM)", "─" * 72]
    cr = crossings
    L += [f"  Oh > 1.0   sens(p={cfg.oh_persist_sens}): {cr['oh_1_sens']}",
          f"  Oh > 1.0   conf(p={cfg.oh_persist_confirm}): {cr['oh_1_confirm']}",
          f"  Oh > q95   sens: {cr['oh_q95_sens']}",
          f"  Oh > q95   conf: {cr['oh_q95_confirm']}",
          f"  Φ > Φ_c    sens: {cr['phi_sens']}",
          f"  Φ > Φ_c    conf: {cr['phi_confirm']}",
          f"  Katashi    conf: {cr['katashi_confirm']}",
          ""]

    L += ["─" * 72, "PROPOSIÇÃO B.15 — IRREVERSIBILIDADE", "─" * 72]
    phi_max = v["phi_max"]   # Φ máximo real da série
    irreversible = v["phi_star"] > 0 and phi_max > v["phi_star"]
    L += [f"  Φ* estimado (joelho): {v['phi_star']:.6f}",
          f"  Φ_max real:           {phi_max:.6f}",
          f"  Irreversibilidade:    {'✅ ATINGIDA (Φ > Φ*)' if irreversible else '❌ Não atingida'}",
          ""]

    L += ["=" * 72]
    return "\n".join(L)


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def run(cfg: ConfigV3, scenario_title: str = "") -> pd.DataFrame:
    ensure_dir(cfg.out)

    # Resolve preset
    if cfg.preset and cfg.preset in ASSET_PRESETS and not cfg.tickers:
        cfg.tickers = ASSET_PRESETS[cfg.preset]

    print("\n" + "=" * 72)
    print("Kappa-FIN v3.0 — Implementação Exata do Método Kappa")
    print("David Ohio · odavidohio@gmail.com")
    print("=" * 72)

    # ── A. Dados ──────────────────────────────────────────────────────────────
    print("[A] Carregando dados...")
    prices  = get_prices(cfg)
    returns = compute_returns(prices)
    print(f"    Returns: {returns.shape}")

    # ── B. Estado estrutural ──────────────────────────────────────────────────
    print("[B] Computando estado estrutural [d_ij = sqrt(2(1-C_ij))]...")
    state = compute_full_state(returns, cfg)
    print(f"    State rows: {len(state)}")

    # ── C. CALM ───────────────────────────────────────────────────────────────
    if cfg.calm_policy == "fixed":
        if not cfg.calm_start or not cfg.calm_end:
            raise ValueError("calm_policy=fixed requer calm_start e calm_end")
        cs, ce = parse_date(cfg.calm_start), parse_date(cfg.calm_end)
        top = [(cs, ce, float("nan"))]
        print(f"[C] CALM fixo: {cs.date()} → {ce.date()}")
    else:
        cs, ce, top = choose_calm(state, cfg)
        print(f"[C] CALM scan: {cs.date()} → {ce.date()}  "
              f"| score={top[0][2]:.4f}")
        for i, (s, e, sc) in enumerate(top[:cfg.calm_topn], 1):
            print(f"    {i:02d}. {s.date()} → {e.date()}  score={sc:.4f}")

    mask = calm_mask(state.index, cs, ce)

    # ── D. 5 Observáveis Formais S(t) = (Oh, Φ, η, Ξ, DEF) ──────────────────
    print("[D] Computando S(t) = (Oh, Φ, η, Ξ, DEF)...")

    # CALM ensemble: agrega top-N candidatos por mediana (portado de kappa_eegs)
    ens = calm_ensemble_params(state, top, cfg)

    Oh, Oh_meta = compute_Oh(state, mask, cfg,
                             eta_ref=ens["eta_ref"], Xi_c=ens["Xi_c"])
    Xi, xi_c   = compute_Xi(state, mask, xi_c=ens["xi_c"])
    Phi, Oh_pre, phi_c = compute_Phi(Oh, mask, cfg, Oh_pre=ens["Oh_pre"])
    DEF        = compute_DEF(Oh, Xi, state["eta"].values, state["v_raw"].values)

    print(f"    Oh: min={Oh.min():.3f}  max={Oh.max():.3f}  "
          f"Oh_c95={Oh_meta['Oh_calm_q95']:.3f}")
    print(f"    Φ:  max={Phi.max():.6f}  phi_c={phi_c:.6f}")
    print(f"    Ξ:  min={Xi.min():.3f}  max={Xi.max():.3f}  Ξ_c={xi_c:.3f}")
    print(f"    DEF:min={DEF.min():.3f}  max={DEF.max():.3f}")

    # ── E. Classificação de Regime ────────────────────────────────────────────
    print("[E] Classificando regimes (Nagare / Utsuroi / Katashi)...")
    regimes = classify_regime(Oh, Phi, DEF, mask, cfg)
    counts = {r: int((regimes == r).sum()) for r in ["Nagare", "Utsuroi", "Katashi"]}
    print(f"    {counts}")

    # ── F. Viscosidade Estrutural ─────────────────────────────────────────────
    print("[F] Computando viscosidade estrutural ν_s...")
    viscosity = compute_viscosity(Phi, Xi, state["eta"].values, regimes)
    print(f"    ν_s = {viscosity['nu_s']:.4f}  |  ∫Φ = {viscosity['phi_integral']:.4f}  "
          f"|  τ_Katashi_max = {viscosity['tau_Katashi_max']:.0f}d")

    # ── G. Monta DataFrame final ──────────────────────────────────────────────
    df = state.copy()
    df["Oh"]     = Oh
    df["Xi"]     = Xi
    df["phi"]    = Phi
    df["DEF"]    = DEF
    df["regime"] = regimes
    df["phi_c"]  = phi_c
    df["Oh_pre"] = Oh_pre
    df["Xi_c"]   = xi_c
    df.attrs["gamma"] = cfg.gamma
    df.attrs["delta"] = cfg.delta

    # ── H. Crossings ──────────────────────────────────────────────────────────
    print("[G] Detectando crossings...")
    eval_start = int(np.where(mask)[0].max() + 1) if mask.sum() > 0 else 0
    crossings = find_crossings(
        Oh, Phi, regimes, state.index,
        Oh_meta["Oh_calm_q95"], phi_c,
        eval_start if cfg.eval_after_calm else 0, cfg
    )
    for k, v in crossings.items():
        if k not in ("Oh_c95", "phi_c"):
            print(f"    {k}: {v}")

    # ── I. Relatório ──────────────────────────────────────────────────────────
    report = build_report(
        cfg, len(prices.columns), len(returns),
        cs, ce, Oh_meta, phi_c, Oh_pre, xi_c,
        crossings, viscosity
    )

    # ── J. Salva outputs ──────────────────────────────────────────────────────
    csv_path = os.path.join(cfg.out, "kappa_v3_state.csv")
    df.to_csv(csv_path)
    print(f"[OK] State CSV: {csv_path}")

    rep_path = os.path.join(cfg.out, "kappa_v3_report.txt")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[OK] Relatório: {rep_path}")

    # Viscosidade CSV
    visc_path = os.path.join(cfg.out, "kappa_v3_viscosity.csv")
    pd.Series(viscosity).to_csv(visc_path, header=["value"])
    print(f"[OK] Viscosidade: {visc_path}")

    # ── K. Dashboard ──────────────────────────────────────────────────────────
    print("[H] Gerando dashboard...")
    plot_dashboard(df, ce, crossings, viscosity, cfg.out,
                   title=scenario_title or f"{cfg.start}→{cfg.end}")

    print("\n" + report)
    print("\n✅ Kappa-FIN v3.0 concluído.")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> ConfigV3:
    p = argparse.ArgumentParser(
        description="Kappa-FIN v3.0 — Implementação exata do Método Kappa"
    )
    p.add_argument("--tickers", type=str, default="")
    p.add_argument("--preset",  type=str, default=None,
                   choices=list(ASSET_PRESETS.keys()),
                   help="Preset de ativos multi-mercado")
    p.add_argument("--start",   type=str, required=True)
    p.add_argument("--end",     type=str, required=True)
    p.add_argument("--interval",type=str, default="1d")
    p.add_argument("--parquet", type=str, default=None,
                   dest="parquet_path", help="Caminho para FNSPID parquet")

    p.add_argument("--window",  type=int,   default=22)
    p.add_argument("--k",       type=int,   default=5)
    p.add_argument("--dep",     type=str,   default="spearman", dest="dep_method")
    p.add_argument("--shrink",  type=float, default=0.05, dest="shrink_lambda")
    p.add_argument("--dist_alpha", type=float, default=1.0)

    p.add_argument("--max_edge_q", type=float, default=0.99, dest="max_edge_quantile")
    p.add_argument("--curv_summary",type=str,  default="median")
    p.add_argument("--eta_floor",  type=float, default=0.05)

    p.add_argument("--gamma",   type=float, default=0.97)
    p.add_argument("--delta",   type=float, default=0.08)
    p.add_argument("--pre_q",   type=float, default=0.968)
    p.add_argument("--phi_q",   type=float, default=0.99)
    p.add_argument("--phi_persist", type=int, default=3)
    p.add_argument("--phi_floor",   type=float, default=1e-6)

    p.add_argument("--calm_policy",      type=str,   default="scan")
    p.add_argument("--calm_start",       type=str,   default=None)
    p.add_argument("--calm_end",         type=str,   default=None)
    p.add_argument("--calm_length_days", type=int,   default=504)
    p.add_argument("--calm_step_days",   type=int,   default=14)
    p.add_argument("--calm_topn",        type=int,   default=5)
    p.add_argument("--calm_search_to",   type=str,   default=None)

    p.add_argument("--calm_knee_weight",    type=float, default=1.0)
    p.add_argument("--calm_knee_tail_frac", type=float, default=0.25)
    p.add_argument("--calm_knee_ratio",     type=float, default=1.50)
    p.add_argument("--calm_knee_slope_ann", type=float, default=0.0)
    p.add_argument("--calm_knee_mix",       type=float, default=0.5)

    p.add_argument("--oh_nagare_c",     type=float, default=0.85)
    p.add_argument("--oh_katashi_c",    type=float, default=1.0)
    p.add_argument("--eval_after_calm", type=bool,  default=True)
    p.add_argument("--oh_persist_sens", type=int,   default=2)
    p.add_argument("--oh_persist_confirm", type=int,default=5)
    p.add_argument("--out", type=str, default="./results/out_v3")
    p.add_argument("--title", type=str, default="")

    a = p.parse_args()
    tickers = [t.strip() for t in a.tickers.split(",") if t.strip()]

    return ConfigV3(
        tickers=tickers,
        start=a.start, end=a.end, interval=a.interval,
        preset=a.preset, parquet_path=a.parquet_path,
        window=a.window, k=a.k,
        dep_method=a.dep_method, shrink_lambda=a.shrink,
        dist_alpha=a.dist_alpha,
        max_edge_quantile=a.max_edge_q,
        curv_summary=a.curv_summary, eta_floor=a.eta_floor,
        gamma=a.gamma, delta=a.delta, pre_q=a.pre_q,
        phi_q=a.phi_q, phi_persist=a.phi_persist, phi_floor=a.phi_floor,
        calm_policy=a.calm_policy,
        calm_start=a.calm_start, calm_end=a.calm_end,
        calm_length_days=a.calm_length_days,
        calm_step_days=a.calm_step_days, calm_topn=a.calm_topn,
        calm_search_to=a.calm_search_to,
        calm_knee_weight=a.calm_knee_weight,
        calm_knee_tail_frac=a.calm_knee_tail_frac,
        calm_knee_ratio=a.calm_knee_ratio,
        calm_knee_slope_ann=a.calm_knee_slope_ann,
        calm_knee_mix=a.calm_knee_mix,
        oh_nagare_c=a.oh_nagare_c, oh_katashi_c=a.oh_katashi_c,
        eval_after_calm=bool(a.eval_after_calm),
        oh_persist_sens=a.oh_persist_sens,
        oh_persist_confirm=a.oh_persist_confirm,
        out=a.out,
    ), a.title

def main() -> None:
    cfg, title = parse_args()
    run(cfg, scenario_title=title)

if __name__ == "__main__":
    main()
