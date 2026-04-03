#!/usr/bin/env python3
"""
Kappa-LLM → Unified Kappa Bridge
==================================
Maps LLM attention observables to the unified Kappa state vector S(t),
enabling cross-domain SIG/LSCC analysis.

Domain mapping:
  LLM omega (entropy)      → Oh  (inverted: low entropy = high Oh)
  LLM phi   (persistence)  → phi (structural memory)
  LLM eta   (rigidity)     → eta (rigidity, same semantics)
  LLM xi    (diversity)    → Xi  (diversity, same semantics, inverted for DEF)
  LLM delta (divergence)   → DEF (structural deficit)
  LLM rscore (composite)   → mean_corr proxy (attention coherence)

David Ohio | odavidohio@gmail.com | Independent Researcher
April 2026
"""
import numpy as np
from typing import Dict, List


def llm_obs_to_kappa_state(obs: Dict[str, float]) -> Dict[str, float]:
    """
    Map a single LLM observable dict to unified Kappa state.
    
    The mapping preserves the semantic direction:
    - High Oh = high structural pressure (low entropy in LLM = concentrated attention)
    - High phi = accumulated structural memory (persistent topological cycles)
    - High eta = high rigidity (same in both domains)
    - Low Xi = low diversity (same in both domains)
    - High DEF = high structural deficit (high divergence from healthy baseline)
    
    Args:
        obs: Dict with keys {omega, phi, eta, xi, delta, rscore}
    Returns:
        Dict with keys {Oh, phi, eta, mean_corr, DEF, Xi} matching FIN format
    """
    return {
        "Oh": 1.0 - obs.get("omega", 0.5),      # Invert: low entropy → high Oh
        "phi": obs.get("phi", 0.0),                # Direct: persistence = memory
        "eta": obs.get("eta", 0.0),                # Direct: rigidity
        "mean_corr": obs.get("rscore", 0.0),       # R-score as coherence proxy
        "DEF": obs.get("delta", 0.0),              # Divergence → deficit
        "Xi": obs.get("xi", 0.5),                  # Direct: diversity
    }


def llm_trajectory_to_states(obs_list: List[Dict[str, float]]) -> np.ndarray:
    """
    Convert a sequence of LLM observables to a state matrix S(t).
    
    Args:
        obs_list: List of observable dicts from token-level extraction
    Returns:
        np.ndarray of shape (T, 6) with columns [Oh, phi, eta, mean_corr, DEF, Xi]
    """
    STATE_COLS = ["Oh", "phi", "eta", "mean_corr", "DEF", "Xi"]
    rows = []
    for obs in obs_list:
        mapped = llm_obs_to_kappa_state(obs)
        rows.append([mapped[c] for c in STATE_COLS])
    return np.array(rows, dtype=np.float32)


def compute_correlation_matrix_from_attention(attention_matrix: np.ndarray) -> np.ndarray:
    """
    Compute a correlation-like matrix from an attention matrix.
    For SIG spectral analysis (Wheeler test, D_MP_KL, etc.)
    
    The attention matrix A[i,j] is treated as a similarity measure.
    We compute the Pearson correlation between attention patterns
    of different tokens (rows of A).
    
    Args:
        attention_matrix: Square attention matrix (n_tokens x n_tokens)
    Returns:
        Correlation matrix (n_tokens x n_tokens)
    """
    A = np.asarray(attention_matrix, dtype=np.float64)
    # Each row is a token's attention distribution
    # Correlation between rows = structural similarity between tokens
    n = A.shape[0]
    if n < 3:
        return np.eye(n)
    # Standardize rows
    means = A.mean(axis=1, keepdims=True)
    stds = A.std(axis=1, keepdims=True)
    stds = np.maximum(stds, 1e-10)
    Z = (A - means) / stds
    C = Z @ Z.T / n
    np.fill_diagonal(C, 1.0)
    return np.clip(C, -1, 1)


def obsessive_coherence_signature(obs: Dict[str, float]) -> Dict[str, float]:
    """
    Compute the Obsessive Coherence signature for an LLM observation.
    
    The obsessive coherence hypothesis states that hallucinations
    manifest as EXCESSIVE structural coherence rather than disorder.
    
    Returns metrics that quantify this:
    - coherence_index: High = obsessively coherent (hallucination-like)
    - disorder_index: High = structurally disordered
    - obsessive_score: coherence_index - disorder_index (positive = obsessive)
    """
    # Coherence indicators (high = obsessive)
    rigidity = obs.get("eta", 0)
    concentration = 1.0 - obs.get("omega", 0.5)
    divergence = obs.get("delta", 0)
    
    # Disorder indicators (high = chaotic)  
    entropy = obs.get("omega", 0.5)
    diversity = obs.get("xi", 0.5)
    
    coherence_index = (rigidity + concentration + divergence) / 3.0
    disorder_index = (entropy + diversity) / 2.0
    obsessive_score = coherence_index - disorder_index
    
    return {
        "coherence_index": float(coherence_index),
        "disorder_index": float(disorder_index),
        "obsessive_score": float(obsessive_score),
    }
