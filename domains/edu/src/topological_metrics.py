"""
Topological Metrics - Computation of structural observables.

Implements the core structural observables for Katashi/Kappa method:
- Ξ (Xi): Structural intensity
- η (eta): Structural friction  
- Φ (phi): Structural memory (computed in main analyzer)
- Ω/Oh: Regime number (normalized post-CALM)
- Entropy: Spectral entropy (NEW from version anterior)
- Dominance: Spectral dominance (NEW from version anterior)

Includes PATCH B: Degeneracy detection for robust analysis.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple


def compute_correlation_matrix(
    window_data: pd.DataFrame,
    method: str = 'spearman'
) -> np.ndarray:
    """
    Compute correlation matrix with simple shrinkage.
    
    Args:
        window_data: DataFrame with engagement metrics
        method: Correlation method ('spearman' or 'pearson')
    
    Returns:
        Correlation matrix (numpy array)
    """
    if method == 'spearman':
        corr_matrix = window_data.corr(method='spearman')
    else:
        corr_matrix = window_data.corr(method='pearson')
    
    # Simple shrinkage towards identity
    shrinkage = 0.1
    n = corr_matrix.shape[0]
    corr_shrunk = (1 - shrinkage) * corr_matrix + shrinkage * np.eye(n)
    
    return corr_shrunk.values


def compute_spectral_properties(corr_matrix: np.ndarray) -> Dict[str, float]:
    """
    Compute basic spectral properties of correlation matrix.
    
    Args:
        corr_matrix: Correlation matrix
    
    Returns:
        Dictionary with eigenvalue statistics
    """
    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order
    eigenvalues = np.clip(eigenvalues, 1e-12, None)  # Avoid negatives
    
    # Spectral dominance: ratio of largest to second-largest eigenvalue
    if len(eigenvalues) > 1 and eigenvalues[1] > 1e-6:
        dominance = eigenvalues[0] / eigenvalues[1]
    else:
        dominance = eigenvalues[0]
    
    return {
        'eigenvalues': eigenvalues,
        'lambda_max': float(eigenvalues[0]),
        'lambda_min': float(eigenvalues[-1]),
        'lambda_mean': float(eigenvalues.mean()),
        'dominance': float(dominance),
        'effective_rank': float(np.sum(eigenvalues)**2 / np.sum(eigenvalues**2))
    }


def compute_spectral_entropy(corr_matrix: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute spectral entropy from eigenvalue distribution.
    
    From version anterior (katashi_run.py):
    entropy = -(w * log(w)).sum() where w = normalized eigenvalues
    
    Entropy measures uncertainty/diversity in spectral decomposition:
    - High entropy = many significant eigenvalues (rich structure)
    - Low entropy = few dominant eigenvalues (simple structure)
    
    Args:
        corr_matrix: Correlation matrix
        eps: Small constant to avoid log(0)
    
    Returns:
        Spectral entropy (in nats)
    
    Example:
        >>> C = np.corrcoef(np.random.randn(10, 5), rowvar=False)
        >>> entropy = compute_spectral_entropy(C)
        >>> print(f"Entropy: {entropy:.3f}")
    """
    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    eigenvalues = np.clip(eigenvalues, eps, None)
    
    # Normalize to probability distribution
    weights = eigenvalues / eigenvalues.sum()
    
    # Shannon entropy: H = -Σ p_i log(p_i)
    entropy = float(-(weights * np.log(weights)).sum())
    
    return entropy


def compute_spectral_dominance(corr_matrix: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute spectral dominance (max eigenvalue weight).
    
    From version anterior (katashi_run.py):
    dominance = w.max() where w = normalized eigenvalues
    
    Dominance measures how much variance is captured by leading eigenvalue:
    - High dominance = one eigenvalue dominates (rigid structure)
    - Low dominance = variance distributed (flexible structure)
    
    Args:
        corr_matrix: Correlation matrix
        eps: Small constant for numerical stability
    
    Returns:
        Dominance coefficient (0 to 1)
    
    Example:
        >>> C = np.corrcoef(np.random.randn(10, 5), rowvar=False)
        >>> dominance = compute_spectral_dominance(C)
        >>> print(f"Dominance: {dominance:.3f}")
    """
    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    eigenvalues = np.clip(eigenvalues, eps, None)
    
    # Normalize to weights
    weights = eigenvalues / eigenvalues.sum()
    
    # Dominance = maximum weight
    dominance = float(weights.max())
    
    return dominance


def detect_degeneracy(
    window_data: pd.DataFrame,
    min_active_features: int = 2,
    variance_threshold: float = 1e-10
) -> Tuple[bool, int]:
    """
    Detect degenerate windows (PATCH B from version anterior).
    
    From katashi_run.py:
    col_var = np.nanvar(Z, axis=0)
    valid = col_var > 1e-10
    if int(valid.sum()) < int(cfg.min_active_features):
        # Mark as degenerate
    
    A window is degenerate if too many features are "dead" (zero variance).
    This prevents numerical issues and meaningless structural metrics.
    
    Args:
        window_data: DataFrame with features for current window
        min_active_features: Minimum number of active features required
        variance_threshold: Minimum variance to consider feature "active"
    
    Returns:
        Tuple of (is_degenerate, n_active_features)
    
    Example:
        >>> # Window with mostly zeros
        >>> data = pd.DataFrame(np.zeros((10, 5)))
        >>> is_degen, n_active = detect_degeneracy(data)
        >>> print(f"Degenerate: {is_degen}, Active: {n_active}")
        Degenerate: True, Active: 0
    """
    # Compute variance of each column
    variances = window_data.var(axis=0, ddof=0)
    
    # Count active features (variance above threshold)
    active = (variances > variance_threshold).sum()
    
    # Degenerate if too few active features
    is_degenerate = int(active) < int(min_active_features)
    
    return is_degenerate, int(active)


def compute_xi(
    corr_matrix: np.ndarray,
    spectral_dom: float,
    k: int = 4
) -> float:
    """
    Compute Ξ (Xi) - Structural Intensity.
    
    Formula:
        Ξ = (1 + |ρ_mean|) × (1 + λ_dominant) × k
    
    where:
    - ρ_mean = mean absolute correlation
    - λ_dominant = spectral dominance
    - k = scale factor (default 4)
    
    Args:
        corr_matrix: Correlation matrix
        spectral_dom: Spectral dominance (λ_max / λ_2)
        k: Scale factor (default 4)
    
    Returns:
        Ξ value
    """
    # Mean absolute correlation (exclude diagonal)
    n = corr_matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    mean_corr = np.abs(corr_matrix[mask]).mean()
    
    # Xi formula
    xi = (1 + mean_corr) * (1 + spectral_dom) * k
    
    return float(xi)


def compute_eta(corr_matrix: np.ndarray) -> float:
    """
    Compute η (eta) - Structural Friction.
    
    Formula:
        η = 1 + log(1 + ||C||_Frobenius)
    
    where ||C||_F is the Frobenius norm of correlation matrix.
    
    Args:
        corr_matrix: Correlation matrix
    
    Returns:
        η value
    """
    frobenius_norm = np.linalg.norm(corr_matrix, ord='fro')
    eta = 1.0 + np.log1p(frobenius_norm)
    
    return float(eta)


def compute_structural_metrics(
    window_data: pd.DataFrame,
    k: int = 4,
    method: str = 'spearman',
    min_active_features: int = 2
) -> Dict[str, float]:
    """
    Compute all structural observables for a window.
    
    UPDATED from version anterior to include:
    - entropy: Spectral entropy
    - dominance: Spectral dominance
    - degenerate: Degeneracy flag (PATCH B)
    - n_active: Number of active features
    
    Args:
        window_data: DataFrame with engagement metrics
        k: Scale factor for Ξ
        method: Correlation method ('spearman' or 'pearson')
        min_active_features: Minimum active features (PATCH B)
    
    Returns:
        Dictionary with all structural observables:
        - Xi: Structural intensity
        - eta: Structural friction
        - entropy: Spectral entropy (NEW!)
        - dominance: Spectral dominance (NEW!)
        - mean_corr: Mean absolute correlation
        - spectral_dom: Spectral dominance ratio
        - lambda_max: Largest eigenvalue
        - degenerate: Boolean flag (NEW!)
        - n_active: Number of active features (NEW!)
    
    Example:
        >>> window = df.iloc[10:20]  # 10-week window
        >>> metrics = compute_structural_metrics(window)
        >>> print(f"Xi={metrics['Xi']:.2f}, entropy={metrics['entropy']:.2f}")
    """
    # Filter numeric columns only
    numeric_cols = window_data.select_dtypes(include=[np.number]).columns
    window_numeric = window_data[numeric_cols]
    
    # PATCH B: Detect degeneracy (dead features)
    is_degenerate, n_active = detect_degeneracy(
        window_numeric,
        min_active_features=min_active_features
    )
    
    if is_degenerate:
        # Return safe defaults for degenerate window
        return {
            'Xi': 0.0,
            'eta': 1.0,
            'entropy': 0.0,
            'dominance': 1.0,
            'mean_corr': 0.0,
            'spectral_dom': 1.0,
            'lambda_max': 0.0,
            'degenerate': True,
            'n_active': n_active
        }
    
    # Remove columns with no variance
    variance = window_numeric.var()
    valid_cols = variance[variance > 1e-10].index
    
    if len(valid_cols) < 2:
        # Not enough valid columns for correlation
        return {
            'Xi': 0.0,
            'eta': 1.0,
            'entropy': 0.0,
            'dominance': 1.0,
            'mean_corr': 0.0,
            'spectral_dom': 1.0,
            'lambda_max': 0.0,
            'degenerate': True,
            'n_active': len(valid_cols)
        }
    
    window_filtered = window_numeric[valid_cols]
    
    # Compute correlation matrix
    corr_matrix = compute_correlation_matrix(window_filtered, method=method)
    
    # Spectral properties
    spectral = compute_spectral_properties(corr_matrix)
    
    # NEW: Spectral entropy and dominance (from version anterior)
    entropy = compute_spectral_entropy(corr_matrix)
    dominance = compute_spectral_dominance(corr_matrix)
    
    # Compute observables
    xi = compute_xi(corr_matrix, spectral['dominance'], k=k)
    eta = compute_eta(corr_matrix)
    
    # Mean absolute correlation (for reference)
    n = corr_matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    mean_corr = np.abs(corr_matrix[mask]).mean()
    
    return {
        'Xi': float(xi),
        'eta': float(eta),
        'entropy': float(entropy),          # NEW!
        'dominance': float(dominance),      # NEW!
        'mean_corr': float(mean_corr),
        'spectral_dom': float(spectral['dominance']),
        'lambda_max': float(spectral['lambda_max']),
        'degenerate': bool(is_degenerate),  # NEW!
        'n_active': int(n_active)           # NEW!
    }


def compute_oh_normalized(xi: float, xi_c: float) -> float:
    """
    Compute Oh (Ω) - Regime Number.
    
    Formula:
        Oh = Ξ / Ξ_c
    
    where Ξ_c is the reference intensity from CALM period (typically P95).
    
    Args:
        xi: Current structural intensity
        xi_c: Reference intensity from CALM period
    
    Returns:
        Oh value (regime number)
    """
    return float(xi / (xi_c + 1e-18))


def compute_phi_update(
    phi_prev: float,
    oh: float,
    gamma: float = 0.97,
    delta: float = 0.08,
    theta: float = 1.0
) -> float:
    """
    Update Φ (phi) - Structural Memory.
    
    Formula:
        Φ(t+1) = γ × Φ(t) + δ × max(0, Oh(t) - θ)
    
    Φ accumulates "damage" only when Oh > θ (critical regime).
    
    Args:
        phi_prev: Previous Φ value
        oh: Current Oh value
        gamma: Forgetting factor (default 0.97)
        delta: Damage sensitivity (default 0.08)
        theta: Critical threshold (default 1.0)
    
    Returns:
        Updated Φ value
    """
    excess = max(0.0, oh - theta)
    phi_new = gamma * phi_prev + delta * excess
    
    return float(phi_new)


# Convenience exports for backward compatibility
__all__ = [
    'compute_correlation_matrix',
    'compute_spectral_properties',
    'compute_spectral_entropy',
    'compute_spectral_dominance',
    'detect_degeneracy',
    'compute_xi',
    'compute_eta',
    'compute_structural_metrics',
    'compute_oh_normalized',
    'compute_phi_update'
]
