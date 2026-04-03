"""
CALM Detection - Collectively Aligned Low-variance Mode identification.

Implements algorithms for detecting stable reference periods in time series
where the system exhibits:
1. Low structural variance
2. No systematic drift
3. Collective synchronization
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from scipy import stats as scipy_stats

from .topological_metrics import detect_degeneracy


def compute_calm_score(
    window_data: pd.Series,
    deg_weight: float = 1.0
) -> float:
    """
    Compute CALM quality score for a candidate period.
    
    Lower score = better CALM candidate
    
    Score components:
    - Coefficient of variation (CV) of the metric
    - Linear trend strength (absolute slope)
    - Degeneracy penalty (optional, if window_data is DataFrame)
    
    Args:
        window_data: Time series data for candidate period
        deg_weight: Weight for degeneracy penalty
    
    Returns:
        CALM score (lower is better)
    """
    values = window_data.values if hasattr(window_data, 'values') else window_data
    
    # Coefficient of variation
    mean_val = np.mean(values)
    std_val = np.std(values)
    
    if mean_val < 1e-10:
        cv = 1e6  # Penalize near-zero mean
    else:
        cv = std_val / mean_val
    
    # Linear trend strength
    n = len(values)
    x = np.arange(n)
    if n > 2:
        slope, _, _, _, _ = scipy_stats.linregress(x, values)
        trend = abs(slope)
    else:
        trend = 0.0
    
    # Base score
    score = cv + trend
    
    # Add degeneracy penalty if data is multivariate
    if isinstance(window_data, pd.DataFrame):
        deg = detect_degeneracy(window_data)
        score += deg_weight * deg
    
    return score


def detect_calm_periods(
    state: pd.DataFrame,
    length: int = 12,
    step: int = 1,
    metric: str = 'Xi',
    deg_weight: float = 1.0
) -> List[Dict]:
    """
    Detect all candidate CALM periods in the time series.
    
    Args:
        state: DataFrame with structural observables
        length: Length of CALM period (weeks)
        step: Step size for sliding window
        metric: Metric to evaluate ('Xi' or 'Oh')
        deg_weight: Weight for degeneracy penalty
    
    Returns:
        List of CALM candidate dictionaries
    """
    candidates = []
    
    # Ensure we have enough data
    if len(state) < length:
        return candidates
    
    # Slide window through time series
    for i in range(0, len(state) - length + 1, step):
        window = state.iloc[i:i + length]
        
        # Compute CALM score
        score = compute_calm_score(window[metric], deg_weight=deg_weight)
        
        candidates.append({
            'start': window.index[0],
            'end': window.index[-1],
            'score': score,
            'length': length,
            'mean_xi': window['Xi'].mean(),
            'std_xi': window['Xi'].std(),
            'mean_oh': window.get('Oh', pd.Series([0])).mean()
        })
    
    return candidates


def rank_calm_candidates(
    candidates: List[Dict],
    top_n: int = 3
) -> List[Dict]:
    """
    Rank CALM candidates by score and return top N.
    
    Args:
        candidates: List of CALM candidate dictionaries
        top_n: Number of top candidates to return
    
    Returns:
        List of top N CALM candidates (sorted by score, ascending)
    """
    if not candidates:
        return []
    
    # Sort by score (lower is better)
    sorted_candidates = sorted(candidates, key=lambda x: x['score'])
    
    return sorted_candidates[:top_n]


def validate_calm_period(
    state: pd.DataFrame,
    calm_start: pd.Timestamp,
    calm_end: pd.Timestamp,
    max_cv: float = 0.15,
    max_trend: float = 0.01
) -> Dict[str, bool]:
    """
    Validate a CALM period against quality criteria.
    
    Args:
        state: DataFrame with structural observables
        calm_start: Start of CALM period
        calm_end: End of CALM period
        max_cv: Maximum allowed coefficient of variation
        max_trend: Maximum allowed trend strength
    
    Returns:
        Dictionary with validation results
    """
    calm_data = state.loc[calm_start:calm_end]
    
    # Check coefficient of variation
    xi_mean = calm_data['Xi'].mean()
    xi_std = calm_data['Xi'].std()
    cv = xi_std / xi_mean if xi_mean > 1e-10 else 1e6
    
    cv_valid = cv < max_cv
    
    # Check trend
    n = len(calm_data)
    x = np.arange(n)
    slope, _, _, _, _ = scipy_stats.linregress(x, calm_data['Xi'].values)
    
    trend_valid = abs(slope) < max_trend
    
    # Overall validity
    valid = cv_valid and trend_valid
    
    return {
        'valid': valid,
        'cv_valid': cv_valid,
        'trend_valid': trend_valid,
        'cv': cv,
        'trend': abs(slope)
    }


def get_calm_statistics(
    state: pd.DataFrame,
    calm_start: pd.Timestamp,
    calm_end: pd.Timestamp
) -> Dict[str, float]:
    """
    Compute descriptive statistics for a CALM period.
    
    Args:
        state: DataFrame with structural observables
        calm_start: Start of CALM period
        calm_end: End of CALM period
    
    Returns:
        Dictionary with CALM statistics
    """
    calm_data = state.loc[calm_start:calm_end]
    
    stats = {
        'duration': len(calm_data),
        'xi_mean': calm_data['Xi'].mean(),
        'xi_std': calm_data['Xi'].std(),
        'xi_min': calm_data['Xi'].min(),
        'xi_max': calm_data['Xi'].max(),
        'eta_mean': calm_data['eta'].mean(),
        'phi_mean': calm_data['phi'].mean() if 'phi' in calm_data else 0.0
    }
    
    # Add Oh statistics if available
    if 'Oh' in calm_data.columns:
        stats.update({
            'oh_mean': calm_data['Oh'].mean(),
            'oh_std': calm_data['Oh'].std()
        })
    
    return stats


def detect_calm_with_ensemble(
    state: pd.DataFrame,
    length: int = 12,
    top_n: int = 3,
    deg_weight: float = 1.0
) -> Dict:
    """
    Complete CALM detection workflow with ensemble.
    
    Args:
        state: DataFrame with structural observables
        length: CALM period length
        top_n: Number of CALMs for ensemble
        deg_weight: Degeneracy penalty weight
    
    Returns:
        Dictionary with best CALM and ensemble
    """
    # Detect all candidates
    candidates = detect_calm_periods(
        state,
        length=length,
        deg_weight=deg_weight
    )
    
    if not candidates:
        return {
            'best': None,
            'ensemble': [],
            'count': 0
        }
    
    # Rank and select top N
    top_calms = rank_calm_candidates(candidates, top_n=top_n)
    
    # Get statistics for best CALM
    best = top_calms[0]
    best_stats = get_calm_statistics(state, best['start'], best['end'])
    best.update(best_stats)
    
    # Validate best CALM
    validation = validate_calm_period(state, best['start'], best['end'])
    best['validation'] = validation
    
    return {
        'best': best,
        'ensemble': top_calms,
        'count': len(candidates)
    }


def compare_calm_periods(
    calms: List[Dict],
    state: pd.DataFrame
) -> pd.DataFrame:
    """
    Compare multiple CALM periods side-by-side.
    
    Args:
        calms: List of CALM period dictionaries
        state: DataFrame with structural observables
    
    Returns:
        Comparison DataFrame
    """
    comparison = []
    
    for i, calm in enumerate(calms):
        stats = get_calm_statistics(state, calm['start'], calm['end'])
        validation = validate_calm_period(state, calm['start'], calm['end'])
        
        comparison.append({
            'rank': i + 1,
            'start': calm['start'],
            'end': calm['end'],
            'score': calm['score'],
            'xi_mean': stats['xi_mean'],
            'xi_std': stats['xi_std'],
            'cv': validation['cv'],
            'trend': validation['trend'],
            'valid': validation['valid']
        })
    
    return pd.DataFrame(comparison)
