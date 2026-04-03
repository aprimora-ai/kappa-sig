"""
DEF Metric - State-Phase Divergence

Implements the fifth structural observable from the Kappa Method:
DEF(t) measures incoherence between where the system is (state x) 
and where it's going (phase dynamics ẋ).

Reference: https://github.com/odavidohio/Kappa-Method

Formula: DEF(t) = |x(t) - P(ẋ(t))|

Where:
- x(t): Current structural state vector
- ẋ(t): Phase velocity (time derivative of state)
- P: Projection operator onto state space
- | · |: Euclidean norm (distance)

High DEF indicates the system is "in the wrong place" for its current 
trajectory - a key predictor of imminent failures invisible to other metrics.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
from scipy.interpolate import interp1d


def compute_state_vector(
    state: pd.DataFrame,
    observables: List[str] = None
) -> np.ndarray:
    """
    Extract state vector from structural observables.
    
    Args:
        state: DataFrame with structural observables
        observables: List of observable names (default: ['Xi', 'Oh', 'phi', 'eta'])
    
    Returns:
        State matrix (T x D) where T=timesteps, D=dimensions
    """
    if observables is None:
        observables = ['Xi', 'Oh', 'phi', 'eta']
    
    # Filter to available observables
    available = [col for col in observables if col in state.columns]
    
    if not available:
        raise ValueError(f"None of {observables} found in state columns: {state.columns.tolist()}")
    
    return state[available].values


def compute_phase_velocity(
    state_vector: np.ndarray,
    dt: float = 1.0,
    method: str = 'centered'
) -> np.ndarray:
    """
    Compute phase velocity ẋ(t) = dx/dt.
    
    The phase velocity represents the instantaneous direction and speed
    of change in the structural state space.
    
    Args:
        state_vector: State matrix (T x D)
        dt: Time step (default 1.0 for weekly data)
        method: Derivative method:
               - 'centered': Central difference (O(dt²), most accurate)
               - 'forward': Forward difference (O(dt))
               - 'backward': Backward difference (O(dt))
    
    Returns:
        Phase velocity matrix ẋ (T x D)
    """
    T, D = state_vector.shape
    velocity = np.zeros_like(state_vector)
    
    if method == 'centered':
        # Central difference: ẋ(t) = [x(t+dt) - x(t-dt)] / (2*dt)
        velocity[1:-1] = (state_vector[2:] - state_vector[:-2]) / (2 * dt)
        # Boundaries: use one-sided differences
        velocity[0] = (state_vector[1] - state_vector[0]) / dt  # Forward
        velocity[-1] = (state_vector[-1] - state_vector[-2]) / dt  # Backward
        
    elif method == 'forward':
        # Forward difference: ẋ(t) = [x(t+dt) - x(t)] / dt
        velocity[:-1] = (state_vector[1:] - state_vector[:-1]) / dt
        velocity[-1] = velocity[-2]  # Extrapolate last
        
    elif method == 'backward':
        # Backward difference: ẋ(t) = [x(t) - x(t-dt)] / dt
        velocity[1:] = (state_vector[1:] - state_vector[:-1]) / dt
        velocity[0] = velocity[1]  # Extrapolate first
        
    else:
        raise ValueError(f"Unknown derivative method: {method}. Use 'centered', 'forward', or 'backward'.")
    
    return velocity


def project_phase_to_state(
    current_state: np.ndarray,
    phase_velocity: np.ndarray,
    projection_horizon: int = 1
) -> np.ndarray:
    """
    Project phase dynamics onto state space: P(ẋ(t)).
    
    Uses first-order Euler projection:
    P(ẋ(t)) = x(t) + ẋ(t) * h
    
    where h is the projection horizon (time steps ahead).
    
    Args:
        current_state: Current state x(t) matrix (T x D)
        phase_velocity: Phase velocity ẋ(t) matrix (T x D)
        projection_horizon: Steps ahead for projection (default 1)
    
    Returns:
        Projected state matrix P(ẋ) (T x D)
    """
    projected = current_state + phase_velocity * projection_horizon
    return projected


def compute_def(
    state: pd.DataFrame,
    observables: List[str] = None,
    dt: float = 1.0,
    method: str = 'centered',
    projection_horizon: int = 1,
    normalize: bool = True
) -> pd.Series:
    """
    Compute DEF - State-Phase Divergence.
    
    Formula: DEF(t) = |x(t) - P(ẋ(t))|
    
    Interpretation:
    - DEF = 0: Perfect coherence (state matches dynamics)
    - DEF > 0: Incoherence (state doesn't match where dynamics point)
    - High DEF: System in "wrong place" for its trajectory → failure imminent
    
    Args:
        state: DataFrame with structural observables (indexed by time)
        observables: List of observables to use (default: ['Xi', 'Oh', 'phi', 'eta'])
        dt: Time step between observations
        method: Derivative method for computing ẋ
        projection_horizon: Steps ahead for phase projection
        normalize: Whether to normalize by state magnitude (recommended)
    
    Returns:
        Series of DEF values indexed by time
    
    Example:
        >>> state = analyzer.analyze(df)
        >>> def_values = compute_def(state)
        >>> print(f"Max DEF: {def_values.max():.3f}")
    """
    # 1. Extract state vector x(t)
    state_vector = compute_state_vector(state, observables)
    
    # 2. Compute phase velocity ẋ(t)
    phase_velocity = compute_phase_velocity(state_vector, dt=dt, method=method)
    
    # 3. Project phase onto state space P(ẋ(t))
    projected_state = project_phase_to_state(state_vector, phase_velocity, projection_horizon)
    
    # 4. Compute divergence |x(t) - P(ẋ(t))|
    divergence = np.linalg.norm(state_vector - projected_state, axis=1)
    
    # 5. Optional normalization (scale-invariant)
    if normalize:
        state_magnitude = np.linalg.norm(state_vector, axis=1)
        # Avoid division by zero
        state_magnitude = np.maximum(state_magnitude, 1e-10)
        divergence = divergence / state_magnitude
    
    # Return as pandas Series with original index
    return pd.Series(divergence, index=state.index, name='DEF')


def compute_def_with_components(
    state: pd.DataFrame,
    observables: List[str] = None
) -> pd.DataFrame:
    """
    Compute DEF with per-observable component breakdown.
    
    Shows which observables contribute most to state-phase divergence.
    
    Args:
        state: DataFrame with structural observables
        observables: List of observables
    
    Returns:
        DataFrame with columns:
        - DEF_Xi, DEF_Oh, DEF_phi, DEF_eta: Component contributions
        - DEF_total: Total divergence (Euclidean norm)
        - DEF_normalized: Normalized by state magnitude
    
    Example:
        >>> components = compute_def_with_components(state)
        >>> print(components[['DEF_Xi', 'DEF_Oh', 'DEF_total']].tail())
    """
    if observables is None:
        observables = ['Xi', 'Oh', 'phi', 'eta']
    
    available = [col for col in observables if col in state.columns]
    
    # Compute state and phase
    state_vector = compute_state_vector(state, available)
    phase_velocity = compute_phase_velocity(state_vector)
    projected_state = project_phase_to_state(state_vector, phase_velocity)
    
    # Component-wise divergence |x_i - P(ẋ)_i|
    component_div = np.abs(state_vector - projected_state)
    
    # Build result DataFrame
    result = pd.DataFrame(
        component_div,
        index=state.index,
        columns=[f'DEF_{col}' for col in available]
    )
    
    # Total DEF (Euclidean norm across components)
    result['DEF_total'] = np.linalg.norm(component_div, axis=1)
    
    # Normalized DEF
    state_magnitude = np.linalg.norm(state_vector, axis=1)
    state_magnitude = np.maximum(state_magnitude, 1e-10)
    result['DEF_normalized'] = result['DEF_total'] / state_magnitude
    
    return result


def interpret_def(def_value: float, percentile_50: float = 0.2) -> str:
    """
    Provide qualitative interpretation of DEF value.
    
    Args:
        def_value: DEF value to interpret
        percentile_50: Median DEF for reference (default 0.2)
    
    Returns:
        Interpretation string
    """
    ratio = def_value / percentile_50 if percentile_50 > 0 else def_value
    
    if ratio < 0.5:
        return "Highly coherent: State perfectly aligned with dynamics"
    elif ratio < 1.0:
        return "Coherent: State matches trajectory well"
    elif ratio < 1.5:
        return "Mild incoherence: Minor state-phase misalignment"
    elif ratio < 2.5:
        return "Moderate incoherence: Significant divergence developing"
    elif ratio < 4.0:
        return "High incoherence: System in wrong place for trajectory"
    else:
        return "CRITICAL: Severe state-phase divergence (failure imminent)"


def detect_def_crossings(
    def_series: pd.Series,
    threshold: float = None,
    percentile: float = 95.0,
    persist_weeks: int = 2
) -> Dict:
    """
    Detect when DEF crosses critical thresholds.
    
    Args:
        def_series: DEF time series
        threshold: Explicit threshold (if None, use percentile from data)
        percentile: Percentile for automatic threshold
        persist_weeks: Weeks of sustained violation for confirmation
    
    Returns:
        Dictionary with:
        - 'threshold': DEF threshold used
        - 'sens': First sensitization date (sustained crossing)
        - 'confirm': Confirmation date (prolonged crossing)
        - 'max_def': Maximum DEF value
        - 'max_def_date': Date of maximum DEF
    
    Example:
        >>> def_cross = detect_def_crossings(state['DEF'])
        >>> print(f"DEF alert: {def_cross['sens']}")
    """
    if threshold is None:
        threshold = np.percentile(def_series.dropna(), percentile)
    
    violations = def_series > threshold
    
    # Find first sustained violation (sens)
    sens_date = None
    for i in range(len(violations) - persist_weeks + 1):
        if violations.iloc[i:i + persist_weeks].all():
            sens_date = violations.index[i]
            break
    
    # Find confirmation (longer persistence)
    confirm_date = None
    persist_confirm = persist_weeks * 2  # Require double persistence for confirmation
    if sens_date is not None:
        for i in range(len(violations) - persist_confirm + 1):
            if violations.iloc[i:i + persist_confirm].all():
                confirm_date = violations.index[i]
                break
    
    return {
        'threshold': threshold,
        'sens': sens_date,
        'confirm': confirm_date,
        'max_def': def_series.max(),
        'max_def_date': def_series.idxmax(),
        'violations_count': violations.sum()
    }


def compute_def_gradient(state: pd.DataFrame, observables: List[str] = None) -> pd.Series:
    """
    Compute rate of change of DEF: d(DEF)/dt.
    
    Rapidly increasing DEF gradient indicates accelerating incoherence.
    
    Args:
        state: DataFrame with structural observables
        observables: List of observables
    
    Returns:
        Series with DEF gradient
    """
    def_values = compute_def(state, observables=observables)
    gradient = def_values.diff()
    return gradient


def analyze_def_regime_correlation(
    state: pd.DataFrame,
    def_values: pd.Series
) -> Dict:
    """
    Analyze relationship between DEF and other regime indicators.
    
    Args:
        state: DataFrame with structural observables
        def_values: DEF series
    
    Returns:
        Dictionary with correlations and statistics
    """
    results = {}
    
    # Correlations with other observables
    for obs in ['Xi', 'Oh', 'phi', 'eta']:
        if obs in state.columns:
            corr = def_values.corr(state[obs])
            results[f'corr_DEF_{obs}'] = corr
    
    # Split by Oh regime
    if 'Oh' in state.columns:
        critical_regime = state['Oh'] > 1.0
        
        def_normal = def_values[~critical_regime]
        def_critical = def_values[critical_regime]
        
        results['mean_DEF_normal_regime'] = def_normal.mean()
        results['mean_DEF_critical_regime'] = def_critical.mean()
        results['std_DEF_normal_regime'] = def_normal.std()
        results['std_DEF_critical_regime'] = def_critical.std()
        
        if def_normal.mean() > 0:
            results['DEF_ratio_critical_to_normal'] = def_critical.mean() / def_normal.mean()
    
    return results


def compute_def_percentiles(def_series: pd.Series) -> Dict[str, float]:
    """
    Compute key percentiles of DEF distribution.
    
    Useful for setting thresholds and interpreting values.
    
    Args:
        def_series: DEF time series
    
    Returns:
        Dictionary with percentiles
    """
    return {
        'p10': def_series.quantile(0.10),
        'p25': def_series.quantile(0.25),
        'p50': def_series.quantile(0.50),
        'p75': def_series.quantile(0.75),
        'p90': def_series.quantile(0.90),
        'p95': def_series.quantile(0.95),
        'p99': def_series.quantile(0.99),
        'mean': def_series.mean(),
        'std': def_series.std()
    }


# Example and testing
if __name__ == '__main__':
    print("=" * 70)
    print("DEF (State-Phase Divergence) - Test Module")
    print("=" * 70)
    print()
    
    # Create synthetic test data with regime transition
    np.random.seed(42)
    dates = pd.date_range('2023-10-01', periods=40, freq='W')
    t = np.linspace(0, 4*np.pi, 40)
    
    # Simulate educational trajectory with crisis
    test_state = pd.DataFrame({
        'Xi': 10 + 2*np.sin(t) + 0.5*np.random.randn(40),
        'Oh': 0.7 + 0.3*t/max(t) + 0.1*np.sin(2*t) + 0.05*np.random.randn(40),
        'phi': 0.02 + 0.03*np.maximum(0, (t-2*np.pi)/np.pi) + 0.005*np.random.randn(40),
        'eta': 2.3 + 0.2*np.cos(t) + 0.1*np.random.randn(40)
    }, index=dates)
    
    print("Test Data Shape:", test_state.shape)
    print()
    
    # Compute DEF
    print("Computing DEF...")
    def_values = compute_def(test_state, normalize=True)
    
    # Statistics
    print("\nDEF Statistics:")
    print(f"  Mean:   {def_values.mean():.4f}")
    print(f"  Median: {def_values.median():.4f}")
    print(f"  Std:    {def_values.std():.4f}")
    print(f"  Min:    {def_values.min():.4f}")
    print(f"  Max:    {def_values.max():.4f}")
    
    # Percentiles
    print("\nDEF Percentiles:")
    percentiles = compute_def_percentiles(def_values)
    for k, v in percentiles.items():
        print(f"  {k}: {v:.4f}")
    
    # Detect crossings
    print("\nDetecting threshold crossings...")
    crossings = detect_def_crossings(def_values, percentile=90)
    print(f"  Threshold (P90): {crossings['threshold']:.4f}")
    print(f"  Sensitization:   {crossings['sens']}")
    print(f"  Confirmation:    {crossings['confirm']}")
    print(f"  Max DEF:         {crossings['max_def']:.4f} on {crossings['max_def_date'].date()}")
    
    # Component analysis
    print("\nComponent Analysis (last 3 timesteps):")
    components = compute_def_with_components(test_state)
    print(components[['DEF_Xi', 'DEF_Oh', 'DEF_phi', 'DEF_eta', 'DEF_total']].tail(3))
    
    # Regime correlation
    print("\nDEF-Regime Correlation:")
    correlations = analyze_def_regime_correlation(test_state, def_values)
    for k, v in correlations.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")
    
    print()
    print("=" * 70)
    print("Test completed successfully!")
    print("=" * 70)
