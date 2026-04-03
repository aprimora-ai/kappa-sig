"""
Kappa-Education: Educational Regime Detection via Topological Analysis.

This package implements the Katashi method for detecting structural regime
transitions in educational systems.
"""

__version__ = "1.2.0"
__author__ = "David Ferreira Ohio Junior"
__license__ = "CC BY 4.0 / MIT"

from .katashi_analyzer import (
    KatashiAnalyzer,
    KatashiConfig,
    load_oulad_data
)

from .topological_metrics import (
    compute_structural_metrics,
    compute_xi,
    compute_eta,
    compute_oh_normalized,
    compute_phi_update,
    compute_spectral_entropy,
    compute_spectral_dominance,
    detect_degeneracy
)

from .calm_detection import (
    detect_calm_periods,
    rank_calm_candidates,
    validate_calm_period,
    get_calm_statistics
)

from .visualization import (
    plot_regime_evolution,
    plot_cohort_comparison,
    plot_phase_space,
    plot_temporal_dynamics,
    plot_lead_time_analysis,
    plot_summary_dashboard
)

from .def_metric import (
    compute_def,
    compute_def_normalized,
    compute_def_timeseries,
    compute_def_components,
    detect_def_threshold_crossings,
    interpret_def
)

__all__ = [
    # Main analyzer
    'KatashiAnalyzer',
    'KatashiConfig',
    'load_oulad_data',
    
    # Metrics
    'compute_structural_metrics',
    'compute_xi',
    'compute_eta',
    'compute_oh_normalized',
    'compute_phi_update',
    'compute_spectral_entropy',      # NEW!
    'compute_spectral_dominance',    # NEW!
    'detect_degeneracy',             # NEW!
    
    # CALM detection
    'detect_calm_periods',
    'rank_calm_candidates',
    'validate_calm_period',
    'get_calm_statistics',
    
    # Visualization
    'plot_regime_evolution',
    'plot_cohort_comparison',
    'plot_phase_space',
    'plot_temporal_dynamics',
    'plot_lead_time_analysis',
    'plot_summary_dashboard',
    
    # DEF metric
    'compute_def',
    'compute_def_normalized',
    'compute_def_timeseries',
    'compute_def_components',
    'detect_def_threshold_crossings',
    'interpret_def'
]
