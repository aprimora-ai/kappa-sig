"""
Kappa-LLM Observables Module.

Provides computation of the 5 canonical observables from attention matrices:
- Ω (omega): Attentional pressure / entropy
- Φ (phi): Persistence / topological cycles
- η (eta): Rigidity / concentration
- Ξ (xi): Diversity / participation ratio
- Δ (delta): Divergence from uniform
"""

from .core import KappaObservables, compute_kappa_observables

__all__ = ["KappaObservables", "compute_kappa_observables"]
