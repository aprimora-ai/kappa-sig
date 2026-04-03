"""
Kappa-LLM: Multi-Observable Topological Detection of Hallucinations
in Large Language Models.

This package implements the Kappa Method framework for LLM attention analysis,
providing tools for computing structural observables, classifying attention regimes,
and detecting hallucinations through multi-observable integration.

Canonical Observables:
- Ω (omega): Attentional pressure / entropy
- Φ (phi): Persistence / topological cycles
- η (eta): Rigidity / concentration
- Ξ (xi): Diversity / participation ratio
- Δ (delta): Divergence from uniform

Regimes:
- NAGARE: Adaptive flow (healthy)
- UTSUROI: Transitional (intermediate)
- KATASHI: Obsessive (pathological/hallucination)
"""

__version__ = "0.1.0"
__author__ = "David Ohio"

from .observables import KappaObservables, compute_kappa_observables
from .regimes import RegimeClassifier, AttentionRegime
from .detection import KappaDetector

__all__ = [
    "KappaObservables",
    "compute_kappa_observables",
    "RegimeClassifier",
    "AttentionRegime",
    "KappaDetector",
]
