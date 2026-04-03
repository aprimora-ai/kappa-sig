"""
Kappa-LLM Regime Classification Module.

Classifies attention patterns into structural regimes:
- NAGARE: Adaptive flow (healthy, diverse attention)
- UTSUROI: Transitional (intermediate state)
- KATASHI: Obsessive (pathological, potential hallucination)
"""

from .classifier import RegimeClassifier, AttentionRegime

__all__ = ["RegimeClassifier", "AttentionRegime"]
