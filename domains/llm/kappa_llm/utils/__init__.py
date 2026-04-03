__init__.py"""
Kappa-LLM Utilities Module.

Provides helper functions for data loading, visualization, and integration
with HaluEval dataset and various LLM backends.
"""

from .data import (
    load_halueval_sample,
    attention_from_model,
    normalize_attention_matrix
)

__all__ = [
    "load_halueval_sample",
    "attention_from_model",
    "normalize_attention_matrix"
]
