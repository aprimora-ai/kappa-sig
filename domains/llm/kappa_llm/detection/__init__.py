"""
Kappa-LLM Detection Module.

Provides multi-observable hallucination detection using the Kappa framework.
Combines multiple structural observables into a composite Kappa Score
for robust detection of pathological attention patterns.
"""

from .detector import KappaDetector, DetectionResult, DetectorConfig

__all__ = ["KappaDetector", "DetectionResult", "DetectorConfig"]
