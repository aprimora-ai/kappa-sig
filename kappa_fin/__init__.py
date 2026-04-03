"""
kappa_fin — Topological Early Warning System for Financial Market Crises
========================================================================

Applies the Kappa Method (persistent homology H1 + Forman-Ricci curvature)
to detect systemic risk precursors in financial markets via rolling
correlation networks.

Author : David Ohio <odavidohio@gmail.com>
License: CC BY 4.0
"""

from kappa_fin.engine import Config, run

__version__ = "0.1.0"
__author__ = "David Ohio"
__email__ = "odavidohio@gmail.com"
__license__ = "CC BY 4.0"

__all__ = ["Config", "run"]
