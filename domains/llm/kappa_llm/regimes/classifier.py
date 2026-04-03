"""
Classificação de regime (Nagare/Utsuroi/Katashi) para atenção LLM.

Este módulo implementa a classificação de regimes estruturais baseada nos
observáveis Kappa. Os três regimes correspondem a diferentes padrões de
distribuição atencional:

- NAGARE (流れ, "fluxo"): Atenção distribuída, adaptativa, saudável
- UTSUROI (移ろい, "transição"): Estado intermediário, em mudança
- KATASHI (形, "forma fixa"): Atenção concentrada, rígida, obsessiva

References:
    - Kappa Method Framework (Radiante Pentadimensional)
    - Regime classification in structural dynamics
"""

from typing import Dict, Tuple, Optional, List
from enum import Enum
from dataclasses import dataclass
import numpy as np


class AttentionRegime(Enum):
    """
    Regimes de atenção no framework Kappa.

    Attributes:
        NAGARE: Fluxo adaptativo - atenção saudável e diversificada
        UTSUROI: Transicional - estado intermediário entre regimes
        KATASHI: Obsessivo - atenção concentrada, potencial hallucination
    """
    NAGARE = "nagare"      # Fluxo adaptativo (saudável)
    UTSUROI = "utsuroi"    # Transicional (intermediário)
    KATASHI = "katashi"    # Obsessivo (patológico)

    def is_healthy(self) -> bool:
        """Returns True if this regime indicates healthy attention."""
        return self == AttentionRegime.NAGARE

    def is_pathological(self) -> bool:
        """Returns True if this regime indicates potential hallucination."""
        return self == AttentionRegime.KATASHI


@dataclass
class RegimeClassification:
    """
    Result of regime classification.

    Attributes:
        regime: The classified regime
        confidence: Confidence score in [0, 1]
        nagare_score: Score for NAGARE regime [0, 1]
        katashi_score: Score for KATASHI regime [0, 1]
        distances: Distances to each regime centroid
    """
    regime: AttentionRegime
    confidence: float
    nagare_score: float
    katashi_score: float
    distances: Dict[str, float]

    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            'regime': self.regime.value,
            'confidence': self.confidence,
            'nagare_score': self.nagare_score,
            'katashi_score': self.katashi_score,
            'distances': self.distances
        }


@dataclass
class RegimeThresholds:
    """
    Thresholds for regime classification.

    Default values are calibrated from HaluEval experiments.
    """
    # NAGARE thresholds (must satisfy ALL)
    nagare_xi_min: float = 0.6     # Diversidade mínima
    nagare_eta_max: float = 0.4    # Rigidez máxima
    nagare_delta_max: float = 0.3  # Divergência máxima

    # KATASHI thresholds (must satisfy ALL)
    katashi_eta_min: float = 0.6   # Rigidez mínima
    katashi_phi_min: float = 0.5   # Persistência mínima
    katashi_xi_max: float = 0.4    # Diversidade máxima

    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            'nagare': {
                'xi_min': self.nagare_xi_min,
                'eta_max': self.nagare_eta_max,
                'delta_max': self.nagare_delta_max
            },
            'katashi': {
                'eta_min': self.katashi_eta_min,
                'phi_min': self.katashi_phi_min,
                'xi_max': self.katashi_xi_max
            }
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'RegimeThresholds':
        """Create from dictionary."""
        return cls(
            nagare_xi_min=d['nagare']['xi_min'],
            nagare_eta_max=d['nagare']['eta_max'],
            nagare_delta_max=d['nagare']['delta_max'],
            katashi_eta_min=d['katashi']['eta_min'],
            katashi_phi_min=d['katashi']['phi_min'],
            katashi_xi_max=d['katashi']['xi_max']
        )


class RegimeClassifier:
    """
    Classifica regime de atenção baseado em observáveis Kappa.

    O classificador usa uma combinação de thresholds e scoring contínuo
    para determinar o regime mais provável. A classificação considera
    múltiplos observáveis simultaneamente para robustez.

    Attributes:
        thresholds: RegimeThresholds com valores de corte
        use_soft_classification: Se True, usa scoring contínuo

    Example:
        >>> classifier = RegimeClassifier()
        >>> obs = compute_kappa_observables(attention_matrix)
        >>> result = classifier.classify(obs)
        >>> if result.regime == AttentionRegime.KATASHI:
        ...     print("Warning: Potential hallucination detected")
    """

    # Regime centroids in observable space [omega, phi, eta, xi, delta]
    REGIME_CENTROIDS = {
        'nagare': np.array([0.8, 0.2, 0.2, 0.8, 0.2]),    # High diversity, low rigidity
        'utsuroi': np.array([0.5, 0.4, 0.5, 0.5, 0.5]),   # Intermediate values
        'katashi': np.array([0.2, 0.7, 0.8, 0.2, 0.8])    # Low diversity, high rigidity
    }

    def __init__(
        self,
        thresholds: Optional[RegimeThresholds] = None,
        use_soft_classification: bool = True
    ):
        """
        Initialize classifier.

        Args:
            thresholds: Custom thresholds. If None, uses defaults.
            use_soft_classification: If True, uses continuous scoring.
        """
        self.thresholds = thresholds or RegimeThresholds()
        self.use_soft_classification = use_soft_classification

    def classify(
        self,
        observables: Dict[str, float]
    ) -> RegimeClassification:
        """
        Classifica regime baseado em observáveis.

        Args:
            observables: Dict with keys: omega, phi, eta, xi, delta

        Returns:
            RegimeClassification with regime, confidence, and scores
        """
        # Compute regime scores
        nagare_score = self._compute_nagare_score(observables)
        katashi_score = self._compute_katashi_score(observables)

        # Compute distances to centroids
        obs_vector = np.array([
            observables.get('omega', 0.5),
            observables.get('phi', 0.5),
            observables.get('eta', 0.5),
            observables.get('xi', 0.5),
            observables.get('delta', 0.5)
        ])

        distances = {
            regime: float(np.linalg.norm(obs_vector - centroid))
            for regime, centroid in self.REGIME_CENTROIDS.items()
        }

        # Classify based on scores
        if self.use_soft_classification:
            regime, confidence = self._soft_classify(nagare_score, katashi_score)
        else:
            regime, confidence = self._hard_classify(observables)

        return RegimeClassification(
            regime=regime,
            confidence=confidence,
            nagare_score=nagare_score,
            katashi_score=katashi_score,
            distances=distances
        )

    def _compute_nagare_score(self, obs: Dict[str, float]) -> float:
        """
        Compute score for NAGARE regime (0-1).

        NAGARE criteria:
        - High diversity (Ξ): Attention spread across many tokens
        - Low rigidity (η): Flexible attention patterns
        - Low divergence (Δ): Close to uniform distribution

        Returns:
            Score in [0, 1] where 1 = perfect NAGARE
        """
        t = self.thresholds

        # Diversity score: higher xi = better
        xi = obs.get('xi', 0.5)
        if xi >= t.nagare_xi_min:
            xi_score = 1.0
        else:
            xi_score = xi / t.nagare_xi_min

        # Rigidity score: lower eta = better
        eta = obs.get('eta', 0.5)
        if eta <= t.nagare_eta_max:
            eta_score = 1.0
        else:
            eta_score = max(0, 1.0 - (eta - t.nagare_eta_max) / (1.0 - t.nagare_eta_max))

        # Divergence score: lower delta = better
        delta = obs.get('delta', 0.5)
        if delta <= t.nagare_delta_max:
            delta_score = 1.0
        else:
            delta_score = max(0, 1.0 - (delta - t.nagare_delta_max) / (1.0 - t.nagare_delta_max))

        # Weighted average (diversity most important)
        weights = [0.4, 0.35, 0.25]  # xi, eta, delta
        score = weights[0] * xi_score + weights[1] * eta_score + weights[2] * delta_score

        return float(np.clip(score, 0, 1))

    def _compute_katashi_score(self, obs: Dict[str, float]) -> float:
        """
        Compute score for KATASHI regime (0-1).

        KATASHI criteria:
        - High rigidity (η): Concentrated attention
        - High persistence (Φ): Stable cyclic patterns
        - Low diversity (Ξ): Few tokens dominate

        Returns:
            Score in [0, 1] where 1 = perfect KATASHI
        """
        t = self.thresholds

        # Rigidity score: higher eta = more katashi
        eta = obs.get('eta', 0.5)
        if eta >= t.katashi_eta_min:
            eta_score = 1.0
        else:
            eta_score = eta / t.katashi_eta_min

        # Persistence score: higher phi = more katashi
        phi = obs.get('phi', 0.5)
        if phi >= t.katashi_phi_min:
            phi_score = 1.0
        else:
            phi_score = phi / t.katashi_phi_min

        # Diversity score: lower xi = more katashi
        xi = obs.get('xi', 0.5)
        if xi <= t.katashi_xi_max:
            xi_score = 1.0
        else:
            xi_score = max(0, 1.0 - (xi - t.katashi_xi_max) / (1.0 - t.katashi_xi_max))

        # Weighted average (rigidity most important for katashi)
        weights = [0.4, 0.3, 0.3]  # eta, phi, xi
        score = weights[0] * eta_score + weights[1] * phi_score + weights[2] * xi_score

        return float(np.clip(score, 0, 1))

    def _soft_classify(
        self,
        nagare_score: float,
        katashi_score: float
    ) -> Tuple[AttentionRegime, float]:
        """Soft classification based on scores."""
        if nagare_score > katashi_score and nagare_score > 0.5:
            return AttentionRegime.NAGARE, nagare_score
        elif katashi_score > nagare_score and katashi_score > 0.5:
            return AttentionRegime.KATASHI, katashi_score
        else:
            # Transitional
            confidence = 1.0 - max(nagare_score, katashi_score)
            return AttentionRegime.UTSUROI, confidence

    def _hard_classify(
        self,
        obs: Dict[str, float]
    ) -> Tuple[AttentionRegime, float]:
        """Hard classification based on thresholds."""
        t = self.thresholds

        # Check NAGARE conditions
        is_nagare = (
            obs.get('xi', 0) >= t.nagare_xi_min and
            obs.get('eta', 1) <= t.nagare_eta_max and
            obs.get('delta', 1) <= t.nagare_delta_max
        )

        # Check KATASHI conditions
        is_katashi = (
            obs.get('eta', 0) >= t.katashi_eta_min and
            obs.get('phi', 0) >= t.katashi_phi_min and
            obs.get('xi', 1) <= t.katashi_xi_max
        )

        if is_nagare and not is_katashi:
            return AttentionRegime.NAGARE, 1.0
        elif is_katashi and not is_nagare:
            return AttentionRegime.KATASHI, 1.0
        else:
            return AttentionRegime.UTSUROI, 0.5

    def classify_trajectory(
        self,
        trajectory: List[Dict[str, float]]
    ) -> List[RegimeClassification]:
        """
        Classify regime for a trajectory of observables.

        Useful for analyzing regime transitions over time.

        Args:
            trajectory: List of observable dicts over time.

        Returns:
            List of classifications for each timestep.
        """
        return [self.classify(obs) for obs in trajectory]

    def detect_transition(
        self,
        trajectory: List[Dict[str, float]]
    ) -> Optional[int]:
        """
        Detect first regime transition in trajectory.

        Returns:
            Index of first transition, or None if no transition.
        """
        classifications = self.classify_trajectory(trajectory)

        for i in range(1, len(classifications)):
            if classifications[i].regime != classifications[i-1].regime:
                return i

        return None


def classify_attention_regime(
    observables: Dict[str, float],
    thresholds: Optional[Dict] = None
) -> Tuple[AttentionRegime, float]:
    """
    Convenience function for quick regime classification.

    Args:
        observables: Dict with observable values.
        thresholds: Optional custom thresholds dict.

    Returns:
        Tuple of (regime, confidence)
    """
    if thresholds:
        t = RegimeThresholds.from_dict(thresholds)
        classifier = RegimeClassifier(thresholds=t)
    else:
        classifier = RegimeClassifier()

    result = classifier.classify(observables)
    return result.regime, result.confidence
