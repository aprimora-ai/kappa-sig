"""
Detector de alucinação multi-observable usando framework Kappa.

Este módulo implementa o detector principal que combina múltiplos observáveis
estruturais em um score Kappa composto para detecção robusta de hallucinations.

A fórmula do Kappa Score é:
    K = w₁·R + w₂·η + w₃·(1-Ξ) + w₄·Δ - w₅·Ω

Onde:
    - R: R-Score (persistência topológica)
    - η: Rigidez (concentração)
    - Ξ: Diversidade (participation ratio)
    - Δ: Divergência (KL from uniform)
    - Ω: Entropia (pressão atencional)

Scores altos indicam padrões obsessivos típicos de hallucinations.
"""

from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
import numpy as np

from ..regimes import RegimeClassifier, AttentionRegime


@dataclass
class DetectorConfig:
    """
    Configuration for KappaDetector.

    Attributes:
        weights: Weight for each observable in Kappa score
        threshold: Detection threshold (scores above = hallucination)
        include_regime: If True, also compute regime classification
    """
    # Weights for Kappa Score formula
    w_rscore: float = 0.30      # R-Score weight
    w_eta: float = 0.25         # Rigidity weight
    w_xi_inv: float = 0.20      # (1 - diversity) weight
    w_delta: float = 0.15       # Divergence weight
    w_omega_inv: float = 0.10   # (-entropy) weight

    # Detection threshold
    threshold: float = 0.5

    # Additional options
    include_regime: bool = True
    normalize_weights: bool = True

    def get_weights_dict(self) -> Dict[str, float]:
        """Return weights as dictionary."""
        weights = {
            'w_rscore': self.w_rscore,
            'w_eta': self.w_eta,
            'w_xi_inv': self.w_xi_inv,
            'w_delta': self.w_delta,
            'w_omega_inv': self.w_omega_inv
        }
        if self.normalize_weights:
            total = sum(abs(v) for v in weights.values())
            if total > 0:
                weights = {k: v/total for k, v in weights.items()}
        return weights

    @classmethod
    def balanced(cls) -> 'DetectorConfig':
        """Balanced weights configuration."""
        return cls(
            w_rscore=0.25,
            w_eta=0.25,
            w_xi_inv=0.20,
            w_delta=0.15,
            w_omega_inv=0.15
        )

    @classmethod
    def rscore_focused(cls) -> 'DetectorConfig':
        """R-Score focused configuration (closer to HEIMDALL)."""
        return cls(
            w_rscore=0.50,
            w_eta=0.15,
            w_xi_inv=0.15,
            w_delta=0.10,
            w_omega_inv=0.10
        )

    @classmethod
    def regime_focused(cls) -> 'DetectorConfig':
        """Regime indicators focused configuration."""
        return cls(
            w_rscore=0.15,
            w_eta=0.30,
            w_xi_inv=0.30,
            w_delta=0.15,
            w_omega_inv=0.10
        )


@dataclass
class DetectionResult:
    """
    Result of hallucination detection.

    Attributes:
        is_hallucination: Binary prediction
        kappa_score: Composite Kappa score
        confidence: Confidence in prediction
        regime: Attention regime (if include_regime=True)
        component_scores: Individual contributions to Kappa score
        observables: Original observable values
    """
    is_hallucination: bool
    kappa_score: float
    confidence: float
    regime: Optional[AttentionRegime] = None
    component_scores: Dict[str, float] = field(default_factory=dict)
    observables: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            'is_hallucination': self.is_hallucination,
            'kappa_score': self.kappa_score,
            'confidence': self.confidence,
            'regime': self.regime.value if self.regime else None,
            'component_scores': self.component_scores,
            'observables': self.observables
        }


class KappaDetector:
    """
    Detector de hallucinations usando múltiplos observáveis Kappa.

    O detector combina informação de todos os 5 observáveis canônicos
    em um score Kappa composto que captura diferentes aspectos de
    padrões atencionais patológicos.

    Attributes:
        config: DetectorConfig with weights and threshold
        regime_classifier: Optional RegimeClassifier for regime detection

    Example:
        >>> detector = KappaDetector()
        >>> obs = compute_kappa_observables(attention_matrix)
        >>> result = detector.detect(obs)
        >>> if result.is_hallucination:
        ...     print(f"Hallucination detected (K={result.kappa_score:.3f})")
    """

    def __init__(
        self,
        config: Optional[DetectorConfig] = None,
        regime_classifier: Optional[RegimeClassifier] = None
    ):
        """
        Initialize detector.

        Args:
            config: Detector configuration. Uses default if None.
            regime_classifier: Optional custom regime classifier.
        """
        self.config = config or DetectorConfig()
        self.weights = self.config.get_weights_dict()

        if self.config.include_regime:
            self.regime_classifier = regime_classifier or RegimeClassifier()
        else:
            self.regime_classifier = None

    def compute_kappa_score(self, observables: Dict[str, float]) -> float:
        """
        Computa score composto Kappa.

        Formula:
            K = w₁·R + w₂·η + w₃·(1-Ξ) + w₄·Δ - w₅·Ω

        High K indicates hallucination-like patterns:
        - High R-Score (persistent topological structure)
        - High η (rigid attention)
        - Low Ξ (low diversity)
        - High Δ (divergent from uniform)
        - Low Ω (low entropy, concentrated)

        Args:
            observables: Dict with omega, phi, eta, xi, delta, rscore

        Returns:
            Kappa score (not bounded, typically in [0, 1] but can exceed)
        """
        w = self.weights
        obs = observables

        score = (
            w['w_rscore'] * obs.get('rscore', 0) +
            w['w_eta'] * obs.get('eta', 0) +
            w['w_xi_inv'] * (1 - obs.get('xi', 0.5)) +
            w['w_delta'] * obs.get('delta', 0) -
            w['w_omega_inv'] * obs.get('omega', 0.5)
        )

        return float(score)

    def compute_component_scores(
        self,
        observables: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute individual component contributions to Kappa score.

        Returns:
            Dict with each component's weighted contribution.
        """
        w = self.weights
        obs = observables

        return {
            'rscore_contrib': w['w_rscore'] * obs.get('rscore', 0),
            'eta_contrib': w['w_eta'] * obs.get('eta', 0),
            'xi_inv_contrib': w['w_xi_inv'] * (1 - obs.get('xi', 0.5)),
            'delta_contrib': w['w_delta'] * obs.get('delta', 0),
            'omega_inv_contrib': -w['w_omega_inv'] * obs.get('omega', 0.5)
        }

    def detect(self, observables: Dict[str, float]) -> DetectionResult:
        """
        Detecta hallucination baseado em observáveis.

        Args:
            observables: Dict with observable values

        Returns:
            DetectionResult with prediction and details
        """
        # Compute Kappa score
        kappa_score = self.compute_kappa_score(observables)

        # Binary prediction
        is_hallu = kappa_score > self.config.threshold

        # Confidence based on distance from threshold
        distance = abs(kappa_score - self.config.threshold)
        max_distance = max(self.config.threshold, 1 - self.config.threshold)
        confidence = min(1.0, distance / max_distance)

        # Compute component contributions
        component_scores = self.compute_component_scores(observables)

        # Regime classification (optional)
        regime = None
        if self.regime_classifier:
            regime_result = self.regime_classifier.classify(observables)
            regime = regime_result.regime

        return DetectionResult(
            is_hallucination=is_hallu,
            kappa_score=kappa_score,
            confidence=confidence,
            regime=regime,
            component_scores=component_scores,
            observables=observables
        )

    def detect_batch(
        self,
        observables_list: List[Dict[str, float]]
    ) -> List[DetectionResult]:
        """
        Detect hallucinations for batch of samples.

        Args:
            observables_list: List of observable dicts

        Returns:
            List of DetectionResult
        """
        return [self.detect(obs) for obs in observables_list]

    def calibrate_threshold(
        self,
        factual_obs: List[Dict[str, float]],
        hallu_obs: List[Dict[str, float]],
        target_fpr: float = 0.05
    ) -> float:
        """
        Calibra threshold para atingir FPR alvo.

        Usa o percentil dos scores de amostras factuais para
        definir um threshold que resulta no FPR desejado.

        Args:
            factual_obs: List of observables from factual responses
            hallu_obs: List of observables from hallucinations (for validation)
            target_fpr: Target false positive rate (default 5%)

        Returns:
            Calibrated threshold value
        """
        # Compute scores for factual samples
        factual_scores = [self.compute_kappa_score(o) for o in factual_obs]

        # Threshold = percentile (1 - FPR) of factual scores
        threshold = float(np.percentile(factual_scores, (1 - target_fpr) * 100))

        return threshold

    def calibrate_weights(
        self,
        factual_obs: List[Dict[str, float]],
        hallu_obs: List[Dict[str, float]],
        n_iterations: int = 100
    ) -> Dict[str, float]:
        """
        Optimize weights using simple grid search.

        Note: For production, use proper optimization (scipy, optuna).

        Args:
            factual_obs: Factual sample observables
            hallu_obs: Hallucination sample observables
            n_iterations: Number of random weight combinations to try

        Returns:
            Optimized weights dictionary
        """
        best_auc = 0
        best_weights = self.weights.copy()

        all_obs = factual_obs + hallu_obs
        labels = [0] * len(factual_obs) + [1] * len(hallu_obs)

        for _ in range(n_iterations):
            # Generate random weights
            raw_weights = np.random.rand(5)
            raw_weights /= raw_weights.sum()  # Normalize

            weights = {
                'w_rscore': raw_weights[0],
                'w_eta': raw_weights[1],
                'w_xi_inv': raw_weights[2],
                'w_delta': raw_weights[3],
                'w_omega_inv': raw_weights[4]
            }

            # Compute scores with these weights
            self.weights = weights
            scores = [self.compute_kappa_score(o) for o in all_obs]

            # Compute AUC
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(labels, scores)
                if auc > best_auc:
                    best_auc = auc
                    best_weights = weights.copy()
            except:
                pass

        self.weights = best_weights
        return best_weights

    def get_feature_importance(
        self,
        observables_list: List[Dict[str, float]],
        labels: List[int]
    ) -> Dict[str, float]:
        """
        Compute feature importance via ablation.

        Removes each observable and measures AUC drop.

        Args:
            observables_list: List of observable dicts
            labels: Binary labels (0=factual, 1=hallucination)

        Returns:
            Dict mapping observable to importance score
        """
        try:
            from sklearn.metrics import roc_auc_score
        except ImportError:
            return {}

        # Baseline AUC
        baseline_scores = [self.compute_kappa_score(o) for o in observables_list]
        baseline_auc = roc_auc_score(labels, baseline_scores)

        importance = {}
        observables_to_test = ['rscore', 'eta', 'xi', 'delta', 'omega']

        for obs_name in observables_to_test:
            # Create modified observables with this one zeroed
            modified_obs = []
            for o in observables_list:
                m = o.copy()
                if obs_name in ['xi', 'omega']:
                    # For inverted observables, set to 0.5 (neutral)
                    m[obs_name] = 0.5
                else:
                    m[obs_name] = 0
                modified_obs.append(m)

            # Compute scores without this observable
            modified_scores = [self.compute_kappa_score(o) for o in modified_obs]
            modified_auc = roc_auc_score(labels, modified_scores)

            # Importance = AUC drop
            importance[obs_name] = baseline_auc - modified_auc

        return importance


def detect_hallucination(
    observables: Dict[str, float],
    threshold: float = 0.5
) -> Tuple[bool, float]:
    """
    Convenience function for quick hallucination detection.

    Args:
        observables: Observable dictionary
        threshold: Detection threshold

    Returns:
        Tuple of (is_hallucination, kappa_score)
    """
    config = DetectorConfig(threshold=threshold)
    detector = KappaDetector(config=config)
    result = detector.detect(observables)
    return result.is_hallucination, result.kappa_score
