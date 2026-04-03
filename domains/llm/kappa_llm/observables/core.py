"""
Observáveis canônicos do método Kappa aplicados a LLMs.
Mapeamento: Atenção → (Ω, Φ, η, Ξ, Δ)

Este módulo implementa a extração dos 5 observáveis estruturais a partir
de matrizes de atenção de LLMs, permitindo análise topológica do regime
atencional e detecção de padrões obsessivos (hallucinations).

References:
    - Kappa Method Framework (Radiante Pentadimensional)
    - HEIMDALL: Topological Hallucination Detection
"""

import numpy as np
from scipy.stats import entropy as scipy_entropy
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class ObservableBundle:
    """Bundle containing all computed observables with metadata."""
    omega: float      # Ω - Pressão atencional (entropia)
    phi: float        # Φ - Persistência (ciclos topológicos)
    eta: float        # η - Rigidez (concentração)
    xi: float         # Ξ - Diversidade (participation ratio)
    delta: float      # Δ - Divergência (KL from uniform)
    rscore: float     # R-Score composto (para compatibilidade)

    # Metadata
    n_tokens: int
    matrix_norm: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format."""
        return {
            'omega': self.omega,
            'phi': self.phi,
            'eta': self.eta,
            'xi': self.xi,
            'delta': self.delta,
            'rscore': self.rscore
        }

    def as_vector(self) -> np.ndarray:
        """Return observables as numpy vector [Ω, Φ, η, Ξ, Δ]."""
        return np.array([self.omega, self.phi, self.eta, self.xi, self.delta])


class KappaObservables:
    """
    Computa todos os observáveis estruturais a partir de uma matriz de atenção.

    Esta classe implementa o mapeamento do método Kappa para LLMs:
    - Matriz de atenção A ∈ R^{n×n} → (Ω, Φ, η, Ξ, Δ) ∈ [0,1]^5

    Attributes:
        A: Matriz de atenção normalizada
        n: Número de tokens
        eps: Epsilon para estabilidade numérica
        use_gudhi: Se True, usa GUDHI para homologia persistente

    Example:
        >>> attention = model.get_attention_matrix(input_ids)
        >>> obs = KappaObservables(attention)
        >>> result = obs.compute_all()
        >>> print(f"Regime entropy (Ω): {result['omega']:.3f}")
    """

    def __init__(
        self,
        attention_matrix: np.ndarray,
        use_gudhi: bool = True,
        eps: float = 1e-10
    ):
        """
        Initialize with attention matrix.

        Args:
            attention_matrix: Square matrix A[i,j] = attention from token i to j.
                             Should be normalized (sum ~= 1 or rows sum to 1).
            use_gudhi: If True, compute persistent homology with GUDHI.
                      If False, use fast approximation for Φ.
            eps: Small constant for numerical stability.
        """
        self.A = np.asarray(attention_matrix, dtype=np.float64)
        self.n = self.A.shape[0]
        self.eps = eps
        self.use_gudhi = use_gudhi

        # Validate input
        if self.A.ndim != 2:
            raise ValueError(f"Expected 2D matrix, got shape {self.A.shape}")
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError(f"Expected square matrix, got shape {self.A.shape}")

        # Ensure non-negative
        self.A = np.maximum(self.A, 0)

        # Store normalization for metadata
        self._matrix_norm = float(np.sum(self.A))

        # Precompute flattened and filtered versions
        self._flat = self.A.flatten()
        self._flat_positive = self._flat[self._flat > self.eps]

    def compute_all(self) -> Dict[str, float]:
        """
        Computa todos os 5 observáveis canônicos + R-score.

        Returns:
            Dictionary with keys: omega, phi, eta, xi, delta, rscore
            All values are in [0, 1] except rscore which can be > 1.
        """
        return {
            'omega': self._compute_omega(),
            'phi': self._compute_phi(),
            'eta': self._compute_eta(),
            'xi': self._compute_xi(),
            'delta': self._compute_delta(),
            'rscore': self._compute_rscore()
        }

    def compute_bundle(self) -> ObservableBundle:
        """
        Computa observáveis e retorna como bundle estruturado.

        Returns:
            ObservableBundle with all observables and metadata.
        """
        obs = self.compute_all()
        return ObservableBundle(
            omega=obs['omega'],
            phi=obs['phi'],
            eta=obs['eta'],
            xi=obs['xi'],
            delta=obs['delta'],
            rscore=obs['rscore'],
            n_tokens=self.n,
            matrix_norm=self._matrix_norm
        )

    def _compute_omega(self) -> float:
        """
        Ω - Pressão atencional (entropia normalizada).

        Mede a distribuição de atenção:
        - Ω ≈ 1: Atenção uniformemente distribuída (baixa pressão)
        - Ω ≈ 0: Atenção altamente concentrada (alta pressão)

        Formula:
            Ω = H(A_flat) / H_max
            where H is Shannon entropy and H_max = log2(n²)
        """
        if len(self._flat_positive) == 0:
            return 0.0

        # Normalize to probability distribution
        probs = self._flat_positive / (np.sum(self._flat_positive) + self.eps)

        # Shannon entropy in bits
        H = scipy_entropy(probs, base=2)

        # Maximum possible entropy for n² elements
        H_max = np.log2(self.n ** 2)

        omega = H / H_max if H_max > 0 else 0.0
        return float(np.clip(omega, 0, 1))

    def _compute_phi(self) -> float:
        """
        Φ - Persistência (máximo lifetime de ciclos H1).

        Mede estruturas cíclicas persistentes na atenção:
        - Φ ≈ 1: Ciclos atencionais fortes e persistentes
        - Φ ≈ 0: Sem estruturas cíclicas significativas

        Implementação:
            - Com GUDHI: Homologia persistente via Rips complex
            - Sem GUDHI: Aproximação baseada em autovalores
        """
        if self.use_gudhi:
            return self._compute_phi_gudhi()
        else:
            return self._compute_phi_approx()

    def _compute_phi_gudhi(self) -> float:
        """Compute Φ using GUDHI persistent homology."""
        try:
            from gudhi import CubicalComplex

            # Usar CubicalComplex que é mais robusto para matrizes 2D
            # Inverter para que alta atenção = baixa "altura"
            inverted = 1.0 - self.A / (self.A.max() + self.eps)

            cc = CubicalComplex(dimensions=list(inverted.shape), top_dimensional_cells=inverted.flatten())
            cc.compute_persistence()

            # Extrair pares H1 (1-dimensional cycles)
            persistence_pairs = cc.persistence_intervals_in_dimension(1)

            if len(persistence_pairs) == 0:
                return 0.0

            # Calcular lifetimes
            lifetimes = []
            for birth, death in persistence_pairs:
                if np.isfinite(death):
                    lifetimes.append(death - birth)

            if len(lifetimes) == 0:
                return 0.0

            max_lifetime = max(lifetimes)

            # Normalizar (max teórico = 1.0 para matriz normalizada)
            return float(np.clip(max_lifetime, 0, 1))

        except ImportError:
            # Fallback para aproximação
            return self._compute_phi_approx()

    def _compute_phi_approx(self) -> float:
        """Approximate Φ without GUDHI using spectral analysis."""
        # Use eigenvalue gap as proxy for persistent structure
        try:
            eigenvalues = np.linalg.eigvalsh(self.A)
            eigenvalues = np.sort(np.abs(eigenvalues))[::-1]

            if len(eigenvalues) < 2:
                return 0.0

            # Spectral gap normalized
            gap = (eigenvalues[0] - eigenvalues[1]) / (eigenvalues[0] + self.eps)

            return float(np.clip(gap, 0, 1))
        except:
            return 0.0

    def _compute_eta(self) -> float:
        """
        η - Rigidez (concentração via Gini ou complemento de entropia).

        Mede a concentração/rigidez da distribuição:
        - η ≈ 1: Atenção muito concentrada (alta rigidez)
        - η ≈ 0: Atenção distribuída (baixa rigidez)

        Formula: η = 1 - Ω (complemento da entropia normalizada)

        Alternativa (Gini coefficient) disponível via _compute_eta_gini().
        """
        omega = self._compute_omega()
        return 1.0 - omega

    def _compute_eta_gini(self) -> float:
        """Alternative η computation using Gini coefficient."""
        flat = np.sort(self._flat)
        n = len(flat)
        if n == 0 or flat.sum() == 0:
            return 0.0

        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * flat)) / (n * np.sum(flat)) - (n + 1) / n

        return float(np.clip(gini, 0, 1))

    def _compute_xi(self) -> float:
        """
        Ξ - Diversidade (inverse participation ratio normalizada).

        Mede quantos elementos "participam efetivamente":
        - Ξ ≈ 1: Muitos elementos contribuem (alta diversidade)
        - Ξ ≈ 0: Poucos elementos dominam (baixa diversidade)

        Formula:
            IPR = 1 / Σᵢ pᵢ⁴
            Ξ = IPR / n²  (normalizado pelo máximo)
        """
        # Normalize to probabilities
        total = np.sum(self._flat) + self.eps
        probs = self._flat / total

        # Inverse Participation Ratio
        ipr = 1.0 / (np.sum(probs ** 4) + self.eps)

        # Normalize by maximum (n² for uniform distribution)
        xi_norm = ipr / (self.n ** 2)

        return float(np.clip(xi_norm, 0, 1))

    def _compute_delta(self) -> float:
        """
        Δ - Divergência (KL divergence da distribuição uniforme).

        Mede quanto a distribuição diverge do uniforme:
        - Δ ≈ 1: Muito diferente de uniforme (alta divergência)
        - Δ ≈ 0: Próximo de uniforme (baixa divergência)

        Formula:
            Δ = D_KL(A || U) / D_KL_max
            where U is uniform distribution
        """
        if len(self._flat_positive) == 0:
            return 0.0

        # Normalize to probabilities
        total = np.sum(self._flat) + self.eps
        probs = self._flat / total

        # Uniform distribution
        uniform = 1.0 / (self.n ** 2)

        # KL divergence: Σ p * log(p/q)
        # Add eps to avoid log(0)
        probs_safe = np.maximum(probs, self.eps)
        kl = np.sum(probs_safe * np.log(probs_safe / uniform))

        # Maximum KL (when all mass on one element)
        # D_KL(delta || uniform) = log(n²)
        kl_max = np.log(self.n ** 2)

        delta_norm = kl / kl_max if kl_max > 0 else 0.0

        return float(np.clip(delta_norm, 0, 1))

    def _compute_rscore(self) -> float:
        """
        R-Score original do HEIMDALL (para compatibilidade e comparação).

        Combina número de ciclos e lifetime máximo via homologia persistente.

        Formula:
            R = log(1 + max_lifetime / (num_cycles + ε))

        Note: R-score pode ser > 1 (não é normalizado como outros observáveis).
        """
        try:
            from gudhi import CubicalComplex

            # Usar CubicalComplex
            inverted = 1.0 - self.A / (self.A.max() + self.eps)

            cc = CubicalComplex(dimensions=list(inverted.shape), top_dimensional_cells=inverted.flatten())
            cc.compute_persistence()

            persistence_pairs = cc.persistence_intervals_in_dimension(1)

            if len(persistence_pairs) == 0:
                return 0.0

            lifetimes = []
            for birth, death in persistence_pairs:
                if np.isfinite(death):
                    lifetimes.append(death - birth)

            if len(lifetimes) == 0:
                return 0.0

            max_lifetime = max(lifetimes)
            num_cycles = len(lifetimes)

            # Fórmula original HEIMDALL
            rscore = np.log(1 + max_lifetime / (num_cycles + self.eps))

            return float(rscore)

        except ImportError:
            # Fallback: use phi as approximation
            return self._compute_phi_approx()


def compute_kappa_observables(attention_matrix: np.ndarray, **kwargs) -> Dict[str, float]:
    """
    Interface simples para computar observáveis Kappa.

    Args:
        attention_matrix: Square attention matrix from LLM.
        **kwargs: Additional arguments passed to KappaObservables.

    Returns:
        Dictionary with all observables: omega, phi, eta, xi, delta, rscore.

    Example:
        >>> attention = model.get_attention_weights(input_ids)
        >>> obs = compute_kappa_observables(attention)
        >>> is_concentrated = obs['eta'] > 0.7
    """
    obs = KappaObservables(attention_matrix, **kwargs)
    return obs.compute_all()


def compute_observables_batch(
    attention_matrices: List[np.ndarray],
    **kwargs
) -> List[Dict[str, float]]:
    """
    Compute observables for multiple attention matrices.

    Args:
        attention_matrices: List of attention matrices.
        **kwargs: Arguments passed to KappaObservables.

    Returns:
        List of observable dictionaries.
    """
    return [compute_kappa_observables(A, **kwargs) for A in attention_matrices]


def observables_to_dataframe(
    observables_list: List[Dict[str, float]],
    labels: Optional[List[int]] = None
):
    """
    Convert list of observables to pandas DataFrame.

    Args:
        observables_list: List of observable dictionaries.
        labels: Optional labels (0=factual, 1=hallucination).

    Returns:
        pandas DataFrame with observables as columns.
    """
    import pandas as pd

    df = pd.DataFrame(observables_list)
    if labels is not None:
        df['label'] = labels

    return df
