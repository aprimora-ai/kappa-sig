"""
Data utilities for Kappa-LLM.

Provides functions for loading HaluEval dataset, extracting attention
matrices from various LLM backends, and data preprocessing.
"""

import numpy as np
from typing import Dict, Optional, List, Tuple, Any


def normalize_attention_matrix(
    attention: np.ndarray,
    method: str = 'sum'
) -> np.ndarray:
    """
    Normalize attention matrix for analysis.

    Args:
        attention: Raw attention matrix (any shape, will be made 2D)
        method: Normalization method
            - 'sum': Divide by sum (makes it a probability distribution)
            - 'row': Normalize each row to sum to 1
            - 'max': Divide by maximum value
            - 'none': No normalization

    Returns:
        Normalized 2D attention matrix
    """
    # Ensure 2D
    if attention.ndim == 1:
        n = int(np.sqrt(len(attention)))
        attention = attention.reshape(n, n)
    elif attention.ndim > 2:
        # Take mean over extra dimensions (e.g., layers, heads)
        while attention.ndim > 2:
            attention = attention.mean(axis=0)

    # Ensure non-negative
    attention = np.maximum(attention, 0)

    if method == 'sum':
        total = attention.sum()
        if total > 0:
            attention = attention / total
    elif method == 'row':
        row_sums = attention.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        attention = attention / row_sums
    elif method == 'max':
        max_val = attention.max()
        if max_val > 0:
            attention = attention / max_val
    # 'none' does nothing

    return attention


def attention_from_model(
    model: Any,
    tokenizer: Any,
    text: str,
    layer: Optional[int] = None,
    head: Optional[int] = None,
    aggregation: str = 'mean'
) -> np.ndarray:
    """
    Extract attention matrix from HuggingFace model.

    Args:
        model: HuggingFace model with output_attentions=True
        tokenizer: Corresponding tokenizer
        text: Input text
        layer: Specific layer to extract (None = all layers)
        head: Specific head to extract (None = all heads)
        aggregation: How to aggregate multiple layers/heads
            - 'mean': Average
            - 'max': Maximum
            - 'last': Use last layer

    Returns:
        Normalized attention matrix [n_tokens x n_tokens]
    """
    import torch

    # Tokenize
    inputs = tokenizer(text, return_tensors='pt')

    # Forward pass with attention output
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # Get attentions: tuple of (batch, heads, seq, seq) per layer
    attentions = outputs.attentions

    # Stack all layers: [layers, heads, seq, seq]
    attn_stack = torch.stack([a.squeeze(0) for a in attentions])

    # Select layer if specified
    if layer is not None:
        attn_stack = attn_stack[layer:layer+1]

    # Select head if specified
    if head is not None:
        attn_stack = attn_stack[:, head:head+1]

    # Aggregate
    if aggregation == 'mean':
        attention = attn_stack.mean(dim=(0, 1)).numpy()
    elif aggregation == 'max':
        attention = attn_stack.amax(dim=(0, 1)).numpy()
    elif aggregation == 'last':
        attention = attn_stack[-1].mean(dim=0).numpy()
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    return normalize_attention_matrix(attention)


def load_halueval_sample(
    index: int = 0,
    split: str = 'test',
    return_label: bool = True
) -> Tuple[Dict[str, str], Optional[int]]:
    """
    Load a single sample from HaluEval dataset.

    Args:
        index: Sample index
        split: Dataset split ('test', 'train')
        return_label: If True, return label (0=factual, 1=hallucination)

    Returns:
        Tuple of (sample_dict, label) or just sample_dict if return_label=False

    Note:
        Requires datasets library: pip install datasets
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    # Load HaluEval
    dataset = load_dataset("pminervini/HaluEval")
    sample = dataset[split][index]

    if return_label:
        label = 1 if sample.get('hallucination', False) else 0
        return sample, label
    return sample, None


def load_halueval_batch(
    n_samples: int = 100,
    split: str = 'test',
    balanced: bool = True
) -> Tuple[List[Dict], List[int]]:
    """
    Load batch of samples from HaluEval.

    Args:
        n_samples: Number of samples to load
        split: Dataset split
        balanced: If True, balance factual/hallucination samples

    Returns:
        Tuple of (samples_list, labels_list)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    dataset = load_dataset("pminervini/HaluEval")
    data = dataset[split]

    samples = []
    labels = []

    if balanced:
        # Get equal numbers of each class
        n_each = n_samples // 2
        factual_count = 0
        hallu_count = 0

        for sample in data:
            is_hallu = sample.get('hallucination', False)

            if is_hallu and hallu_count < n_each:
                samples.append(sample)
                labels.append(1)
                hallu_count += 1
            elif not is_hallu and factual_count < n_each:
                samples.append(sample)
                labels.append(0)
                factual_count += 1

            if factual_count >= n_each and hallu_count >= n_each:
                break
    else:
        for i, sample in enumerate(data):
            if i >= n_samples:
                break
            samples.append(sample)
            labels.append(1 if sample.get('hallucination', False) else 0)

    return samples, labels


def create_synthetic_attention(
    n_tokens: int = 64,
    regime: str = 'uniform'
) -> np.ndarray:
    """
    Create synthetic attention matrix for testing.

    Args:
        n_tokens: Number of tokens (matrix size)
        regime: Pattern type
            - 'uniform': Uniform attention
            - 'diagonal': Attention on diagonal
            - 'concentrated': All attention on one element
            - 'causal': Lower triangular (causal attention)
            - 'random': Random attention

    Returns:
        Normalized attention matrix
    """
    if regime == 'uniform':
        A = np.ones((n_tokens, n_tokens))
    elif regime == 'diagonal':
        A = np.eye(n_tokens) + 0.1 * np.ones((n_tokens, n_tokens))
    elif regime == 'concentrated':
        A = np.zeros((n_tokens, n_tokens))
        A[0, 0] = 1.0
    elif regime == 'causal':
        A = np.tril(np.ones((n_tokens, n_tokens)))
    elif regime == 'random':
        A = np.random.rand(n_tokens, n_tokens)
    else:
        raise ValueError(f"Unknown regime: {regime}")

    return normalize_attention_matrix(A)
