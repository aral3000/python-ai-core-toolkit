"""
Challenge 02: High-Performance NumPy Vectorization Kernels for AI
Author: M. Athallah Rafif Aldera

Task:
Implement core mathematical operations using vectorized NumPy expressions.
NO PYTHON FOR-LOOPS over matrix dimensions allowed.
"""

from typing import Optional
import numpy as np


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Computes pairwise cosine similarity between two matrices of embeddings.
    
    Parameters:
        A: np.ndarray of shape (M, D)
        B: np.ndarray of shape (N, D)
        eps: Small float to prevent division by zero
        
    Returns:
        np.ndarray of shape (M, N) containing cosine similarity scores in [-1.0, 1.0].
    """
    a_norm = np.linalg.norm(A, axis=1, keepdims=True)
    b_norm = np.linalg.norm(B, axis=1, keepdims=True)
    return (A @ B.T) / np.maximum(a_norm @ b_norm.T, eps)


def stable_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Computes numerically stable softmax to prevent exp(x) floating point overflow.
    Uses the identity: softmax(x) = softmax(x - max(x)).
    """
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Computes Scaled Dot-Product Attention (Vaswani et al., 2017):
        Attention(Q, K, V) = softmax( (Q @ K.T) / sqrt(d_k) + mask ) @ V
    """
    d_k = Q.shape[-1]
    scores = (Q @ np.swapaxes(K, -1, -2)) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask == 1, -1e9, scores)
    weights = stable_softmax(scores, axis=-1)
    return weights @ V
