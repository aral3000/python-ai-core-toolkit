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
        
    Constraints:
        Must be fully vectorized (NO python for-loops).
    """
    # TODO: Athallah - implement vectorized cosine similarity
    raise NotImplementedError("TODO: Athallah - implement cosine_similarity_matrix")


def stable_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Computes numerically stable softmax to prevent exp(x) floating point overflow.
    Uses the identity: softmax(x) = softmax(x - max(x)).
    
    Parameters:
        logits: np.ndarray of arbitrary shape
        axis: Int, axis along which softmax is applied (default: -1)
        
    Returns:
        np.ndarray of the same shape as logits, where elements along `axis`
        are non-negative and sum to 1.0.
    """
    # TODO: Athallah - implement numerically stable softmax
    raise NotImplementedError("TODO: Athallah - implement stable_softmax")


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Computes Scaled Dot-Product Attention (Vaswani et al., 2017):
        Attention(Q, K, V) = softmax( (Q @ K.T) / sqrt(d_k) + mask ) @ V
        
    Parameters:
        Q: Queries array of shape (batch_size, seq_len_q, d_k)
        K: Keys array of shape (batch_size, seq_len_k, d_k)
        V: Values array of shape (batch_size, seq_len_k, d_v)
        mask: Optional mask array of shape (batch_size, seq_len_q, seq_len_k)
              Positions where mask == 1 (or True) should have attention scores set to -1e9.
              
    Returns:
        np.ndarray of shape (batch_size, seq_len_q, d_v)
        
    Constraints:
        Vectorized batched matrix multiplication using np.matmul or @ operator.
    """
    # TODO: Athallah - implement scaled dot product attention
    raise NotImplementedError("TODO: Athallah - implement scaled_dot_product_attention")
