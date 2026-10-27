"""
Test suite for Challenge 02: NumPy Vectorization Kernels
Run with: pytest challenges/c02_numpy_vectorization/test_math_kernels.py
"""

import pytest
import numpy as np
from challenges.c02_numpy_vectorization.math_kernels import (
    cosine_similarity_matrix,
    stable_softmax,
    scaled_dot_product_attention
)


def test_cosine_similarity_matrix():
    A = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    B = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [1.0, 1.0, 0.0]
    ])
    sim = cosine_similarity_matrix(A, B)
    
    assert sim.shape == (2, 3)
    # Cosine of identical vector is 1.0
    assert np.isclose(sim[0, 0], 1.0)
    # Cosine of orthogonal vector is 0.0
    assert np.isclose(sim[0, 1], 0.0)
    # Cosine of opposite vector is -1.0
    assert np.isclose(sim[1, 1], -1.0)
    # Cosine with (1, 1, 0)
    assert np.isclose(sim[0, 2], 1.0 / np.sqrt(2))


def test_stable_softmax_overflow_prevention():
    # Large numbers that would overflow standard exp(x) in float64
    large_logits = np.array([1000.0, 1001.0, 1002.0])
    probs = stable_softmax(large_logits)
    
    assert not np.any(np.isnan(probs))
    assert not np.any(np.isinf(probs))
    assert np.isclose(np.sum(probs), 1.0)
    assert probs[2] > probs[1] > probs[0]


def test_stable_softmax_2d_axis():
    logits_2d = np.array([
        [1.0, 2.0, 3.0],
        [10.0, 20.0, 30.0]
    ])
    probs = stable_softmax(logits_2d, axis=-1)
    
    assert probs.shape == (2, 3)
    assert np.allclose(np.sum(probs, axis=-1), np.array([1.0, 1.0]))


def test_scaled_dot_product_attention_shape_and_values():
    batch_size = 2
    seq_len = 4
    d_k = 8
    d_v = 16

    np.random.seed(42)
    Q = np.random.randn(batch_size, seq_len, d_k)
    K = np.random.randn(batch_size, seq_len, d_k)
    V = np.random.randn(batch_size, seq_len, d_v)

    out = scaled_dot_product_attention(Q, K, V)
    assert out.shape == (batch_size, seq_len, d_v)
    assert not np.any(np.isnan(out))


def test_scaled_dot_product_attention_with_mask():
    batch_size = 1
    seq_len = 3
    d_k = 4
    d_v = 3

    Q = np.ones((batch_size, seq_len, d_k))
    K = np.ones((batch_size, seq_len, d_k))
    V = np.eye(seq_len).reshape((batch_size, seq_len, d_v))

    # Mask out the last token (index 2)
    mask = np.zeros((batch_size, seq_len, seq_len))
    mask[:, :, 2] = 1.0  # mask the 3rd key

    out = scaled_dot_product_attention(Q, K, V, mask=mask)
    # The output attention weights for the 3rd token should be essentially zero
    assert np.allclose(out[:, :, 2], 0.0, atol=1e-4)
