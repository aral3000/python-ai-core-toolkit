"""
Test suite for Challenge 03: Pandas Data Pipeline
Run with: pytest challenges/c03_pandas_pipeline/test_transformer.py
"""

import pytest
import numpy as np
import pandas as pd
from challenges.c03_pandas_pipeline.transformer import (
    clean_and_impute,
    encode_cyclical_features,
    compute_rolling_features
)


def test_clean_and_impute():
    df = pd.DataFrame({
        "age": [20.0, 30.0, np.nan, 40.0],
        "category": ["A", "B", "A", np.nan],
        "score": [10.0, np.nan, 30.0, 40.0]
    })
    strategy = {
        "age": "mean",      # (20 + 30 + 40)/3 = 30.0
        "category": "mode",  # 'A'
        "score": "zero"      # 0.0
    }
    cleaned = clean_and_impute(df, strategy)
    
    assert cleaned["age"].iloc[2] == 30.0
    assert cleaned["category"].iloc[3] == "A"
    assert cleaned["score"].iloc[1] == 0.0
    # Immutability check
    assert np.isnan(df["age"].iloc[2])


def test_encode_cyclical_features():
    df = pd.DataFrame({
        "hour": [0.0, 6.0, 12.0, 18.0]
    })
    encoded = encode_cyclical_features(df, "hour", max_val=24.0)
    
    assert "hour_sin" in encoded.columns
    assert "hour_cos" in encoded.columns
    
    # At hour 0: sin(0) = 0, cos(0) = 1
    assert np.isclose(encoded["hour_sin"].iloc[0], 0.0)
    assert np.isclose(encoded["hour_cos"].iloc[0], 1.0)
    
    # At hour 6: sin(pi/2) = 1, cos(pi/2) = 0
    assert np.isclose(encoded["hour_sin"].iloc[1], 1.0)
    assert np.isclose(encoded["hour_cos"].iloc[1], 0.0, atol=1e-7)
    
    # At hour 12: sin(pi) = 0, cos(pi) = -1
    assert np.isclose(encoded["hour_sin"].iloc[2], 0.0, atol=1e-7)
    assert np.isclose(encoded["hour_cos"].iloc[2], -1.0)


def test_compute_rolling_features():
    df = pd.DataFrame({
        "device": ["A", "A", "A", "A", "B", "B"],
        "metric": [10.0, 20.0, 30.0, 40.0, 100.0, 200.0]
    })
    res = compute_rolling_features(df, group_col="device", target_col="metric", window=2)
    
    mean_col = "metric_rolling_mean_2"
    assert mean_col in res.columns
    # Device A index 1 should be mean of [10, 20] = 15.0
    assert res[mean_col].iloc[1] == 15.0
    # Device A index 2 should be mean of [20, 30] = 25.0
    assert res[mean_col].iloc[2] == 25.0
    # Device B index 5 should be mean of [100, 200] = 150.0 (not bleeding into device A)
    assert res[mean_col].iloc[5] == 150.0
