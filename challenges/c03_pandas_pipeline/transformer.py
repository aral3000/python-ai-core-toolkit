"""
Challenge 03: Production Feature Engineering & Data Pipeline with Pandas
Author: M. Athallah Rafif Aldera

Task:
Implement pure, chainable DataFrame transformations for machine learning preprocessing.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd


def clean_and_impute(df: pd.DataFrame, strategy: Dict[str, str]) -> pd.DataFrame:
    """
    Imputes missing values in `df` based on `strategy`.
    
    Parameters:
        df: Input DataFrame (must NOT be mutated in-place)
        strategy: Dict mapping column name to imputation method:
                  - 'mean': Replace NaN with column arithmetic mean
                  - 'median': Replace NaN with column median
                  - 'mode': Replace NaN with the most frequent value
                  - 'zero': Replace NaN with 0
                  
    Returns:
        New pandas DataFrame with imputed columns.
    """
    # TODO: Athallah - implement clean_and_impute without mutating input df
    raise NotImplementedError("TODO: Athallah - implement clean_and_impute")


def encode_cyclical_features(df: pd.DataFrame, col: str, max_val: float) -> pd.DataFrame:
    """
    Encodes a periodic/cyclical feature (e.g. hour 0-23, month 1-12)
    into two continuous orthogonal circular coordinates using sine and cosine:
        sin_val = sin(2 * pi * x / max_val)
        cos_val = cos(2 * pi * x / max_val)
        
    Parameters:
        df: Input DataFrame
        col: Target column name
        max_val: Maximum cyclical period (e.g. 24.0 for hours, 12.0 for months)
        
    Returns:
        New DataFrame containing two additional columns:
        f"{col}_sin" and f"{col}_cos".
    """
    # TODO: Athallah - implement cyclical sin/cos transformation
    raise NotImplementedError("TODO: Athallah - implement encode_cyclical_features")


def compute_rolling_features(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    window: int
) -> pd.DataFrame:
    """
    Computes rolling mean and rolling standard deviation per group.
    
    Parameters:
        df: Input DataFrame (assumed pre-sorted chronologically)
        group_col: Grouping entity column (e.g. 'user_id', 'sensor_id')
        target_col: Numeric feature column to aggregate
        window: Window size for rolling statistics
        
    Returns:
        New DataFrame containing original columns plus:
        f"{target_col}_rolling_mean_{window}" and f"{target_col}_rolling_std_{window}"
        (with NaN filled with 0.0 or backward-filled).
    """
    # TODO: Athallah - implement rolling window per group
    raise NotImplementedError("TODO: Athallah - implement compute_rolling_features")
