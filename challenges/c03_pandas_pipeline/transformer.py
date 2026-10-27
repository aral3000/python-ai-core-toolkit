"""
Challenge 03: Production Feature Engineering & Data Pipeline with Pandas
Author: M. Athallah Rafif Aldera

Task:
Implement pure, chainable DataFrame transformations for machine learning preprocessing.
"""

from typing import Dict
import numpy as np
import pandas as pd


def clean_and_impute(df: pd.DataFrame, strategy: Dict[str, str]) -> pd.DataFrame:
    """
    Imputes missing values in `df` based on `strategy` without mutating `df` in-place.
    """
    out = df.copy()
    for col, method in strategy.items():
        if col not in out.columns:
            continue
        if method == "mean":
            fill_val = out[col].mean()
        elif method == "median":
            fill_val = out[col].median()
        elif method == "mode":
            modes = out[col].mode(dropna=True)
            fill_val = modes.iloc[0] if not modes.empty else np.nan
        elif method == "zero":
            fill_val = 0.0
        else:
            raise ValueError(f"Unsupported imputation strategy: {method}")
        out[col] = out[col].fillna(fill_val)
    return out


def encode_cyclical_features(df: pd.DataFrame, col: str, max_val: float) -> pd.DataFrame:
    """
    Encodes a periodic/cyclical feature into two continuous orthogonal circular coordinates
    using sine and cosine.
    """
    out = df.copy()
    angle = 2.0 * np.pi * out[col] / max_val
    out[f"{col}_sin"] = np.sin(angle)
    out[f"{col}_cos"] = np.cos(angle)
    return out


def compute_rolling_features(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    window: int
) -> pd.DataFrame:
    """
    Computes rolling mean and rolling standard deviation per group.
    """
    out = df.copy()
    grouped_rolling = out.groupby(group_col)[target_col].rolling(window=window)
    out[f"{target_col}_rolling_mean_{window}"] = (
        grouped_rolling.mean().reset_index(level=0, drop=True).fillna(0.0)
    )
    out[f"{target_col}_rolling_std_{window}"] = (
        grouped_rolling.std().reset_index(level=0, drop=True).fillna(0.0)
    )
    return out
