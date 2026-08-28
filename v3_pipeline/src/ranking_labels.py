"""
Ranking labels computation for V3 pipeline.

Converts continuous returns into cross-sectional percentile ranks [0-1]
for LightGBM lambdarank training.
"""

import pandas as pd
import numpy as np
from typing import Optional


def compute_cross_sectional_ranks(
    df: pd.DataFrame,
    return_col: str,
    timestamp_col: str = 'timestamp'
) -> pd.Series:
    """
    Compute cross-sectional percentile ranks for each timestamp.

    Groups data by timestamp and ranks stocks by their return within each day.
    Returns percentile ranks in [0, 1] where:
    - 0 = lowest return that day
    - 1 = highest return that day
    - 0.5 = median (or single stock, or all tied)

    Args:
        df: DataFrame with timestamp and return columns
        return_col: Column name containing continuous returns
        timestamp_col: Column name containing timestamps (default: 'timestamp')

    Returns:
        Series of percentile ranks [0-1], same index as input df
        NaN returns are propagated as NaN ranks

    Examples:
        >>> df = pd.DataFrame({
        ...     'timestamp': ['2020-01-01', '2020-01-01', '2020-01-01'],
        ...     'return': [0.05, 0.10, 0.02]
        ... })
        >>> compute_cross_sectional_ranks(df, 'return')
        0    0.5
        1    1.0
        2    0.0
        dtype: float64
    """
    def rank_group(group: pd.Series) -> pd.Series:
        """Rank within a single timestamp group."""
        # Get integer ranks (1-based)
        # method='average' handles ties by assigning average rank
        int_ranks = group.rank(method='average', na_option='keep')

        # Normalize to [0, 1]
        # Special case: single stock gets 0.5
        n_valid = group.notna().sum()
        if n_valid <= 1:
            # Single stock or all NaN: assign 0.5 to non-NaN
            result = pd.Series(index=group.index, dtype=float)
            result[group.notna()] = 0.5
            result[group.isna()] = np.nan
            return result

        # Normalize: (rank - 1) / (n - 1) maps [1, n] to [0, 1]
        normalized = (int_ranks - 1) / (n_valid - 1)
        return normalized

    # Group by timestamp and apply ranking
    # This automatically propagates NaN (they don't affect ranking)
    ranks = df.groupby(timestamp_col)[return_col].transform(rank_group)

    return ranks


def validate_ranks(ranks: pd.Series, allow_nan: bool = True) -> None:
    """
    Validate that ranks are in [0, 1] range.

    Args:
        ranks: Series of computed ranks
        allow_nan: Whether NaN values are acceptable

    Raises:
        ValueError: If ranks are outside [0, 1] or contain unexpected NaNs
    """
    valid_ranks = ranks.dropna()

    if len(valid_ranks) == 0:
        raise ValueError("All ranks are NaN")

    if not allow_nan and ranks.isna().any():
        raise ValueError(f"Found {ranks.isna().sum()} NaN ranks")

    if valid_ranks.min() < 0 or valid_ranks.max() > 1:
        raise ValueError(
            f"Ranks outside [0, 1]: min={valid_ranks.min():.4f}, max={valid_ranks.max():.4f}"
        )


def compute_all_ranking_labels(
    df: pd.DataFrame,
    horizons: list[str] = ['3d', '5d', '10d', '15d', '20d', '25d', '30d'],
    timestamp_col: str = 'timestamp',
    validate: bool = True
) -> pd.DataFrame:
    """
    Compute ranking labels for all return horizons.

    Args:
        df: DataFrame with timestamp and future_return_* columns
        horizons: List of horizon suffixes (e.g., ['3d', '5d', '10d'])
        timestamp_col: Column name containing timestamps
        validate: Whether to validate rank ranges

    Returns:
        DataFrame with new rank_future_return_* columns added
    """
    df = df.copy()

    for horizon in horizons:
        return_col = f'future_return_{horizon}'
        rank_col = f'rank_future_return_{horizon}'

        if return_col not in df.columns:
            raise ValueError(f"Column {return_col} not found in dataframe")

        print(f"Computing ranks for {horizon}...")
        ranks = compute_cross_sectional_ranks(df, return_col, timestamp_col)

        if validate:
            validate_ranks(ranks, allow_nan=True)

        df[rank_col] = ranks

        # Print summary statistics
        null_pct = ranks.isna().sum() / len(ranks) * 100
        valid_ranks = ranks.dropna()
        if len(valid_ranks) > 0:
            print(f"  {rank_col}: "
                  f"min={valid_ranks.min():.4f}, "
                  f"mean={valid_ranks.mean():.4f}, "
                  f"max={valid_ranks.max():.4f}, "
                  f"null={null_pct:.2f}%")
        else:
            print(f"  {rank_col}: all NaN")

    return df
