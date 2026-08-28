"""
Tests for ranking labels computation.

Run with: python -m pytest v3_pipeline/tests/test_ranking_labels.py -v
"""

import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from v3_pipeline.src.ranking_labels import (
    compute_cross_sectional_ranks,
    validate_ranks,
    compute_all_ranking_labels
)


class TestComputeCrossSectionalRanks:
    """Test cross-sectional ranking computation."""

    def test_basic_ranking(self):
        """Test basic ranking with three stocks."""
        df = pd.DataFrame({
            'timestamp': ['2020-01-01', '2020-01-01', '2020-01-01'],
            'return': [0.02, 0.10, 0.05]
        })
        ranks = compute_cross_sectional_ranks(df, 'return')

        # Lowest return (0.02) → 0.0
        # Middle return (0.05) → 0.5
        # Highest return (0.10) → 1.0
        expected = pd.Series([0.0, 1.0, 0.5])
        pd.testing.assert_series_equal(ranks, expected, check_names=False)

    def test_single_stock_per_day(self):
        """Single stock on a date gets rank 0.5."""
        df = pd.DataFrame({
            'timestamp': ['2020-01-01'],
            'return': [0.05]
        })
        ranks = compute_cross_sectional_ranks(df, 'return')

        assert ranks.iloc[0] == 0.5

    def test_all_tied_returns(self):
        """All stocks with same return get rank 0.5."""
        df = pd.DataFrame({
            'timestamp': ['2020-01-01', '2020-01-01', '2020-01-01'],
            'return': [0.05, 0.05, 0.05]
        })
        ranks = compute_cross_sectional_ranks(df, 'return')

        # method='average' assigns average rank to ties
        # Ranks would be [0, 0.5, 1.0], average = 0.5
        assert all(ranks == 0.5)

    def test_nan_propagation(self):
        """NaN returns are propagated as NaN ranks."""
        df = pd.DataFrame({
            'timestamp': ['2020-01-01', '2020-01-01', '2020-01-01'],
            'return': [0.02, np.nan, 0.10]
        })
        ranks = compute_cross_sectional_ranks(df, 'return')

        # Stock with NaN return should have NaN rank
        assert pd.isna(ranks.iloc[1])

        # Other stocks ranked normally (0.0 and 1.0)
        assert ranks.iloc[0] == 0.0
        assert ranks.iloc[2] == 1.0

    def test_multiple_dates(self):
        """Ranks are computed independently per timestamp."""
        df = pd.DataFrame({
            'timestamp': ['2020-01-01', '2020-01-01', '2020-01-02', '2020-01-02'],
            'return': [0.05, 0.10, 0.02, 0.08]
        })
        ranks = compute_cross_sectional_ranks(df, 'return')

        # Day 1: 0.05 → 0.0, 0.10 → 1.0
        assert ranks.iloc[0] == 0.0
        assert ranks.iloc[1] == 1.0

        # Day 2: 0.02 → 0.0, 0.08 → 1.0
        assert ranks.iloc[2] == 0.0
        assert ranks.iloc[3] == 1.0

    def test_higher_return_higher_rank(self):
        """Validate that higher returns always get higher ranks within each day."""
        df = pd.DataFrame({
            'timestamp': ['2020-01-01'] * 5,
            'return': [0.01, 0.03, 0.02, 0.05, 0.04]
        })
        ranks = compute_cross_sectional_ranks(df, 'return')

        # Check that rank order matches return order
        return_order = df['return'].argsort()
        rank_order = ranks.argsort()
        pd.testing.assert_series_equal(return_order, rank_order, check_names=False)


class TestValidateRanks:
    """Test rank validation."""

    def test_valid_ranks(self):
        """Valid ranks pass without error."""
        ranks = pd.Series([0.0, 0.25, 0.5, 0.75, 1.0])
        validate_ranks(ranks, allow_nan=False)  # Should not raise

    def test_ranks_with_nan(self):
        """NaN ranks pass when allow_nan=True."""
        ranks = pd.Series([0.0, np.nan, 0.5, np.nan, 1.0])
        validate_ranks(ranks, allow_nan=True)  # Should not raise

    def test_ranks_with_nan_disallowed(self):
        """NaN ranks fail when allow_nan=False."""
        ranks = pd.Series([0.0, np.nan, 0.5])
        with pytest.raises(ValueError, match="NaN ranks"):
            validate_ranks(ranks, allow_nan=False)

    def test_ranks_below_zero(self):
        """Ranks below 0 fail validation."""
        ranks = pd.Series([-0.1, 0.5, 1.0])
        with pytest.raises(ValueError, match="outside"):
            validate_ranks(ranks)

    def test_ranks_above_one(self):
        """Ranks above 1 fail validation."""
        ranks = pd.Series([0.0, 0.5, 1.1])
        with pytest.raises(ValueError, match="outside"):
            validate_ranks(ranks)

    def test_all_nan(self):
        """All NaN ranks fail validation."""
        ranks = pd.Series([np.nan, np.nan, np.nan])
        with pytest.raises(ValueError, match="All ranks are NaN"):
            validate_ranks(ranks)


class TestComputeAllRankingLabels:
    """Test batch ranking label computation."""

    def test_compute_all_labels(self):
        """Test computing labels for all horizons."""
        df = pd.DataFrame({
            'timestamp': ['2020-01-01', '2020-01-01', '2020-01-01'],
            'future_return_3d': [0.02, 0.10, 0.05],
            'future_return_5d': [0.01, 0.08, 0.03],
            'future_return_10d': [0.04, 0.12, 0.07]
        })

        df_result = compute_all_ranking_labels(
            df,
            horizons=['3d', '5d', '10d'],
            validate=True
        )

        # Check new columns were added
        assert 'rank_future_return_3d' in df_result.columns
        assert 'rank_future_return_5d' in df_result.columns
        assert 'rank_future_return_10d' in df_result.columns

        # Check 3d ranks
        assert df_result['rank_future_return_3d'].iloc[0] == 0.0
        assert df_result['rank_future_return_3d'].iloc[1] == 1.0
        assert df_result['rank_future_return_3d'].iloc[2] == 0.5

    def test_missing_return_column(self):
        """Test error when return column is missing."""
        df = pd.DataFrame({
            'timestamp': ['2020-01-01'],
            'future_return_3d': [0.05]
        })

        with pytest.raises(ValueError, match="not found"):
            compute_all_ranking_labels(df, horizons=['5d'])

    def test_original_columns_preserved(self):
        """Test that original columns are preserved."""
        df = pd.DataFrame({
            'timestamp': ['2020-01-01', '2020-01-01'],
            'stock_id': ['A', 'B'],
            'future_return_3d': [0.02, 0.10]
        })

        df_result = compute_all_ranking_labels(df, horizons=['3d'])

        # Original columns should still exist
        assert 'stock_id' in df_result.columns
        assert 'future_return_3d' in df_result.columns
        pd.testing.assert_series_equal(
            df['stock_id'],
            df_result['stock_id'],
            check_names=False
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
