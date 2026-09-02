#!/usr/bin/env python3
"""
Regression test for issue #12: _get_default_combined_features must return DataFrame.

When _calculate_market_features catches an exception, it calls _get_default_combined_features.
This must return a DataFrame (not a dict) so the merge at line 468 succeeds.
"""
import sys
import pandas as pd

sys.path.insert(0, '/home/karl/repos/personal/stock_qt_nd/src')

def test_get_default_combined_features_returns_dataframe():
    """Test that _get_default_combined_features returns a DataFrame with timestamp index."""
    from feature_pipeline_v2 import FeaturePipeline

    # Create a minimal pipeline instance (needs mocks for required args)
    class MockDetector:
        pass

    pipeline = FeaturePipeline(MockDetector(), {})

    timestamp = pd.Timestamp('2010-01-04')
    result = pipeline._get_default_combined_features(timestamp)

    # Must be a DataFrame
    assert isinstance(result, pd.DataFrame), f"Expected DataFrame, got {type(result)}"

    # Must have timestamp as index
    assert result.index.name == 'timestamp', f"Expected index name 'timestamp', got {result.index.name}"
    assert timestamp in result.index, f"Expected {timestamp} in index"

    # Must be mergeable with a target DataFrame
    target_df = pd.DataFrame({
        'timestamp': [timestamp],
        'symbol': ['test'],
        'value': [42]
    })

    merged = target_df.merge(result, left_on='timestamp', right_index=True, how='left')
    assert len(merged) == 1, f"Expected 1 row after merge, got {len(merged)}"
    assert 'market_sentiment' in merged.columns, "Expected market features in merged result"


if __name__ == '__main__':
    try:
        test_get_default_combined_features_returns_dataframe()
        print("PASS")
        sys.exit(0)
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
