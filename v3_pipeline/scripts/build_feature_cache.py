#!/usr/bin/env python3
"""
Build V3 feature cache with ranking labels.

Loads v2's feature_cache_all.parquet, computes cross-sectional rank labels
for all return horizons, and saves to v3_pipeline/feature_cache_v3.parquet.

Usage:
    python v3_pipeline/scripts/build_feature_cache.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from v3_pipeline.src.ranking_labels import compute_all_ranking_labels


def main():
    """Build V3 feature cache with ranking labels."""
    # Paths
    v2_cache_path = project_root / "feature_cache_all.parquet"
    v3_cache_path = project_root / "v3_pipeline" / "feature_cache_v3.parquet"

    print(f"Loading v2 feature cache from {v2_cache_path}...")
    if not v2_cache_path.exists():
        raise FileNotFoundError(f"V2 cache not found: {v2_cache_path}")

    df = pd.read_parquet(v2_cache_path)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Unique dates: {df['timestamp'].nunique():,}")

    # Check for required return columns
    horizons = ['3d', '5d', '10d', '15d', '20d', '25d', '30d']
    return_cols = [f'future_return_{h}' for h in horizons]
    missing = [col for col in return_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing return columns: {missing}")

    print(f"\nComputing cross-sectional ranking labels for {len(horizons)} horizons...")
    df_v3 = compute_all_ranking_labels(df, horizons=horizons, validate=True)

    # Verify new columns were added
    rank_cols = [f'rank_future_return_{h}' for h in horizons]
    added_cols = [col for col in rank_cols if col in df_v3.columns]
    print(f"\nAdded {len(added_cols)} ranking label columns:")
    for col in added_cols:
        print(f"  - {col}")

    # Save v3 cache
    print(f"\nSaving v3 cache to {v3_cache_path}...")
    v3_cache_path.parent.mkdir(parents=True, exist_ok=True)
    df_v3.to_parquet(v3_cache_path, index=False)

    # Print final summary
    print(f"\n✓ V3 cache created successfully")
    print(f"  Rows: {len(df_v3):,}")
    print(f"  Columns: {len(df_v3.columns)} (original: {len(df.columns)}, added: {len(df_v3.columns) - len(df.columns)})")
    print(f"  Size: {v3_cache_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Sample rank distribution
    print(f"\nSample rank distributions (first 1000 valid ranks per horizon):")
    for horizon in horizons:
        rank_col = f'rank_future_return_{horizon}'
        sample = df_v3[rank_col].dropna().head(1000)
        if len(sample) > 0:
            print(f"  {horizon}: "
                  f"p10={sample.quantile(0.1):.3f}, "
                  f"p50={sample.quantile(0.5):.3f}, "
                  f"p90={sample.quantile(0.9):.3f}")


if __name__ == '__main__':
    main()
