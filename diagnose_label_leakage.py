#!/usr/bin/env python3
"""
诊断标签泄漏：截断的return导致rank label泄漏。

问题：当future_return被截断到0.15时，rank label仍然基于截断后的值计算。
这导致所有截断的股票获得相同的rank（接近1.0），模型学会识别这个模式。

验证：检查截断股票的rank分布。
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("LABEL LEAKAGE DIAGNOSIS")
print("=" * 80)

# Load cache
cache = pd.read_parquet('v3_pipeline/feature_cache_v3.parquet')

# Focus on validation period
val_mask = (cache['timestamp'] >= '2022-01-01') & (cache['timestamp'] <= '2025-07-31')
val_df = cache[val_mask].copy()

print(f"\nValidation set: {len(val_df):,} rows")

# Identify capped stocks
val_df['is_capped'] = (np.abs(val_df['future_return_3d'] - 0.15) < 0.001)

print(f"Capped stocks: {val_df['is_capped'].sum():,} ({val_df['is_capped'].mean():.2%})")

# Check rank distribution for capped vs non-capped
capped_ranks = val_df[val_df['is_capped']]['rank_future_return_3d'].dropna()
non_capped_ranks = val_df[~val_df['is_capped']]['rank_future_return_3d'].dropna()

print("\n" + "-" * 80)
print("RANK DISTRIBUTION: Capped stocks (return == 0.15)")
print("-" * 80)
print(capped_ranks.describe())

print("\n" + "-" * 80)
print("RANK DISTRIBUTION: Non-capped stocks (return < 0.15)")
print("-" * 80)
print(non_capped_ranks.describe())

print("\n" + "=" * 80)
print("KEY FINDING")
print("=" * 80)

# Sample one period to show the issue
sample_ts = '2022-01-04'
sample = val_df[val_df['timestamp'] == sample_ts].copy()
sample = sample.sort_values('future_return_3d', ascending=False)

print(f"\nSample period: {sample_ts}")
print(f"Total stocks: {len(sample)}")

capped_stocks = sample[sample['is_capped']]
print(f"\nCapped stocks ({len(capped_stocks)}):")
print(capped_stocks[['symbol', 'future_return_3d', 'rank_future_return_3d']].head(10))

print("\nTop 10 by rank:")
print(sample[['symbol', 'future_return_3d', 'rank_future_return_3d', 'is_capped']].head(10))

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("""
If capped stocks have rank ≈ 1.0:
  ❌ LABEL LEAKAGE: Model learns to identify capped stocks, not predict returns

If capped stocks have diverse ranks:
  ✓ NO LEAKAGE: Ranks are computed correctly from truncated returns

The issue is:
  1. Real return is 20%, 30%, or even 100%
  2. Truncated to 15%
  3. All these stocks get similar high rank (tied at 0.15)
  4. Model learns: "These features → will be capped → rank=1.0"
  5. This is NOT return prediction, it's capped-stock detection

Solution: Remove truncation or add noise to break ties.
""")
