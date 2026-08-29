#!/usr/bin/env python3
"""
V3 修复方案1：移除所有截断样本，重新训练。

策略：过滤掉所有future_return_*d == 0.15的样本，
重新计算rank labels，重新训练模型。

如果结果仍然好 → 证明模型有真实预测能力
如果结果崩溃 → 证明之前的高收益来自标签泄漏
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

from v3_pipeline.src.ranking_labels import compute_all_ranking_labels

print("=" * 80)
print("V3 FIX: REMOVE CAPPED SAMPLES")
print("=" * 80)

# Load original cache
cache_path = REPO_ROOT / "feature_cache_all.parquet"
print(f"\n1. Loading cache from {cache_path}...")
df = pd.read_parquet(cache_path)
print(f"   Original: {len(df):,} rows")

# Identify capped samples for all horizons
horizons = ['3d', '5d', '10d', '15d', '20d', '25d', '30d']
capped_mask = pd.Series([False] * len(df), index=df.index)

print("\n2. Identifying capped samples...")
for horizon in horizons:
    col = f'future_return_{horizon}'
    capped = np.abs(df[col] - 0.15) < 0.001
    capped_count = capped.sum()
    print(f"   {horizon}: {capped_count:,} samples at 0.15 ({capped_count/len(df)*100:.2f}%)")
    capped_mask |= capped

total_capped = capped_mask.sum()
print(f"\n   Total samples with ANY horizon capped: {total_capped:,} ({total_capped/len(df)*100:.2f}%)")

# Remove capped samples
print("\n3. Removing capped samples...")
df_clean = df[~capped_mask].copy()
print(f"   Remaining: {len(df_clean):,} rows ({len(df_clean)/len(df)*100:.2f}% of original)")

# Verify no capped samples remain
print("\n4. Verification:")
for horizon in horizons:
    col = f'future_return_{horizon}'
    max_val = df_clean[col].max()
    at_15 = (np.abs(df_clean[col] - 0.15) < 0.001).sum()
    print(f"   {horizon}: max={max_val:.6f}, at_0.15={at_15}")

# Recompute rank labels on clean data
print("\n5. Recomputing rank labels on clean data...")
df_clean_v3 = compute_all_ranking_labels(df_clean, horizons=horizons, validate=True)

# Save cleaned cache
output_path = REPO_ROOT / "v3_pipeline/feature_cache_v3_no_cap.parquet"
print(f"\n6. Saving cleaned cache to {output_path}...")
df_clean_v3.to_parquet(output_path, index=False)

file_size = output_path.stat().st_size / 1024 / 1024
print(f"   Size: {file_size:.1f} MB")

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Original rows:     {len(df):,}")
print(f"Capped rows:       {total_capped:,} ({total_capped/len(df)*100:.2f}%)")
print(f"Clean rows:        {len(df_clean_v3):,} ({len(df_clean_v3)/len(df)*100:.2f}%)")
print(f"\nDate range:")
print(f"  Original: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"  Clean:    {df_clean_v3['timestamp'].min()} to {df_clean_v3['timestamp'].max()}")

print("\n✓ Clean cache created: feature_cache_v3_no_cap.parquet")
print("\nNext: Retrain models with clean cache")
print("  python v3_pipeline/scripts/train_ranking.py --cache v3_pipeline/feature_cache_v3_no_cap.parquet \\")
print("         --output v3_pipeline/models/v3_0_3_no_cap")
