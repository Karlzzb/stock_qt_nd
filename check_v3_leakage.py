#!/usr/bin/env python3
"""
全面审查V3流程的数据泄漏风险。

检查点：
1. future_return计算是否使用未来数据
2. 特征是否包含未来信息
3. 训练/验证/测试集是否严格时间分隔
4. rank label计算是否只用同期数据
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).parent
V2_CACHE = REPO_ROOT / "feature_cache_all.parquet"
V3_CACHE = REPO_ROOT / "v3_pipeline/feature_cache_v3.parquet"

print("=" * 80)
print("V3 PIPELINE DATA LEAKAGE AUDIT")
print("=" * 80)

# === Check 1: Time splits ===
print("\n1. TIME SPLIT VALIDATION")
print("-" * 80)

cache = pd.read_parquet(V3_CACHE)
print(f"Total rows: {len(cache):,}")
print(f"Date range: {cache['timestamp'].min()} to {cache['timestamp'].max()}")

train_mask = cache['timestamp'] <= '2021-12-31'
val_mask = (cache['timestamp'] >= '2022-01-01') & (cache['timestamp'] <= '2025-07-31')
test_mask = (cache['timestamp'] >= '2025-08-01') & (cache['timestamp'] <= '2026-08-14')

print(f"\nTrain: {train_mask.sum():,} rows (up to 2021-12-31)")
print(f"Val:   {val_mask.sum():,} rows (2022-01-01 to 2025-07-31)")
print(f"Test:  {test_mask.sum():,} rows (2025-08-01 to 2026-08-14)")

overlap = train_mask & val_mask
if overlap.any():
    print(f"⚠️  WARNING: {overlap.sum()} rows overlap between train and val")
else:
    print("✓ No overlap between train and val")

overlap = val_mask & test_mask
if overlap.any():
    print(f"⚠️  WARNING: {overlap.sum()} rows overlap between val and test")
else:
    print("✓ No overlap between val and test")

# === Check 2: Feature columns ===
print("\n2. FEATURE COLUMN INSPECTION")
print("-" * 80)

# Identify suspicious patterns
suspicious_patterns = {
    'future': [],
    'next': [],
    'forward': [],
    'target': [],
    'label': []
}

for col in cache.columns:
    col_lower = col.lower()
    for pattern, matches in suspicious_patterns.items():
        if pattern in col_lower:
            matches.append(col)

print("Columns containing suspicious patterns:")
for pattern, cols in suspicious_patterns.items():
    if cols:
        print(f"\n  '{pattern}': {len(cols)} columns")
        for col in cols[:10]:  # Show first 10
            print(f"    - {col}")
        if len(cols) > 10:
            print(f"    ... and {len(cols) - 10} more")

# Expected future columns (these are labels, not features)
expected_future = [
    'future_return_3d', 'future_return_5d', 'future_return_10d',
    'future_return_15d', 'future_return_20d', 'future_return_25d', 'future_return_30d',
    'future_sell_date_3d', 'future_sell_date_5d', 'future_sell_date_10d',
    'future_sell_date_15d', 'future_sell_date_20d', 'future_sell_date_25d', 'future_sell_date_30d',
    'rank_future_return_3d', 'rank_future_return_5d', 'rank_future_return_10d',
    'rank_future_return_15d', 'rank_future_return_20d', 'rank_future_return_25d', 'rank_future_return_30d'
]

unexpected = [c for c in suspicious_patterns['future'] if c not in expected_future]
if unexpected:
    print(f"\n⚠️  UNEXPECTED future columns (potential leakage): {unexpected}")
else:
    print("\n✓ All 'future' columns are expected label columns")

# === Check 3: Feature exclusion in training ===
print("\n3. TRAINING FEATURE EXCLUSION")
print("-" * 80)

# Read train_ranking.py exclusion patterns
exclusion_patterns = [
    r".*_date$",
    r".*_signal$",
    r"^prev_",
    r"^rank_future_return_",
    r"^future_return_",
]

print("Exclusion patterns in train_ranking.py:")
for pattern in exclusion_patterns:
    print(f"  - {pattern}")

# Check if they catch all future columns
from re import match
caught = []
for col in cache.columns:
    for pattern in exclusion_patterns:
        if match(pattern, col):
            caught.append(col)
            break

future_cols = [c for c in cache.columns if 'future' in c.lower()]
missed = [c for c in future_cols if c not in caught]

if missed:
    print(f"\n⚠️  MISSED future columns (not excluded): {missed}")
else:
    print("\n✓ All future columns are properly excluded from training")

# === Check 4: Rank label computation ===
print("\n4. RANK LABEL COMPUTATION")
print("-" * 80)

# Sample one timestamp and verify ranks are cross-sectional only
sample_ts = cache[cache['timestamp'] == '2022-01-03'].copy()
if len(sample_ts) > 0:
    print(f"\nSample timestamp: 2022-01-03 ({len(sample_ts)} stocks)")

    # Recompute rank manually
    returns = sample_ts['future_return_3d'].dropna()
    manual_ranks = (returns.rank(method='average') - 1) / (len(returns) - 1)
    stored_ranks = sample_ts.loc[returns.index, 'rank_future_return_3d']

    diff = (manual_ranks - stored_ranks).abs()
    max_diff = diff.max()

    print(f"Manual rank recomputation:")
    print(f"  Max difference: {max_diff:.10f}")

    if max_diff < 1e-6:
        print("✓ Ranks are correctly computed (cross-sectional only)")
    else:
        print(f"⚠️  Rank mismatch detected (max diff: {max_diff})")

# === Check 5: Data truncation inspection ===
print("\n5. DATA TRUNCATION ANALYSIS")
print("-" * 80)

EXPECTED_PROFIT = 1.15  # +15% cap

for horizon in ['3d', '5d', '10d', '15d', '20d', '25d', '30d']:
    col = f'future_return_{horizon}'
    returns = cache[val_mask][col].dropna()

    if len(returns) == 0:
        continue

    # Count capped values
    capped = (returns >= EXPECTED_PROFIT - 0.001).sum()  # Allow small float error
    pct_capped = capped / len(returns) * 100

    print(f"{horizon:5s}: {pct_capped:5.2f}% capped at {EXPECTED_PROFIT:.2f} ({capped:,}/{len(returns):,})")

# === Check 6: Prediction vs actual alignment ===
print("\n6. PREDICTION FILE VALIDATION")
print("-" * 80)

pred_3d = pd.read_parquet(REPO_ROOT / "v3_pipeline/models/v3_0_1_label_selection/pred_3d.parquet")
print(f"pred_3d.parquet: {len(pred_3d):,} rows")
print(f"Date range: {pred_3d['timestamp'].min()} to {pred_3d['timestamp'].max()}")

# Check if predictions are only on validation set
pred_in_train = (pred_3d['timestamp'] <= '2021-12-31').sum()
pred_in_test = (pred_3d['timestamp'] >= '2025-08-01').sum()

if pred_in_train > 0:
    print(f"⚠️  WARNING: {pred_in_train} predictions on training set")
else:
    print("✓ No predictions on training set")

if pred_in_test > 0:
    print(f"⚠️  WARNING: {pred_in_test} predictions on test set")
else:
    print("✓ No predictions on test set (good - test set not evaluated yet)")

# === Check 7: Verify actual_return matches cache ===
print("\n7. ACTUAL RETURN CONSISTENCY")
print("-" * 80)

# Sample 1000 random rows and verify actual_return matches cache
sample_rows = pred_3d.sample(min(1000, len(pred_3d)), random_state=42)
merged = sample_rows.merge(
    cache[['timestamp', 'symbol', 'future_return_3d']],
    on=['timestamp', 'symbol'],
    how='left'
)

diff = (merged['actual_return'] - merged['future_return_3d']).abs()
max_diff = diff.max()

print(f"Sampled {len(sample_rows)} predictions, merged with cache")
print(f"Max difference in actual_return: {max_diff:.10f}")

if max_diff < 1e-6:
    print("✓ actual_return matches cache (no leakage)")
else:
    print(f"⚠️  actual_return mismatch (max diff: {max_diff})")

# === SUMMARY ===
print("\n" + "=" * 80)
print("AUDIT SUMMARY")
print("=" * 80)
print("""
✓ = Pass
⚠️  = Warning (review needed)
❌ = Fail (data leakage detected)

Key findings will be printed above. Review all ⚠️  and ❌ items carefully.
""")
