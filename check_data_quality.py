#!/usr/bin/env python3
"""检查 v3 feature cache 的数据质量问题"""

import pandas as pd
import numpy as np

# 读取 v3 缓存
df = pd.read_parquet('v3_pipeline/feature_cache_v3.parquet')

print(f"Total rows: {len(df):,}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# 检查收益率列
ret_cols = [c for c in df.columns if 'future_return' in c and 'rank' not in c]

print("\n" + "="*80)
print("收益率列统计（检查是否被截断到 ±15%）")
print("="*80)

for col in ret_cols:
    data = df[col].dropna()
    print(f"\n{col}:")
    print(f"  min:     {data.min():.6f}")
    print(f"  max:     {data.max():.6f}")
    print(f"  mean:    {data.mean():.6f}")
    print(f"  std:     {data.std():.6f}")
    print(f"  > 0.15:  {(data > 0.15).sum():,} ({(data > 0.15).sum() / len(data) * 100:.2f}%)")
    print(f"  < -0.15: {(data < -0.15).sum():,} ({(data < -0.15).sum() / len(data) * 100:.2f}%)")
    print(f"  = 0.15:  {(data == 0.15).sum():,}")
    print(f"  = -0.15: {(data == -0.15).sum():,}")

# 检查是否真的被截断
print("\n" + "="*80)
print("结论")
print("="*80)

has_clipping = False
for col in ret_cols:
    data = df[col].dropna()
    if data.max() > 0.14 and (data > 0.15).sum() == 0 and (data == 0.15).sum() > 100:
        print(f"✗ {col} 被截断到 +15%（{(data == 0.15).sum():,} 个样本正好等于 0.15）")
        has_clipping = True
    if data.min() < -0.14 and (data < -0.15).sum() == 0 and (data == -0.15).sum() > 100:
        print(f"✗ {col} 被截断到 -15%（{(data == -0.15).sum():,} 个样本正好等于 -0.15）")
        has_clipping = True

if not has_clipping:
    print("✓ 收益率未被截断，数据质量 OK")
else:
    print("\n⚠️  数据被截断会导致：")
    print("   1. 排名指标（Rank IC）虚高 - 极端收益被压缩导致排名更容易正确")
    print("   2. 盈利预测失真 - 模型学不到真实收益的分布")
    print("   3. 回测失败 - 训练数据和实际交易的收益分布不一致")
