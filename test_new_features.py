#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试新增的流动性和微观结构特征
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from feature_pipeline_v2 import FeaturePipeline
from divergence_detector import DivergenceDetector

print("=" * 60)
print("测试新增特征生成")
print("=" * 60)

# 创建测试数据
dates = pd.date_range('2022-01-01', periods=100, freq='D')
test_data = pd.DataFrame({
    'timestamp': dates,
    'symbol': 'TEST.SZ',
    'open': 10.0 + np.random.randn(100) * 0.5,
    'high': 11.0 + np.random.randn(100) * 0.5,
    'low': 9.0 + np.random.randn(100) * 0.5,
    'close': 10.0 + np.random.randn(100) * 0.5,
    'volume': 1000000 + np.random.randint(-100000, 100000, 100)
})
test_data['close'] = test_data['close'].clip(lower=1.0)
test_data['high'] = test_data[['high', 'close']].max(axis=1)
test_data['low'] = test_data[['low', 'close']].min(axis=1)

print(f"测试数据: {len(test_data)} 行")

# 初始化pipeline
detector = DivergenceDetector()
full_stocks = {'TEST.SZ': test_data.set_index('timestamp')}
pipeline = FeaturePipeline(
    divergence_detector=detector,
    full_stocks_inner=full_stocks
)

# 计算基础技术特征
print("\n计算基础技术特征...")
result = pipeline._calculate_basic_technical_features(test_data.copy())
print(f"基础特征后列数: {len(result.columns)}")

# 计算高级技术特征（包含新特征）
print("\n计算高级技术特征（包含新增特征）...")
result = pipeline._calculate_advance_technical_features(result)
print(f"高级特征后列数: {len(result.columns)}")

# 检查新特征
new_features = [
    'amihud_illiq_intraday',
    'hl_spread',
    'effective_spread',
    'alpha12',
    'price_volume_divergence',
    'volume_momentum',
    'price_impact'
]

print("\n" + "=" * 60)
print("新增特征验证")
print("=" * 60)

all_present = True
for feat in new_features:
    if feat in result.columns:
        non_null_count = result[feat].notna().sum()
        print(f"✓ {feat}: {non_null_count}/{len(result)} 非空值")

        # 显示统计信息
        if non_null_count > 0:
            print(f"    min={result[feat].min():.6f}, max={result[feat].max():.6f}, mean={result[feat].mean():.6f}")
    else:
        print(f"✗ {feat}: 未找到")
        all_present = False

if all_present:
    print("\n✅ 所有新特征生成成功！")
else:
    print("\n❌ 部分特征缺失")
    sys.exit(1)

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
